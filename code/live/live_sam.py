"""
live_sam.py (2026-11-10): SAM 3.1 text-prompt tracking one frame at a time, for the live SO-101 pipeline (run_live.py).

The offline lift (lift_so101.Segmenter, track_both_ends) gives SAM 3.1 a whole recorded video. Three parts of that pass
read frames after frame t, and none of them exists on the robot:
  hotstart_delay 15     the masks of frame t are held back until frame t + 14 (tracks unmatched or duplicated in their
                        first 15 frames are removed before they are shown)
  batched grounding     the detector runs on 16 frames at once (frame t is detected together with t + 1 .. t + 15)
  backward pass         when the forward pass misses an object in more than 2% of the frames, a second pass prompted
                        at the last frame
Here each session (one per object phrase, as offline) runs the same per-frame step of the base propagation
(Sam3MultiplexTracking.propagate_in_video: _run_single_frame_inference, _cache_frame_outputs, _postprocess_output) on
the newest frame only, so detector and tracker state carry over from frame to frame as in one long forward pass, with
hotstart_delay 0 and a grounding batch of 1. The frames live in a LiveFrames video that grows as the camera delivers
them, preprocessed exactly as sam3.model.io_utils.load_resource_as_video_frames does for in-memory frames.
SAM's bf16 autocast and TF32 matmuls are scoped to its own calls: the SAM 3.1 predictor switches both on for the whole
thread when it is built, which would change the numerics of the world model and the policy (evaluated in fp32).
"""
import contextlib
import os
import time
import uuid

import numpy as np
import torch
from PIL import Image


class LiveFrames:
    """A SAM 3.1 video that grows during the episode. Frame i is pushed before the tracker reads it; reading a frame
    that was not pushed yet raises (it would mean the model looks ahead)."""

    def __init__(self, n_max, image_size=1008, offload=True):
        self.n_max, self.size, self.offload = int(n_max), int(image_size), offload
        self.frames = [None] * self.n_max
        self.mean = torch.tensor((0.5, 0.5, 0.5), dtype=torch.float16)[:, None, None]
        self.std = torch.tensor((0.5, 0.5, 0.5), dtype=torch.float16)[:, None, None]
        self.n, self.hw = 0, None

    def push(self, rgb):
        """rgb (H, W, 3) uint8 -> its frame index. Same arithmetic as the list-of-PIL path of
        sam3.model.io_utils.load_resource_as_video_frames (PIL resize, / 255, float16, (x - 0.5) / 0.5)."""
        if self.n >= self.n_max:
            raise RuntimeError(f"LiveFrames full ({self.n_max} frames): raise --max_frames")
        if self.hw is None:
            self.hw = tuple(rgb.shape[:2])
        img_np = np.array(Image.fromarray(rgb).convert("RGB").resize((self.size, self.size)))
        img_np = img_np / 255.0
        img = torch.from_numpy(img_np).permute(2, 0, 1).to(dtype=torch.float16)
        img -= self.mean
        img /= self.std
        self.frames[self.n] = img if self.offload else img.cuda()
        self.n += 1
        return self.n - 1

    def __len__(self):
        return self.n_max

    def _one(self, i):
        i = int(i)
        x = self.frames[i] if 0 <= i < self.n_max else None
        if x is None:
            raise IndexError(f"SAM read frame {i}, but only frames 0..{self.n - 1} exist (look-ahead)")
        return x.cuda(non_blocking=True)

    def __getitem__(self, i):
        if isinstance(i, torch.Tensor):
            i = i.tolist()
        if isinstance(i, np.ndarray):
            i = i.tolist()
        if isinstance(i, slice):
            i = list(range(*i.indices(self.n_max)))
        if isinstance(i, (list, tuple)):
            return torch.stack([self._one(j) for j in i])
        return self._one(i)


def _patch_loader():
    """init_state loads its video through load_resource_as_video_frames: pass a LiveFrames through unchanged."""
    import sam3.model.sam3_multiplex_tracking as smt
    if getattr(smt, "_live_frames_patch", False):
        return
    orig = smt.load_resource_as_video_frames

    def load(resource_path, **kw):
        if isinstance(resource_path, LiveFrames):
            h, w = resource_path.hw
            return resource_path, h, w
        return orig(resource_path=resource_path, **kw)

    smt.load_resource_as_video_frames = load
    if hasattr(smt, "is_image_type"):
        orig_is_image = smt.is_image_type
        smt.is_image_type = lambda r: False if isinstance(r, LiveFrames) else orig_is_image(r)
    # the non-batched detector path (add_prompt) pre-computes the detection of the NEXT frame to overlap it with
    # multi-GPU transfers; live, that frame does not exist yet. Skip the prefetch: the detector computes a frame that
    # is missing from its buffer when that frame is processed (forward_video_grounding_multigpu, step 1)
    from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
    orig_next = Sam3MultiplexDetector._build_multigpu_buffer_next_chunk

    def next_chunk(self, backbone_out, find_inputs, geometric_prompt, frame_idx_begin, frame_idx_end, *a, **kw):
        lf = getattr(backbone_out.get("img_batch_all_stages"), "tensors", None)
        if isinstance(lf, LiveFrames) and frame_idx_begin >= lf.n:
            return None
        return orig_next(self, backbone_out, find_inputs, geometric_prompt, frame_idx_begin, frame_idx_end, *a, **kw)

    Sam3MultiplexDetector._build_multigpu_buffer_next_chunk = next_chunk
    smt._live_frames_patch = True


class LiveSegmenter:
    """One SAM 3.1 session per object phrase over a shared LiveFrames video.
      start(rgb0)  new episode: frame 0 and the text prompts
      step(rgb)    the next frame (call start() first; start() already consumed frame 0's image)
    Both return, per phrase, {obj_id: (mask (H, W) bool, prob)} of that frame, the format of lift_so101.Segmenter.track."""

    def __init__(self, phrases, ckpt=None, max_frames=1800, offload=True, det_thresh=None, keep_frames=None,
                 share_backbone=True, new_det_thresh=None, trim_keep=32):
        from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTracking
        from sam3.model_builder import build_sam3_predictor
        _patch_loader()
        tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        self.p = build_sam3_predictor(checkpoint_path=ckpt, version="sam3.1", use_fa3=False, compile=False)
        # the predictor's constructor entered a thread-wide bf16 autocast and set TF32 globally: undo both here and
        # apply them only around SAM's own calls (self._ctx)
        self.p.bf16_context.__exit__(None, None, None)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
        m = self.p.model
        m.hotstart_delay, m.hotstart_unmatch_thresh, m.hotstart_dup_thresh = 0, 0, 0
        m.postprocess_batch_size = 1
        m.batched_grounding_batch_size = 1
        th = det_thresh if det_thresh is not None else os.environ.get("SAM_DET_THRESH")
        if th:   # as lift_so101.Segmenter: lower detection / new-track thresholds (default off)
            for a_ in ("score_threshold_detection", "new_det_thresh"):
                if hasattr(m, a_):
                    setattr(m, a_, float(th))
        if new_det_thresh is not None:
            # only the threshold for starting a new track (default 0.65): live has no backward pass, which offline
            # found objects hidden at the start with (the red cube in the pot)
            m.new_det_thresh = float(new_det_thresh)
        self._base_cls = Sam3MultiplexTracking
        # share_backbone: every session runs the image encoder (detector.backbone.forward_image) on the same frame;
        # compute it once per frame and hand the same output to the other sessions (same input, same result)
        self.share_backbone, self._bb = share_backbone, None
        if share_backbone:
            import weakref
            bb = m.detector.backbone
            orig_fwd = bb.forward_image
            me = weakref.ref(self)    # no reference cycle model -> segmenter: a closed segmenter is freed at once

            def forward_image(image, *a, **kw):
                s = me()
                memo = s._bb if s is not None else None
                if memo is not None and not a and not kw and memo[0] is not None \
                        and image.shape == memo[0].shape and torch.equal(image, memo[0]):
                    return memo[1]
                out = orig_fwd(image, *a, **kw)
                if memo is not None and not a and not kw:
                    s._bb = (image, out)
                return out

            bb.forward_image = forward_image
        self.phrases = list(phrases)
        self.max_frames, self.offload, self.keep_frames, self.trim_keep = max_frames, offload, keep_frames, trim_keep
        self.frames, self.sids, self.t = None, [], -1

    @contextlib.contextmanager
    def _ctx(self):
        old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.inference_mode():
                yield
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old

    def close(self):
        for sid in self.sids:
            try:
                self.p.handle_request(dict(type="close_session", session_id=sid))
            except Exception:  # noqa: BLE001
                pass
        self.sids, self.frames, self.t = [], None, -1
        torch.cuda.empty_cache()

    def start(self, rgb0):
        self.close()
        self.frames = LiveFrames(self.max_frames, self.p.model.image_size, self.offload)
        self.frames.push(rgb0)
        for phrase in self.phrases:
            with self._ctx():
                st = self.p.model.init_state(resource_path=self.frames, offload_video_to_cpu=self.offload,
                                             async_loading_frames=False)
            sid = str(uuid.uuid4())
            self.p._all_inference_states[sid] = dict(state=st, session_id=sid, start_time=time.time(),
                                                      last_use_time=time.time())
            self.sids.append(sid)
            # the predictor's add_prompt (its own bf16 autocast), as offline
            self.p.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=phrase))
        self.t = 0
        return self._run(0)

    def step(self, rgb):
        t = self.frames.push(rgb)
        assert t == self.t + 1
        self.t = t
        return self._run(t)

    def _run(self, t):
        """One frame of the base forward propagation in every session (Sam3MultiplexTracking.propagate_in_video with
        hotstart_delay 0: the frame's output is post-processed and returned at once)."""
        m, res = self.p.model, []
        if self.share_backbone:
            self._bb = (None, None)          # this frame's encoder output, filled by the first session
        for sid in self.sids:
            st = self.p._all_inference_states[sid]["state"]
            with self._ctx():
                m._compile_model()
                # the offline pass ran with max_frame_num_to_track None: the detector's valid range is the whole video
                st["feature_cache"]["tracking_bounds"] = {"max_frame_num_to_track": None,
                                                          "propagate_in_video_start_frame_idx": 0}
                out = m._run_single_frame_inference(st, t, False, is_instance_processing=False)
                sup = out["suppressed_obj_ids"]
                m._cache_frame_outputs(st, t, out["obj_id_to_mask"], suppressed_obj_ids=sup, removed_obj_ids=set(),
                                       unconfirmed_obj_ids=None)
                o = m._postprocess_output(st, out, set(), sup, None)
            st.get("cached_frame_outputs", {}).pop(t, None)   # as the predictor's evict_cached_frame_outputs
            d = {}
            probs = o.get("out_probs", np.ones(len(o["out_obj_ids"])))
            for i, oid in enumerate(np.asarray(o["out_obj_ids"]).tolist()):
                mk = np.asarray(o["out_binary_masks"][i]).astype(bool)
                if mk.any():
                    d[oid] = (mk, float(np.asarray(probs)[i]))
            res.append(d)
        self._bb = None
        if self.keep_frames and t - self.keep_frames >= 1:   # optional: free old frame tensors (memory)
            self.frames.frames[t - self.keep_frames] = None
        if self.trim_keep:
            self._trim(t)
        return res

    def _trim(self, t):
        """Free the tracker outputs of non-conditioning frames older than trim_keep frames. The tracker reads at most
        max(num_maskmem - 1, max_obj_ptrs_in_encoder - 1) = 15 frames back (memory features and object pointers); the
        outputs of every older frame otherwise stay on the GPU (~12 MB per frame for three phrases), so a 600-step
        episode would add ~7 GB. Conditioning frames (where a track started or was re-conditioned) are kept."""
        old = t - self.trim_keep
        if old < 0:
            return
        for sid in self.sids:
            st = self.p._all_inference_states[sid]["state"]
            for s2 in st.get("sam2_inference_states", []):
                dicts = [s2.get("output_dict")] + list(s2.get("output_dict_per_obj", {}).values())
                for d in dicts:
                    nc = d.get("non_cond_frame_outputs") if isinstance(d, dict) else None
                    if nc:
                        for f in [f for f in nc if f <= old]:
                            del nc[f]
