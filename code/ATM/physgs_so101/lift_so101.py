"""
lift_so101.py (2026-11-06): SO-101 LeRobot episodes -> PhysGO episode files (physgs.data.format), the same per-object
Gaussian format the joint world model was pre-trained / adapted on, so physgs.data.format.pack and train_joint run
unchanged. Separate version for the SO-101 work; nothing in physgs is modified.

Per episode (15 fps = every 2nd 30 fps frame, the PhysicalAI pre-training frame rate):
  top view    sensor depth (LeRobot log-quantized 12-bit, full 640x360), registered to the RGB camera (depth
              intrinsics + baseline fitted by register_depth.py); per-dataset RGB intrinsics (median Depth Anything 3); world frame from the table
              plane of the episode's first frame (z up, gravity -z, origin under the image centre)
  wrist view  Depth Anything 3 (Nested-Giant-Large-1.1) on chunks of [top frames padded to 4:3, wrist frames] (same
              size, so DA3 does not crop the wrist views); DA3 is scaled to the sensor by the median sensor/DA3 ratio
              on the top frames of the chunk; wrist pose in the top camera = E_top @ inv(E_wrist) (DA3 extrinsics are
              world->camera). A wrist frame is used only when its table points land on the sensor table
              (median distance below 3x the episode median, adaptive) and per object only when its points agree with
              the top-view points of that object (median distance below the object's radius) or the object is unseen
              from the top
  masks       SAM 3.1 (text prompts, one session per prompt) on the top and wrist videos; per prompt the most
              confident instance (best phrase per object and view, PROMPTS); robot tokens as LIBERO's: the teal
              left arm's gripper ("gripper of the teal robot arm", kept inside the arm mask), the rest of that arm
              ("teal robot arm" minus the gripper) and the idle white arm ("robot arm on the right", right image
              half, no overlap with the teal arm); robots are lifted from the top view only
  lifting     physgs.data.lift._object_points (each masked pixel -> Gaussian [xyz, rgb, footprint]) per view, fused
              per object, farthest-point resampled to 1024 stored points (physgs.data.lift.resample_many; training
              reads a prefix, 512 as in pre-training)
No physics labels exist: pos / vel / angvel / contact / props are NaN (rel_lab without labels: --physics 0).
  python lift_so101.py --data_root <dir> --datasets <a>,<b> --out <dir> --calib_dir <dir> --emb_dir <dir>
                       [--worker i --nworkers n]
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import av
import cv2
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from physgs.data.format import save_episode
from physgs.data.lift import _object_points, resample_many
from physgs.model.layers import farthest_point_sample

FPS_IN, STRIDE = 30, 2          # 30 fps recordings -> 15 fps
# stored Gaussians per object, in farthest-point order: training reads a prefix (--num_points, 512 = pre-training)
NUM_POINTS, MIN_POINTS = 1024, 16

# objects per dataset: (object name, SAM 3.1 phrase on the top view, phrase on the wrist view), the best-detected
# phrase per view from tests/sam3_prompts.py (2026-11-06: e.g. "steel pot" / "pot" / "drawer" are never detected,
# "silver pot" / "plastic storage box" are). The arms come from "robot arm" on the top view.
PROMPTS = {
    "so101-left_arm_task_red_cube_out_of_pot_into_the_orange_hollow_cube": [
        ("red_cube", "red block", "red block"), ("pot", "silver pot", "silver pot"),
        ("orange_cube", "orange box", "orange hollow cube")],
    "so101_left_arm_open_drawer_pick_red_cube_stack_on_blue_cube": [
        ("drawer_unit", "plastic storage box", "plastic storage box"), ("red_cube", "red block", "red cube"),
        ("blue_cube", "blue block", "blue block")],
    "so101-left_arm_task_drawer_placing_red_squre_block": [
        ("drawer_unit", "plastic storage box", "plastic storage box"), ("red_block", "red block", "red square block")],
}
ARM_PROMPT = "robot arm"
# robot tokens, as LIBERO's (gripper, arm): the acting left arm is split into its gripper and the rest of the arm, so
# the gripper gets its own Gaussian budget (tests/gauss_budget.py 2026-11-06: the whole 60 cm arm at 512 points keeps
# 2.8 mm median detail, the gripper alone 1.5 mm; the gripper phrase lands 99% inside the teal-arm mask). The idle
# right arm stays one token.
ROBOT_NAMES = ("gripper", "left_arm", "right_arm")
ARM_PROMPTS = ("teal robot arm", "robot arm on the right")   # tests/sam3_arms.py, tests/arm_check (2026-11-06)
GRIPPER_PROMPT = "gripper of the teal robot arm"
N_ROBOT = len(ROBOT_NAMES)


# ----------------------------------------------------------------------------------------------------------- data
def load_meta(root):
    info = json.load(open(f"{root}/meta/info.json"))
    eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/chunk-*/file-*.parquet"))])
    tasks = pd.read_parquet(f"{root}/meta/tasks.parquet").index.tolist()
    return info, eps.set_index("episode_index").sort_index(), tasks


def video_path(root, row, key):
    return (f"{root}/videos/{key}/chunk-{int(row[f'videos/{key}/chunk_index']):03d}/"
            f"file-{int(row[f'videos/{key}/file_index']):03d}.mp4")


def decode_episode(root, row, key, n_frames, depth=False):
    """Frames 0, STRIDE, 2*STRIDE, ... of one episode (decoded in order from the episode start)."""
    t0 = float(row[f"videos/{key}/from_timestamp"])
    c = av.open(video_path(root, row, key))
    s = c.streams.video[0]
    c.seek(int(max(0.0, t0 - 0.2) / s.time_base), stream=s)
    want = list(range(0, n_frames, STRIDE))
    out, k = [], 0
    for fr in c.decode(s):
        ts = float(fr.pts * s.time_base) - t0
        idx = int(round(ts * FPS_IN))
        if idx < 0:
            continue
        if idx >= n_frames:
            break
        if k < len(want) and idx == want[k]:
            a = fr.to_ndarray(format="gray12le" if depth else "rgb24")
            out.append(a[..., 0] if (depth and a.ndim == 3) else a)
            k += 1
    c.close()
    assert len(out) == len(want), f"{key}: decoded {len(out)} of {len(want)} frames"
    return np.stack(out)


def depth_from_codes(q, info):
    """LeRobot depth_utils log quantization (stored unit mm; depth_min / max / shift in metres) -> metres, NaN = none."""
    k = 1000.0 if info.get("depth_unit", "mm") == "mm" else 1.0
    lo, hi, sh = info["video.depth_min"] * k, info["video.depth_max"] * k, info["video.shift"] * k
    a, b = math.log(lo + sh), math.log(hi + sh)
    d = (np.exp(q.astype(np.float64) / 4095 * (b - a) + a) - sh) / k
    d[q == 0] = np.nan
    return d.astype(np.float32)


def register_depth(depth, reg, Kc, hw):
    """Sensor depth (H, W) m -> depth in the RGB camera (hw), with the fitted registration (register_depth.py): each
    depth pixel is unprojected with the depth intrinsics, moved by the baseline t, projected with the RGB intrinsics
    and splatted over the RGB pixels it covers (ceil of the focal ratio); nearest depth wins. NaN = no depth."""
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    m = np.isfinite(depth) & (depth > 0)
    z = depth[m]
    f_d, cx_d, cy_d, t = reg["f_d"], reg["cx_d"], reg["cy_d"], reg["t"]
    X = np.stack([(u[m] - cx_d) * z / f_d + t[0], (v[m] - cy_d) * z / f_d + t[1], z + t[2]], 1)
    X = X[X[:, 2] > 0.05]
    uc = Kc[0, 0] * X[:, 0] / X[:, 2] + Kc[0, 2]
    vc = Kc[1, 1] * X[:, 1] / X[:, 2] + Kc[1, 2]
    h, w = hw
    out = np.full(h * w, np.inf)
    r = max(1, int(np.ceil(Kc[0, 0] / f_d)))
    for du in range(r):
        for dv in range(r):
            ui = np.floor(uc - (r - 1) / 2 + du + 0.5).astype(np.int64)
            vi = np.floor(vc - (r - 1) / 2 + dv + 0.5).astype(np.int64)
            ok = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            np.minimum.at(out, vi[ok] * w + ui[ok], X[ok, 2])
    out[np.isinf(out)] = np.nan
    return out.reshape(h, w).astype(np.float32)


@torch.no_grad()
def register_depth_gpu(depths, reg, Kc, hw, device="cuda"):
    """register_depth for all frames at once on the GPU (float64, the same arithmetic): depths (T, H, W) m."""
    D = torch.from_numpy(np.asarray(depths, np.float64)).to(device)
    T, H, W = D.shape
    v, u = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float64),
                          torch.arange(W, device=device, dtype=torch.float64), indexing="ij")
    f_d, cx_d, cy_d, t = reg["f_d"], reg["cx_d"], reg["cy_d"], reg["t"]
    m = torch.isfinite(D) & (D > 0)
    tt, vv, uu = torch.nonzero(m, as_tuple=True)
    z = D[m]
    X = (u[vv, uu] - cx_d) * z / f_d + t[0]
    Y = (v[vv, uu] - cy_d) * z / f_d + t[1]
    Z = z + t[2]
    ok = Z > 0.05
    tt, X, Y, Z = tt[ok], X[ok], Y[ok], Z[ok]
    uc = Kc[0, 0] * X / Z + Kc[0, 2]
    vc = Kc[1, 1] * Y / Z + Kc[1, 2]
    h, w = hw
    out = torch.full((T * h * w,), float("inf"), dtype=torch.float64, device=device)
    r = max(1, int(np.ceil(Kc[0, 0] / f_d)))
    for du in range(r):
        for dv in range(r):
            ui = torch.floor(uc - (r - 1) / 2 + du + 0.5).long()
            vi = torch.floor(vc - (r - 1) / 2 + dv + 0.5).long()
            ok = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            out.scatter_reduce_(0, (tt[ok] * h + vi[ok]) * w + ui[ok], Z[ok], reduce="amin")
    out[torch.isinf(out)] = float("nan")
    return out.view(T, h, w).float().cpu().numpy()


def shift_image(a, dx, dy, fill=np.nan):
    """Integer shift (dx right, dy down) with fill."""
    out = np.full_like(a, fill)
    H, W = a.shape[:2]
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[yd, xd] = a[ys, xs]
    return out


# ------------------------------------------------------------------------------------------------------- geometry
def unproject_np(depth, K):
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    return np.stack([(u - K[0, 2]) * depth / K[0, 0], (v - K[1, 2]) * depth / K[1, 1], depth], -1)


def table_frame(depth, K, exclude, rng):
    """cam2world (4x4) with the table plane as z = 0, +z towards the camera, origin below the image centre."""
    P = unproject_np(depth, K)
    m = np.isfinite(P[..., 2]) & ~exclude
    P = P[m]
    P = P[rng.choice(len(P), min(len(P), 40000), replace=False)]
    best, bi = None, 0
    for _ in range(600):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((P - a) @ n) < 0.008
        if inl.sum() > bi:
            best, bi = (n, a), inl.sum()
    n, a = best
    Q = P[np.abs((P - a) @ n) < 0.008]                       # refine on the inliers (least squares)
    c0 = Q.mean(0)
    n = np.linalg.svd(Q - c0)[2][-1]
    if n @ (-c0) < 0:                                         # normal towards the camera (camera at the origin)
        n = -n
    z = n
    x = np.array([1.0, 0, 0]) - z * z[0]
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_wc = np.stack([x, y, z])                                # rows: world axes in camera coordinates
    ray = np.array([0.0, 0.0, 1.0])                           # image centre ray hits the plane at the origin
    o = ray * (n @ c0) / (n @ ray)
    T = np.eye(4)                                             # cam2world: x_world = R_wc (x_cam - o)
    T[:3, :3] = R_wc
    T[:3, 3] = -R_wc @ o
    return T, bi / len(P)


# ---------------------------------------------------------------------------------------------------------- SAM 3.1
class Segmenter:
    """SAM 3.1 video predictor (text prompts); one session per prompt over in-memory frames."""

    def __init__(self, ckpt=None):
        from sam3.model_builder import build_sam3_predictor
        self.p = build_sam3_predictor(checkpoint_path=ckpt, version="sam3.1", use_fa3=False, compile=False)
        # SAM_DET_THRESH (2026-11-07): lower detection / new-track thresholds (default 0.5) for a retry on episodes where
        # an object was never found (task A: the red cube only shows as a small patch inside the shiny pot)
        th = os.environ.get("SAM_DET_THRESH")
        if th:
            m = self.p.model
            for a_ in ("score_threshold_detection", "new_det_thresh"):
                if hasattr(m, a_):
                    setattr(m, a_, float(th))
            print("SAM thresholds", {a_: getattr(m, a_, None) for a_ in ("score_threshold_detection", "new_det_thresh")}, flush=True)

    def _start(self, pil):
        """Sam3BasePredictor.start_session without offload_state_to_cpu: sam3 git f1647e8 (2026-10-23) passes it to
        Sam3MultiplexTrackingWithInteractivity.init_state, which does not take it (TypeError). Same session record."""
        import uuid
        # SAM_OFFLOAD_VIDEO=1 (2026-11-07): frames stay in CPU memory; a 600-frame episode took ~27 GB GPU otherwise
        # and 3 workers per 80 GB GPU ran out of memory
        st = self.p.model.init_state(resource_path=pil, offload_video_to_cpu=os.environ.get("SAM_OFFLOAD_VIDEO", "0") == "1",
                                     async_loading_frames=getattr(self.p, "async_loading_frames", False))
        sid = str(uuid.uuid4())
        self.p._all_inference_states[sid] = dict(state=st, session_id=sid, start_time=time.time(),
                                                  last_use_time=time.time())
        return sid

    def track(self, frames, prompt, frame=0, direction="both"):
        """frames (T, H, W, 3) uint8 -> list over T of {obj_id: (mask (H, W) bool, prob)}. The text prompt is given at
        `frame` and propagated in `direction` from there (2026-11-07: frame T-1 + "backward" finds an object that is
        hidden at the start, e.g. the red cube inside the pot, which the forward pass from frame 0 never picked up)."""
        from PIL import Image
        pil = [Image.fromarray(f) for f in frames]
        sid = self._start(pil)
        try:
            self.p.handle_request(dict(type="add_prompt", session_id=sid, frame_index=frame, text=prompt))
            out = [dict() for _ in frames]
            for r in self.p.handle_stream_request(dict(type="propagate_in_video", session_id=sid, start_frame_index=frame,
                                                       propagation_direction=direction)):
                o = r["outputs"]
                probs = o.get("out_probs", np.ones(len(o["out_obj_ids"])))
                for i, oid in enumerate(np.asarray(o["out_obj_ids"]).tolist()):
                    m = np.asarray(o["out_binary_masks"][i]).astype(bool)
                    if m.any():
                        out[r["frame_index"]][oid] = (m, float(np.asarray(probs)[i]))
        finally:
            self.p.handle_request(dict(type="close_session", session_id=sid))
        return out


def track_both_ends(seg, frames, prompt, min_cover=0.98):
    """Forward pass prompted at frame 0; if the object is missing in more than 1 - min_cover of the frames, also a
    backward pass prompted at the last frame. Per frame, the candidates of both passes (best_instance then keeps the
    most confident). Returns (track, number of passes)."""
    # (2026-11-07) SAM 3.1's tracker can raise "No points are provided; please add points first" inside propagation
    # (seen once in the full lift): a failed pass counts as finding nothing instead of ending the worker
    try:
        fw = seg.track(frames, prompt)
    except RuntimeError as ex:
        print(f"SAM_PASS_FAILED forward '{prompt}': {ex}", flush=True)
        fw = [dict() for _ in frames]
    if np.mean([bool(d) for d in fw]) >= min_cover:
        return fw, 1
    try:
        bw = seg.track(frames, prompt, frame=len(frames) - 1, direction="backward")
    except RuntimeError as ex:
        print(f"SAM_PASS_FAILED backward '{prompt}': {ex}", flush=True)
        return fw, 1
    return [{**{("f", k): v for k, v in a.items()}, **{("b", k): v for k, v in b.items()}} for a, b in zip(fw, bw)], 2


def best_instance_src(track):
    """best_instance and, per frame, which pass the chosen instance came from: 0 none, 1 forward, 2 backward
    (track_both_ends keys ("f", id) / ("b", id); plain tracks count as forward)."""
    m = best_instance(track)
    src = np.zeros(len(track), np.int8)
    for t, d in enumerate(track):
        if d:
            k = max(d, key=lambda kk: d[kk][1])
            src[t] = 2 if (isinstance(k, tuple) and k[0] == "b") else 1
    return m, src


def best_instance(track):
    """Per frame the most confident instance -> (T, H, W) bool (an object occluded in a frame has an empty mask)."""
    T = len(track)
    shape = next((m.shape for t in track for m, _ in t.values()), None)
    if shape is None:
        return None
    out = np.zeros((T,) + shape, bool)
    for t, d in enumerate(track):
        if d:
            out[t] = max(d.values(), key=lambda mp: mp[1])[0]
    return out


def split_arms(track):
    """'robot arm' instances -> (left, right) masks over time: the two most persistent track ids, ordered by their mean
    image x (the acting left arm is on the image left in every dataset)."""
    ids = {}
    for d in track:
        for oid, (m, _) in d.items():
            ids.setdefault(oid, []).append(np.nonzero(m)[1].mean())
    keep = sorted(ids, key=lambda k: -len(ids[k]))[:2]
    keep = sorted(keep, key=lambda k: np.mean(ids[k]))
    T = len(track)
    shape = next(m.shape for d in track for m, _ in d.values())
    masks = []
    for k in keep:
        a = np.zeros((T,) + shape, bool)
        for t, d in enumerate(track):
            if k in d:
                a[t] = d[k][0]
        masks.append(a)
    while len(masks) < 2:
        masks.append(np.zeros((T,) + shape, bool))
    return masks[0], masks[1]


# ------------------------------------------------------------------------------------------------- Depth Anything 3
class WristGeometry:
    """Wrist depth + pose in the top camera frame, at sensor scale, from DA3 on [top padded to 4:3, wrist] chunks."""

    def __init__(self, process_res=644, chunk=8):
        from depth_anything_3.api import DepthAnything3
        self.m = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1").to("cuda").eval()
        self.res, self.chunk = process_res, chunk

    @torch.no_grad()
    def run(self, top, wrist, sensor):
        """top (T, 360, 640, 3), wrist (T, 480, 640, 3), sensor (T, 360, 640) m -> per t: wrist depth (h, w) m, wrist K
        (3x3, pixels of (h, w)), M (4x4 wrist camera -> top camera), and the chunk scale / agreement."""
        T, Ht, Wt = top.shape[:3]
        Hw, Ww = wrist.shape[1:3]
        pad = (Hw * Wt // Ww - Ht) // 2                       # 360 -> 480 rows: 60 above and below
        topp = np.zeros((T, Ht + 2 * pad, Wt, 3), np.uint8)
        topp[:, pad:pad + Ht] = top
        res = []
        for s in range(0, T, self.chunk):
            idx = list(range(s, min(T, s + self.chunk)))
            n = len(idx)
            p = self.m.inference([topp[i] for i in idx] + [wrist[i] for i in idx], ref_view_strategy="first",
                                 process_res=self.res)
            hp, wp = p.depth.shape[1:]
            E = np.stack([np.vstack([e[:3], [0, 0, 0, 1]]) if e.shape == (3, 4) else e for e in p.extrinsics])
            # DA3 top depth of the unpadded rows vs the sensor -> scale to metres of the sensor
            r0, r1 = int(round(pad * hp / (Ht + 2 * pad))), int(round((pad + Ht) * hp / (Ht + 2 * pad)))
            ratios = []
            for j, i in enumerate(idx):
                d = p.depth[j, r0:r1]
                sen = cv2.resize(sensor[i], (d.shape[1], d.shape[0]), interpolation=cv2.INTER_NEAREST)
                ok = np.isfinite(sen) & (sen > 0.05) & (d > 0)
                ratios.append(np.median(sen[ok] / d[ok]))
            sc = float(np.median(ratios))
            for j, i in enumerate(idx):
                M = E[j] @ np.linalg.inv(E[n + j])           # wrist camera -> DA3 world -> top camera j (DA3 units)
                M[:3, 3] *= sc
                Kw = p.intrinsics[n + j].copy()
                res.append(dict(depth=(p.depth[n + j] * sc).astype(np.float32), K=Kw, M=M, scale=sc,
                                hw=(hp, wp)))
        return res


# ------------------------------------------------------------------------------------------------------- episode
def lift_episode(root, info, row, ep, prompts, calib, task_emb, seg, wg, out_path, rng, device="cuda"):
    n30 = int(row["length"])
    t_start = time.time()
    tim = {}
    top = decode_episode(root, row, "observation.images.top", n30)
    wrist = decode_episode(root, row, "observation.images.wrist", n30)
    q = decode_episode(root, row, "observation.images.top_depth", n30, depth=True)
    dinfo = info["features"]["observation.images.top_depth"]["info"]
    T = len(top)
    Kt = np.array(calib["K_top"], np.float32)
    tim["decode"] = time.time() - t_start
    t0 = time.time()
    raw = np.stack([depth_from_codes(x, dinfo) for x in q])
    sensor = np.concatenate([register_depth_gpu(raw[s:s + 64], calib["depth_reg"], Kt.astype(np.float64), top.shape[1:3])
                             for s in range(0, T, 64)])
    tim["register"] = time.time() - t0

    # masks: objects on top + wrist; each arm with its own phrase on the top view (2026-11-06: "robot arm" alone
    # misses the white idle arm in some episodes; "teal robot arm" / "robot arm on the right" find each arm in every
    # frame of the tested clips, with no overlap)
    t0 = time.time()
    top_masks, wr_masks = {}, {}
    for name, p_top, p_wr in prompts:
        top_masks[name] = best_instance(seg.track(top, p_top))
        wr_masks[name] = best_instance(seg.track(wrist, p_wr))
    left = best_instance(seg.track(top, ARM_PROMPTS[0]))
    right = best_instance(seg.track(top, ARM_PROMPTS[1]))
    W_img = top.shape[2]
    if left is None:
        left = np.zeros(top.shape[:3], bool)
    if right is not None:      # the white arm: right half of the image, not the teal arm
        for t in range(T):
            if right[t].any():
                inter = (right[t] & left[t]).sum() / max(1, (right[t] | left[t]).sum())
                if np.nonzero(right[t])[1].mean() < W_img / 2 or inter > 0.3:
                    right[t] = False
    # gripper: inside the teal arm (pixels outside it are dropped); the arm token is the teal arm without the gripper
    grip = best_instance(seg.track(top, GRIPPER_PROMPT))
    grip = np.zeros(top.shape[:3], bool) if grip is None else grip & left
    arm = left & ~grip
    tim["sam"] = time.time() - t0
    names = list(ROBOT_NAMES) + [n for n, _, _ in prompts]
    tops = [grip, arm, right] + [top_masks[n] for n, _, _ in prompts]
    wrs = [None] * N_ROBOT + [wr_masks[n] for n, _, _ in prompts]
    O = len(names)
    for k in range(O):
        if tops[k] is None:
            tops[k] = np.zeros(top.shape[:3], bool)

    # world frame from the first frame's table (objects and arms excluded)
    excl = np.any(np.stack([m[0] for m in tops]), 0)
    c2w_top, table_frac = table_frame(sensor[0], Kt, excl, rng)

    # wrist geometry and its agreement with the sensor table
    t0 = time.time()
    geo = wg.run(top, wrist, sensor)
    tim["da3"] = time.time() - t0
    t0 = time.time()
    table_pts = unproject_np(sensor[0], Kt)[np.isfinite(sensor[0]) & ~excl]
    tree = cKDTree(table_pts[rng.choice(len(table_pts), min(len(table_pts), 60000), replace=False)])
    agree = np.full(T, np.inf)
    for t in range(T):
        g = geo[t]
        wmask = np.zeros(g["depth"].shape, bool)
        for k in range(N_ROBOT, O):
            if wrs[k] is not None:
                wmask |= cv2.resize(wrs[k][t].astype(np.uint8), g["depth"].shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
        P = unproject_np(g["depth"], g["K"])[~wmask & (g["depth"] > 0)]
        if len(P) > 100:
            P = P[rng.choice(len(P), min(len(P), 8000), replace=False)] @ g["M"][:3, :3].T + g["M"][:3, 3]
            agree[t] = np.median(tree.query(P)[0])
    use_wrist = agree <= 3 * np.median(agree[np.isfinite(agree)])
    tim["agree"] = time.time() - t0
    t0 = time.time()

    # lift
    gauss = np.zeros((T, O, NUM_POINTS, 7), np.float16)
    npts = np.zeros((T, O), np.int32)
    gens = [torch.Generator(device=device).manual_seed(int(ep) * 100003 + t) for t in range(T)]
    c2w_t = torch.tensor(c2w_top, dtype=torch.float32, device=device)[None]
    Kt_t = torch.tensor(Kt, device=device)[None]
    wrist_used = np.zeros((T, O), bool)
    for t in range(T):
        rgb = torch.from_numpy(top[t]).to(device)[None]
        dep = torch.from_numpy(np.nan_to_num(sensor[t], nan=0.0)).to(device)[None]
        segm = np.zeros(top.shape[1:3], np.int64)
        for k in range(O):
            segm[tops[k][t]] = k + 1
        sels, _ = _object_points(rgb, dep, torch.from_numpy(segm).to(device)[None], Kt_t, c2w_t, list(range(1, O + 1)), 5.0)
        if use_wrist[t]:
            g = geo[t]
            h, w = g["depth"].shape
            wr_rgb = torch.from_numpy(cv2.resize(wrist[t], (w, h), interpolation=cv2.INTER_AREA)).to(device)[None]
            wsegm = np.zeros((h, w), np.int64)
            for k in range(N_ROBOT, O):
                if wrs[k] is not None:
                    wsegm[cv2.resize(wrs[k][t].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0] = k + 1
            c2w_w = torch.tensor(c2w_top @ g["M"], dtype=torch.float32, device=device)[None]
            wsels, _ = _object_points(wr_rgb, torch.from_numpy(g["depth"]).to(device)[None],
                                      torch.from_numpy(wsegm).to(device)[None],
                                      torch.tensor(g["K"], dtype=torch.float32, device=device)[None],
                                      c2w_w, list(range(1, O + 1)), 5.0)
            for k in range(N_ROBOT, O):
                ws, ts_ = wsels[k], sels[k]
                if ws.shape[0] < MIN_POINTS:
                    continue
                if ts_.shape[0] >= MIN_POINTS:   # the wrist must agree with the top view of this object
                    rad = (ts_[:, :3] - ts_[:, :3].mean(0)).norm(dim=-1).pow(2).mean().sqrt()
                    d = torch.cdist(ws[::max(1, ws.shape[0] // 2000), :3], ts_[::max(1, ts_.shape[0] // 4000), :3]).min(1).values
                    if d.median() > rad:
                        continue
                sels[k] = torch.cat([ts_, ws], 0)
                wrist_used[t, k] = True
        sets, owner = [], []
        for k in range(O):
            npts[t, k] = sels[k].shape[0]
            if sels[k].shape[0] >= MIN_POINTS:
                s_ = sels[k]
                if s_.shape[0] < NUM_POINTS:   # fewer points than stored: FPS-order them before the padding copies, so
                    s_ = s_[farthest_point_sample(s_[None, :, :3], s_.shape[0])[0]]   # any prefix is an even subsample
                sets.append(s_); owner.append(k)
        for k, g_ in zip(owner, resample_many(sets, NUM_POINTS, [gens[t]] * len(sets))):
            gauss[t, k] = g_.cpu().numpy().astype(np.float16)
    tim["lift"] = time.time() - t0

    nan3 = np.full((T, O, 3), np.nan, np.float32)
    data = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "action"])
                      for f in sorted(glob.glob(f"{root}/data/chunk-*/file-*.parquet"))])  # small; actions at 15 fps
    a = data[data["episode_index"] == ep].sort_values("frame_index")
    actions = np.stack(a["action"].values)[::STRIDE][:T].astype(np.float32)
    save_episode(out_path, gauss=gauss, npts=npts, pos=nan3, vel=nan3.copy(), angvel=nan3.copy(),
                 contact=np.full((T, O), np.nan, np.float32), props=np.full((O, 3), np.nan, np.float32),
                 is_robot=np.array([True] * N_ROBOT + [False] * (O - N_ROBOT)), obj_names=np.array(names), task_emb=task_emb,
                 fps=np.float32(FPS_IN / STRIDE), source=np.array("so101"), actions=actions)
    np.savez_compressed(out_path.replace(".npz", "_cam.npz"), K_top=Kt, c2w_top=c2w_top,
                        c2w_wrist=np.stack([c2w_top @ g["M"] for g in geo]), K_wrist=np.stack([g["K"] for g in geo]),
                        da3_scale=np.array([g["scale"] for g in geo]), wrist_agree=agree, use_wrist=use_wrist,
                        wrist_used=wrist_used)
    return dict(ep=int(ep), T=T, objects=names, table_inliers=round(float(table_frac), 3),
                visible=dict(zip(names, np.round((npts >= MIN_POINTS).mean(0), 3).tolist())),
                wrist_frames=round(float(use_wrist.mean()), 3), wrist_obj_frac=round(float(wrist_used[:, N_ROBOT:].mean()), 3),
                seconds=round(time.time() - t_start, 1), stages={k: round(v, 1) for k, v in tim.items()})


def main():
    """One worker: the jobs (dataset, episode) of all datasets, longest episode first, every n-th from index i, so
    several workers per GPU share the list evenly. Models load once per worker; finished episodes are skipped (a
    killed run continues where it stopped)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="folder of the LeRobot datasets")
    ap.add_argument("--datasets", required=True, help="comma-separated dataset folder names")
    ap.add_argument("--episodes", default="all", help="'all' or comma-separated episode ids (same for every dataset)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib_dir", required=True, help="calibrate_so101.py + register_depth.py outputs <dataset>.json")
    ap.add_argument("--emb_dir", required=True, help="prep_so101.py task_emb outputs <dataset>.npy")
    ap.add_argument("--sam3_ckpt", default=None)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    args = ap.parse_args()
    jobs, meta = [], {}
    for name in args.datasets.split(","):
        root = f"{args.data_root.rstrip('/')}/{name}"
        info, eps, _ = load_meta(root)
        meta[name] = (root, info, eps, json.load(open(f"{args.calib_dir}/{name}.json")),
                      np.load(f"{args.emb_dir}/{name}.npy").astype(np.float32))
        ids = eps.index.tolist() if args.episodes == "all" else [int(x) for x in args.episodes.split(",")]
        jobs += [(int(eps.loc[e, "length"]), name, int(e)) for e in ids]
    jobs = sorted(jobs, key=lambda j: (-j[0], j[1], j[2]))[args.worker::args.nworkers]
    print(f"worker {args.worker}/{args.nworkers}: {len(jobs)} episodes", flush=True)
    seg, wg = Segmenter(args.sam3_ckpt), WristGeometry()
    for _, name, e in jobs:
        out = f"{args.out}/{name}/episode_{e:04d}.npz"
        if os.path.exists(out):
            continue
        root, info, eps, calib, emb = meta[name]
        r = lift_episode(root, info, eps.loc[e], e, PROMPTS[name], calib, emb, seg, wg, out, np.random.default_rng(e))
        print(json.dumps(dict(dataset=name[:40], **r)), flush=True)


if __name__ == "__main__":
    sys.exit(main())
