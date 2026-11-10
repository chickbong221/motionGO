"""
BC dataset for PhysGSViLTPolicy.

Reads the re-rendered demo hdf5 files (scripts/render_libero_multiview.py; same keys as ATM's
preprocessed files except that there are no tracks) and, for every demo, the forecast array written
by physgs.engine.precompute_forecast at  <forecast_dir>/<task>/<demo>.npy  with shape (T, H, O, F).
Items keep ATM's order: (obs, forecast, dummy_track, task_emb, actions, extra_states).
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from atm.dataloader.base_dataset import BaseDataset


class ForecastBCDataset(BaseDataset):
    def __init__(self, forecast_dir, *args, views=("agentview", "eye_in_hand"), object_slots=None, state_shift=0,
                 action_chunk=None, relative_actions=False, rel_scale=None, action_rate=1, target_mean=None,
                 target_std=None, **kwargs):
        # action_rate / target_mean / target_std (SO-101 version, 2026-11-08; defaults = no change): LeRobot's recommended
        # smolVLA setting, chunks of the 30 fps actions (action_rate 2: root/actions30[2k : 2k + chunk] at 15 fps frame
        # k) and MEAN_STD-normalized targets (target - mean) / std (after the relative step when relative_actions).
        self.action_rate = action_rate
        self.target_mean = torch.tensor(target_mean, dtype=torch.float32) if target_mean is not None else None
        self.target_std = torch.tensor(target_std, dtype=torch.float32) if target_std is not None else None
        # relative_actions (SO-101 version, 2026-11-07; default False = the original absolute targets): the target is
        # (action - state) / rel_scale, the motion from the current joint state (the SO-101 analogue of LIBERO's delta
        # end-effector actions); evaluation adds the state back. Offline test: "command the current state" beat the
        # absolute-action policies on the first action (1.7 vs 2.3-10 deg arm error).
        self.relative_actions = relative_actions
        self.rel_scale = torch.tensor(rel_scale, dtype=torch.float32) if rel_scale is not None else None
        self.forecast_dir = forecast_dir
        # action_chunk (SO-101 version, 2026-11-07; default None = the original items): the actions of the next
        # action_chunk steps from the window's last frame and their padding mask (extra "action_is_pad"), for an
        # action-chunk backbone (physgs.policy.physgs_smolvla). Needs num_track_ts >= action_chunk (end padding).
        self.action_chunk = action_chunk
        self.object_slots = object_slots  # keep only the first object slots (the policy reads no others)
        # LIBERO's recorded obs[t] (copied into extra_states by older renders) is the state AFTER action t
        # (= states[t + 1]), while our frames are rendered from states[t]; state_shift=1 moves those robot states
        # back one step so image, robot state and forecast describe the same instant, as in the online evaluation
        # (frame 0 repeats obs[0]). Files whose extra_states were taken at the rendered instant
        # (attrs extra_states_source = "render") are never shifted.
        self.state_shift = state_shift
        kwargs.pop("track_obs_fs", None)
        kwargs.pop("augment_track", None)
        super().__init__(*args, views=list(views), augment_track=False, **kwargs)

    def forecast_path(self, demo_path):
        task = os.path.basename(os.path.dirname(os.path.realpath(demo_path)))
        return os.path.join(self.forecast_dir, task, os.path.basename(demo_path).replace(".hdf5", ".npy"))

    def load_h5(self, fn):
        """Read only what the policy needs (the re-rendered files also hold depth / segmentation of 4 views)."""
        import h5py
        with h5py.File(fn, "r") as f:
            r = f["root"]
            views = self.views if self.views is not None else ["agentview", "eye_in_hand"]
            root = {v: {"video": r[v]["video"][:]} for v in views}
            root["actions"] = r["actions"][:]
            if self.action_rate > 1:
                root["actions30"] = r["actions30"][:]
            root["task_emb_bert"] = r["task_emb_bert"][:]
            root["extra_states"] = {k: r["extra_states"][k][:] for k in r["extra_states"]}
            aligned = f.attrs.get("extra_states_source") == "render"
        return {"root": root, "forecast": np.load(self.forecast_path(fn)).astype(np.float32), "aligned": aligned}

    def process_demo(self, demo):
        pad = self.frame_stack + self.num_track_ts
        for v in self.views:
            # cached as uint8 (ATM keeps float32: 4x the host memory, ~45 GB for LIBERO-Long at 256)
            vids = torch.from_numpy(np.ascontiguousarray(demo["root"][v]["video"][0]))
            vids = torch.cat([vids, vids[-1:].repeat(pad, 1, 1, 1)], 0)
            if vids.shape[-2:] != tuple(self.img_size):
                vids = F.interpolate(vids.float(), size=self.img_size, mode="bilinear", align_corners=False)
                vids = vids.round().clamp(0, 255).to(torch.uint8)
            demo["root"][v]["video"] = vids
        actions = demo["root"]["actions"]
        actions = actions[0] if actions.ndim == 3 else actions
        demo["root"]["actions"] = torch.Tensor(np.concatenate([actions, np.zeros((pad, actions.shape[-1]), actions.dtype)]))
        demo["root"]["task_emb_bert"] = torch.Tensor(demo["root"]["task_emb_bert"])
        extra = {k: demo["root"]["extra_states"][k] for k in self.extra_state_keys}
        if self.state_shift and not demo.get("aligned", False):
            extra = {k: np.concatenate([np.repeat(v[:1], self.state_shift, 0), v[:-self.state_shift]]) for k, v in extra.items()}
        demo["root"]["extra_states"] = {k: torch.Tensor(np.concatenate([v, np.repeat(v[-1:], pad, 0)]))
                                        for k, v in extra.items()}
        fc = demo["forecast"]                                                                   # T H O F
        if self.object_slots:
            fc = fc[:, :, :self.object_slots]
        demo["forecast"] = torch.Tensor(np.concatenate([fc, np.repeat(fc[-1:], pad, 0)]))
        if "actions30" in demo["root"]:   # action_rate 2: the 30 fps actions, zero-padded by one chunk (masked)
            a30 = demo["root"]["actions30"]
            demo["root"]["actions30"] = torch.Tensor(np.concatenate([a30, np.zeros(((self.action_chunk or 1) * self.action_rate, a30.shape[-1]), a30.dtype)]))
            demo["len30"] = len(a30)
        # drop arrays the policy never reads to keep the cache small
        for k in list(demo["root"].keys()):
            if k not in self.views and k not in ("actions", "task_emb_bert", "extra_states", "actions30"):
                del demo["root"][k]
        return demo

    def __getitem__(self, index):
        demo_id = self._index_to_demo_id[index]
        t0 = index - self._demo_id_to_start_indices[demo_id]
        demo = self._cache[demo_id] if self.cache_all else self.process_demo(self.load_h5(self._demo_id_to_path[demo_id]))
        obs = torch.stack([self._load_image_list_from_demo(demo, v, t0) for v in self.views], 0)  # v t c h w
        if np.random.rand() < self.aug_prob:  # back to uint8 (as the frames were stored): a 4x smaller batch
            obs = self.augmentor((obs.float() / 255.0, torch.zeros(obs.shape[0], obs.shape[1], 1, 1, 2)))[0]
            obs = (obs * 255.0).round().clamp(0, 255).to(torch.uint8)
        forecast = demo["forecast"][t0:t0 + self.frame_stack]                                  # t H O F
        actions = demo["root"]["actions"][t0:t0 + self.frame_stack]
        extra = {k: v[t0:t0 + self.frame_stack] for k, v in demo["root"]["extra_states"].items()}
        if self.action_chunk:
            a0 = t0 + self.frame_stack - 1
            if self.action_rate > 1:   # chunk of the 30 fps actions from the frame's 30 fps index
                i0 = a0 * self.action_rate
                actions = demo["root"]["actions30"][i0:i0 + self.action_chunk]
                extra["action_is_pad"] = torch.arange(i0, i0 + self.action_chunk) >= demo["len30"]
            else:
                actions = demo["root"]["actions"][a0:a0 + self.action_chunk]
                extra["action_is_pad"] = torch.arange(a0, a0 + self.action_chunk) >= self._demo_id_to_demo_length[demo_id]
        if self.relative_actions:
            # relative to the robot state at the frame each action is issued from (chunk: the window's last frame):
            # (action - state) / rel_scale; steps past the episode end: 0 = hold (ATM pads absolute zeros there)
            es = demo["root"]["extra_states"]
            st = torch.cat([es["joint_states"][:, :5], es["gripper_states"][:, :1]], -1)
            n = actions.shape[0]
            if self.action_chunk:
                ref = st[a0].expand(n, -1)
                past = extra["action_is_pad"]
            else:
                ref = st[t0:t0 + n]
                past = torch.arange(t0, t0 + n) >= self._demo_id_to_demo_length[demo_id]
            actions = actions - ref
            if self.rel_scale is not None:
                actions = actions / self.rel_scale
            actions[past] = 0.0
        if self.target_mean is not None:   # MEAN_STD normalization of the targets (LeRobot's for smolVLA)
            actions = (actions - self.target_mean) / self.target_std
        dummy_track = torch.zeros(1)
        return obs, forecast, dummy_track, demo["root"]["task_emb_bert"], actions, extra
