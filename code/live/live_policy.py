"""
live_policy.py (2026-11-10): world-model forecast + smolVLA policy for one live 15 fps step, with the inputs built as in
the offline evaluation the checkpoints were selected with (eval_policy_so101.py, setting 2: one relative action per
step, a fresh random start noise per step):
  forecast   infer_check_so101.OnlineForecaster (the world model filters the newest lifted frame and imagines 15 steps;
             it matches the precomputed training forecasts to 2e-5); scene centre and length scale from the first frame
  images     top and wrist letterboxed to the run's img_size x img_size (so101_policy_h5.letterbox, cv2 INTER_AREA):
             512
  state      the 6 joint values scaled to [-1, 1] with norm.json, in ATM's extra-state layout (5 arm joints + 2 zero
             pad; gripper twice)
  language   the task sentence + "\\n", smolVLA's tokenizer, right padding to 48 (tokenize_h5.py)
  action     one-action runs (act): predict_chunk (10 flow steps) -> clamp [-1, 1] -> x rel_scale + scaled state ->
             joint units (LeRobot so_follower values: arm degrees, gripper 0..100)
             chunked runs (plan, 2026-11-11; smolVLA's default setting, action_rate 2): the whole chunk of 50 actions at
             30 fps from one random start noise -> x target_std + target_mean (MEAN_STD) -> + scaled state of the
             planning frame (relative runs) -> clip [-1.5, 1.5] -> joint units, as eval_policy_so101.chunk_exec_eval
The plain smolVLA (forecast_cfg.enabled false) skips the forecast.
"""
import json
import os

import numpy as np
import torch

import live_common as C  # first: it sets the code paths the next imports need
import infer_check_so101 as IC  # noqa: E402
from eval_policy_so101 import load_policy  # noqa: E402
from infer_check_so101 import OnlineForecaster  # noqa: E402
from so101_policy_h5 import letterbox

LANG_LEN = 48


class LivePolicy:
    def __init__(self, kind, name, meta, ckpt=None, num_steps=0):
        run, ck, wm = C.POLICIES[kind]
        ckpt = ckpt or ck
        if ckpt is None:   # v3 runs: the checkpoint the offline selection chose
            sel = C.asset(f"{run}/selection.json")
            if not os.path.exists(sel):
                raise FileNotFoundError(f"{sel} not found: pass --ckpt <model_N>")
            ckpt = json.load(open(sel))["chosen"]
        self.kind, self.ckpt, self.wm = kind, ckpt, wm
        self.policy, self.cfg = load_policy(C.asset(run), self.ckpt)
        if num_steps:
            self.policy.policy.config.num_steps = int(num_steps)
        self.use_fc = bool(getattr(self.policy, "enabled", False))
        if self.use_fc and wm is None:
            raise ValueError(f"policy {kind} reads forecasts: live_common.POLICIES needs its world model")
        nrm = json.load(open(C.asset("so101/policy_h5/norm.json")))
        self.lo, self.hi = np.array(nrm["min"]), np.array(nrm["max"])
        dc = self.cfg.dataset_cfg
        self.img = int(self.cfg.get("img_size", 256))
        self.chunk = int(self.policy.chunk)
        # chunked runs (smolVLA's default setting): actions at 30 fps (action_rate 2), MEAN_STD targets
        self.rate = int(dc.get("action_rate", 1))
        self.chunked = self.cfg.get("backbone", "vilt") == "smolvla" and self.rate > 1
        self.relf = bool(dc.get("relative_actions", False))
        if self.chunked:
            self.tm, self.ts = np.array(dc.target_mean, np.float32), np.array(dc.target_std, np.float32)
        self.rel = np.array(dc.rel_scale, np.float32) if self.relf and dc.get("rel_scale") is not None else None
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(SmolVLAConfig().vlm_model_name)
        t = tok(meta[name]["task"] + "\n", padding="max_length", max_length=LANG_LEN, truncation=True,
                padding_side="right", return_tensors="np")
        self.lang = (t["input_ids"][0].astype(np.float32), t["attention_mask"][0].astype(np.float32))
        self.slots = int(self.cfg.model_cfg.forecast_cfg.object_slots)
        self.O_e = 1 + len(meta[name]["prompts"])
        self.fc = None
        if self.use_fc:
            is_robot = np.zeros(self.slots, bool)
            is_robot[0] = True                                   # slot 0 = the gripper token (lift order)
            present = np.arange(self.slots) < self.O_e
            emb = np.load(C.asset(f"so101/task_emb/{name}.npy")).astype(np.float32)
            self.fc = OnlineForecaster(C.asset(wm), is_robot, present, emb, float(C.FPS), self.slots)

    def reset(self):
        if self.fc is not None:
            self.fc.state, self.fc.center, self.fc.L, self.fc.t = None, None, None, 0
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def scale(self, x):
        return (2 * (np.asarray(x, np.float64) - self.lo) / (self.hi - self.lo) - 1).astype(np.float32)

    def unit(self, a):
        return (np.asarray(a, np.float64) + 1) / 2 * (self.hi - self.lo) + self.lo

    def forecast(self, gauss, npts):
        """gauss (O_e, >=512, 7), npts (O_e,) of the newest frame -> forecast (16, slots, 143), or None (plain)."""
        if self.fc is None:
            return None
        g = np.zeros((self.slots,) + gauss.shape[1:], np.float32)
        g[:self.O_e] = gauss[:self.slots]
        n = np.zeros(self.slots, np.int64)
        n[:self.O_e] = npts[:self.slots]
        return self.fc.step(g, n)

    @torch.no_grad()
    def observe(self, gauss, npts):
        """A world-model frame whose forecast nobody reads (chunked mode, the frames between planning frames): the
        filter update of OnlineForecaster.step without its 15-step imagination, so the belief is the same as after
        forecast() at a fraction of the time."""
        f = self.fc
        if f is None:
            return
        if f.center is None:     # the episode's first frame (scene centre, length scale): the full step
            self.forecast(gauss, npts)
            return
        g = np.zeros((self.slots,) + gauss.shape[1:], np.float32)
        g[:self.O_e] = gauss[:self.slots]
        n = np.zeros(self.slots, np.int64)
        n[:self.O_e] = npts[:self.slots]
        gg = torch.as_tensor(g, device=f.dev)[None][:, :, :f.P]
        vis = (torch.as_tensor(n, device=f.dev)[None] >= IC.MIN_POINTS) & f.present
        gg = torch.cat([gg[..., :3] - f.center[:, None, None], gg[..., 3:]], -1)
        obs = f.m.observations(f.m.to_units(gg, f.L), vis)
        f.state, _ = f.m.step(f.state, f.ctx, obs, first=(f.t == 0), sample=False)
        f.t += 1

    def _chunk(self, top, wrist, state, fc, noise_seed):
        """The network's raw chunk (chunk, 6) for one frame, and the scaled state."""
        obs = np.stack([letterbox(top[None], size=self.img)[0].transpose(1, 2, 0),
                        letterbox(wrist[None], size=self.img)[0].transpose(1, 2, 0)])[None]
        ss = self.scale(state)
        extra = {"joint_states": np.concatenate([ss[:5], np.zeros(2, np.float32)])[None],
                 "gripper_states": np.array([[ss[5], ss[5]]], np.float32),
                 "lang_tokens": self.lang[0][None], "lang_mask": self.lang[1][None]}
        shape = (1, self.chunk, self.policy.policy.config.max_action_dim)
        noise = torch.randn(shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(int(noise_seed)))
        with torch.no_grad():
            ch = self.policy.predict_chunk(obs, extra, None if fc is None else np.asarray(fc, np.float32)[None],
                                           noise=noise).float()
        return ch[0].cpu().numpy(), np.concatenate([ss[:5], ss[5:6]])

    def act(self, top, wrist, state, fc, noise_seed):
        """One-action runs: top (H, W, 3), wrist (H', W', 3) uint8 RGB, state (6,) joint values, fc from forecast() ->
        target (6,) in joint values."""
        assert not self.chunked, "chunked run: use plan()"
        ch, ss = self._chunk(top, wrist, state, fc, noise_seed)
        a = np.clip(ch[0], -1, 1)
        if self.rel is not None:          # relative-action run: the state of this frame added back
            a = a * self.rel + ss
        return self.unit(a)

    def plan(self, top, wrist, state, fc, noise_seed):
        """Chunked runs: the same inputs -> the chunk's targets (chunk, 6) in joint values, one per 1 / (15 x rate) s."""
        assert self.chunked, "one-action run: use act()"
        ch, ss = self._chunk(top, wrist, state, fc, noise_seed)
        ch = ch * self.ts + self.tm
        if self.relf:                     # relative targets: the planning frame's state added back
            ch = ch + ss[None]
        return self.unit(np.clip(ch, -1.5, 1.5))
