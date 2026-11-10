"""
live_policy.py (2026-11-10): world-model forecast + smolVLA policy for one live 15 fps step, with the inputs built as in
the offline evaluation the checkpoints were selected with (eval_policy_so101.py, setting 2: one relative action per
step, a fresh random start noise per step):
  forecast   infer_check_so101.OnlineForecaster (the world model filters the newest lifted frame and imagines 15 steps;
             it matches the precomputed training forecasts to 2e-5); scene centre and length scale from the first frame
  images     top and wrist letterboxed to 256 x 256 (so101_policy_h5.letterbox, cv2 INTER_AREA)
  state      the 6 joint values scaled to [-1, 1] with norm.json, in ATM's extra-state layout (5 arm joints + 2 zero
             pad; gripper twice)
  language   the task sentence + "\\n", smolVLA's tokenizer, right padding to 48 (tokenize_h5.py)
  action     predict_chunk (10 flow steps) -> clamp [-1, 1] -> x rel_scale + scaled state -> joint units (LeRobot
             so_follower values: arm degrees, gripper 0..100)
The plain smolVLA (forecast_cfg.enabled false) skips the forecast.
"""
import json

import numpy as np
import torch

import live_common as C
from eval_policy_so101 import load_policy
from infer_check_so101 import OnlineForecaster
from so101_policy_h5 import letterbox

LANG_LEN = 48


class LivePolicy:
    def __init__(self, kind, name, meta, ckpt=None, num_steps=0):
        run, ck = C.POLICIES[kind]
        self.kind, self.ckpt = kind, ckpt or ck
        self.policy, self.cfg = load_policy(C.asset(run), self.ckpt)
        if num_steps:
            self.policy.policy.config.num_steps = int(num_steps)
        self.use_fc = bool(getattr(self.policy, "enabled", False))
        nrm = json.load(open(C.asset("so101/policy_h5/norm.json")))
        self.lo, self.hi = np.array(nrm["min"]), np.array(nrm["max"])
        dc = self.cfg.dataset_cfg
        self.rel = np.array(dc.rel_scale, np.float32) if dc.get("relative_actions", False) else None
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
            self.fc = OnlineForecaster(C.asset(C.WM), is_robot, present, emb, float(C.FPS), self.slots)

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

    def act(self, top, wrist, state, fc, noise_seed):
        """top (H, W, 3), wrist (H', W', 3) uint8 RGB, state (6,) joint values, fc from forecast() -> target (6,) in
        joint values."""
        obs = np.stack([letterbox(top[None])[0].transpose(1, 2, 0), letterbox(wrist[None])[0].transpose(1, 2, 0)])[None]
        ss = self.scale(state)
        extra = {"joint_states": np.concatenate([ss[:5], np.zeros(2, np.float32)])[None],
                 "gripper_states": np.array([[ss[5], ss[5]]], np.float32),
                 "lang_tokens": self.lang[0][None], "lang_mask": self.lang[1][None]}
        shape = (1, self.policy.chunk, self.policy.policy.config.max_action_dim)
        noise = torch.randn(shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(int(noise_seed)))
        with torch.no_grad():
            ch = self.policy.predict_chunk(obs, extra, None if fc is None else np.asarray(fc, np.float32)[None],
                                           noise=noise).clamp(-1, 1).cpu().numpy()
        a = ch[0, 0]
        if self.rel is not None:          # relative-action run: the state of this frame added back
            a = a * self.rel + np.concatenate([ss[:5], ss[5:6]])
        return self.unit(a)
