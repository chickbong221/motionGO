"""
Online LIBERO evaluation of PhysGSViLTPolicy. At every control step:
render the lift views (RGB-D + instance segmentation) -> per-object Gaussians -> object encoder tokens
-> RSSM filter step + H-step imagination -> forecast -> policy action.

    python -m physgs.engine.eval_policy --suite libero_spatial --task <task_name> \
        --policy runs/policy/x/model_best.ckpt --encoder runs/encoder_libero_spatial/last.pt \
        --rssm runs/rssm_libero_spatial/last.pt --packed data/packed/libero_spatial --episodes 20
"""
import argparse
import json
import os

import numpy as np
import torch
from omegaconf import OmegaConf

from libero import benchmark, get_libero_path
from libero.envs import OffScreenRenderEnv
from robosuite.utils import camera_utils

from physgs.data.format import MIN_POINTS
from physgs.data.libero_parts import part_layout, part_segmentation
from physgs.data.libero_process import ROBOT_INSTANCES, robot_instances
from physgs.data.lift import lift_objects
from physgs.model.object_encoder import ObjectAutoencoder
from physgs.model.object_rssm import ObjectRSSM
from physgs.policy.physgs_vilt import PhysGSViLTPolicy
from scripts.render_libero_multiview import add_placeholder_cameras, place_extra_cameras

POLICY_VIEWS = ("agentview", "robot0_eye_in_hand")
OBS_KEYS = {"gripper_states": "robot0_gripper_qpos", "joint_states": "robot0_joint_pos"}


def load(cls, path, device):
    ck = torch.load(path, map_location=device)
    m = cls(**ck["cfg"]).to(device).eval()
    m.load_state_dict(ck["model"])
    return m


class ForecastPipeline:
    """Keeps the per-episode state (previous Gaussians, RSSM state, scene centre)."""

    def __init__(self, env, encoder, rssm, lift_views, size, max_objects, task_emb, horizon, device, seed=0,
                 fixtures=False, include_now=False, arm=False, num_points=512):
        self.env, self.encoder, self.rssm = env, encoder, rssm
        self.include_now = include_now  # step 0 = the current filtered state (precompute_forecast --include_now)
        self.gen = torch.Generator(device=device).manual_seed(seed)  # point subsampling, reproducible per episode
        self.lift_views, self.size, self.horizon, self.device = lift_views, size, horizon, device
        instances = list(env.env.model.instances_to_ids.keys())
        # robot tokens as in the training data: gripper + arm (robot_instances) when it has an arm token, else legacy
        rob = robot_instances(instances) if arm else [(i, l) for i, l in ROBOT_INSTANCES.items() if i in instances]
        robots = [(lbl, instances.index(inst) + 1) for inst, lbl in rob]
        self.num_points = num_points  # Gaussians per token, as packed for training
        self.fixtures, self.parts = fixtures, part_layout(env, fixtures)   # objects (+ fixtures and their moving parts)
        objs = [(p["name"], p["seg_id"]) for p in self.parts]
        order = robots + objs                     # same order as format.pack(sort_by_visibility=False)
        self.seg_ids = [i for _, i in order][:max_objects]
        n = len(self.seg_ids)
        self.O = max_objects
        self.is_robot = torch.zeros(1, self.O, dtype=torch.bool, device=device)
        self.is_robot[0, :len(robots)] = True
        self.present = torch.zeros(1, self.O, dtype=torch.bool, device=device)
        self.present[0, :n] = True
        self.ctx = rssm.context(self.is_robot, self.present, torch.as_tensor(task_emb, device=device).view(1, -1).float())
        self.reset()

    def reset(self):
        self.hist, self.state, self.center, self.t = [], None, None, 0

    def lift(self, obs):
        sim = self.env.sim
        rgb, depth, seg, K, c2w = [], [], [], [], []
        for v in self.lift_views:
            rgb.append(obs[f"{v}_image"][::-1].copy())
            depth.append(camera_utils.get_real_depth_map(sim, obs[f"{v}_depth"])[::-1, :, 0].copy())
            s = obs[f"{v}_segmentation_instance"][::-1, :, 0]
            if self.fixtures:
                s = part_segmentation(s, obs[f"{v}_segmentation_element"][::-1, :, 0], self.parts)
            seg.append(s.astype(np.int64).copy())
            K.append(camera_utils.get_camera_intrinsic_matrix(sim, v, self.size, self.size))
            c2w.append(camera_utils.get_camera_extrinsic_matrix(sim, v))
        t = lambda x, dt=torch.float32: torch.as_tensor(np.stack(x), dtype=dt, device=self.device)
        g, npts = lift_objects(t(rgb, torch.uint8), t(depth), t(seg, torch.long), t(K), t(c2w), self.seg_ids,
                               num_points=self.num_points, max_depth=5.0, generator=self.gen)
        gauss = torch.zeros(self.O, g.shape[1], 7, device=self.device)
        gauss[:g.shape[0]] = g
        vis = torch.zeros(self.O, dtype=torch.bool, device=self.device)
        vis[:g.shape[0]] = npts.to(self.device) >= MIN_POINTS
        return gauss, vis

    @torch.no_grad()
    def step(self, obs):
        g, vis = self.lift(obs)
        # encoder window t-K+1 .. t; frames where an object was not visible take the nearest newer frame
        # (same rule as physgs.data.dataset.history_indices)
        K = self.encoder.num_frames
        frames, cur = [g], g
        for j in range(1, K):
            if j <= len(self.hist):
                gp, vp = self.hist[-j]
                cur = torch.where(vp[:, None, None], gp, cur)
            frames.append(cur)
        g_hist = torch.stack(frames[::-1], 1)                                           # (O, K, P, 7)
        tok = torch.zeros(self.O, self.rssm.obs_embed[0].in_features, device=self.device)
        if vis.any():
            tok[vis] = self.encoder.tokens(g_hist[vis]).float()
        if self.center is None:
            self.center = g[vis, :, :3].mean(1).mean(0) if vis.any() else torch.zeros(3, device=self.device)
        tok[:, 0:3] -= self.center
        tok[~vis] = 0
        if self.state is None:
            self.state = self.rssm.initial(1, self.O, self.device)
        self.state, _ = self.rssm.obs_step(self.state, tok[None], vis[None], self.ctx, first=(self.t == 0), sample=False)
        belief = self.rssm.heads(self.state)
        steps = []
        imag = self.rssm.imagine({k: self.state[k] for k in ("h", "s", "pos", "seen")}, self.ctx, self.horizon, sample=False)
        for s in ([self.state] if self.include_now else []) + list(imag):
            p = self.rssm.heads(s)
            steps.append(torch.cat([p["pos"], p["vel"], p["angvel"], torch.sigmoid(p["contact"])[..., None], p["z"],
                                    belief["belief_mu"], belief["belief_logstd"],
                                    self.is_robot[..., None].float(), (self.present & self.state["seen"])[..., None].float()], -1))  # unseen objects are absent
        self.hist = (self.hist + [(g, vis)])[-(K - 1):]
        self.t += 1
        return torch.stack(steps, 1).cpu().numpy()                                     # (1, H, O, F+2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--rssm", required=True)
    ap.add_argument("--packed", required=True, help="packed LIBERO dataset (for the padded object count)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--start", type=int, default=0, help="first init state (split long evaluations over processes)")
    ap.add_argument("--horizon_steps", type=int, default=600)
    ap.add_argument("--lift_views", default="agentview,mv_left,mv_right,mv_top")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--libero_root", default="./data/libero/")
    ap.add_argument("--fixtures", action="store_true", help="tokens for fixtures and their moving parts (as rendered)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = "cuda"
    lift_views = args.lift_views.split(",")

    policy_cfg = OmegaConf.load(os.path.join(os.path.dirname(args.policy), "config.yaml"))
    policy = PhysGSViLTPolicy(**OmegaConf.to_container(policy_cfg.model_cfg, resolve=True)).to(device)
    policy.load(args.policy)
    policy.eval()  # ATM's eval() returns None, so it cannot be chained
    encoder, rssm = load(ObjectAutoencoder, args.encoder, device), load(ObjectRSSM, args.rssm, device)
    meta = json.load(open(os.path.join(args.packed, "meta.json")))   # the training data decides tokens and points
    max_objects = meta["max_objects"]
    arm = any("arm" in e["obj_names"] for e in meta["episodes"])

    add_placeholder_cameras()
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_id = suite.get_task_names().index(args.task)
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    views = list(dict.fromkeys(list(POLICY_VIEWS) + lift_views))
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_names=views, camera_heights=args.size,
                             camera_widths=args.size, camera_depths=True,
                             camera_segmentations=[["instance", "element"]] * len(views) if args.fixtures else "instance")
    init_states = suite.get_task_init_states(task_id)
    from scripts.preprocess_libero import get_task_bert_embs
    task_emb = get_task_bert_embs(args.libero_root)[task.language]

    successes, lengths = [], {}
    for ep in range(args.start, args.start + args.episodes):
        env.reset()
        obs = env.set_init_state(init_states[ep % len(init_states)])
        place_extra_cameras(env, lift_views)
        dummy = np.zeros(7)
        dummy[-1] = -1.0
        for _ in range(5):
            obs, _, _, _ = env.step(dummy)
        pipe = ForecastPipeline(env, encoder, rssm, lift_views, args.size, max_objects, task_emb,
                                policy.horizon, device, seed=ep, fixtures=args.fixtures,
                                include_now=getattr(policy, "include_now", False), arm=arm,
                                num_points=meta.get("num_points", 512))
        policy.reset()
        success = False
        for step in range(args.horizon_steps):
            forecast = pipe.step(obs)
            rgb = np.stack([obs[f"{v}_image"][::-1] for v in POLICY_VIEWS])[None]
            extra = {k: obs[OBS_KEYS[k]][None] for k in policy.extra_state_keys}
            action, _ = policy.act(rgb, task_emb[None], extra, forecast)
            obs, _, done, _ = env.step(action[0])
            if env.check_success():
                success, lengths[str(ep)] = True, step + 1
                break
        successes.append(success)
        print(f"episode {ep}: {'success' if success else 'fail'}  (running {np.mean(successes):.2f})", flush=True)
    result = dict(suite=args.suite, task=args.task, success_rate=float(np.mean(successes)), episodes=len(successes),
                  start=args.start, successes=[bool(s) for s in successes], lengths=lengths)
    print(result)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(result, open(args.out, "w"))


if __name__ == "__main__":
    main()
