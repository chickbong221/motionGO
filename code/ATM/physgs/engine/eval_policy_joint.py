"""
Batched online LIBERO evaluation of PhysGSViLTPolicy with forecasts from the joint world model (physgs.model.joint_wm):
the environment workers, lifting and rollout of physgs.engine.eval_policy_vec, with lift -> joint world model (filter,
imagine) -> policy at every step.

    python -m physgs.engine.eval_policy_joint --suite libero_spatial --task <task_name> \
        --policy runs/x/policy/model_final.ckpt --model runs/joint/libero/best.pt \
        --packed data/packed/libero_spatial --episodes 5 --start 20 --num_envs 5 --out runs/x/eval/<task>.json
"""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np
import torch
from omegaconf import OmegaConf

from physgs.engine.eval_policy_vec import POLICY_VIEWS, _worker
from physgs.model.joint_wm import length_scale


class BatchedForecastJoint:
    def __init__(self, model, max_objects, task_emb, horizon, device, include_now=True, num_points=512, fps=20.0):
        self.model, self.O, self.horizon, self.device, self.fps = model, max_objects, horizon, device, fps
        self.num_points, self.include_now = num_points, include_now
        self.task_emb = torch.as_tensor(task_emb, device=device).view(1, -1).float()

    def reset(self, first_obs, episodes):
        B, O = len(first_obs), self.O
        self.gens = [torch.Generator(device=self.device).manual_seed(ep) for ep in episodes]
        self.seg_ids = [o["seg_ids"][:O] for o in first_obs]
        self.is_robot = torch.zeros(B, O, dtype=torch.bool, device=self.device)
        self.present = torch.zeros(B, O, dtype=torch.bool, device=self.device)
        for i, o in enumerate(first_obs):
            self.is_robot[i, :o["num_robots"]] = True
            self.present[i, :len(self.seg_ids[i])] = True
        self.ctx = self.model.context(self.is_robot, self.present, self.task_emb.expand(B, -1),
                                      torch.full((B,), float(self.fps), device=self.device))
        self.state, self.center, self.L, self.t = None, None, None, 0

    def lift(self, obs):
        from physgs.data.format import MIN_POINTS
        from physgs.data.lift import lift_objects_batch
        t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=self.device)
        scenes = [(t(o["lift_rgb"], torch.uint8), t(o["lift_depth"]), t(o["lift_seg"], torch.long), t(o["K"]), t(o["c2w"]),
                   ids, gen) for o, ids, gen in zip(obs, self.seg_ids, self.gens)]
        gs, vis = [], []
        for g, npts in lift_objects_batch(scenes, num_points=self.num_points, max_depth=5.0):
            gauss = torch.zeros(self.O, g.shape[1], 7, device=self.device)
            gauss[:g.shape[0]] = g
            v = torch.zeros(self.O, dtype=torch.bool, device=self.device)
            v[:g.shape[0]] = npts.to(self.device) >= MIN_POINTS
            gs.append(gauss)
            vis.append(v)
        return torch.stack(gs), torch.stack(vis)                                         # (B, O, P, 7), (B, O)

    @torch.no_grad()
    def step(self, obs):
        g, vis = self.lift(obs)
        vis = vis & self.present
        if self.center is None:  # scene centre: mean centroid of the objects visible in the first frame (as pack)
            cen = (g[..., :3].mean(2) * vis[..., None]).sum(1) / vis.sum(1, keepdim=True).clamp(min=1)
            self.center = cen
        g = torch.cat([g[..., :3] - self.center[:, None, None], g[..., 3:]], -1)
        B = g.shape[0]
        if self.L is None:  # length scale from the first frame, as in training (physgs.data.joint_dataset)
            self.L = length_scale(g, vis, self.is_robot)
            self.ctx = self.model.context(self.is_robot, self.present, self.task_emb.expand(B, -1),
                                          torch.full((B,), float(self.fps), device=self.device), self.L)
        obs = self.model.observations(self.model.to_units(g, self.L), vis)
        if self.state is None:
            self.state = self.model.initial(B, self.O, self.device)
        self.state, _ = self.model.step(self.state, self.ctx, obs, first=(self.t == 0), sample=False)
        steps = ([self.state] if self.include_now else []) + self.model.imagine(self.state, self.ctx, self.horizon,
                                                                                sample=False)
        self.t += 1
        return torch.stack([self.model.forecast_features(s, self.state, self.is_robot, self.present) for s in steps],
                           1).cpu().numpy()                                               # (B, H, O, 143)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--model", required=True, help="joint world model checkpoint")
    ap.add_argument("--packed", required=True, help="packed LIBERO dataset (object slots, Gaussians per token)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--num_envs", type=int, default=10)
    ap.add_argument("--horizon_steps", type=int, default=600)
    ap.add_argument("--lift_views", default="agentview,mv_left,mv_right,mv_top")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--libero_root", default="./data/libero/")
    ap.add_argument("--fixtures", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from libero import benchmark, get_libero_path
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_id = suite.get_task_names().index(args.task)
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    init_states = suite.get_task_init_states(task_id)
    lift_views = args.lift_views.split(",")
    views = list(dict.fromkeys(list(POLICY_VIEWS) + lift_views))
    meta = json.load(open(os.path.join(args.packed, "meta.json")))
    arm = any("arm" in e["obj_names"] for e in meta["episodes"])
    ctx = mp.get_context("spawn")
    n = min(args.num_envs, args.episodes)
    conns, procs = [], []
    for _ in range(n):
        a, b = ctx.Pipe()
        p = ctx.Process(target=_worker, args=(b, bddl, views, lift_views, args.size, args.fixtures, arm), daemon=True)
        p.start()
        conns.append(a)
        procs.append(p)

    from physgs.engine.precompute_forecast_joint import load_model
    from physgs.policy.physgs_vilt import PhysGSViLTPolicy
    from scripts.preprocess_libero import get_task_bert_embs
    device = "cuda"
    policy_cfg = OmegaConf.load(os.path.join(os.path.dirname(args.policy), "config.yaml"))
    policy = PhysGSViLTPolicy(**OmegaConf.to_container(policy_cfg.model_cfg, resolve=True)).to(device)
    policy.load(args.policy)
    policy.eval()
    task_emb = get_task_bert_embs(args.libero_root)[task.language]
    fc = BatchedForecastJoint(load_model(args.model, device), meta["max_objects"], task_emb, policy.horizon, device,
                              include_now=getattr(policy, "include_now", True), num_points=meta.get("num_points", 512))

    episodes = list(range(args.start, args.start + args.episodes))
    successes, lengths = {}, {}
    for r in range(0, len(episodes), n):
        eps = episodes[r:r + n]
        for c, ep in zip(conns, eps):
            c.send(("reset", init_states[ep % len(init_states)]))
        obs = [c.recv() for c in conns[:len(eps)]]
        fc.reset(obs, eps)
        policy.reset()
        done = [False] * len(eps)
        for step in range(args.horizon_steps):
            forecast = fc.step(obs)
            rgb = np.stack([o["rgb"] for o in obs])
            extra = {k: np.stack([o["extra"][k] for o in obs]) for k in policy.extra_state_keys}
            action, _ = policy.act(rgb, np.repeat(task_emb[None], len(eps), 0), extra, forecast)
            for i, c in enumerate(conns[:len(eps)]):
                if not done[i]:
                    c.send(("step", action[i]))
            for i, c in enumerate(conns[:len(eps)]):
                if not done[i]:
                    o = c.recv()
                    obs[i] = o
                    if o["success"]:
                        done[i], lengths[eps[i]] = True, step + 1
            if all(done):
                break
        for i, ep in enumerate(eps):
            successes[ep] = done[i]
            print(f"episode {ep}: {'success' if done[i] else 'fail'}", flush=True)
        print(f"running success {np.mean(list(successes.values())):.2f} ({len(successes)} episodes)", flush=True)
    for c in conns:
        c.send(("close", None))
    for p in procs:
        p.join(timeout=30)
    s = [successes[e] for e in episodes]
    result = dict(suite=args.suite, task=args.task, success_rate=float(np.mean(s)), episodes=len(s), start=args.start,
                  successes=[bool(x) for x in s], lengths={str(k): v for k, v in lengths.items()})
    print(result)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(result, open(args.out, "w"))


if __name__ == "__main__":
    main()
