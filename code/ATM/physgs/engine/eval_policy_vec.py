"""
Batched online LIBERO evaluation of PhysGSViLTPolicy: same episodes / init states as eval_policy.py, but
`--num_envs` environments run in worker processes and the lift -> encoder -> RSSM -> policy chain runs once
per step for all of them on the GPU. ATM's own rollout also keeps stepping until every env is done or the
horizon is reached; here an env that succeeded is no longer stepped.

    python -m physgs.engine.eval_policy_vec --suite libero_spatial --task <task_name> \
        --policy runs/x/policy_s0/model_best.ckpt --encoder runs/enc/libero/last.pt --rssm runs/x/rssm_libero/last.pt \
        --packed data/packed/libero_spatial --episodes 20 --num_envs 10 --out runs/x/eval/s0_<task>.json
"""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np
import torch
from omegaconf import OmegaConf

POLICY_VIEWS = ("agentview", "robot0_eye_in_hand")
OBS_KEYS = {"gripper_states": "robot0_gripper_qpos", "joint_states": "robot0_joint_pos"}


# ---------------------------------------------------------------------------------------------- env worker
def _worker(conn, bddl, views, lift_views, size, fixtures=False, arm=False):
    from robosuite.utils import camera_utils
    from libero.envs import OffScreenRenderEnv
    from physgs.data.libero_parts import part_layout, part_segmentation
    from physgs.data.libero_process import ROBOT_INSTANCES, robot_instances
    from scripts.render_libero_multiview import add_placeholder_cameras, place_extra_cameras

    add_placeholder_cameras()
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_names=views, camera_heights=size, camera_widths=size,
                             camera_depths=True, camera_segmentations=[["instance", "element"]] * len(views) if fixtures else "instance")
    parts = None

    def seg_of(obs, v):
        s = obs[f"{v}_segmentation_instance"][::-1, :, 0]
        if fixtures:
            s = part_segmentation(s, obs[f"{v}_segmentation_element"][::-1, :, 0], parts)
        return s.astype(np.int64)

    def pack(obs):
        sim = env.sim
        out = dict(rgb=np.stack([obs[f"{v}_image"][::-1] for v in POLICY_VIEWS]),
                   extra={k: obs[o] for k, o in OBS_KEYS.items()},
                   lift_rgb=np.stack([obs[f"{v}_image"][::-1] for v in lift_views]),
                   lift_depth=np.stack([camera_utils.get_real_depth_map(sim, obs[f"{v}_depth"])[::-1, :, 0] for v in lift_views]),
                   lift_seg=np.stack([seg_of(obs, v) for v in lift_views]),
                   K=np.stack([camera_utils.get_camera_intrinsic_matrix(sim, v, size, size) for v in lift_views]),
                   c2w=np.stack([camera_utils.get_camera_extrinsic_matrix(sim, v) for v in lift_views]))
        return {k: (np.ascontiguousarray(v) if isinstance(v, np.ndarray) else v) for k, v in out.items()}

    while True:
        cmd, arg = conn.recv()
        if cmd == "reset":
            env.reset()
            obs = env.set_init_state(arg)
            place_extra_cameras(env, lift_views)
            dummy = np.zeros(7)
            dummy[-1] = -1.0
            for _ in range(5):
                obs, _, _, _ = env.step(dummy)
            instances = list(env.env.model.instances_to_ids.keys())
            # robot tokens as in the training data: gripper + arm (robot_instances) when it has an arm token, else the
            # legacy gripper-only lookup of the earlier runs
            rob = robot_instances(instances) if arm else [(i, l) for i, l in ROBOT_INSTANCES.items() if i in instances]
            robots = [instances.index(inst) + 1 for inst, _ in rob]
            parts = part_layout(env, fixtures)    # objects (+ fixtures and their moving parts), as rendered
            objs = [p["seg_id"] for p in parts]
            conn.send(dict(pack(obs), seg_ids=robots + objs, num_robots=len(robots)))
        elif cmd == "step":
            obs, _, _, _ = env.step(arg)
            conn.send(dict(pack(obs), success=bool(env.check_success())))
        elif cmd == "close":
            env.close()
            conn.close()
            return


# ---------------------------------------------------------------------------------------------- batched forecast
class BatchedForecast:
    def __init__(self, encoder, rssm, max_objects, task_emb, horizon, device, include_now=False, num_points=512):
        self.encoder, self.rssm, self.O, self.horizon, self.device = encoder, rssm, max_objects, horizon, device
        self.num_points = num_points  # Gaussians per token, as packed for training
        self.include_now = include_now  # step 0 = the current filtered state (precompute_forecast --include_now)
        self.task_emb = torch.as_tensor(task_emb, device=device).view(1, -1).float()

    def reset(self, first_obs, episodes):
        B, O = len(first_obs), self.O
        # one point-subsampling generator per episode, seeded by the episode index (as in eval_policy.py)
        self.gens = [torch.Generator(device=self.device).manual_seed(ep) for ep in episodes]
        self.seg_ids = [o["seg_ids"][:O] for o in first_obs]
        self.is_robot = torch.zeros(B, O, dtype=torch.bool, device=self.device)
        self.present = torch.zeros(B, O, dtype=torch.bool, device=self.device)
        for i, o in enumerate(first_obs):
            self.is_robot[i, :o["num_robots"]] = True
            self.present[i, :len(self.seg_ids[i])] = True
        self.ctx = self.rssm.context(self.is_robot, self.present, self.task_emb.expand(B, -1))
        self.hist, self.state, self.center, self.t = [], None, None, 0

    def lift(self, obs):
        from physgs.data.format import MIN_POINTS
        from physgs.data.lift import lift_objects_batch
        t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=self.device)
        scenes = [(t(o["lift_rgb"], torch.uint8), t(o["lift_depth"]), t(o["lift_seg"], torch.long), t(o["K"]), t(o["c2w"]),
                   ids, gen) for o, ids, gen in zip(obs, self.seg_ids, self.gens)]
        gs, vis = [], []
        for g, npts in lift_objects_batch(scenes, num_points=self.num_points, max_depth=5.0):  # all objects of all envs in one FPS loop
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
        B = g.shape[0]
        # encoder window t-K+1 .. t; frames where an object was not visible take the nearest newer frame
        K = self.encoder.num_frames
        frames, cur = [g], g
        for j in range(1, K):
            if j <= len(self.hist):
                gp, vp = self.hist[-j]
                cur = torch.where(vp[..., None, None], gp, cur)
            frames.append(cur)
        g_hist = torch.stack(frames[::-1], 2)                                           # (B, O, K, P, 7)
        tok = torch.zeros(B, self.O, self.rssm.obs_embed[0].in_features, device=self.device)
        if vis.any():
            tok[vis] = self.encoder.tokens(g_hist[vis]).float()
        if self.center is None:
            cen = (g[..., :3].mean(2) * vis[..., None]).sum(1) / vis.sum(1, keepdim=True).clamp(min=1)
            self.center = cen                                                             # (B, 3)
        tok[..., 0:3] -= self.center[:, None]
        tok[~vis] = 0
        if self.state is None:
            self.state = self.rssm.initial(B, self.O, self.device)
        self.state, _ = self.rssm.obs_step(self.state, tok, vis, self.ctx, first=(self.t == 0), sample=False)
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
        return torch.stack(steps, 1).cpu().numpy()                                      # (B, H, O, F+2)


def load(cls, path, device):
    ck = torch.load(path, map_location=device)
    m = cls(**ck["cfg"]).to(device).eval()
    m.load_state_dict(ck["model"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--rssm", required=True)
    ap.add_argument("--packed", required=True, help="packed LIBERO dataset (for the padded object count)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--num_envs", type=int, default=10)
    ap.add_argument("--horizon_steps", type=int, default=600)
    ap.add_argument("--lift_views", default="agentview,mv_left,mv_right,mv_top")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--libero_root", default="./data/libero/")
    ap.add_argument("--fixtures", action="store_true", help="tokens for fixtures and their moving parts (as rendered)")
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

    # the training data decides the robot tokens and the Gaussians per token
    meta = json.load(open(os.path.join(args.packed, "meta.json")))
    arm = any("arm" in e["obj_names"] for e in meta["episodes"])
    # workers first (spawned, before CUDA is initialised here)
    ctx = mp.get_context("spawn")
    n = min(args.num_envs, args.episodes)
    conns, procs = [], []
    for _ in range(n):
        a, b = ctx.Pipe()
        p = ctx.Process(target=_worker, args=(b, bddl, views, lift_views, args.size, args.fixtures, arm), daemon=True)
        p.start()
        conns.append(a)
        procs.append(p)

    from physgs.model.object_encoder import ObjectAutoencoder
    from physgs.model.object_rssm import ObjectRSSM
    from physgs.policy.physgs_vilt import PhysGSViLTPolicy
    from scripts.preprocess_libero import get_task_bert_embs
    device = "cuda"
    policy_cfg = OmegaConf.load(os.path.join(os.path.dirname(args.policy), "config.yaml"))
    policy = PhysGSViLTPolicy(**OmegaConf.to_container(policy_cfg.model_cfg, resolve=True)).to(device)
    policy.load(args.policy)
    policy.eval()  # ATM's eval() returns None, so it cannot be chained
    encoder, rssm = load(ObjectAutoencoder, args.encoder, device), load(ObjectRSSM, args.rssm, device)
    max_objects = meta["max_objects"]
    task_emb = get_task_bert_embs(args.libero_root)[task.language]
    fc = BatchedForecast(encoder, rssm, max_objects, task_emb, policy.horizon, device,
                         include_now=getattr(policy, "include_now", False), num_points=meta.get("num_points", 512))

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
