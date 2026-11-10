"""
infer_check_so101.py (2026-11-07): inference check of a trained SO-101 PhysGO policy (either backbone) on one episode,
running the inference chain step by step as on the robot:
  1. online forecast: at frame t the world model filters frame t's lifted Gaussians (frames <= t only) and imagines 15
     steps (as physgs.engine.eval_policy_joint.BatchedForecastJoint does online on LIBERO; scene centre and length scale
     from the first frame, as in training). Checked against the precomputed forecasts the policy was trained on
     (physgs.engine.precompute_forecast_joint): they must agree to numerical precision.
  2. policy: vilt -> one action per frame through act() (its frame history kept); smolvla -> a chunk re-planned every
     --replan frames and executed from (LeRobot's way on the robot; --replan 1 = re-plan every frame).
  3. outputs in --out: actions_<episode>.png (6 joints: recorded vs predicted target, degrees / gripper units),
     check_<episode>.json (action errors per joint, forecast agreement, per-step latency of forecast and policy),
     actions_<episode>.npz (predicted and recorded actions in joint units).
Not part of the check: the live SO-101 lift (SAM 3.1 + sensor depth + kinematics, lift_so101_v2.py), which runs offline
over whole videos today; its output for the episode (lifted_v2/<dataset>/episode_XXXX.npz) is read instead.
  python infer_check_so101.py <run dir> --dataset A|B|C|<name> --episode N [--ckpt model_final] [--replan 8]
         [--wm /dev/shm/physgo/runs/so101/so101_rel4_lab/wm/best.pt] [--out <dir>]
Run with the backbone's env (vilt: the original env; smolvla: env_vla).
"""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, "/dev/shm/physgo/code_so101/ATM")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_policy_so101 import load_policy  # noqa: E402
from physgs.data.format import MIN_POINTS  # noqa: E402
from physgs.model.joint_wm import JointWorldModel, length_scale  # noqa: E402

PG = "/dev/shm/physgo/so101/policy_h5"
LIFT = "/dev/shm/physgo/so101/lifted_v2"
NAMES = {"A": "so101-left_arm_task_red_cube_out_of_pot_into_the_orange_hollow_cube",
         "B": "so101_left_arm_open_drawer_pick_red_cube_stack_on_blue_cube",
         "C": "so101-left_arm_task_drawer_placing_red_squre_block"}
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class OnlineForecaster:
    """The world model run frame by frame on one episode's lifted Gaussians (no look-ahead)."""

    def __init__(self, wm_path, is_robot, present, task_emb, fps, slots, horizon=15, num_points=512, device="cuda"):
        ck = torch.load(wm_path, map_location=device)
        self.m = JointWorldModel(**ck["cfg"]).to(device).eval()
        self.m.load_state_dict(ck["model"])
        self.dev, self.H, self.P, self.O = device, horizon, num_points, slots
        self.is_robot = torch.as_tensor(is_robot, device=device)[None]
        self.present = torch.as_tensor(present, device=device)[None]
        self.task_emb = torch.as_tensor(task_emb, device=device, dtype=torch.float32).view(1, -1)
        self.fps = torch.tensor([float(fps)], device=device)
        self.state, self.center, self.L, self.t = None, None, None, 0

    @torch.no_grad()
    def step(self, gauss, npts):
        """gauss (O, >=P, 7), npts (O,) of the current frame -> forecast (H + 1, O, 143)."""
        g_all = torch.as_tensor(np.asarray(gauss, np.float32), device=self.dev)[None]
        g = g_all[:, :, :self.P]
        vis = (torch.as_tensor(np.asarray(npts), device=self.dev)[None] >= MIN_POINTS) & self.present
        if self.center is None:   # first frame: scene centre = mean centroid of the visible objects over ALL stored
            # points, as physgs.data.format.pack computes it (packed with every stored point, 1024; the model reads 512)
            self.center = (g_all[..., :3].mean(2) * vis[..., None]).sum(1) / vis.sum(1, keepdim=True).clamp(min=1)
        g = torch.cat([g[..., :3] - self.center[:, None, None], g[..., 3:]], -1)
        if self.L is None:
            self.L = length_scale(g, vis, self.is_robot)
            self.ctx = self.m.context(self.is_robot, self.present, self.task_emb, self.fps, self.L)
            self.state = self.m.initial(1, self.O, self.dev)
        obs = self.m.observations(self.m.to_units(g, self.L), vis)
        self.state, _ = self.m.step(self.state, self.ctx, obs, first=(self.t == 0), sample=False)
        steps = [self.state] + self.m.imagine(self.state, self.ctx, self.H, sample=False)
        self.t += 1
        return torch.stack([self.m.forecast_features(s, self.state, self.is_robot, self.present) for s in steps],
                           1)[0].float().cpu().numpy()


def gpu_busy():
    """GPU utilization % (other jobs on the GPU inflate the latencies), or None if it cannot be read."""
    try:
        return int(torch.cuda.utilization())
    except Exception:  # noqa: BLE001  (needs pynvml, absent in the original env)
        try:
            import subprocess
            dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
            return int(subprocess.check_output(["nvidia-smi", "-i", dev, "--query-gpu=utilization.gpu",
                                                "--format=csv,noheader,nounits"], text=True).strip())
        except Exception:  # noqa: BLE001
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--ckpt", default="model_final")
    ap.add_argument("--num_steps", type=int, default=0, help="smolvla: flow-matching steps at inference (0: the config's, 10)")
    ap.add_argument("--bf16", action="store_true", help="policy inference under bf16 autocast (speed test)")
    ap.add_argument("--replan", type=int, default=8, help="smolvla: frames executed from each chunk")
    ap.add_argument("--wm", default="/dev/shm/physgo/runs/so101/so101_rel4_lab/wm/best.pt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    name = NAMES.get(args.dataset, args.dataset)
    out = args.out or f"{args.run}/infer_check"
    os.makedirs(out, exist_ok=True)
    tag = f"{name[:20]}_ep{args.episode}_{args.ckpt}"
    policy, cfg = load_policy(args.run, args.ckpt)
    if args.num_steps and cfg.get("backbone", "vilt") == "smolvla":
        policy.policy.config.num_steps = args.num_steps
    variant = (f"_ns{args.num_steps}" if args.num_steps else "") + ("_bf16" if args.bf16 else "")
    tag = tag + variant
    smol = cfg.get("backbone", "vilt") == "smolvla"
    keys = list(cfg.extra_state_keys)
    nrm = json.load(open(f"{PG}/norm.json"))
    lo, hi = np.array(nrm["min"]), np.array(nrm["max"])
    unit = lambda a: (np.asarray(a) + 1) / 2 * (hi - lo) + lo  # noqa: E731

    with h5py.File(f"{PG}/{name}_demo/demo_{args.episode}.hdf5", "r") as h:
        r = h["root"]
        top, wrist = r["top/video"][0].transpose(0, 2, 3, 1), r["wrist/video"][0].transpose(0, 2, 3, 1)
        act_rec, emb = r["actions"][:], r["task_emb_bert"][:]
        extra = {k: r["extra_states"][k][:] for k in keys}
    lift = np.load(f"{LIFT}/{name}/episode_{args.episode:04d}.npz")
    slots = int(cfg.model_cfg.forecast_cfg.object_slots)
    O_e = lift["npts"].shape[1]
    gauss = np.zeros((lift["npts"].shape[0], slots) + lift["gauss"].shape[2:], np.float32)
    gauss[:, :O_e] = lift["gauss"][:, :slots]
    npts = np.zeros((lift["npts"].shape[0], slots), np.int64)
    npts[:, :O_e] = lift["npts"][:, :slots]
    is_robot = np.zeros(slots, bool)
    is_robot[:O_e] = lift["is_robot"][:slots]
    present = np.arange(slots) < O_e
    fc_pre = np.load(f"{cfg.forecast_dir}/{name}_demo/demo_{args.episode}.npy")
    T = len(act_rec)
    assert len(gauss) == T == len(fc_pre) == len(top), (len(gauss), T, len(fc_pre), len(top))
    fcr = OnlineForecaster(args.wm, is_robot, present, lift["task_emb"], float(lift["fps"]), slots)

    rel = np.array(cfg.dataset_cfg.rel_scale, np.float32) if cfg.dataset_cfg.get("relative_actions", False) else None
    state_t = lambda t: np.concatenate([extra["joint_states"][t, :5], extra["gripper_states"][t, :1]])  # noqa: E731
    pred, t_fc, t_pol, t_plan, fc_err, chunk, t0 = [], [], [], [], [], None, 0
    if hasattr(policy, "reset"):
        policy.reset()
    for t in range(T):
        torch.cuda.synchronize()
        a0 = time.perf_counter()
        fc = fcr.step(gauss[t], npts[t])
        torch.cuda.synchronize()
        t_fc.append(time.perf_counter() - a0)
        fc_err.append(float(np.abs(fc - fc_pre[t]).max()))
        obs = np.stack([top[t], wrist[t]])[None]
        ex = {k: v[t][None].astype(np.float32) for k, v in extra.items()}
        a0 = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            if smol:
                plan = chunk is None or t - t0 >= args.replan
                if plan:
                    chunk = policy.predict_chunk(obs, ex, fc[None]).clamp(-1, 1).cpu().numpy()[0]
                    if rel is not None:   # relative-action run: the state at the planning frame added back
                        chunk = chunk * rel + state_t(t)
                    t0 = t
                a = chunk[min(t - t0, len(chunk) - 1)]
            else:
                plan = True
                a = policy.act(obs, emb[None].astype(np.float32), ex, fc[None])[0][0]
                if rel is not None:
                    a = a * rel + state_t(t)
        torch.cuda.synchronize()
        t_pol.append(time.perf_counter() - a0)
        if plan and t > 0:          # the first call includes CUDA / kernel warm-up
            t_plan.append(t_pol[-1])
        pred.append(a)
    pred = np.array(pred)
    P, Rr = unit(pred), unit(act_rec)
    err = np.abs(P - Rr)
    import glob as _g   # MSE on normalized actions: [-1, 1] scale and z-score with the 15 demos' per-joint mean / std
    demo = np.concatenate([h5py.File(f, "r")["root/actions"][:] for f in sorted(_g.glob(f"{PG}/*_demo/bc_train_5/*.hdf5"))])
    d = pred - act_rec
    res = dict(run=args.run, ckpt=args.ckpt, backbone=cfg.get("backbone", "vilt"), dataset=name, episode=args.episode,
               frames=T, replan=args.replan if smol else 1,
               action_error={j: round(float(err[:, i].mean()), 3) for i, j in enumerate(JOINTS)},
               arm_deg=round(float(err[:, :5].mean()), 3),
               mse=round(float(np.mean(d ** 2)), 5), mse_z=round(float(np.mean((d / (demo.std(0) + 1e-8)) ** 2)), 5),
               forecast_online_vs_precomputed_max_abs=round(float(np.max(fc_err)), 6),
               forecast_scale_max_abs=round(float(np.abs(fc_pre).max()), 3),
               latency_ms=dict(forecast_step_median=round(1000 * float(np.median(t_fc[1:])), 2),
                               policy_call_median=round(1000 * float(np.median(t_plan)), 2),
                               policy_call_p90=round(1000 * float(np.percentile(t_plan, 90)), 2),
                               gpu_busy=gpu_busy(),
                               note="per 15 fps step (66.7 ms budget). policy_call = one act() (vilt) or one chunk "
                                    "(smolvla, every --replan steps; the steps between cost ~0). First call excluded. "
                                    "Inflated when the GPU is shared (gpu_busy %, sampled at the end)."))
    json.dump(res, open(f"{out}/check_{tag}.json", "w"), indent=1)
    np.savez(f"{out}/actions_{tag}.npz", predicted=P, recorded=Rr)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    ts = np.arange(T) / 15.0
    for i, j in enumerate(JOINTS):
        a_ = ax.flat[i]
        a_.plot(ts, Rr[:, i], color="0.3", lw=1.6, label="recorded")
        a_.plot(ts, P[:, i], color="tab:blue", lw=1.0, label="predicted")
        a_.set_title(f"{j}: mean |err| {err[:, i].mean():.2f} {'' if i == 5 else 'deg'}", fontsize=10)
        a_.grid(alpha=0.3)
    ax.flat[0].legend(fontsize=8)
    for a_ in ax[1]:
        a_.set_xlabel("time (s)")
    fig.suptitle(f"{res['backbone']} {args.ckpt} | {name[:50]} ep {args.episode} | arm {res['arm_deg']} deg | "
                 f"forecast online vs precomputed {res['forecast_online_vs_precomputed_max_abs']:.1e}", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/actions_{tag}.png", dpi=110)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
