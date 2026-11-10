"""
Policy forecasts from the joint world model (physgs.model.joint_wm): for every step t of every LIBERO demo, filter
the Gaussians of steps 0..t and imagine H steps ahead (with --include_now the filtered state of step t is step 0; ATM:
the current + 15 future = 16 entries). Writes <out>/<task>/<demo>.npy with shape (T, H [+ 1], O, 143), float32
(JointWorldModel.forecast_features): pos, vel (units of the episode length scale L, relative to the scene centre),
e = E s (128), belief mean / log-std of the filtered state (normalized, kept during imagination), log radius, is_robot,
present & seen.

    python -m physgs.engine.precompute_forecast_joint --data data/packed/libero_spatial \
        --model runs/joint/libero/best.pt --out data/forecast_joint/libero_spatial --horizon 15 --include_now 1
"""
import argparse
import os

import numpy as np
import torch

from physgs.data.format import MIN_POINTS, Packed
from physgs.data.joint_dataset import episode_length_scale
from physgs.model.joint_wm import JointWorldModel


def load_model(path, device):
    ck = torch.load(path, map_location=device)
    m = JointWorldModel(**ck["cfg"]).to(device).eval()
    m.load_state_dict(ck["model"])
    return m


@torch.no_grad()
def forecast_episodes(model, d, eps, horizon, device, include_now=True, num_points=512):
    T_max = max(d.ep_start[e + 1] - d.ep_start[e] for e in eps)
    B, O = len(eps), d.gauss.shape[1]
    P = min(num_points, d.gauss.shape[2])
    g = np.zeros((B, T_max, O, P, 7), np.float32)
    vis = np.zeros((B, T_max, O), bool)
    for i, e in enumerate(eps):
        s, t = d.ep_start[e], d.ep_start[e + 1]
        x = np.array(d.gauss[s:t, :, :P], np.float32)  # a copy: a float32 pack would give a read-only view
        x[..., :3] -= d.center[e]
        g[i, :t - s], vis[i, :t - s] = x, np.asarray(d.npts[s:t]) >= MIN_POINTS
    present = torch.from_numpy(d.present[eps]).to(device)
    is_robot = torch.from_numpy(d.is_robot[eps]).to(device)
    vis_t = torch.from_numpy(vis).to(device) & present[:, None]
    fps = torch.tensor([float(d.meta["episodes"][e].get("fps", 20.0)) for e in eps], device=device)
    L = torch.tensor([episode_length_scale(d, e, P) for e in eps], device=device)
    obs = model.observations(model.to_units(torch.from_numpy(g).to(device), L), vis_t)
    ctx = model.context(is_robot, present, torch.from_numpy(d.task_emb[eps]).to(device), fps, L)
    st = model.initial(B, O, device)
    out = []
    for t in range(T_max):
        st, _ = model.step(st, ctx, {k: v[:, t] for k, v in obs.items()}, first=(t == 0), sample=False)
        steps = ([st] if include_now else []) + model.imagine(st, ctx, horizon, sample=False)
        out.append(torch.stack([model.forecast_features(s, st, is_robot, present) for s in steps], 1).float().cpu())
    out = torch.stack(out, 1).numpy()                                                    # (B, T, H, O, 143)
    return [out[i, :d.ep_start[e + 1] - d.ep_start[e]] for i, e in enumerate(eps)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--include_now", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--num_points", type=int, default=512)
    ap.add_argument("--episodes", default=None, help="comma-separated episode indices (default: all)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device)
    d = Packed(args.data)
    todo = [int(e) for e in args.episodes.split(",")] if args.episodes else list(range(d.num_episodes))
    for s in range(0, len(todo), args.batch):
        eps = todo[s:s + args.batch]
        for e, fc in zip(eps, forecast_episodes(model, d, eps, args.horizon, device, bool(args.include_now), args.num_points)):
            src = d.meta["episodes"][e]["file"]
            dst = os.path.join(args.out, os.path.basename(os.path.dirname(src)), os.path.basename(src).replace(".npz", ".npy"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            np.save(dst, fc)
        print(f"{min(s + args.batch, len(todo))}/{len(todo)} episodes", flush=True)


if __name__ == "__main__":
    main()
