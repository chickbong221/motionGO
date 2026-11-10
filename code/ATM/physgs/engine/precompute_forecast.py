"""
Policy inputs: for every step t of every LIBERO demo, filter the object RSSM on tokens[0..t] and imagine
H steps ahead (with --include_now the current filtered state is prepended as step 0, as ATM tracks start at the
current point positions). Writes <out>/<task>/<demo>.npy with shape (T, H [+ 1], O, F + 2), F + 2 = forecast +
is_robot + present, float32.

    python -m physgs.engine.precompute_forecast --data data/packed/libero_spatial \
        --rssm runs/rssm_libero_spatial/last.pt --out data/forecast/libero_spatial
"""
import argparse
import os

import numpy as np
import torch

from physgs.data.format import MIN_POINTS, Packed
from physgs.model.object_rssm import ObjectRSSM


@torch.no_grad()
def forecast_episodes(model, d, tokens, eps, horizon, device, include_now=False):
    """Batched over episodes (padded in time). Returns list of (T_e, H, O, F) arrays."""
    T_max = max(d.ep_start[e + 1] - d.ep_start[e] for e in eps)
    B, O = len(eps), tokens.shape[1]
    tok = np.zeros((B, T_max, O, tokens.shape[2]), np.float32)
    vis = np.zeros((B, T_max, O), bool)
    for i, e in enumerate(eps):
        s, t = d.ep_start[e], d.ep_start[e + 1]
        x = np.asarray(tokens[s:t], np.float32).copy()
        x[..., 0:3] -= d.center[e]
        v = np.asarray(d.npts[s:t]) >= MIN_POINTS
        x[~v] = 0
        tok[i, :t - s], vis[i, :t - s] = x, v
    tok, vis = torch.from_numpy(tok).to(device), torch.from_numpy(vis).to(device)
    present = torch.from_numpy(d.present[eps]).to(device)
    is_robot = torch.from_numpy(d.is_robot[eps]).to(device)
    task = torch.from_numpy(d.task_emb[eps]).to(device)
    ctx = model.context(is_robot, present, task)

    state = model.initial(B, O, device)
    out = []
    for t in range(T_max):
        state, _ = model.obs_step(state, tok[:, t], vis[:, t], ctx, first=(t == 0), sample=False)
        belief = model.heads(state)
        imag = model.imagine({k: state[k] for k in ("h", "s", "pos", "seen")}, ctx, horizon, sample=False)
        steps = []
        for s_ in ([state] if include_now else []) + list(imag):  # step 0 = the current (filtered) state
            p = model.heads(s_)
            steps.append(torch.cat([p["pos"], p["vel"], p["angvel"], torch.sigmoid(p["contact"])[..., None], p["z"],
                                    belief["belief_mu"], belief["belief_logstd"],
                                    is_robot[..., None].float(), (present & state["seen"])[..., None].float()], -1))  # unseen objects are absent
        out.append(torch.stack(steps, 1).float().cpu())                                 # (B, H, O, F+2), float32
    out = torch.stack(out, 1).numpy()                                                   # (B, T, H, O, F+2)
    return [out[i, :d.ep_start[e + 1] - d.ep_start[e]] for i, e in enumerate(eps)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--rssm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--include_now", type=int, default=0, help="1: prepend the current state as step 0")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.rssm, map_location=device)
    model = ObjectRSSM(**ck["cfg"]).to(device).eval()
    model.load_state_dict(ck["model"])
    d = Packed(args.data)
    tokens = np.load(os.path.join(args.data, "tokens.npy"), mmap_mode="r")
    for s in range(0, d.num_episodes, args.batch):
        eps = list(range(s, min(s + args.batch, d.num_episodes)))
        for e, fc in zip(eps, forecast_episodes(model, d, tokens, eps, args.horizon, device, bool(args.include_now))):
            src = d.meta["episodes"][e]["file"]
            task = os.path.basename(os.path.dirname(src))
            dst = os.path.join(args.out, task, os.path.basename(src).replace(".npz", ".npy"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            np.save(dst, fc)
        print(f"{min(s + args.batch, d.num_episodes)}/{d.num_episodes} episodes", flush=True)


if __name__ == "__main__":
    main()
