"""
Sanity check of a trained object RSSM: H-step open-loop prediction error of object positions on validation
episodes, against two baselines (objects stay still; objects keep their last velocity).

    python -m physgs.engine.eval_rssm --data data/packed/physai --rssm runs/rssm_physai/last.pt --horizon 16
"""
import argparse

import numpy as np
import torch

from physgs.data.dataset import TokenSeqDataset
from physgs.data.format import Packed
from physgs.engine.common import dataset_split, to_device
from physgs.model.object_rssm import ObjectRSSM


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--rssm", required=True)
    ap.add_argument("--context", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--batches", type=int, default=20)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.rssm, map_location=device)
    model = ObjectRSSM(**ck["cfg"]).to(device).eval()
    model.load_state_dict(ck["model"])
    _, va = dataset_split(args.data, Packed(args.data).num_episodes)
    ds = TokenSeqDataset(args.data, args.context + args.horizon, step=args.context + args.horizon, episodes=va)
    dl = torch.utils.data.DataLoader(ds, 32, shuffle=True)
    C, H = args.context, args.horizon
    err = {"rssm": [], "static": [], "const_vel": []}
    med = {"rssm": [], "static": [], "const_vel": []}
    for i, b in enumerate(dl):
        if i == args.batches:
            break
        b = to_device(b, device)
        ctx = model.context(b["is_robot"], b["present"], b["task"])
        posts, _ = model.observe(b["tokens"][:, :C], b["visible"][:, :C], ctx, sample=False)
        imag = model.imagine({k: posts[-1][k] for k in ("h", "s", "pos", "seen")}, ctx, H, sample=False)
        pred = torch.stack([model.heads(s)["pos"] for s in imag], 1)                      # (B, H, O, 3)
        target = b["pos"][:, C:C + H]                                                       # simulator CoM
        last = b["pos"][:, C - 1:C]
        dt = 1.0                                                                             # in frames
        vel_frame = b["pos"][:, C - 1:C] - b["pos"][:, C - 2:C - 1]
        steps = torch.arange(1, H + 1, device=device).view(1, H, 1, 1).float()
        cands = {"rssm": pred, "static": last.expand_as(target), "const_vel": last + vel_frame * steps * dt}
        mask = torch.isfinite(target).all(-1) & torch.isfinite(last).all(-1) & b["present"][:, None]
        mask &= torch.isfinite(vel_frame).all(-1)
        mask &= posts[-1]["seen"][:, None]  # objects never observed in the context have no position
        for k, p in cands.items():
            e = (p - torch.nan_to_num(target)).norm(dim=-1)                                 # (B, H, O)
            err[k].append(torch.stack([e[:, h][mask[:, h]].mean() for h in range(H)]).cpu().numpy())
            med[k].append(torch.stack([e[:, h][mask[:, h]].median() for h in range(H)]).cpu().numpy())
    for name, table in (("mean", err), ("median", med)):
        for k, v in table.items():
            v = np.nanmean(np.stack(v), 0)
            print(f"{k:10s} {name:6s} position error (m) at steps 1/4/8/{H}: " + " / ".join(f"{v[h - 1]:.3f}" for h in (1, 4, 8, H)))


if __name__ == "__main__":
    main()
