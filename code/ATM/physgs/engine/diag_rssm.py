"""
Diagnose the object-RSSM open-loop error (companion of eval_rssm.py, same windows and model).

Splits the H-step position error by whether the object was visible at the last context frame, reports
percentiles, and adds two baselines that use the encoder's token positions (what the RSSM sees) instead of
simulator positions, so that perception error and dynamics error can be told apart.

    python -m physgs.engine.diag_rssm --data data/packed/physai --rssm runs/x/rssm_physai/last.pt --device cpu
"""
import argparse

import numpy as np
import torch

from physgs.data.dataset import TokenSeqDataset
from physgs.data.format import Packed
from physgs.engine.common import dataset_split, to_device
from physgs.model.object_rssm import ObjectRSSM, TOKEN_SLICES


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--rssm", required=True)
    ap.add_argument("--context", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    ck = torch.load(a.rssm, map_location=a.device)
    model = ObjectRSSM(**ck["cfg"]).to(a.device).eval()
    model.load_state_dict(ck["model"])
    _, va = dataset_split(a.data, Packed(a.data).num_episodes)
    ds = TokenSeqDataset(a.data, a.context + a.horizon, step=a.context + a.horizon, episodes=va)
    dl = torch.utils.data.DataLoader(ds, 32, shuffle=True, generator=torch.Generator().manual_seed(0))
    C, H = a.context, a.horizon
    rows = {k: [] for k in ("rssm", "static_gt", "constvel_gt", "static_tok", "constvel_tok")}
    groups = []          # per (window, step, object): visible-at-last-context, n visible context frames
    speeds = []
    for i, b in enumerate(dl):
        if i == a.batches:
            break
        b = to_device(b, a.device)
        ctx = model.context(b["is_robot"], b["present"], b["task"])
        posts, _ = model.observe(b["tokens"][:, :C], b["visible"][:, :C], ctx, sample=False)
        imag = model.imagine({k: posts[-1][k] for k in ("h", "s", "pos", "seen")}, ctx, H, sample=False)
        pred = torch.stack([model.heads(s)["pos"] for s in imag], 1)                       # (B,H,O,3)
        target = b["pos"][:, C:C + H]
        steps = torch.arange(1, H + 1, device=a.device).view(1, H, 1, 1).float()
        gt_last, gt_prev = b["pos"][:, C - 1:C], b["pos"][:, C - 2:C - 1]
        tok_pos = b["tokens"][..., TOKEN_SLICES["pos"]]
        tk_last, tk_prev = tok_pos[:, C - 1:C], tok_pos[:, C - 2:C - 1]
        cands = dict(rssm=pred, static_gt=gt_last.expand_as(target), constvel_gt=gt_last + (gt_last - gt_prev) * steps,
                     static_tok=tk_last.expand_as(target), constvel_tok=tk_last + (tk_last - tk_prev) * steps)
        mask = torch.isfinite(target).all(-1) & torch.isfinite(gt_last).all(-1) & torch.isfinite(gt_prev).all(-1) \
            & b["present"][:, None]
        vis_last = b["visible"][:, C - 1]                                                   # (B,O)
        vis_prev = b["visible"][:, C - 2]
        n_vis = b["visible"][:, :C].sum(1)                                                  # (B,O)
        for k, p in cands.items():
            e = (p - torch.nan_to_num(target)).norm(dim=-1)                                 # (B,H,O)
            rows[k].append(e[mask].cpu().numpy())
        Bv = (vis_last & vis_prev)[:, None].expand_as(mask)[mask]
        groups.append(np.stack([Bv.cpu().numpy(), n_vis[:, None].expand_as(mask)[mask].cpu().numpy(),
                                steps.squeeze(-1).squeeze(-1).expand(mask.shape[0], H)[:, :, None].expand_as(mask)[mask].cpu().numpy()], 1))
        speeds.append(((gt_last - gt_prev).norm(dim=-1).expand_as(mask))[mask].cpu().numpy())
    g = np.concatenate(groups)
    sp = np.concatenate(speeds)
    E = {k: np.concatenate(v) for k, v in rows.items()}
    vis2, nvis, step = g[:, 0].astype(bool), g[:, 1], g[:, 2]
    print(f"objects x steps evaluated: {len(step)}; visible in the last 2 context frames: {vis2.mean():.2f}; "
          f"never visible in context: {(nvis == 0).mean():.2f}")
    for name, sel in (("all", np.ones_like(vis2)), ("visible (last 2 ctx frames)", vis2), ("not visible", ~vis2),
                      ("visible & moving (>1 cm/frame)", vis2 & (sp > 0.01))):
        print(f"\n== {name}: {sel.mean():.2f} of samples")
        for k, e in E.items():
            line = []
            for h in (1, 4, 8, 16):
                s = sel & (step == h)
                if s.sum() == 0:
                    line.append("   -  ")
                    continue
                line.append(f"{e[s].mean():.3f}/{np.median(e[s]):.3f}/{np.percentile(e[s], 90):.3f}")
            print(f"  {k:13s} mean/median/p90 at steps 1,4,8,16: " + "  ".join(line))


if __name__ == "__main__":
    main()
