"""
How well does an object encoder recover angular velocity (and velocity)? Validation objects of a packed dataset.

Reports, per object group, the L1 error of the encoder's angular velocity against two trivial predictors
(always zero; the training-set mean), the fraction of variance explained (R^2 against the zero predictor) and the
cosine between predicted and true spin axes, split into shapes whose spin is visible and symmetric shapes
(sphere, cylinder, torus, cone, ball), whose spin about the symmetry axis is invisible at any window length.

    python -m physgs.engine.eval_angvel --data data/packed/physai_v2 --ckpt runs/x/encoder/last.pt [--batches 150]
"""
import argparse
import re

import numpy as np
import torch

from physgs.data.dataset import ObjectFrameDataset
from physgs.data.format import Packed
from physgs.engine.common import split_episodes, to_device
from physgs.model.object_encoder import ObjectAutoencoder

SYMMETRIC = re.compile(r"sphere|cylinder|torus|cone|ball|orange|lemon|apple|pomegranate|avocado|pumpkin|can|bottle|projectile", re.I)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batches", type=int, default=150)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cuda")
    model = ObjectAutoencoder(**ck["cfg"]).cuda().eval()
    model.load_state_dict(ck["model"])
    K = model.num_frames
    d = Packed(a.data)
    _, va = split_episodes(d.num_episodes)
    ds = ObjectFrameDataset(a.data, va, frames=K)
    names = [e["obj_names"] for e in d.meta["episodes"]]
    rng = np.random.default_rng(0)
    pick = rng.choice(len(ds), min(len(ds), a.batches * a.batch), replace=False)
    pred_w, true_w, pred_v, true_v, sym = [], [], [], [], []
    for s in range(0, len(pick), a.batch):
        items = [ds[int(i)] for i in pick[s:s + a.batch]]
        batch = {k: torch.stack([it[k] for it in items]) for k in items[0]}
        batch = to_device(batch, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.encoder(batch["g_hist"])
        pred_w.append(out["angvel"].float().cpu().numpy()); true_w.append(batch["angvel"].cpu().numpy())
        pred_v.append(out["vel"].float().cpu().numpy()); true_v.append(batch["vel"].cpu().numpy())
        for i in pick[s:s + a.batch]:
            f, o = ds.items[int(i)]
            sym.append(bool(SYMMETRIC.search(names[d.frame_episode[f]][o])))
    pw, tw, pv, tv = map(np.concatenate, (pred_w, true_w, pred_v, true_v))
    sym = np.array(sym)
    ok = np.isfinite(tw).all(1) & np.isfinite(tv).all(1)
    print(f"encoder frames {K}; validation objects with labels: {ok.sum()} (symmetric shapes {sym[ok].mean():.2f})")
    mean_w = np.nanmean(tw[ok], 0)
    for label, sel in (("all", ok), ("visible-spin shapes", ok & ~sym), ("symmetric shapes", ok & sym),
                       ("visible-spin, |w| > 2 rad/s", ok & ~sym & (np.linalg.norm(tw, axis=1) > 2))):
        if sel.sum() == 0:
            continue
        e = np.abs(pw[sel] - tw[sel]).sum(1).mean()
        e0 = np.abs(tw[sel]).sum(1).mean()
        em = np.abs(mean_w - tw[sel]).sum(1).mean()
        r2 = 1 - ((pw[sel] - tw[sel]) ** 2).sum() / max((tw[sel] ** 2).sum(), 1e-9)
        nz = np.linalg.norm(tw[sel], axis=1) > 0.5
        cos = np.median((pw[sel][nz] * tw[sel][nz]).sum(1) / (np.linalg.norm(pw[sel][nz], axis=1) * np.linalg.norm(tw[sel][nz], axis=1) + 1e-9)) if nz.any() else np.nan
        ev = np.abs(pv[sel] - tv[sel]).sum(1).mean()
        print(f"  {label:30s} n={sel.sum():6d}  angvel L1 {e:.3f} (zero {e0:.3f}, mean {em:.3f})  R2 {r2:+.3f}  "
              f"median cos {cos:+.3f}   vel L1 {ev:.3f}")


if __name__ == "__main__":
    main()
