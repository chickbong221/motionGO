"""
Contact labels from geometry, for data without simulator contacts (PhysicalAI).

An object is in contact when at least --k of its Gaussians lie within --tau of another token (a robust minimum: at
occlusion boundaries a few pixels take the depth of the object behind). Support surfaces are not
tokens: LIBERO's table and PhysicalAI's ground are excluded, as in LIBERO's simulator labels, which ignore the table).
Distances use the first --points stored Gaussians of each object (farthest-point order, so an even subsample).

    # calibrate tau where the simulator labels exist (LIBERO): precision / recall / F1 per threshold
    python -m physgs.data.geo_contact --data data/packed/libero_spatial256 --calibrate   # -> best k, tau
    # write contact.npy of a PhysicalAI pack (replaces a symlinked contact.npy by a real file)
    python -m physgs.data.geo_contact --data data/packed/physai1080_v2 --k 24 --tau 0.015

Invisible objects (fewer than MIN_POINTS pixels) get NaN; robot tokens keep their (NaN) label in LIBERO.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from physgs.data.format import MIN_POINTS, Packed


@torch.no_grad()
def nearest_other(gauss, npts, points, batch, device, k=1, frames=None):
    """
    Per (frame, object): distance to the nearest other visible object, robust to stray points: for each other object
    the k-th smallest distance from this object's Gaussians to it (k = 1 is the plain minimum; at occlusion boundaries
    a few pixels take the depth of the object behind and put isolated points on its surface). (F, O), inf if none.
    """
    Fn, O = npts.shape
    Fn = min(Fn, frames) if frames else Fn
    out = np.full((Fn, O), np.inf, np.float32)
    for a in range(0, Fn, batch):
        b = min(a + batch, Fn)
        g = torch.from_numpy(np.asarray(gauss[a:b, :, :points, :3], np.float32)).to(device)        # (B, O, p, 3)
        vis = torch.from_numpy(np.asarray(npts[a:b]) >= MIN_POINTS).to(device)                     # (B, O)
        B = g.shape[0]
        d = torch.cdist(g.reshape(B, O * points, 3), g.reshape(B, O * points, 3))                  # (B, Op, Op)
        d = d.reshape(B, O, points, O, points).amin(dim=4)                                         # (B, O, p, O)
        d = d.kthvalue(k, dim=2).values if k > 1 else d.amin(dim=2)                                # (B, O, O)
        bad = ~(vis[:, :, None] & vis[:, None, :]) | torch.eye(O, dtype=torch.bool, device=device)[None]
        d = d.masked_fill(bad, float("inf")).amin(dim=2)
        out[a:b] = d.cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tau", type=float, default=0.01, help="contact distance threshold (m)")
    ap.add_argument("--points", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--k", type=int, default=1, help="k-th nearest point (robust minimum), see nearest_other")
    ap.add_argument("--frames", type=int, default=0, help="--calibrate: use the first N frames only (0 = all)")
    ap.add_argument("--calibrate", action="store_true", help="compare with the pack's own contact labels, write nothing")
    ap.add_argument("--k_grid", default="16,24,32,48", help="--calibrate: k values tried (with every tau)")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = Packed(a.data)
    if a.calibrate:  # pick (k, tau) by F1 against the simulator's labels (LIBERO: contact with anything but the table)
        best = None
        for k in [int(x) for x in a.k_grid.split(",")]:
            t0 = time.time()
            dist = nearest_other(d.gauss, d.npts, a.points, a.batch, device, k=k, frames=a.frames or None)
            n = dist.shape[0]
            vis = np.asarray(d.npts[:n]) >= MIN_POINTS
            lab = np.asarray(d.contact[:n])
            m = vis & np.isfinite(lab) & ~d.is_robot[d.frame_episode[:n]]
            y, x = lab[m] > 0.5, dist[m]
            print(f"k {k}: {n:,} frames in {time.time() - t0:.0f} s, {m.sum():,} labelled object-frames, "
                  f"{y.mean():.3f} in contact (simulator)", flush=True)
            for tau in (0.004, 0.006, 0.008, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.04):
                p = x < tau
                tp, fp, fn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum()
                prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-9)
                print(f"  tau {tau:.4f}: precision {prec:.3f} recall {rec:.3f} F1 {f1:.3f} accuracy {(p == y).mean():.3f}")
                best = max(best or (f1, k, tau), (f1, k, tau))
        print(f"BEST_TAU {best[2]} BEST_K {best[1]} (F1 {best[0]:.3f})")
        json.dump(dict(best_tau=best[2], best_k=best[1], f1=float(best[0]), points=a.points),
                  open(os.path.join(a.data, "geo_contact_calibration.json"), "w"))
        return
    t0 = time.time()
    dist = nearest_other(d.gauss, d.npts, a.points, a.batch, device, k=a.k)
    vis = np.asarray(d.npts) >= MIN_POINTS
    print(f"nearest-other distances for {dist.shape[0]:,} frames in {time.time() - t0:.0f} s", flush=True)
    contact = np.where(vis, (dist < a.tau).astype(np.float32), np.nan).astype(np.float32)
    path = os.path.join(a.data, "contact.npy")
    np.save(path + ".tmp.npy", contact)
    os.replace(path + ".tmp.npy", path)  # replaces a symlink (label-fixed packs link contact.npy) by the new file
    json.dump(dict(tau=a.tau, points=a.points, k=a.k, contact_rate=float(np.nanmean(contact))),
              open(os.path.join(a.data, "geo_contact.json"), "w"))
    print(f"contact.npy written: tau {a.tau} m, {np.nanmean(contact):.3f} of visible object-frames in contact")


if __name__ == "__main__":
    main()
