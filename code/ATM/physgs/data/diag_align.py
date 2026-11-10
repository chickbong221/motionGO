"""
Time alignment between lifted Gaussians and simulator labels in a packed dataset.

For moving, small objects (extent < 0.6 m, so the centroid tracks the CoM), finds the label shift k (in stored
frames, fractional via linear interpolation) that best matches the Gaussian centroid at t, and checks whether the
velocity labels are consistent with the CoM labels and with the centroid motion.

    python -m physgs.data.diag_align --data data/packed/physai
"""
import argparse

import numpy as np

from physgs.data.format import MIN_POINTS, Packed
from physgs.engine.common import split_episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--frames", type=int, default=6000)
    a = ap.parse_args()
    d = Packed(a.data)
    gauss = np.load(f"{a.data}/gauss.npy", mmap_mode="r")
    fps = float(d.meta["episodes"][0].get("fps", 15.0))
    _, va = split_episodes(d.num_episodes)
    rng = np.random.default_rng(0)
    frames = [(e, f) for e in va for f in range(d.ep_start[e] + 3, d.ep_start[e + 1] - 3)]
    pick = rng.choice(len(frames), min(a.frames, len(frames)), replace=False)
    shifts = np.arange(-3.0, 2.01, 0.25)
    err = {s: [] for s in shifts}
    vel_lab, vel_pos_fd, vel_cen_fd = [], [], []
    for i in pick:
        e, f = frames[i]
        for o in range(d.npts.shape[1]):
            if d.npts[f, o] < MIN_POINTS or d.npts[f - 1, o] < MIN_POINTS:
                continue
            P = d.pos[f - 3:f + 4, o]
            if not np.isfinite(P).all() or not np.isfinite(d.vel[f, o]).all():
                continue
            if np.linalg.norm(d.vel[f, o]) < 0.3:
                continue
            g = np.asarray(gauss[f, o, :, :3], np.float32)
            if np.linalg.norm(g.max(0) - g.min(0)) > 0.6:
                continue
            cen = g.mean(0)
            cen_prev = np.asarray(gauss[f - 1, o, :, :3], np.float32).mean(0)
            for s in shifts:
                k = 3 + s
                lo = int(np.floor(k)); w = k - lo
                p = P[lo] * (1 - w) + P[min(lo + 1, 6)] * w
                err[s].append(np.linalg.norm(cen - p))
            vel_lab.append(d.vel[f, o])
            vel_pos_fd.append((P[4] - P[2]) * fps / 2)           # central difference of CoM labels
            vel_cen_fd.append((cen - cen_prev) * fps)               # backward difference of centroids
    print(f"moving small objects analysed: {len(vel_lab)} (fps {fps})")
    print("label shift (stored frames) -> median / mean distance centroid(t) vs CoM label(t + shift):")
    best = min(shifts, key=lambda s: np.median(err[s]))
    for s in shifts:
        print(f"  {s:+5.2f}: {np.median(err[s]):.3f} / {np.mean(err[s]):.3f}" + ("   <- best" if s == best else ""))
    vl, vp, vc = map(np.array, (vel_lab, vel_pos_fd, vel_cen_fd))
    cos = lambda x, y: np.median(np.sum(x * y, -1) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1) + 1e-9))
    print(f"velocity label vs CoM-label central difference: median |diff| {np.median(np.linalg.norm(vl - vp, axis=-1)):.3f} m/s, "
          f"median cos {cos(vl, vp):.3f}, median speed ratio {np.median(np.linalg.norm(vl, axis=-1) / (np.linalg.norm(vp, axis=-1) + 1e-9)):.3f}")
    print(f"velocity label vs centroid backward difference: median |diff| {np.median(np.linalg.norm(vl - vc, axis=-1)):.3f} m/s, "
          f"median cos {cos(vl, vc):.3f}")


if __name__ == "__main__":
    main()
