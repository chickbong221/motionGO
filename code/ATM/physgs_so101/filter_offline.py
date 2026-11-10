"""filter_offline.py (2026-11-07): lift_so101_v2's per-object consistency filter applied to an already lifted episode
(no SAM re-run): frames dropped per object and the radius / jitter / jumps after the filter.
  python filter_offline.py <episode.npz> ..."""
import sys

import numpy as np

MIN_POINTS = 16
for path in sys.argv[1:]:
    e = dict(np.load(path))
    names = [str(n) for n in e["obj_names"]]
    g, npts = e["gauss"][:, :, :512, :3].astype(np.float32), e["npts"].copy()
    print(path.split("/")[-2][:40], path.split("/")[-1])
    for k in range(1, len(names)):
        vis = npts[:, k] >= MIN_POINTS
        if vis.sum() < 3:
            print(f"  {names[k]:13s} not visible")
            continue
        c = g[:, k].mean(1)
        rad = np.sqrt(((g[:, k] - c[:, None]) ** 2).sum(-1).mean(1))
        rmed = float(np.median(rad[vis]))
        bad = vis & (rad > 2 * rmed)
        ok = vis & ~bad
        d = lambda a_, b_: np.linalg.norm(a_ - b_, axis=1)
        mid = ok[1:-1] & ok[:-2] & ok[2:]
        jump = mid & (np.minimum(d(c[1:-1], c[:-2]), d(c[1:-1], c[2:])) > 2 * rmed) & (d(c[2:], c[:-2]) < rmed)
        bad[1:-1] |= jump
        keep = vis & ~bad
        both = keep[2:] & keep[1:-1] & keep[:-2]
        acc = np.linalg.norm(c[2:] - 2 * c[1:-1] + c[:-2], axis=1)[both] * 1000
        j2 = both & (d(c[1:-1], c[:-2]) > 0.02) & (d(c[1:-1], c[2:]) > 0.02)
        print(f"  {names[k]:13s} median radius {rmed * 1000:5.1f} mm: dropped {int((bad & (rad > 2 * rmed)).sum())} size + "
              f"{int(jump.sum())} jump frames; after: visible {keep.mean():5.1%}, radius p90 {np.percentile(rad[keep], 90) * 1000:5.1f} mm, "
              f"jitter p95 {np.percentile(acc, 95) if len(acc) else np.nan:5.1f} mm, 2-cm jumps {int(j2.sum())}")
