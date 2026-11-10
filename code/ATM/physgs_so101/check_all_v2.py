"""
check_all_v2.py (2026-11-07): dataset-wide quality summary of the lifted SO-101 episodes (lift_so101_v2 + postfilter_arm),
using the first 512 Gaussians as training does. Per task and object over all episodes: visibility (median and range
over episodes), median RMS radius, centroid jitter (|second difference| at 15 fps: median and p95 over all frames),
2-cm jump frames per 1000 visible frames, frames dropped by the arm filter; gripper the same; the worst episodes.
  python check_all_v2.py /dev/shm/physgo/so101/lifted_v2
"""
import glob
import sys

import numpy as np

MIN_POINTS = 16
root = sys.argv[1]
TASK = lambda d: "A" if "pot" in d else ("B" if "open_drawer" in d else "C")  # noqa: E731
stats, worst = {}, []
for path in sorted(glob.glob(f"{root}/*/episode_*.npz")):
    if path.endswith("_cam.npz"):
        continue
    e = np.load(path)
    task = TASK(path.split("/")[-2])
    names = [str(n) for n in e["obj_names"]]
    g, npts = e["gauss"][:, :, :512, :3].astype(np.float32), e["npts"]
    arm = e["arm_postfilter"] if "arm_postfilter" in e.files else np.zeros_like(npts, bool)
    ep = int(path.split("_")[-1].split(".")[0])
    for k, n in enumerate(names):
        s = stats.setdefault((task, n), dict(vis=[], rad=[], acc=[], jumps=0, nvis=0, arm=0, eps=0))
        vis = npts[:, k] >= MIN_POINTS
        s["vis"].append(vis.mean())
        s["eps"] += 1
        s["arm"] += int(arm[:, k].sum())
        if vis.sum() < 3:
            worst.append((task, ep, n, f"visible {vis.mean():.0%}"))
            continue
        c = g[:, k].mean(1)
        rad = np.sqrt(((g[:, k] - c[:, None]) ** 2).sum(-1).mean(1))
        s["rad"].append(np.median(rad[vis]))
        both = vis[2:] & vis[1:-1] & vis[:-2]
        acc = np.linalg.norm(c[2:] - 2 * c[1:-1] + c[:-2], axis=1)[both]
        s["acc"].append(acc)
        d = lambda a, b: np.linalg.norm(a - b, axis=1)  # noqa: E731
        j = both & (d(c[1:-1], c[:-2]) > 0.02) & (d(c[1:-1], c[2:]) > 0.02)
        s["jumps"] += int(j.sum())
        s["nvis"] += int(vis.sum())
        if k and (vis.mean() < 0.1 or (len(acc) and np.percentile(acc, 95) > 0.03)):
            worst.append((task, ep, n, f"visible {vis.mean():.0%}, jitter p95 {np.percentile(acc, 95) * 1000 if len(acc) else 0:.0f} mm"))
print(f"{'task':4s} {'object':12s} {'eps':>3s} {'visible med [min,max]':>22s} {'radius':>7s} {'jitter med':>10s} {'p95':>6s} {'jumps/1k':>8s} {'arm-dropped':>11s}")
for (task, n), s in sorted(stats.items()):
    acc = np.concatenate(s["acc"]) * 1000 if s["acc"] else np.array([np.nan])
    v = np.array(s["vis"])
    print(f"{task:4s} {n:12s} {s['eps']:3d} {np.median(v):8.0%} [{v.min():4.0%},{v.max():4.0%}] {np.median(s['rad']) * 1000 if s['rad'] else np.nan:6.1f}mm "
          f"{np.median(acc):8.1f}mm {np.percentile(acc, 95):5.1f} {1000 * s['jumps'] / max(s['nvis'], 1):8.1f} {s['arm']:11d}")
print(f"\nflagged object-episodes ({len(worst)}): visible < 10% or jitter p95 > 30 mm")
for w in sorted(worst):
    print("  ", *w)
