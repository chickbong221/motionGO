"""
postfilter_track.py (2026-11-07): physical track consistency of each lifted object, after postfilter_arm.py. Found in the
dataset-wide check (tests/check_all_v2.py): task A red cube jitter p95 32 mm, single episodes up to 430 mm; e.g. task A
episode 11, frames 154-164: while the cube was carried in the gripper, SAM 3.1's most confident "red block" instance
jumped to another red thing on the table 0.5 m away for 9 frames and back. The lift's jump filter only removes single
frames. Rule: an object moves only when the gripper moves it, so between two sightings a and b its centroid cannot
move more than allowed(a, b) = 4 x its median RMS radius + 2 x the path length of the fastest gripper surface point from
a to b (margin 2x: occlusion shifts a partly seen centroid; wrong-instance jumps are 5-10x over); falling is free (forward kinematics of the recorded joints, kin_v5: covers gripper rotation, not only its centre).
  kept = the largest set of visible frames whose consecutive members are all within allowed (dynamic programming,
  longest path over the frames in time order); the other frames: the object is unseen there
  python postfilter_track.py --kin kin_v5.json [--apply] <episode.npz> ...
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from physgs.data.format import save_episode  # noqa: E402
from so101_kin import GRIPPER_LINKS, SO101  # noqa: E402

URDF = "/dev/shm/physgo/so101/robot/so101_new_calib_camera.urdf"
MIN_POINTS = 16


def keep_mask(c, vis, path, tol, splits=None):
    """c (T, 3) centroids, vis (T,) bool, path (T,) cumulative gripper path, tol -> (T,) bool frames kept: the largest
    set of visible frames in which every pair of consecutive kept frames is physically consistent (longest path in the
    time-ordered DAG, dynamic programming). (2026-11-07: the first, segment-based version linked a single wrong frame
    after an occlusion gap to the earlier segment and then threw away the true later half, task A episode 3 orange cube.)"""
    ts = np.nonzero(vis)[0]
    P = c[ts]
    pa = path[ts]
    n = len(ts)
    best = np.ones(n, int)
    prev = np.full(n, -1)
    for j in range(1, n):
        d = P[j] - P[:j]
        # a downward move is free (falling when the gripper opens, test A3); horizontal and upward motion must come from
        # the gripper. Margin 2x: occlusion shifts a partly seen centroid (A3 grasp 49 mm vs 42); wrong-instance jumps
        # are 5-10x over (A11 314-456 mm vs ~40, C54 180-400 vs ~32)
        dist = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2 + np.maximum(d[:, 2], 0.0) ** 2)
        ok = dist <= 2 * tol + 2 * (pa[j] - pa[:j])
        if ok.any():
            i = np.nonzero(ok)[0]
            k = i[np.argmax(best[i])]
            best[j], prev[j] = best[k] + 1, k
    kept = np.zeros(len(vis), bool)
    j = int(np.argmax(best))
    while j >= 0:
        kept[ts[j]] = True
        j = prev[j]
    if splits is not None:
        dropped = np.nonzero(vis & ~kept)[0]
        if len(dropped):
            runs = np.split(dropped, np.nonzero(np.diff(dropped) > 1)[0] + 1)
            splits.extend((int(r[0]), int(r[-1])) for r in runs)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kin", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("episodes", nargs="+")
    args = ap.parse_args()
    kc = json.load(open(args.kin))
    kin = SO101(URDF, device="cpu", points_per_link=1000)
    s = torch.tensor(kc["joint_scale"], dtype=torch.float64)
    o = torch.tensor(kc["joint_offset"], dtype=torch.float64)
    total = {}
    for path in args.episodes:
        e = dict(np.load(path))
        if "track_postfilter" in e and args.apply:
            print(path, "already filtered")
            continue
        names = [str(n) for n in e["obj_names"]]
        g, npts = e["gauss"][:, :, :512, :3].astype(np.float32), e["npts"]
        with torch.no_grad():
            G = kin.points(torch.tensor(e["state"], dtype=torch.float64) * s + o, GRIPPER_LINKS)[0].numpy()
        step = np.concatenate([[0.0], np.linalg.norm(G[1:] - G[:-1], axis=-1).max(1)])
        cum = np.cumsum(step)
        drop = np.zeros_like(npts, bool)
        for k in range(1, len(names)):
            vis = npts[:, k] >= MIN_POINTS
            if vis.sum() < 2:
                continue
            c = g[:, k].mean(1)
            rad = np.sqrt(((g[:, k] - c[:, None]) ** 2).sum(-1).mean(1))
            splits = []
            kept = keep_mask(c, vis, cum, 2 * float(np.median(rad[vis])), splits)
            drop[:, k] = vis & ~kept
            n = int(drop[:, k].sum())
            total[names[k]] = total.get(names[k], 0) + n
            if n:
                print(f"{path.split('/')[-2][:30]} {path.split('/')[-1]}: {names[k]} {n} of {int(vis.sum())} frames off the track"
                      + (f"; dropped runs {splits[:10]}" if args.verbose else ""), flush=True)
        if args.apply:
            e["gauss"][drop] = 0
            e["npts"][drop] = 0
            e["track_postfilter"] = drop
            save_episode(path, **e)
    print("frames dropped per object:", total, "(applied)" if args.apply else "(report only)")


if __name__ == "__main__":
    main()
