"""
postfilter_arm.py (2026-11-07): after lift_so101_v2.py, drop object frames whose points lie on the robot arm instead of
the object. Found in task C episode 41: while the red block was hidden behind the arm (frames 60-121), SAM 3.1's "red
block" mask lay on red wiring of the arm, 25 cm above the table; the lift's robot-point removal (8 mm from the model
surface) keeps wires that stick out further. Rule: in a frame, if more than half of an object's points lie within
`--k` x the object's median RMS radius (capped at the gripper token's RMS radius) of the posed arm links (base .. wrist, NOT the gripper links, so a held cube
stays), the object is unseen in that frame. Scale = the object's own size. Robot pose = kinematics of the recorded
joint states (kin_v5, the lift's calibration), so no images are needed.
  python postfilter_arm.py --kin kin_v5.json <episode.npz> ...           report only
  python postfilter_arm.py --kin kin_v5.json --apply <episode.npz> ...   rewrite the episodes (atomic), mark them
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from physgs.data.format import save_episode  # noqa: E402
from so101_kin import ARM_LINKS, GRIPPER_LINKS, SO101  # noqa: E402

URDF = "/dev/shm/physgo/so101/robot/so101_new_calib_camera.urdf"
MIN_POINTS = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kin", required=True)
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("episodes", nargs="+")
    args = ap.parse_args()
    kc = json.load(open(args.kin))
    dev = "cpu"   # the lift workers fill the GPUs; a KD-tree per frame on CPU is fast enough
    kin = SO101(URDF, device=dev, points_per_link=1500)
    s = torch.tensor(kc["joint_scale"], dtype=torch.float64, device=dev)
    o = torch.tensor(kc["joint_offset"], dtype=torch.float64, device=dev)
    arm_links = tuple(l for l in ARM_LINKS if l not in GRIPPER_LINKS)
    total = {}
    for path in args.episodes:
        e = dict(np.load(path))
        if "arm_postfilter" in e and args.apply:
            print(path, "already filtered")
            continue
        names = [str(n) for n in e["obj_names"]]
        g = e["gauss"][:, :, :512, :3].astype(np.float32)
        npts = e["npts"]
        drop = np.zeros_like(npts, bool)
        with torch.no_grad():
            A = kin.points(torch.tensor(e["state"], dtype=torch.float64, device=dev) * s + o, arm_links)[0].numpy()
        G = g[:, 0]
        r_grip = float(np.median(np.sqrt(((G - G.mean(1, keepdims=True)) ** 2).sum(-1).mean(1))))
        trees = {}
        for k in range(1, len(names)):
            vis = npts[:, k] >= MIN_POINTS
            if vis.sum() < 3:
                continue
            P = g[:, k]
            c = P.mean(1, keepdims=True)
            rad = np.sqrt(((P - c) ** 2).sum(-1).mean(1))
            # tolerance: k x the object's median radius, capped by the gripper token's RMS radius (a robot scale, ~36 mm):
            # uncapped, the drawer (radius 63 mm) counted as "on the arm" whenever the arm reached into it
            rmed = min(float(np.median(rad[vis])), r_grip / args.k)
            for t in np.nonzero(vis)[0]:
                if t not in trees:
                    trees[t] = cKDTree(A[t])
                d = trees[t].query(P[t], distance_upper_bound=args.k * rmed)[0]
                if np.mean(np.isfinite(d)) > 0.5:
                    drop[t, k] = True
            n = int(drop[:, k].sum())
            total[names[k]] = total.get(names[k], 0) + n
            if n:
                fr = np.nonzero(drop[:, k])[0]
                print(f"{path.split('/')[-2][:30]} {path.split('/')[-1]}: {names[k]} {n} of {int(vis.sum())} visible frames on the arm"
                      + (f" (frames {fr.tolist()})" if args.verbose else f" ({fr.min()}..{fr.max()})"), flush=True)
        if args.apply:
            e["gauss"][drop] = 0
            e["npts"][drop] = 0
            e["arm_postfilter"] = drop
            save_episode(path, **e)
    print("frames dropped per object over all episodes:", total, "(applied)" if args.apply else "(report only)")


if __name__ == "__main__":
    main()
