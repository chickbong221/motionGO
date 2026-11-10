"""
Position / velocity labels for the arm token of an already packed LIBERO dataset, from the arm's own Gaussians
(physgs.data.libero_process.point_centroid_labels; lifts since 2026-10-23 write them directly). Rewrites pos.npy /
vel.npy in place for the arm slot only, after copying the originals to pos.before_arm.npy / vel.before_arm.npy.

    python -m physgs.data.arm_labels --data data/packed/libero_10256_v2
"""
import argparse
import json
import os
import shutil

import numpy as np

from physgs.data.format import MIN_POINTS
from physgs.data.libero_process import point_centroid_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    ep = np.load(os.path.join(a.data, "ep_start.npy"))
    for n in ("pos", "vel"):
        bak = os.path.join(a.data, f"{n}.before_arm.npy")
        if not os.path.exists(bak):
            shutil.copy2(os.path.join(a.data, f"{n}.npy"), bak)
    pos = np.load(os.path.join(a.data, "pos.npy"), mmap_mode="r+")
    vel = np.load(os.path.join(a.data, "vel.npy"), mmap_mode="r+")
    gauss = np.load(os.path.join(a.data, "gauss.npy"), mmap_mode="r")
    npts = np.load(os.path.join(a.data, "npts.npy"), mmap_mode="r")
    done, speeds = 0, []
    for e, m in enumerate(meta["episodes"]):
        if "arm" not in m["obj_names"]:
            continue
        k = m["obj_names"].index("arm")
        s, t = ep[e], ep[e + 1]
        p, v = point_centroid_labels(np.asarray(gauss[s:t, k, :, :3], np.float32), np.asarray(npts[s:t, k]), min_points=MIN_POINTS)
        pos[s:t, k], vel[s:t, k] = p, v
        speeds.append(np.nanmedian(np.linalg.norm(v, axis=-1)))
        done += 1
    pos.flush(); vel.flush()
    print(f"arm labels written for {done} of {len(meta['episodes'])} episodes; median arm-centroid speed "
          f"{np.nanmedian(speeds):.3f} m/s; labelled frames {np.isfinite(np.asarray(pos)[:, :, 0]).mean():.3f} of all slots")


if __name__ == "__main__":
    main()
