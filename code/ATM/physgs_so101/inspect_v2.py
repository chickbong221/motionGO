"""
inspect_v2.py (2026-11-06): the four new SO-101 left-arm datasets. Per dataset: episodes, lengths, task text; one frame
of top RGB / top depth / wrist (decoded with LeRobot's depth formula); depth validity and range; whether the top
camera is the same across datasets (phase correlation of the first top frames). Writes a picture grid.
"""
import glob
import json
import math
import os
import sys

import av
import cv2
import numpy as np
import pandas as pd

ROOT = "/dev/shm/data/so101_v2"
QMAX = 4095


def depth_from_codes(q, info):
    """LeRobot depth_utils: log quantization in the stored unit (mm); depth_min / max / shift in metres."""
    k = 1000.0 if info.get("depth_unit", "mm") == "mm" else 1.0
    lo, hi, sh = info["video.depth_min"] * k, info["video.depth_max"] * k, info["video.shift"] * k
    if info.get("video.use_log", True):
        a, b = math.log(lo + sh), math.log(hi + sh)
        d = np.exp(q.astype(np.float64) / QMAX * (b - a) + a) - sh
    else:
        d = q.astype(np.float64) / QMAX * (hi - lo) + lo
    d = d / k                                   # metres
    d[q == 0] = np.nan                          # code 0 = no reading
    return d


def frame_at(path, t, depth=False):
    c = av.open(path)
    s = c.streams.video[0]
    c.seek(int(max(0.0, t - 0.2) / s.time_base), stream=s)
    out = None
    for fr in c.decode(s):
        if fr.pts * s.time_base >= t - 1e-3:
            out = fr.to_ndarray(format="gray12le" if depth else "rgb24") if depth else fr.to_ndarray(format="rgb24")
            break
    c.close()
    return out


def main():
    tiles, firsts = [], {}
    for ds in sorted(os.listdir(ROOT)):
        R = f"{ROOT}/{ds}"
        info = json.load(open(f"{R}/meta/info.json"))
        feats = info["features"]
        eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{R}/meta/episodes/chunk-*/file-*.parquet"))])
        eps = eps.set_index("episode_index").sort_index()
        L = eps["length"].to_numpy()
        tasks = pd.read_parquet(f"{R}/meta/tasks.parquet").index.tolist()
        print(f"\n=== {ds}\n  task: {tasks}\n  episodes {len(eps)}, frames {L.sum()}, length s min/median/max "
              f"{L.min()/30:.1f}/{np.median(L)/30:.1f}/{L.max()/30:.1f}, short (<3 s): {list(eps.index[L < 90])}")
        e0 = eps.iloc[1]

        def vpath(key):
            return f"{R}/videos/{key}/chunk-{int(e0[f'videos/{key}/chunk_index']):03d}/file-{int(e0[f'videos/{key}/file_index']):03d}.mp4"

        t = float(e0["videos/observation.images.top/from_timestamp"]) + 1.0
        top = frame_at(vpath("observation.images.top"), t)
        wr = frame_at(vpath("observation.images.wrist"), float(e0["videos/observation.images.wrist/from_timestamp"]) + 1.0)
        firsts[ds] = cv2.cvtColor(top, cv2.COLOR_RGB2GRAY).astype(np.float32)
        row = [cv2.resize(top, (320, 180)), cv2.resize(wr, (240, 180))]
        if "observation.images.top_depth" in feats:
            dinfo = feats["observation.images.top_depth"]["info"]
            q = frame_at(vpath("observation.images.top_depth"), float(e0["videos/observation.images.top_depth/from_timestamp"]) + 1.0, depth=True)
            q = q[..., 0] if q.ndim == 3 else q
            d = depth_from_codes(q, dinfo)
            v = np.isfinite(d)
            print(f"  depth frame {q.shape} codes {q.min()}..{q.max()}; valid {v.mean():.1%}; depth m: p1 {np.nanpercentile(d,1):.3f} "
                  f"median {np.nanmedian(d):.3f} p99 {np.nanpercentile(d,99):.3f}; centre 20x20 median {np.nanmedian(d[170:190,310:330]):.3f}")
            dv = np.nan_to_num((d - np.nanpercentile(d, 2)) / (np.nanpercentile(d, 98) - np.nanpercentile(d, 2)), nan=0).clip(0, 1)
            row.append(cv2.resize(cv2.applyColorMap((dv * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1], (320, 180)))
            # RGB / depth registration: edges of RGB vs edges of depth
            e_rgb = cv2.Canny(cv2.cvtColor(top, cv2.COLOR_RGB2GRAY), 60, 150) > 0
            e_d = cv2.Canny((dv * 255).astype(np.uint8), 30, 90) > 0
            dist = cv2.distanceTransform((~e_rgb).astype(np.uint8), cv2.DIST_L2, 3)
            print(f"  depth edges -> nearest RGB edge: median {np.median(dist[e_d]):.1f} px (alignment check)")
        else:
            print("  no depth stream")
            row.append(np.zeros((180, 320, 3), np.uint8))
        tiles.append(np.concatenate(row, 1))
    names = list(firsts)
    win = cv2.createHanningWindow(firsts[names[0]].shape[::-1], cv2.CV_32F)
    print("\ntop camera vs the first dataset (phase correlation of a top frame):")
    for n in names:
        (dx, dy), r = cv2.phaseCorrelate(firsts[names[0]], firsts[n], win)
        print(f"  {n[:60]:60s} shift ({dx:+.1f},{dy:+.1f}) px response {r:.2f}")
    cv2.imwrite("/dev/shm/so101_v2_grid.jpg", np.concatenate(tiles, 0)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    sys.exit(main())
