"""
calibrate_so101.py (2026-11-06): per-dataset constants for lift_so101.py, estimated from the data (no camera
calibration ships with the datasets).
  K_top        top-camera intrinsics in pixels of the 640x360 image: median of Depth Anything 3 estimates over many
               frames of many episodes (DA3 on [top padded to 4:3, wrist] chunks, the lifting setting)
  depth_shift  integer (dx, dy) that moves the sensor depth onto the RGB: the shift minimising the mean distance from
               depth edges to RGB edges over many frames (search +-24 px); the residual is reported, and a scale
               search (0.96..1.04 about the centre) tells whether the depth is a plain shift of the RGB or not
  python calibrate_so101.py --dataset <dir> --out <calib.json> [--episodes 10 --frames 4]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lift_so101 import decode_episode, depth_from_codes, load_meta  # noqa: E402


def edge_cost(dep_list, rgb_list, dx, dy, s=1.0):
    costs = []
    for d, rgb in zip(dep_list, rgb_list):
        H, W = d.shape
        M = np.float32([[s, 0, (1 - s) * W / 2 + dx], [0, s, (1 - s) * H / 2 + dy]])
        dd = cv2.warpAffine(np.nan_to_num(d, nan=0.0), M, (W, H), flags=cv2.INTER_NEAREST)
        v = np.nan_to_num((dd - np.percentile(dd[dd > 0], 2)) / (np.percentile(dd[dd > 0], 98) - np.percentile(dd[dd > 0], 2)), nan=0).clip(0, 1)
        e_d = cv2.Canny((v * 255).astype(np.uint8), 30, 90) > 0
        e_d[:24] = e_d[-24:] = False
        e_d[:, :24] = e_d[:, -24:] = False
        e_r = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 60, 150) > 0
        dist = cv2.distanceTransform((~e_r).astype(np.uint8), cv2.DIST_L2, 3)
        if e_d.any():
            costs.append(np.median(dist[e_d]))
    return float(np.mean(costs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--process_res", type=int, default=644)
    args = ap.parse_args()
    root = args.dataset.rstrip("/")
    info, eps, tasks = load_meta(root)
    ids = eps.index[np.linspace(0, len(eps) - 1, args.episodes).round().astype(int)].tolist()
    dinfo = info["features"]["observation.images.top_depth"]["info"]
    tops, wrs, deps = [], [], []
    for e in ids:
        row = eps.loc[e]
        n30 = int(row["length"])
        top = decode_episode(root, row, "observation.images.top", n30)
        wr = decode_episode(root, row, "observation.images.wrist", n30)
        q = decode_episode(root, row, "observation.images.top_depth", n30, depth=True)
        pick = np.linspace(len(top) * 0.1, len(top) * 0.9, args.frames).round().astype(int)
        tops.append(top[pick]); wrs.append(wr[pick]); deps.append(np.stack([depth_from_codes(x, dinfo) for x in q[pick]]))
    rgb_all = [f for t in tops for f in t]
    dep_all = [f for d in deps for f in d]

    # depth -> RGB shift (coarse 4 px grid, then 1 px), then a scale check
    best = min(((edge_cost(dep_all, rgb_all, dx, dy), dx, dy) for dx in range(-24, 25, 4) for dy in range(-24, 25, 4)))
    _, bx, by = best
    best = min(((edge_cost(dep_all, rgb_all, dx, dy), dx, dy) for dx in range(bx - 4, bx + 5) for dy in range(by - 4, by + 5)))
    c0 = edge_cost(dep_all, rgb_all, 0, 0)
    sc = {s: edge_cost(dep_all, rgb_all, best[1], best[2], s) for s in (0.96, 0.98, 1.0, 1.02, 1.04)}
    print(f"depth->RGB edge distance: unshifted {c0:.2f} px, best shift ({best[1]},{best[2]}) {best[0]:.2f} px; "
          f"scale check {json.dumps({k: round(v, 2) for k, v in sc.items()})}")

    # intrinsics from Depth Anything 3 (lifting setting: top padded to 4:3 + wrist, one call per episode)
    from depth_anything_3.api import DepthAnything3
    m = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1").to("cuda").eval()
    Ks = []
    for top, wr in zip(tops, wrs):
        T, Ht, Wt = top.shape[:3]
        pad = (wr.shape[1] * Wt // wr.shape[2] - Ht) // 2
        topp = np.zeros((T, Ht + 2 * pad, Wt, 3), np.uint8)
        topp[:, pad:pad + Ht] = top
        with torch.no_grad():
            p = m.inference(list(topp) + list(wr), ref_view_strategy="first", process_res=args.process_res)
        hp, wp = p.depth.shape[1:]
        for j in range(T):
            K = p.intrinsics[j].astype(np.float64).copy()
            sx, sy = Wt / wp, (Ht + 2 * pad) / hp
            Ks.append([K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy - pad])
    Ks = np.array(Ks)
    med = np.median(Ks, 0)
    print(f"DA3 top intrinsics over {len(Ks)} frames: fx {med[0]:.1f} fy {med[1]:.1f} cx {med[2]:.1f} cy {med[3]:.1f}; "
          f"rel std fx {Ks[:, 0].std() / Ks[:, 0].mean():.1%}")
    f = float((med[0] + med[1]) / 2)
    out = dict(dataset=os.path.basename(root), K_top=[[f, 0, float(med[2])], [0, f, float(med[3])], [0, 0, 1]],
               depth_shift=[int(best[1]), int(best[2])], edge_dist_px=dict(unshifted=c0, shifted=best[0], scale=sc),
               fx_rel_std=float(Ks[:, 0].std() / Ks[:, 0].mean()), episodes=ids)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    sys.exit(main())
