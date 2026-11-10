"""
da3_test.py (2026-11-06): does Depth Anything 3 (Nested-Giant-Large-1.1) give usable metric geometry for SO-101 lifting
from both cameras? No ground truth exists, so the checks are internal:
  - the top camera is fixed: its estimated focal length and pose should not change across time (run B: all frames of
    an episode in one call),
  - the table is a plane: RANSAC plane residual on the top view, camera height above it (same rig in every episode ->
    the height should agree across episodes and tasks),
  - top and wrist views must agree: wrist points moved into the top camera frame should land on top points (median
    nearest-neighbour distance), for the pose convention that fits (w2c or c2w).
  /pfss/.../env_vision/bin/python da3_test.py --root /dev/shm/data/so101-multitask
"""
import argparse
import time

import av
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree


def frames_at(root, ep_row, cam, times):
    f = f"{root}/videos/observation.images.{cam}/chunk-{int(ep_row[f'videos/observation.images.{cam}/chunk_index']):03d}/" \
        f"file-{int(ep_row[f'videos/observation.images.{cam}/file_index']):03d}.mp4"
    t0 = float(ep_row[f"videos/observation.images.{cam}/from_timestamp"])
    c = av.open(f)
    s = c.streams.video[0]
    out = []
    for t in times:
        target = t0 + t
        c.seek(int(max(0.0, target - 0.2) / s.time_base), stream=s)
        img = None
        for fr in c.decode(s):
            if fr.pts * s.time_base >= target - 1e-3:
                img = fr.to_ndarray(format="rgb24")
                break
        out.append(img)
    c.close()
    return out


def points(depth, K, conf, pct=40):
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    m = (conf >= np.percentile(conf, pct)) & np.isfinite(depth) & (depth > 0)
    z = depth[m]
    x = (u[m] - K[0, 2]) * z / K[0, 0]
    y = (v[m] - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], 1)


def to4(E):
    E = np.asarray(E, np.float64)
    if E.shape == (3, 4):
        E = np.vstack([E, [0, 0, 0, 1]])
    return E


def ransac_plane(P, thr=0.01, iters=500, rng=np.random.default_rng(0)):
    best, bi = None, 0
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        d = -n @ a
        inl = np.abs(P @ n + d) < thr
        if inl.sum() > bi:
            best, bi = (n, d), inl.sum()
    n, d = best
    inl = np.abs(P @ n + d) < thr
    return n, d, inl.mean(), np.sqrt(np.mean((P[inl] @ n + d) ** 2))


def rot_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/dev/shm/data/so101-multitask")
    ap.add_argument("--model", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    ap.add_argument("--episodes", default="0,60,110")
    ap.add_argument("--times", default="0.5,2.5,4.5,6.5,8.5")
    ap.add_argument("--res", type=int, default=504)
    args = ap.parse_args()
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(args.model).to("cuda").eval()
    eps = pd.read_parquet(f"{args.root}/meta/episodes/chunk-000/file-000.parquet").set_index("episode_index")
    times = [float(t) for t in args.times.split(",")]
    heights = []
    for e in [int(x) for x in args.episodes.split(",")]:
        row = eps.loc[e]
        L = float(row["length"]) / 30.0
        ts = [t for t in times if t < L - 0.2]
        top, wrist = frames_at(args.root, row, "top", ts), frames_at(args.root, row, "wrist", ts)
        print(f"\n=== episode {e} ({row['tasks'][0][:45]}), {L:.1f} s, frames at {ts}")
        # run B: every top + wrist frame of the episode in one call, the first top frame as reference
        t0 = time.time()
        with torch.no_grad():
            pr = model.inference(top + wrist, ref_view_strategy="first", process_res=args.res)
        dt = time.time() - t0
        n = len(ts)
        E = np.stack([to4(x) for x in pr.extrinsics])
        K = pr.intrinsics
        print(f"run B: {2 * n} views in {dt:.1f} s, metric={pr.is_metric}, depth {pr.depth.shape}, E[0] - I max "
              f"{np.abs(E[0] - np.eye(4)).max():.3f}")
        fx_top = K[:n, 0, 0]
        print(f"  top focal (px at {pr.depth.shape[2]}x{pr.depth.shape[1]}): mean {fx_top.mean():.1f}, rel std "
              f"{fx_top.std() / fx_top.mean():.2%}; wrist focal mean {K[n:, 0, 0].mean():.1f}")
        drift = [(rot_deg(E[i][:3, :3] @ E[0][:3, :3].T), np.linalg.norm(E[i][:3, 3] - E[0][:3, 3])) for i in range(n)]
        print("  top pose vs first top frame (deg, cm): " + ", ".join(f"{r:.2f}/{t * 100:.1f}" for r, t in drift))
        # table plane in the first top view
        Pt = points(pr.depth[0], K[0], pr.conf[0])
        nrm, d, frac, rms = ransac_plane(Pt[np.random.default_rng(0).choice(len(Pt), min(len(Pt), 60000), False)])
        heights.append(abs(d))
        print(f"  table plane: inliers {frac:.0%} (1 cm), residual rms {rms * 1000:.1f} mm, camera height {abs(d):.3f} m, "
              f"median top depth {np.median(pr.depth[0]):.3f} m")
        # cross-view agreement: wrist points in the top camera frame vs the top points of the same instant
        for conv in ("w2c", "c2w"):
            res = []
            for i in range(n):
                Ti, Tw = E[i], E[n + i]
                if conv == "w2c":   # x_cam = E x_world -> wrist cam -> world -> top cam
                    M = Ti @ np.linalg.inv(Tw)
                else:               # x_world = E x_cam
                    M = np.linalg.inv(Ti) @ Tw
                Pw = points(pr.depth[n + i], K[n + i], pr.conf[n + i])
                Pw = Pw @ M[:3, :3].T + M[:3, 3]
                tree = cKDTree(points(pr.depth[i], K[i], pr.conf[i]))
                dist, _ = tree.query(Pw[np.random.default_rng(i).choice(len(Pw), min(len(Pw), 20000), False)])
                res.append(np.median(dist))
            print(f"  wrist->top median NN distance ({conv}): " + ", ".join(f"{r * 100:.1f}" for r in res) + " cm")
        # run A: pairs (top_t, wrist_t), the setting of a per-frame pipeline
        fx_a, h_a = [], []
        for i in range(n):
            with torch.no_grad():
                pa = model.inference([top[i], wrist[i]], ref_view_strategy="first", process_res=args.res)
            fx_a.append(pa.intrinsics[0, 0, 0])
            Pa = points(pa.depth[0], pa.intrinsics[0], pa.conf[0])
            h_a.append(abs(ransac_plane(Pa[np.random.default_rng(0).choice(len(Pa), min(len(Pa), 30000), False)])[1]))
        print(f"run A (pairs): top focal {np.mean(fx_a):.1f} rel std {np.std(fx_a) / np.mean(fx_a):.2%}; camera height "
              f"{np.mean(h_a):.3f} m, rel std {np.std(h_a) / np.mean(h_a):.2%}")
    print(f"\ncamera height across episodes (run B): {np.round(heights, 3)} -> rel std {np.std(heights) / np.mean(heights):.2%}")


if __name__ == "__main__":
    main()
