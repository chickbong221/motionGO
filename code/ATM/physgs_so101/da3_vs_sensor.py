"""
da3_vs_sensor.py (2026-11-06): how good is Depth Anything 3 (Nested-Giant-Large-1.1) metric depth on the SO-101 top
camera, measured against the sensor depth of the three datasets that have it, and which recipe to use for banana_v2
(no sensor). Per episode, DA3 sees top + wrist frames at several instants in one call (the setting that kept the top
camera stable on the old data; DA3 centre-crops the 640x480 wrist views to the top's 16:9 shape).
Measured on the top view, pixels valid in both:
  scale     median(sensor / DA3) per episode: the metric scale error (and its spread across episodes)
  AbsRel    raw, and after a per-episode median scale (shape error only); delta1 raw
Correction tested: "table rescale" = multiply each episode by median_episodes(table distance) / its table distance
(the camera is fixed within a dataset, so the table is equally far in every episode) -> spread of the remaining scale.
For banana_v2: focal and table distance spread across episodes, and whether the camera moved between episodes.
  python da3_vs_sensor.py [--episodes 6] [--times 5]
"""
import argparse
import glob
import json
import math

import av
import cv2
import numpy as np
import pandas as pd
import torch

ROOT = "/dev/shm/data/so101_v2"
DS = ["so101-left_arm_task_red_cube_out_of_pot_into_the_orange_hollow_cube",
      "so101_left_arm_open_drawer_pick_red_cube_stack_on_blue_cube",
      "so101-left_arm_task_drawer_placing_red_squre_block",
      "so101-left_arm_task_banana_v2"]


def load_meta(R):
    info = json.load(open(f"{R}/meta/info.json"))
    eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{R}/meta/episodes/chunk-*/file-*.parquet"))])
    return info, eps.set_index("episode_index").sort_index()


def decode(R, row, key, times, depth=False):
    path = f"{R}/videos/{key}/chunk-{int(row[f'videos/{key}/chunk_index']):03d}/file-{int(row[f'videos/{key}/file_index']):03d}.mp4"
    t0 = float(row[f"videos/{key}/from_timestamp"])
    c = av.open(path)
    s = c.streams.video[0]
    out = []
    for t in times:
        c.seek(int(max(0.0, t0 + t - 0.2) / s.time_base), stream=s)
        img = None
        for fr in c.decode(s):
            if fr.pts * s.time_base >= t0 + t - 1e-3:
                img = fr.to_ndarray(format="gray12le" if depth else "rgb24")
                break
        out.append(img[..., 0] if (depth and img is not None and img.ndim == 3) else img)
    c.close()
    return out


def depth_from_codes(q, info):
    """LeRobot depth_utils log quantization (stored unit mm; depth_min / max / shift in metres) -> metres, NaN = none."""
    k = 1000.0 if info.get("depth_unit", "mm") == "mm" else 1.0
    lo, hi, sh = info["video.depth_min"] * k, info["video.depth_max"] * k, info["video.shift"] * k
    a, b = math.log(lo + sh), math.log(hi + sh)
    d = (np.exp(q.astype(np.float64) / 4095 * (b - a) + a) - sh) / k
    d[q == 0] = np.nan
    return d


def plane_dist(depth, K, rng, conf=None):
    """Distance from the camera to the dominant plane (the table), RANSAC with a 1 cm band."""
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    m = np.isfinite(depth) & (depth > 0)
    if conf is not None:
        m &= conf >= np.percentile(conf, 40)
    z = depth[m]
    P = np.stack([(u[m] - K[0, 2]) * z / K[0, 0], (v[m] - K[1, 2]) * z / K[1, 1], z], 1)
    P = P[rng.choice(len(P), min(len(P), 30000), replace=False)]
    best, bi = None, 0
    for _ in range(400):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((P - a) @ n) < 0.01
        if inl.sum() > bi:
            best, bi = (n, -n @ a), inl.sum()
    return abs(best[1])


def compare(da3, sensor):
    s = cv2.resize(sensor.astype(np.float32), da3.shape[::-1], interpolation=cv2.INTER_NEAREST)
    m = np.isfinite(s) & (s > 0.05) & np.isfinite(da3) & (da3 > 0)
    s, d = s[m], da3[m]
    scale = np.median(s / d)
    return scale, np.mean(np.abs(d - s) / s), np.mean(np.abs(d * scale - s) / s), np.mean(np.maximum(d / s, s / d) < 1.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--times", type=int, default=5)
    ap.add_argument("--res", type=int, default=504)
    args = ap.parse_args()
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1").to("cuda").eval()
    rng = np.random.default_rng(0)
    summary = []
    for ds in DS:
        R = f"{ROOT}/{ds}"
        info, eps = load_meta(R)
        has_depth = "observation.images.top_depth" in info["features"]
        dinfo = info["features"].get("observation.images.top_depth", {}).get("info")
        L = eps["length"].to_numpy()
        ok = [int(i) for i in eps.index[L > 150]]
        ids = [ok[int(round(x))] for x in np.linspace(0, len(ok) - 1, args.episodes)]
        print(f"\n=== {ds} ({'sensor depth' if has_depth else 'NO sensor'}), episodes {ids}")
        fx, h_da3, h_sen, met, firsts = [], [], [], [], []
        for e in ids:
            row = eps.loc[e]
            ts = list(np.linspace(1.0, row["length"] / 30.0 - 1.0, args.times))
            top = decode(R, row, "observation.images.top", ts)
            wr = decode(R, row, "observation.images.wrist", ts)
            firsts.append(cv2.cvtColor(top[0], cv2.COLOR_RGB2GRAY).astype(np.float32))
            with torch.no_grad():
                p = model.inference(top + wr, ref_view_strategy="first", process_res=args.res)
            n = len(ts)
            Hp, Wp = p.depth.shape[1:]
            W0 = top[0].shape[1]
            fx.append(np.median(p.intrinsics[:n, 0, 0]) * W0 / Wp)
            h_da3.append(np.median([plane_dist(p.depth[i], p.intrinsics[i], rng, p.conf[i]) for i in range(n)]))
            if has_depth:
                sen = [depth_from_codes(q, dinfo) for q in decode(R, row, "observation.images.top_depth", ts, depth=True)]
                met.append([compare(p.depth[i], sen[i]) for i in range(n)])
                # sensor table distance, unprojected with DA3's focal (the plane distance depends little on it)
                Ks = p.intrinsics[0].copy() * np.array([[W0 / Wp], [sen[0].shape[0] / Hp], [1]])
                h_sen.append(np.median([plane_dist(s, Ks, rng) for s in sen]))
        fx, h_da3 = np.array(fx), np.array(h_da3)
        win = cv2.createHanningWindow(firsts[0].shape[::-1], cv2.CV_32F)
        sh = [cv2.phaseCorrelate(firsts[0], f, win) for f in firsts]
        print("  camera vs first episode (shift px / response): " + ", ".join(f"({s[0][0]:+.1f},{s[0][1]:+.1f})/{s[1]:.2f}" for s in sh))
        print(f"  DA3 focal per episode {np.round(fx, 1)} px (rel std {fx.std() / fx.mean():.1%})")
        print(f"  DA3 table distance per episode {np.round(h_da3, 3)} m (rel std {h_da3.std() / h_da3.mean():.1%})")
        if has_depth:
            a = np.array(met)                      # episodes x times x 4
            sc = np.median(a[..., 0], 1)           # per-episode scale sensor / DA3
            corr = np.median(h_da3) / h_da3        # table rescale factor per episode
            sc_c = sc / corr                       # remaining scale after the table rescale
            print(f"  sensor table distance per episode {np.round(h_sen, 3)} m")
            print(f"  scale sensor/DA3 per episode {np.round(sc, 3)} -> spread {sc.std() / sc.mean():.1%}")
            print(f"  after the table rescale        {np.round(sc_c, 3)} -> spread {sc_c.std() / sc_c.mean():.1%}, "
                  f"global factor {np.median(sc_c):.3f}")
            print(f"  AbsRel raw {a[..., 1].mean():.1%} | after a per-episode scale (shape only) {a[..., 2].mean():.1%} | "
                  f"delta1 raw {a[..., 3].mean():.1%}")
            summary.append((ds[:40], sc.mean(), sc.std() / sc.mean(), sc_c.std() / sc_c.mean(), np.median(sc_c), a[..., 2].mean()))
    print("\nSUMMARY (datasets with sensor): name, mean scale, spread raw, spread after table rescale, global factor, shape AbsRel")
    for s in summary:
        print(f"  {s[0]:40s} {s[1]:.3f} {s[2]:.1%} {s[3]:.1%} {s[4]:.3f} {s[5]:.1%}")


if __name__ == "__main__":
    main()
