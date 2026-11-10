"""
register_depth.py (2026-11-06): register the SO-101 top depth stream to the top RGB. The calibration showed the depth
is not aligned to the RGB (a 2D shift barely helps, a larger scale keeps helping): the recordings look like an
unaligned RealSense (depth camera with its own, wider field of view, a few cm beside the RGB camera). Model:
  sensor pixel (u, v), depth z -> X_d = ((u - cx_d) z / f_d, (v - cy_d) z / f_d, z) -> X_c = X_d + t
  -> RGB pixel (f_c X_c / Z_c + cx_c, f_c Y_c / Z_c + cy_c), depth Z_c (z-buffer: nearest wins)
(rotation taken as identity: the two sensors of one module are parallel). The RGB intrinsics come from the
calibration (Depth Anything 3 median); f_d, cx_d, cy_d, t are fitted so the re-projected sensor depth matches the
shape of Depth Anything 3's depth of the same RGB frames (aligned to the RGB by construction): cost = median absolute
log-depth difference after the best global scale, over many frames. Grid start, then Nelder-Mead.
  python register_depth.py --dataset <dir> --calib <calib.json>     (adds "depth_reg" to the calib file)
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lift_so101 import decode_episode, depth_from_codes, load_meta  # noqa: E402


def reproject(depth, f_d, cx_d, cy_d, t, Kc, out_hw):
    """Sensor depth (H, W) m -> depth in the RGB camera (out_hw), z-buffered; NaN where nothing lands."""
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    m = np.isfinite(depth) & (depth > 0)
    z = depth[m]
    X = np.stack([(u[m] - cx_d) * z / f_d + t[0], (v[m] - cy_d) * z / f_d + t[1], z + t[2]], 1)
    ok = X[:, 2] > 0.05
    X = X[ok]
    uc = np.round(Kc[0, 0] * X[:, 0] / X[:, 2] + Kc[0, 2]).astype(np.int64)
    vc = np.round(Kc[1, 1] * X[:, 1] / X[:, 2] + Kc[1, 2]).astype(np.int64)
    h, w = out_hw
    inb = (uc >= 0) & (uc < w) & (vc >= 0) & (vc < h)
    uc, vc, zc = uc[inb], vc[inb], X[inb, 2]
    out = np.full(h * w, np.inf)
    np.minimum.at(out, vc * w + uc, zc)
    out[np.isinf(out)] = np.nan
    return out.reshape(h, w)


def cost(params, sensors, da3s, Kc, hw):
    f_d, cx_d, cy_d, tx, ty, tz = params
    if f_d <= 50:
        return 10.0
    errs = []
    for s, a in zip(sensors, da3s):
        r = reproject(s, f_d, cx_d, cy_d, (tx, ty, tz), Kc, hw)
        m = np.isfinite(r) & np.isfinite(a) & (a > 0)
        if m.sum() < 1000:
            return 10.0
        d = np.log(r[m]) - np.log(a[m])
        errs.append(np.median(np.abs(d - np.median(d))))
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--frames", type=int, default=4)
    args = ap.parse_args()
    root = args.dataset.rstrip("/")
    calib = json.load(open(args.calib))
    Kc = np.array(calib["K_top"], np.float64)
    info, eps, _ = load_meta(root)
    dinfo = info["features"]["observation.images.top_depth"]["info"]
    ids = eps.index[np.linspace(0, len(eps) - 1, args.episodes).round().astype(int)].tolist()
    tops, wrs, sens = [], [], []
    for e in ids:
        row = eps.loc[e]
        n30 = int(row["length"])
        top = decode_episode(root, row, "observation.images.top", n30)
        wr = decode_episode(root, row, "observation.images.wrist", n30)
        q = decode_episode(root, row, "observation.images.top_depth", n30, depth=True)
        pick = np.linspace(len(top) * 0.1, len(top) * 0.9, args.frames).round().astype(int)
        tops.append(top[pick]); wrs.append(wr[pick]); sens += [depth_from_codes(x, dinfo) for x in q[pick]]
    # Depth Anything 3 depth of the RGB frames, in the lifting setting (top padded to 4:3 + wrist), cut back to 360x640
    from depth_anything_3.api import DepthAnything3
    m = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1").to("cuda").eval()
    da3s = []
    for top, wr in zip(tops, wrs):
        T, Ht, Wt = top.shape[:3]
        pad = (wr.shape[1] * Wt // wr.shape[2] - Ht) // 2
        topp = np.zeros((T, Ht + 2 * pad, Wt, 3), np.uint8)
        topp[:, pad:pad + Ht] = top
        with torch.no_grad():
            p = m.inference(list(topp) + list(wr), ref_view_strategy="first", process_res=644)
        for j in range(T):
            d = cv2.resize(p.depth[j], (Wt, Ht + 2 * pad), interpolation=cv2.INTER_LINEAR)[pad:pad + Ht]
            da3s.append(d)
    del m
    torch.cuda.empty_cache()
    hw = sens[0].shape
    # start: identity registration vs the best of a coarse grid (focal ratio, x baseline)
    f_c = Kc[0, 0]
    ident = cost([f_c, Kc[0, 2], Kc[1, 2], 0, 0, 0], sens, da3s, Kc, hw)
    grid = []
    for r in (0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0):
        for tx in (-0.05, -0.03, -0.015, 0.0, 0.015, 0.03, 0.05):
            grid.append((cost([f_c * r, Kc[0, 2], Kc[1, 2], tx, 0, 0], sens, da3s, Kc, hw), r, tx))
    g = min(grid)
    print(f"identity (aligned) cost {ident:.4f}; best grid cost {g[0]:.4f} at f_d/f_c {g[1]}, tx {g[2]}")
    x0 = [f_c * g[1], Kc[0, 2], Kc[1, 2], g[2], 0.0, 0.0]
    res = minimize(cost, x0, args=(sens, da3s, Kc, hw), method="Nelder-Mead",
                   options=dict(xatol=0.05, fatol=1e-5, maxiter=600, initial_simplex=None))
    f_d, cx_d, cy_d, tx, ty, tz = res.x
    print(f"fitted: f_d {f_d:.1f} (f_c {f_c:.1f}, ratio {f_d / f_c:.3f}), c_d ({cx_d:.1f},{cy_d:.1f}), "
          f"t ({tx * 1000:.1f},{ty * 1000:.1f},{tz * 1000:.1f}) mm, cost {res.fun:.4f} (identity {ident:.4f})")
    # coverage: share of RGB pixels that receive a depth after registration (vs before)
    cov = np.mean([np.isfinite(reproject(s, f_d, cx_d, cy_d, (tx, ty, tz), Kc, hw)).mean() for s in sens])
    calib["depth_reg"] = dict(f_d=float(f_d), cx_d=float(cx_d), cy_d=float(cy_d), t=[float(tx), float(ty), float(tz)],
                              cost=float(res.fun), cost_identity=float(ident), rgb_coverage=float(cov), episodes=ids)
    json.dump(calib, open(args.calib, "w"), indent=1)
    print(f"RGB pixels with a registered depth: {cov:.1%}; saved to {args.calib}")


if __name__ == "__main__":
    sys.exit(main())
