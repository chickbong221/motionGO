"""
check_lift_v2.py (2026-11-07): quality check of lift_so101_v2.py episodes (robot base frame, top sensor + kinematic
gripper). Per object, over the frames where it is visible (>= 16 points), using the first 512 Gaussians as training
does: visibility, points, RMS radius, centroid height, centroid jitter (median |second difference| at 15 fps, mm: noise
of the object position the dynamics learn from), and frames whose centroid jumps more than 2 cm from both neighbours.
Saves a strip: top RGB with each object's Gaussians projected back (gripper yellow), plus a top-down x-y scatter.
  python check_lift_v2.py <episode.npz> <dataset dir> <out.jpg>
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, "/dev/shm/physgo/code_so101/ATM/physgs_so101")
import lift_so101 as L  # noqa: E402

COLS = [(255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 0, 255), (0, 255, 255)]


def main():
    ep_path, root, out = sys.argv[1:4]
    e = dict(np.load(ep_path))
    cam = dict(np.load(ep_path.replace(".npz", "_cam.npz")))
    names = [str(n) for n in e["obj_names"]]
    g, npts = e["gauss"][:, :, :512].astype(np.float32), e["npts"]
    T, O = npts.shape
    print(f"{ep_path.split('/')[-2][:40]} {ep_path.split('/')[-1]}: T {T} at {float(e['fps'])} fps, actions {e['actions'].shape}, "
          f"state {e['state'].shape}")
    for k, n in enumerate(names):
        vis = npts[:, k] >= 16
        if not vis.any():
            print(f"  {n:13s} never visible")
            continue
        P = g[:, k, :, :3]
        c = P.mean(1)
        rad = np.sqrt(((P - c[:, None]) ** 2).sum(-1).mean(1))
        both = vis[2:] & vis[1:-1] & vis[:-2]
        acc = np.linalg.norm(c[2:] - 2 * c[1:-1] + c[:-2], axis=1)[both] * 1000
        jump = both & (np.linalg.norm(c[1:-1] - c[:-2], axis=1) > 0.02) & (np.linalg.norm(c[1:-1] - c[2:], axis=1) > 0.02)
        print(f"  {n:13s} visible {vis.mean():5.1%}  points med {np.median(npts[vis, k]):6.0f}  radius {np.median(rad[vis]) * 1000:5.1f} mm "
              f"(p90 {np.percentile(rad[vis], 90) * 1000:5.1f})  z {np.median(c[vis, 2]) * 1000:6.1f} mm  "
              f"jitter {np.median(acc) if len(acc) else np.nan:4.1f} mm (p95 {np.percentile(acc, 95) if len(acc) else np.nan:5.1f})  "
              f"jump frames {int(jump.sum())}")
    info, eps, _ = L.load_meta(root)
    ep = int(ep_path.split("_")[-1].split(".")[0])
    top = L.decode_episode(root, eps.loc[ep], "observation.images.top", int(eps.loc[ep, "length"]))
    K, T_bt = cam["K_top"], cam["T_base_to_top"]
    rows = []
    for t in np.linspace(0, T - 1, 6).round().astype(int):
        im = top[t].copy()
        bev = np.full((270, 270, 3), 30, np.uint8)
        for k in range(O):
            if npts[t, k] < 16:
                continue
            P = g[t, k, :, :3].astype(np.float64)
            Pc = P @ T_bt[:3, :3].T + T_bt[:3, 3]
            u = (K[0, 0] * Pc[:, 0] / Pc[:, 2] + K[0, 2]).round().astype(int)
            v = (K[1, 1] * Pc[:, 1] / Pc[:, 2] + K[1, 2]).round().astype(int)
            ok = (u >= 0) & (u < im.shape[1]) & (v >= 0) & (v < im.shape[0])
            for du in (0, 1):
                im[np.clip(v[ok] + du, 0, im.shape[0] - 1), u[ok]] = COLS[k % len(COLS)]
            bu = (135 + P[:, 1] * 450).round().astype(int)        # base frame: x forward, y left; 0.6 m across
            bv = (260 - P[:, 0] * 450).round().astype(int)
            ok = (bu >= 0) & (bu < 270) & (bv >= 0) & (bv < 270)
            bev[bv[ok], bu[ok]] = COLS[k % len(COLS)]
        cv2.putText(im, f"t={t}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        rows.append(np.concatenate([cv2.resize(im, (480, 270)), bev], 1))
    leg = np.zeros((30, 750, 3), np.uint8)
    for k, n in enumerate(names):
        cv2.putText(leg, n, (5 + 125 * k, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLS[k % len(COLS)], 1)
    cv2.imwrite(out, np.concatenate([leg] + rows, 0)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("saved", out)


if __name__ == "__main__":
    main()
