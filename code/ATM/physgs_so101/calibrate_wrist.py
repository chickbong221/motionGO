"""
calibrate_wrist.py (2026-11-06): the wrist-camera mount X (gripper_link -> wrist optical frame, OpenCV: x right,
y down, z forward), shared by all datasets (one arm, one camera), and the wrist focal length.
Per cached frame (calibrate_kin_v2.py cache): wrist pose in the top camera = T_base_to_topcam @ FK(gripper_link) @ X.
Wrist depth = Depth Anything 3 (single view) x a per-frame scale: the top sensor's points, moved into the wrist camera
and projected, give true depth where both cameras see the same surface; scale = median(true / DA3) there.
Fit X (6) + log f_w so the scaled wrist points land on the top sensor's points (Huber 1 cm, nearest neighbour), from
the URDF camera link with each of the 24 axis-aligned optical-frame conventions as starts (best kept).
Reports the median wrist->top distance per dataset, the per-frame scale spread, and overlays (sensor points projected
into the wrist image).
  python calibrate_wrist.py --datasets a,b,c --kin kin_v2.json --out wrist.json
"""
import argparse
import itertools
import json
import math
import os
import sys

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lift_so101 as L  # noqa: E402
from so101_kin import SO101, pose_from_vec  # noqa: E402

URDF = "/dev/shm/physgo/so101/robot/so101_new_calib_camera.urdf"
DEV = "cuda"


def huber(x, d):
    a = x.abs()
    return torch.where(a < d, 0.5 * a * a / d, a - 0.5 * d)


def axis_rotations():
    """The 24 proper rotations with axis-aligned columns."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                R[p, i] = s
            if np.linalg.det(R) > 0:
                out.append(R)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--kin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--viz_dir", default="/dev/shm/physgo/so101/wrist_viz")
    args = ap.parse_args()
    os.makedirs(args.viz_dir, exist_ok=True)
    kc = json.load(open(args.kin))
    kin = SO101(URDF, device=DEV, points_per_link=200)
    s = torch.tensor(kc["joint_scale"], dtype=torch.float64, device=DEV)
    o = torch.tensor(kc["joint_offset"], dtype=torch.float64, device=DEV)
    from depth_anything_3.api import DepthAnything3
    da3 = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1").to(DEV).eval()
    rng = np.random.default_rng(0)
    cpath = "/dev/shm/physgo/so101/kin_cache/wrist_da3.npz"
    da3_cache = {}
    if os.path.exists(cpath):
        zc_ = np.load(cpath)
        da3_cache = {k: (zc_[k + "|d"], zc_[k + "|K"]) for k in set(x.rsplit("|", 1)[0] for x in zc_.files)}
    F = []   # per frame: top sensor points in the top camera, T_base_to_top, T_base_to_grip, wrist DA3 depth, K guess
    fw0 = []
    for n in args.datasets.split(","):
        z = np.load(f"/dev/shm/physgo/so101/kin_cache/{n}.npz")
        dk = kc["datasets"][n]
        T_bt = np.array(dk["T_base_to_cam"])
        f_t, cx, cy = dk["f"], dk["cx"], dk["cy"]
        ang = torch.tensor(z["state"], dtype=torch.float64, device=DEV) * s + o
        fkd = kin.fk(ang)
        T_bg = fkd["gripper_link"].cpu().numpy()                # gripper_link -> base
        T_bj = fkd["moving_jaw_so101_v1_link"].cpu().numpy()    # moving jaw -> base (jaw check in the wrist image)
        for i in range(len(z["state"])):
            d = z["depth"][i]
            v, u = np.nonzero(np.isfinite(d))
            k = rng.choice(len(u), min(len(u), 20000), replace=False)
            zz = d[v[k], u[k]]
            Pt = np.stack([(u[k] - cx) * zz / f_t, (v[k] - cy) * zz / f_t, zz], 1)
            key = f"{n}/{i}"
            if key not in da3_cache:
                with torch.no_grad():
                    p = da3.inference([z["wrist"][i]], process_res=644)
                da3_cache[key] = (p.depth[0].astype(np.float32), p.intrinsics[0].astype(np.float32))
            dw_, Kw_ = da3_cache[key]
            p = type("P", (), dict(depth=[dw_], intrinsics=[Kw_]))
            F.append(dict(ds=n, Pt=Pt, T_bt=T_bt, T_bg=T_bg[i], T_bj=T_bj[i], g=float(z["state"][i][5]),
                          dw=p.depth[0].astype(np.float64), Kw=p.intrinsics[0].astype(np.float64), wrist=z["wrist"][i]))
            fw0.append(p.intrinsics[0][0, 0])
    np.savez(cpath, **{f"{k}|d": v[0] for k, v in da3_cache.items()}, **{f"{k}|K": v[1] for k, v in da3_cache.items()})
    print(f"{len(F)} frames; DA3 wrist focal median {np.median(fw0):.1f} px at {F[0]['dw'].shape[::-1]}", flush=True)
    hw = F[0]["dw"].shape
    # URDF camera link in the gripper frame
    def link_T(child):
        parent, To, _, _, _ = kin.joints[child]
        return To.cpu().numpy()
    X_urdf = link_T("wrist_camera_mount_link") @ link_T("wrist_camera_link")
    trees = [cKDTree(fr["Pt"]) for fr in F]
    Pt_t = [torch.tensor(fr["Pt"], device=DEV) for fr in F]
    Tbt = [torch.tensor(fr["T_bt"], device=DEV) for fr in F]
    Tbg = [torch.tensor(fr["T_bg"], device=DEV) for fr in F]
    dws = [torch.tensor(fr["dw"], device=DEV) for fr in F]
    H, W = hw
    vv, uu = np.mgrid[0:H:4, 0:W:4]
    uv = torch.tensor(np.stack([uu.ravel(), vv.ravel()], 1), device=DEV, dtype=torch.float64)

    def frame_terms(i, Xv, flog):
        """Visible (z-buffered) top sensor points projected into the wrist image vs Depth Anything 3's depth there:
        log-depth agreement after the per-frame scale (scale-invariant), the scale, and the scaled wrist points in the
        top camera (for the report)."""
        X = pose_from_vec(Xv)
        fw = torch.exp(flog)
        T_tw = Tbt[i] @ Tbg[i] @ X                          # wrist camera -> top camera
        T_wt = torch.linalg.inv(T_tw)
        P = Pt_t[i] @ T_wt[:3, :3].T + T_wt[:3, 3]          # top sensor points in the wrist camera
        zc = P[:, 2]
        ok = zc > 0.03
        P, zc = P[ok], zc[ok]
        pu, pv = fw * P[:, 0] / zc + W / 2, fw * P[:, 1] / zc + H / 2
        ins = (pu >= 0) & (pu < W - 1) & (pv >= 0) & (pv < H - 1)
        if ins.sum() < 100:
            return None
        pu, pv, zc = pu[ins], pv[ins], zc[ins]
        pix = (pv.detach().long() // 2) * W + (pu.detach().long() // 2)      # 2x2 px cells for the z-buffer
        zmin = torch.full((H * W,), float("inf"), dtype=zc.dtype, device=DEV).scatter_reduce(0, pix, zc.detach(), reduce="amin")
        vis = zc.detach() <= zmin[pix] * 1.02
        if vis.sum() < 100:
            return None
        g = torch.stack([pu[vis] / (W - 1) * 2 - 1, pv[vis] / (H - 1) * 2 - 1], -1)[None, None]
        dda = torch.nn.functional.grid_sample(dws[i][None, None], g, align_corners=True)[0, 0, 0]
        ratio = zc[vis] / dda
        sc = ratio.median().detach()
        dz = huber(torch.log(ratio / sc), 0.05).mean()
        dsamp = torch.nn.functional.grid_sample(dws[i][None, None], (uv / torch.tensor([W - 1, H - 1], device=DEV) * 2 - 1)[None, None],
                                                align_corners=True)[0, 0, 0] * sc
        Q = torch.stack([(uv[:, 0] - W / 2) * dsamp / fw, (uv[:, 1] - H / 2) * dsamp / fw, dsamp], 1)
        Q = Q @ T_tw[:3, :3].T + T_tw[:3, 3]
        return dz, Q, sc

    t_urdf = torch.tensor(X_urdf[:3, 3], device=DEV)

    def total(idx, Xv, flog):
        l = Xv.sum() * 0.0
        for i in idx:
            r = frame_terms(i, Xv, flog)
            l = l + (0.5 if r is None else r[0])
        # the camera sits on a printed mount: keep the translation within a few cm of the URDF mount
        return l / len(idx) + 0.05 * ((Xv[3:] - t_urdf) ** 2).sum() / 0.02 ** 2

    fl = torch.tensor(math.log(np.median(fw0)), dtype=torch.float64, device=DEV)   # wrist focal fixed (DA3, stable)
    probe = rng.choice(len(F), min(len(F), 40), replace=False).tolist()
    scored = []
    for si, R in enumerate(axis_rotations()):
        Xs = X_urdf.copy()
        Xs[:3, :3] = X_urdf[:3, :3] @ R
        sv = torch.tensor(np.concatenate([cv2.Rodrigues(Xs[:3, :3])[0][:, 0], Xs[:3, 3]]), device=DEV)
        with torch.no_grad():
            scored.append((total(probe, sv, fl).item(), si, sv))
    scored.sort(key=lambda x: x[0])
    print("orientation scores (best 5):", [(round(a, 4), b) for a, b, _ in scored[:5]], flush=True)
    best = None
    for sc0, si, sv in scored[:3]:
        Xv = sv.clone().requires_grad_(True)
        opt = torch.optim.Adam([Xv], lr=0.002)
        for it in range(args.iters // 3):
            l = total(rng.choice(len(F), 8, replace=False).tolist(), Xv, fl)
            opt.zero_grad(); l.backward(); opt.step()
        with torch.no_grad():
            l = total(probe, Xv, fl).item()
        print(f"orientation {si}: start {sc0:.4f} -> refined {l:.4f}", flush=True)
        if best is None or l < best[0]:
            best = (l, Xv.detach().clone(), si)
    Xv = best[1]
    with torch.no_grad():
        X = pose_from_vec(Xv).cpu().numpy()
        fw = float(torch.exp(fl))
        print(f"chosen orientation {best[2]}, loss {best[0]:.4f}", flush=True)
        per_ds, scales = {}, {}
        for i, fr in enumerate(F):
            r = frame_terms(i, Xv, fl)
            if r is None:
                continue
            _, Q, sc = r
            dist = trees[i].query(Q.cpu().numpy())[0]
            per_ds.setdefault(fr["ds"], []).append(np.median(dist))
            scales.setdefault(fr["ds"], []).append(float(sc))
            if i % 20 == 0:   # overlay: top sensor points projected into the wrist image, coloured by depth
                T_tw = fr["T_bt"] @ fr["T_bg"] @ X
                P = fr["Pt"] @ np.linalg.inv(T_tw)[:3, :3].T + np.linalg.inv(T_tw)[:3, 3]
                P = P[P[:, 2] > 0.02]
                sx, sy = fr["wrist"].shape[1] / W, fr["wrist"].shape[0] / H
                u = (fw * P[:, 0] / P[:, 2] + W / 2) * sx
                v = (fw * P[:, 1] / P[:, 2] + H / 2) * sy
                im = fr["wrist"].copy()
                ok = (u >= 0) & (u < im.shape[1]) & (v >= 0) & (v < im.shape[0])
                c = cv2.applyColorMap((np.clip(P[ok, 2] / 0.6, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, 0, ::-1]
                im[v[ok].astype(int), u[ok].astype(int)] = c
                # jaw check: the model's fixed jaw (gripper_link, white) and moving jaw (yellow) in the wrist image
                Xi = np.linalg.inv(X)
                for T_lb, pts, col in ((fr["T_bg"], kin.link_pts["gripper_link"], (255, 255, 255)),
                                       (fr["T_bj"], kin.link_pts["moving_jaw_so101_v1_link"], (255, 255, 0))):
                    Pl = pts.cpu().numpy() @ T_lb[:3, :3].T + T_lb[:3, 3]                 # base
                    Tg = np.linalg.inv(fr["T_bg"])
                    Pw = (Pl @ Tg[:3, :3].T + Tg[:3, 3]) @ Xi[:3, :3].T + Xi[:3, 3]       # wrist camera
                    Pw = Pw[Pw[:, 2] > 0.005]
                    u = (fw * Pw[:, 0] / Pw[:, 2] + W / 2) * sx
                    v = (fw * Pw[:, 1] / Pw[:, 2] + H / 2) * sy
                    ok = (u >= 0) & (u < im.shape[1]) & (v >= 0) & (v < im.shape[0])
                    im[v[ok].astype(int), u[ok].astype(int)] = col
                cv2.imwrite(f"{args.viz_dir}/{fr['ds'][:30]}_{i:03d}_g{fr['g']:.0f}.jpg", im[..., ::-1])
    out = dict(X_gripper_to_wrist=X.tolist(), f_wrist=fw, wrist_hw=list(hw), wrist_image_hw=list(F[0]["wrist"].shape[:2]),
               note="pixels of the DA3 output (h, w); principal point at the centre",
               per_dataset={k: dict(median_dist_mm=float(np.median(v) * 1000), p90_mm=float(np.percentile(v, 90) * 1000),
                                    scale_median=float(np.median(scales[k])), scale_rel_std=float(np.std(scales[k]) / np.mean(scales[k])))
                            for k, v in per_ds.items()})
    for k, v in out["per_dataset"].items():
        print(f"{k[:45]}: wrist->top median {v['median_dist_mm']:.1f} mm (p90 {v['p90_mm']:.1f}); DA3 scale {v['scale_median']:.3f} "
              f"(spread {v['scale_rel_std']:.1%})", flush=True)
    print(f"f_wrist {fw:.1f} px (DA3 {np.median(fw0):.1f}); X translation {np.round(X[:3, 3] * 1000, 1).tolist()} mm "
          f"(URDF {np.round(X_urdf[:3, 3] * 1000, 1).tolist()})")
    json.dump(out, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    sys.exit(main())
