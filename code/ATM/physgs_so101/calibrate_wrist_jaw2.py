"""
calibrate_wrist_jaw2.py (2026-11-06): wrist camera = wide-angle lens with visible barrel distortion, so a pinhole fit
on the fixed jaw alone was ambiguous (calibrate_wrist_jaw.py: free focal 253 px matched both jaws but put the camera 21
cm from its mount; focal fixed at DA3's 524 px matched the fixed jaw but missed the moving jaw by 100 px).
Model: OpenCV-style radial distortion  x_d = x (1 + k1 r^2 + k2 r^4),  u = f x_d + W/2,  v = f y_d + H/2.
Fitted: mount X (gripper_link -> camera, 6), log f, k1, k2, and a small jaw-angle offset d (prior 0, 10 deg) on top of
the LeRobot jaw convention (kin_v5). Evidence:
  fixed jaw   teal pixels present in > 85% of wrist frames (the camera is bolted to the gripper)
  moving jaw  per frame, teal pixels outside the fixed-jaw mask, in components touching the image bottom (excludes the
              light-blue drawer), on 40 frames spread over the recorded gripper values
for each: projected model points must lie on the mask (distance transform) and cover it (mask pixel -> nearest
projected point). Prior: translation within ~1.5 cm of the printed URDF mount. 24 orientation starts.
  python calibrate_wrist_jaw2.py --datasets a,b,c --kin kin_v5.json --out wrist_jaw2.json
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from so101_kin import SO101, _axis_angle, pose_from_vec  # noqa: E402

URDF = "/dev/shm/physgo/so101/robot/so101_new_calib_camera.urdf"
DEV = "cuda"


def axis_rotations():
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                R[p, i] = s
            if np.linalg.det(R) > 0:
                out.append(R)
    return out


def teal(img):
    h = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return (h[..., 0] >= 80) & (h[..., 0] <= 105) & (h[..., 1] >= 90) & (h[..., 2] >= 60)


def bottom_components(m):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    keep = np.zeros_like(m)
    H = m.shape[0]
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if y + h >= H - 3 and a > 300:
            keep |= lab == i
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--kin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--viz_dir", default="/dev/shm/physgo/so101/wrist_jaw2_viz")
    args = ap.parse_args()
    os.makedirs(args.viz_dir, exist_ok=True)
    kc = json.load(open(args.kin))
    kin = SO101(URDF, device=DEV, points_per_link=3000)
    s = torch.tensor(kc["joint_scale"], dtype=torch.float64, device=DEV)
    o = torch.tensor(kc["joint_offset"], dtype=torch.float64, device=DEV)
    imgs, states = [], []
    for n in args.datasets.split(","):
        z = np.load(f"/dev/shm/physgo/so101/kin_cache/{n}.npz")
        imgs.append(z["wrist"]); states.append(z["state"])
    imgs, states = np.concatenate(imgs), np.concatenate(states)
    H, W = imgs.shape[1:3]
    T_all = np.stack([teal(im) for im in imgs])
    static = cv2.morphologyEx((T_all.mean(0) > 0.85).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    rng = np.random.default_rng(0)

    def mask_terms(m, n_pix):
        dt = torch.tensor(cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3), dtype=torch.float64, device=DEV)
        v, u = np.nonzero(m)
        k = rng.choice(len(u), min(len(u), n_pix), replace=False)
        return dt, torch.tensor(np.stack([u[k], v[k]], 1), dtype=torch.float64, device=DEV)

    fixed_dt, fixed_pix = mask_terms(static, 3000)
    # moving-jaw frames spread over the gripper values
    order = np.argsort(states[:, 5])
    mov = []
    for idx in order[np.linspace(0, len(order) - 1, 60).round().astype(int)]:
        m = bottom_components(cv2.morphologyEx((T_all[idx] & ~cv2.dilate(static.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)).astype(np.uint8),
                                               cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0)
        if m.sum() > 800:
            dt, pix = mask_terms(m, 800)
            mov.append(dict(idx=int(idx), g=float(states[idx, 5]), dt=dt, pix=pix, m=m))
        if len(mov) >= 40:
            break
    print(f"{len(imgs)} wrist frames; fixed-jaw pixels {static.sum()}; {len(mov)} moving-jaw frames (gripper {mov[0]['g']:.0f}..{mov[-1]['g']:.0f})", flush=True)

    Pg = kin.link_pts["gripper_link"]
    Pj = kin.link_pts["moving_jaw_so101_v1_link"]
    jaw_parent, jaw_To, jaw_axis, _, _ = kin.joints["moving_jaw_so101_v1_link"]

    def jaw_pts(gval, d):
        ang = (gval * s[5] + o[5] + d).reshape(1)
        Tj = jaw_To @ _axis_angle(jaw_axis, ang)[0]      # moving jaw -> gripper_link
        return Pj @ Tj[:3, :3].T + Tj[:3, 3]

    def link_T(child):
        return kin.joints[child][1].cpu().numpy()
    X_urdf = link_T("wrist_camera_mount_link") @ link_T("wrist_camera_link")
    t_urdf = torch.tensor(X_urdf[:3, 3], device=DEV)

    def project(p, P):
        X = pose_from_vec(p[:6])
        Xi = torch.linalg.inv(X)
        Q = P @ Xi[:3, :3].T + Xi[:3, 3]
        Q = Q[Q[:, 2] > 0.003]
        x, y = Q[:, 0] / Q[:, 2], Q[:, 1] / Q[:, 2]
        r2 = x * x + y * y
        d = 1 + p[7] * r2 + p[8] * r2 * r2
        f = torch.exp(p[6])
        return torch.stack([f * x * d + W / 2, f * y * d + H / 2], 1)

    def term(uv, dt, pix):
        ins = (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
        if ins.sum() < 30:
            return torch.tensor(50.0, dtype=torch.float64, device=DEV) + uv.sum() * 0, torch.tensor(50.0, device=DEV)
        g = torch.stack([uv[ins, 0] / (W - 1) * 2 - 1, uv[ins, 1] / (H - 1) * 2 - 1], -1)[None, None]
        on = torch.nn.functional.grid_sample(dt[None, None], g, align_corners=True)[0, 0, 0].clamp(max=50).mean()
        cover = torch.cdist(pix, uv[ins]).min(1).values.clamp(max=50).mean()
        return on, cover

    def loss(p, frames, detail=False):
        a, b = term(project(p, Pg), fixed_dt, fixed_pix)
        mm = [term(project(p, jaw_pts(fr["g"], p[9])), fr["dt"], fr["pix"]) for fr in frames]
        ma = torch.stack([x[0] for x in mm]).mean()
        mb = torch.stack([x[1] for x in mm]).mean()
        prior = ((p[3:6] - t_urdf) ** 2).sum() / 0.015 ** 2 + (p[9] / math.radians(10)) ** 2 + (p[7] ** 2 + p[8] ** 2)
        l = a + b + ma + mb + prior
        return (l, a.item(), b.item(), ma.item(), mb.item()) if detail else l

    best = None
    for si, R in enumerate(axis_rotations()):
        Xs = X_urdf.copy()
        Xs[:3, :3] = X_urdf[:3, :3] @ R
        p = torch.tensor(np.concatenate([cv2.Rodrigues(Xs[:3, :3])[0][:, 0], Xs[:3, 3], [math.log(300.0), 0.0, 0.0, 0.0]]),
                         dtype=torch.float64, device=DEV, requires_grad=True)
        opt = torch.optim.Adam([p], lr=0.003)
        for it in range(args.iters):
            fr = [mov[i] for i in rng.choice(len(mov), 8, replace=False)]
            l = loss(p, fr)
            opt.zero_grad(); l.backward(); opt.step()
        with torch.no_grad():
            l = loss(p, mov).item()
        if best is None or l < best[0]:
            best = (l, p.detach().clone(), si)
    p = best[1].requires_grad_(True)
    opt = torch.optim.Adam([p], lr=0.001)
    for it in range(800):
        l = loss(p, [mov[i] for i in rng.choice(len(mov), 10, replace=False)])
        opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad():
        l, a, b, ma, mb = loss(p, mov, detail=True)
        X = pose_from_vec(p[:6]).cpu().numpy()
        f, k1, k2, dj = math.exp(p[6].item()), p[7].item(), p[8].item(), math.degrees(p[9].item())
        print(f"start {best[2]}: fixed jaw on {a:.1f} px cover {b:.1f} px; moving jaw on {ma:.1f} px cover {mb:.1f} px; f {f:.1f} px, "
              f"k1 {k1:.3f}, k2 {k2:.3f}; jaw offset {dj:.1f} deg; translation {np.round(X[:3, 3] * 1000, 1).tolist()} mm "
              f"(URDF {np.round(X_urdf[:3, 3] * 1000, 1).tolist()})", flush=True)
        for fr in mov[::5]:
            im = imgs[fr["idx"]].copy()
            for P_, col in ((project(p, Pg), (255, 255, 255)), (project(p, jaw_pts(fr["g"], p[9])), (255, 255, 0))):
                P_ = P_.cpu().numpy()
                ok = (P_[:, 0] >= 0) & (P_[:, 0] < W) & (P_[:, 1] >= 0) & (P_[:, 1] < H)
                im[P_[ok, 1].astype(int), P_[ok, 0].astype(int)] = col
            for m, col in ((static, (255, 0, 255)), (fr["m"], (0, 255, 0))):
                cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(im, cnt, -1, col, 1)
            cv2.imwrite(f"{args.viz_dir}/jaw_g{fr['g']:04.1f}_{fr['idx']:03d}.jpg", im[..., ::-1])
    out = dict(X_gripper_to_wrist=X.tolist(), f_wrist=f, k1=k1, k2=k2, principal_point=[W / 2, H / 2], wrist_image_hw=[H, W],
               jaw_offset_rad=p[9].item(), fit_px=dict(fixed_on=a, fixed_cover=b, moving_on=ma, moving_cover=mb),
               model="x_d = x (1 + k1 r^2 + k2 r^4), u = f x_d + W/2 (pixels of the original wrist image)")
    json.dump(out, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    sys.exit(main())
