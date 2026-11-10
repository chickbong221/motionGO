"""
calibrate_wrist_jaw.py (2026-11-06): the wrist-camera mount X (gripper_link -> wrist camera, OpenCV axes) and the wrist
intrinsics from the jaws themselves. The camera is bolted to the gripper, so the fixed jaw (teal, black tip) occupies
the same pixels in every wrist frame: those pixels (teal in > 85% of frames) are matched by the projected fixed-jaw
model (gripper_link surface points), a 2D silhouette fit that needs neither the scene, nor Depth Anything 3, nor the arm
kinematics. (calibrate_wrist.py's scene-based fits were ill-conditioned: focal 2837 px / camera 1 m away, then the
model jaws fell outside the image.) The moving jaw then checks the LeRobot jaw convention (kin_v5): its projection must
follow the moving teal pixels as the gripper value changes.
Loss: projected fixed-jaw points that land in the image must lie on the static-jaw mask (distance transform, px) and
the mask must be covered by them (mask pixel -> nearest projected point, px); 24 axis-aligned orientation starts at the
URDF mount position, Adam on (X, log f), the best kept. Reports both distances (px) and overlays.
  python calibrate_wrist_jaw.py --datasets a,b,c --kin kin_v5.json --out wrist_jaw.json
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
from so101_kin import SO101, pose_from_vec  # noqa: E402

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--kin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--viz_dir", default="/dev/shm/physgo/so101/wrist_jaw_viz")
    ap.add_argument("--fix_f", type=float, default=None,
                    help="fixed wrist focal (px of the wrist image): a small silhouette alone trades focal against "
                         "distance (v1 of this fit: 253 px at 21 cm); Depth Anything 3 gives ~524 px over 427 frames")
    args = ap.parse_args()
    os.makedirs(args.viz_dir, exist_ok=True)
    kc = json.load(open(args.kin))
    kin = SO101(URDF, device=DEV, points_per_link=4000)
    s = torch.tensor(kc["joint_scale"], dtype=torch.float64, device=DEV)
    o = torch.tensor(kc["joint_offset"], dtype=torch.float64, device=DEV)
    imgs, states = [], []
    for n in args.datasets.split(","):
        z = np.load(f"/dev/shm/physgo/so101/kin_cache/{n}.npz")
        imgs.append(z["wrist"]); states.append(z["state"])
    imgs, states = np.concatenate(imgs), np.concatenate(states)
    H, W = imgs.shape[1:3]
    T_all = np.stack([teal(im) for im in imgs])
    static = T_all.mean(0) > 0.85
    static = cv2.morphologyEx(static.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    print(f"{len(imgs)} wrist frames {W}x{H}; static teal (fixed jaw) pixels {static.sum()} ({static.mean():.1%} of the image)", flush=True)
    cv2.imwrite(f"{args.viz_dir}/static_mask.jpg", np.where(static[..., None], imgs[0], imgs[0] // 4)[..., ::-1])
    dt = torch.tensor(cv2.distanceTransform((~static).astype(np.uint8), cv2.DIST_L2, 3), dtype=torch.float64, device=DEV)
    vs, us = np.nonzero(static)
    k = np.random.default_rng(0).choice(len(us), min(len(us), 3000), replace=False)
    mpix = torch.tensor(np.stack([us[k], vs[k]], 1), dtype=torch.float64, device=DEV)
    Pg = kin.link_pts["gripper_link"]                          # fixed jaw (+ wrist roll housing), gripper_link frame

    def link_T(child):
        return kin.joints[child][1].cpu().numpy()
    X_urdf = link_T("wrist_camera_mount_link") @ link_T("wrist_camera_link")

    def project(Xv, flog, P):
        X = pose_from_vec(Xv)
        Xi = torch.linalg.inv(X)
        Q = P @ Xi[:3, :3].T + Xi[:3, 3]
        f = torch.exp(flog)
        front = Q[:, 2] > 0.005
        Q = Q[front]
        return torch.stack([f * Q[:, 0] / Q[:, 2] + W / 2, f * Q[:, 1] / Q[:, 2] + H / 2], 1)

    def loss(Xv, flog):
        uv = project(Xv, flog, Pg)
        ins = (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
        if ins.sum() < 50:
            return torch.tensor(1e3, dtype=torch.float64, device=DEV) + Xv.sum() * 0, None, None
        g = torch.stack([uv[ins, 0] / (W - 1) * 2 - 1, uv[ins, 1] / (H - 1) * 2 - 1], -1)[None, None]
        on = torch.nn.functional.grid_sample(dt[None, None], g, align_corners=True)[0, 0, 0]
        cover = torch.cdist(mpix, uv[ins]).min(1).values
        a, b = on.mean(), cover.mean()
        return a + b, a, b

    best = None
    f0 = math.log(args.fix_f or 520.0)
    for si, R in enumerate(axis_rotations()):
        Xs = X_urdf.copy()
        Xs[:3, :3] = X_urdf[:3, :3] @ R
        Xv = torch.tensor(np.concatenate([cv2.Rodrigues(Xs[:3, :3])[0][:, 0], Xs[:3, 3]]), device=DEV, requires_grad=True)
        fl = torch.tensor(f0, dtype=torch.float64, device=DEV, requires_grad=True)
        opt = torch.optim.Adam([{"params": [Xv], "lr": 0.003}] + ([] if args.fix_f else [{"params": [fl], "lr": 0.003}]))
        for it in range(args.iters):
            l, _, _ = loss(Xv, fl)
            opt.zero_grad(); l.backward(); opt.step()
        with torch.no_grad():
            l, a, b = loss(Xv, fl)
        if a is not None and (best is None or l.item() < best[0]):
            best = (l.item(), Xv.detach().clone(), fl.detach().clone(), si, a.item(), b.item())
    _, Xv, fl, si, a, b = best
    X = pose_from_vec(Xv).cpu().numpy()
    f = float(torch.exp(fl))
    print(f"best start {si}: projected fixed jaw -> mask {a:.1f} px, mask -> projected jaw {b:.1f} px; f {f:.1f} px; "
          f"translation {np.round(X[:3, 3] * 1000, 1).tolist()} mm (URDF {np.round(X_urdf[:3, 3] * 1000, 1).tolist()})", flush=True)

    # moving-jaw check: projected moving jaw (kin_v5 jaw convention) vs the non-static teal pixels near it
    ang = torch.tensor(states, dtype=torch.float64, device=DEV) * s + o
    fk = kin.fk(ang)
    Tg, Tj = fk["gripper_link"], fk["moving_jaw_so101_v1_link"]
    Pj = kin.link_pts["moving_jaw_so101_v1_link"]
    agree = []
    order = np.argsort(states[:, 5])
    for idx in list(order[::max(1, len(order) // 12)]):
        Tgj = torch.linalg.inv(Tg[idx]) @ Tj[idx]               # moving jaw -> gripper_link
        Pjg = Pj @ Tgj[:3, :3].T + Tgj[:3, 3]
        with torch.no_grad():
            uv = project(Xv, fl, Pjg).cpu().numpy()
            uvg = project(Xv, fl, Pg).cpu().numpy()
        mov = T_all[idx] & ~static
        mov = cv2.morphologyEx(mov.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
        ins = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        if ins.sum() > 30 and mov.sum() > 200:
            d = cv2.distanceTransform((~mov).astype(np.uint8), cv2.DIST_L2, 3)
            agree.append(np.median(d[uv[ins, 1].astype(int), uv[ins, 0].astype(int)]))
        im = imgs[idx].copy()
        for P_, col in ((uvg, (255, 255, 255)), (uv, (255, 255, 0))):
            ok = (P_[:, 0] >= 0) & (P_[:, 0] < W) & (P_[:, 1] >= 0) & (P_[:, 1] < H)
            im[P_[ok, 1].astype(int), P_[ok, 0].astype(int)] = col
        cnt, _ = cv2.findContours(static.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im, cnt, -1, (255, 0, 255), 1)
        cv2.imwrite(f"{args.viz_dir}/jaw_g{states[idx, 5]:04.1f}_{idx:03d}.jpg", im[..., ::-1])
    print(f"moving jaw: projected model -> moving teal pixels, median {np.median(agree) if agree else float('nan'):.1f} px over "
          f"{len(agree)} frames (gripper values {states[order[0], 5]:.0f}..{states[order[-1], 5]:.0f})", flush=True)
    out = dict(X_gripper_to_wrist=X.tolist(), f_wrist=f, wrist_image_hw=[H, W], principal_point=[W / 2, H / 2],
               fixed_jaw_px=dict(on_mask=a, coverage=b), moving_jaw_px_median=float(np.median(agree)) if agree else None,
               note="pixels of the original wrist image; fitted on the fixed-jaw silhouette")
    json.dump(out, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    sys.exit(main())
