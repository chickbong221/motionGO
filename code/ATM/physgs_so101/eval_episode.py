"""
eval_episode.py (2026-11-06): whole-episode accuracy of the SO-101 geometry, on episodes NOT used for any calibration.
Every 15 fps frame of the episode:
  arm      the teal arm's sensor points (SAM 3.1 "teal robot arm" x registered top depth) -> nearest point of the posed
           kinematic model (kin_v5: LeRobot convention + fitted offsets, base -> top camera): median distance (mm)
  gripper  the same for the arm points within 6 cm of the model gripper (fixed + moving jaw)
  wrist    wrist-camera points (Depth Anything 3 depth x a per-frame scale from the top sensor, lens model and mount from
           calibrate_wrist_jaw2.py, pose = FK(gripper_link) @ X) -> nearest top-sensor point of the same instant, over the
           surfaces both cameras see: median distance (mm)
Reports, per episode, the distribution over frames (median, p90, share of frames under 10 mm) and saves a strip of
frames: top image with the kinematic gripper (yellow) / arm (magenta), wrist image with the top-sensor points projected.
  python eval_episode.py <dataset> <episode> <sam3 ckpt>
"""
import json
import math
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, "/dev/shm/physgo/code_so101/ATM/physgs_so101")
import lift_so101 as L  # noqa: E402
from so101_kin import ARM_LINKS, GRIPPER_LINKS, SO101  # noqa: E402

DEV = "cuda"
W_ = "/dev/shm/physgo/so101"
URDF = f"{W_}/robot/so101_new_calib_camera.urdf"
CAM_LINKS = ("wrist_camera_mount_link", "wrist_camera_link")


def undistort(xd, yd, k1, k2, it=8):
    x, y = xd.copy(), yd.copy()
    for _ in range(it):
        r2 = x * x + y * y
        d = 1 + k1 * r2 + k2 * r2 * r2
        x, y = xd / d, yd / d
    return x, y


def distort(x, y, k1, k2):
    r2 = x * x + y * y
    d = 1 + k1 * r2 + k2 * r2 * r2
    return x * d, y * d


def main():
    name, ep, ckpt = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    root = f"/dev/shm/data/so101_v2/{name}"
    info, eps, _ = L.load_meta(root)
    row = eps.loc[ep]
    n30 = int(row["length"])
    top = L.decode_episode(root, row, "observation.images.top", n30)
    wrist = L.decode_episode(root, row, "observation.images.wrist", n30)
    q = L.decode_episode(root, row, "observation.images.top_depth", n30, depth=True)
    T = len(top)
    calib = json.load(open(f"{W_}/calib/{name}.json"))
    kc = json.load(open(f"{W_}/calib/kin_v5.json"))
    wc = json.load(open(f"{W_}/calib/wrist_jaw2.json"))
    dk = kc["datasets"][name]
    Kt = np.array([[dk["f"], 0, dk["cx"]], [0, dk["f"], dk["cy"]], [0, 0, 1]])
    T_bt = np.array(dk["T_base_to_cam"])
    dinfo = info["features"]["observation.images.top_depth"]["info"]
    sensor = np.concatenate([L.register_depth_gpu(np.stack([L.depth_from_codes(x, dinfo) for x in q[s:s + 64]]),
                                                  calib["depth_reg"], np.array(calib["K_top"], np.float64), top.shape[1:3])
                             for s in range(0, T, 64)])
    data = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"])
                      for f in sorted(__import__("glob").glob(f"{root}/data/chunk-*/file-*.parquet"))])
    S = np.stack(data[data["episode_index"] == ep].sort_values("frame_index")["observation.state"].values)[::2][:T]
    kin = SO101(URDF, device=DEV, points_per_link=1500)
    s = torch.tensor(kc["joint_scale"], dtype=torch.float64, device=DEV)
    o = torch.tensor(kc["joint_offset"], dtype=torch.float64, device=DEV)
    o[5] = o[5] + wc.get("jaw_offset_rad", 0.0)
    seg = L.Segmenter(ckpt)
    arm = L.best_instance(seg.track(top, L.ARM_PROMPTS[0]))
    del seg
    torch.cuda.empty_cache()
    from depth_anything_3.api import DepthAnything3
    da3 = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1").to(DEV).eval()
    X = np.array(wc["X_gripper_to_wrist"])
    fw, k1, k2 = wc["f_wrist"], wc["k1"], wc["k2"]
    Hw, Ww = wrist.shape[1:3]
    T_tb = np.linalg.inv(T_bt)
    res_arm, res_grip, res_wr, scales = [], [], [], []
    strip = []
    gi = [(ARM_LINKS + CAM_LINKS).index(l) for l in GRIPPER_LINKS]
    for t in range(T):
        ang = torch.tensor(S[t:t + 1], dtype=torch.float64, device=DEV) * s + o
        P, pidx = kin.points(ang, ARM_LINKS + CAM_LINKS)
        Pb = P[0].cpu().numpy()
        Pc = Pb @ T_bt[:3, :3].T + T_bt[:3, 3]
        gsel = np.isin(pidx.cpu().numpy(), gi)
        if arm is not None and arm[t].sum() > 300:
            O = L.unproject_np(np.where(arm[t], sensor[t], np.nan), Kt)
            O = O[np.isfinite(O[..., 2])]
            if len(O) > 100:
                d = cKDTree(Pc).query(O)[0]
                res_arm.append(np.median(d))
                near = cKDTree(Pc[gsel]).query(O)[0] < 0.06
                if near.sum() > 60:
                    res_grip.append(np.median(d[near]))
        # wrist: DA3 depth in the distorted image, scale from the visible top-sensor points
        with torch.no_grad():
            p = da3.inference([wrist[t]], process_res=644)
        dw = cv2.resize(p.depth[0], (Ww, Hw), interpolation=cv2.INTER_LINEAR).astype(np.float64)
        c2b = kin.fk(ang)["gripper_link"][0].cpu().numpy() @ X
        b2c = np.linalg.inv(c2b)
        Ps = L.unproject_np(sensor[t], Kt)
        Ps = Ps[np.isfinite(Ps[..., 2])]
        Ps = Ps[np.random.default_rng(t).choice(len(Ps), min(len(Ps), 40000), replace=False)]
        Pw = (Ps @ T_tb[:3, :3].T + T_tb[:3, 3]) @ b2c[:3, :3].T + b2c[:3, 3]
        Pw = Pw[Pw[:, 2] > 0.02]
        xd, yd = distort(Pw[:, 0] / Pw[:, 2], Pw[:, 1] / Pw[:, 2], k1, k2)
        u = (fw * xd + Ww / 2).round().astype(int)
        v = (fw * yd + Hw / 2).round().astype(int)
        ok = (u >= 0) & (u < Ww) & (v >= 0) & (v < Hw)
        if ok.sum() > 300:
            zb = np.full((Hw, Ww), np.inf)
            np.minimum.at(zb, (v[ok], u[ok]), Pw[ok, 2])
            sel = np.isfinite(zb) & (dw > 0)
            if sel.sum() > 200:
                sc = np.median(zb[sel] / dw[sel])
                scales.append(sc)
                vv, uu = np.nonzero(sel)
                xn, yn = undistort((uu - Ww / 2) / fw, (vv - Hw / 2) / fw, k1, k2)
                z = dw[vv, uu] * sc
                Q = np.stack([xn * z, yn * z, z], 1)
                Qw = Pw[ok]
                res_wr.append(np.median(cKDTree(Qw).query(Q)[0]))
        if t % max(1, T // 8) == 0 and len(strip) < 8:
            im = top[t].copy()
            for sel_, col in ((~gsel, (255, 0, 255)), (gsel, (255, 255, 0))):
                Q_ = Pc[sel_]
                uu = (Kt[0, 0] * Q_[:, 0] / Q_[:, 2] + Kt[0, 2]).round().astype(int)
                vv = (Kt[1, 1] * Q_[:, 1] / Q_[:, 2] + Kt[1, 2]).round().astype(int)
                m = (uu >= 0) & (uu < im.shape[1]) & (vv >= 0) & (vv < im.shape[0])
                im[vv[m], uu[m]] = col
            wi = wrist[t].copy()
            if ok.sum():
                c = cv2.applyColorMap((np.clip(Pw[ok, 2] / 0.5, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, 0, ::-1]
                wi[v[ok], u[ok]] = (0.5 * wi[v[ok], u[ok]] + 0.5 * c).astype(np.uint8)
            strip.append(np.concatenate([cv2.resize(im, (480, 270)), cv2.resize(wi, (360, 270))], 1))

    def summ(a):
        a = np.array(a) * 1000
        return f"median {np.median(a):.1f} mm, p90 {np.percentile(a, 90):.1f} mm, frames < 10 mm {np.mean(a < 10):.0%} (n={len(a)} of {T})"
    print(f"=== {name[:45]} episode {ep} ({T} frames at 15 fps)")
    print(f"  arm (kinematics vs top sensor):        {summ(res_arm)}")
    print(f"  gripper region:                        {summ(res_grip)}")
    print(f"  wrist points vs top sensor:            {summ(res_wr)}; DA3 scale median {np.median(scales):.3f} spread {np.std(scales) / np.mean(scales):.0%}")
    os.makedirs(f"{W_}/eval", exist_ok=True)
    cv2.imwrite(f"{W_}/eval/{name[:30]}_ep{ep}.jpg", np.concatenate(strip, 0)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 75])
    json.dump(dict(arm_mm=[float(x * 1000) for x in res_arm], gripper_mm=[float(x * 1000) for x in res_grip],
                   wrist_mm=[float(x * 1000) for x in res_wr]), open(f"{W_}/eval/{name[:30]}_ep{ep}.json", "w"))


if __name__ == "__main__":
    main()
