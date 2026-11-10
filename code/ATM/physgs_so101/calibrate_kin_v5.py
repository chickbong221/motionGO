"""
calibrate_kin_v5.py (2026-11-06): the SO-101 joint conversion fixed by LeRobot's own convention (lerobot 0.7
robots/so_follower: use_degrees = True by default -> the five arm joints are recorded in degrees, (raw - mid) * 360 /
4095; the gripper is MotorNormMode.RANGE_0_100 -> 0 = one end, 100 = the other end of the calibrated jaw range).
  arm joints   angle = v * pi / 180 + offset_j      (offsets fitted: the URDF zero vs the calibration mid)
  jaw          angle = JAW_LO + v / 100 * (JAW_HI - JAW_LO), the URDF jaw limits (-10 .. 100 deg)
v1 fitted free scales (0.99-1.01 deg / unit for joints 0-2, but wrist_flex 0.85, wrist_roll 1.54, jaw ~0): the free
fit is ill-posed at the wrist; with the scales known only the offsets, camera poses and focal lengths are refitted
(Adam, from v1), on the teal-arm mask evidence (3D nearest-model distance, Huber 1 cm, + projected silhouette).
Reports: arm residual, near-gripper residual (arm points within 6 cm of the model gripper), overlays.
  python calibrate_kin_v5.py --datasets a,b,c --v1 kin.json --out kin_v5.json
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from so101_kin import ARM_LINKS, GRIPPER_LINKS, SO101, pose_from_vec  # noqa: E402

URDF = "/dev/shm/physgo/so101/robot/so101_new_calib_camera.urdf"
DEV = "cuda"
DEG = math.pi / 180
CAM_LINKS = ("wrist_camera_mount_link", "wrist_camera_link")
MODEL = ARM_LINKS + CAM_LINKS
JAW_LO, JAW_HI = -0.174533, 1.74533


def huber(x, d):
    a = x.abs()
    return torch.where(a < d, 0.5 * a * a / d, a - 0.5 * d)


def conv_tensors(off):
    """scale (6,), offset (6,) with the arm offsets `off` (5,) and LeRobot's jaw convention."""
    s = torch.cat([torch.full((5,), DEG, dtype=torch.float64, device=DEV),
                   torch.tensor([(JAW_HI - JAW_LO) / 100], dtype=torch.float64, device=DEV)])
    o = torch.cat([off, torch.tensor([JAW_LO], dtype=torch.float64, device=DEV)])
    return s, o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--v1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--viz_dir", default="/dev/shm/physgo/so101/kin_viz_v5")
    args = ap.parse_args()
    os.makedirs(args.viz_dir, exist_ok=True)
    v1 = json.load(open(args.v1))
    kin = SO101(URDF, device=DEV, points_per_link=1200)
    rng = np.random.default_rng(0)
    names = args.datasets.split(",")
    D = {}
    for n in names:
        z = np.load(f"/dev/shm/physgo/so101/kin_cache/{n}.npz")
        dk = v1["datasets"][n]
        obs, dts = [], []
        for i in range(len(z["state"])):
            m = z["mask"][i] & np.isfinite(z["depth"][i])
            v, u = np.nonzero(m)
            k = rng.choice(len(u), min(len(u), 1500), replace=False)
            obs.append(torch.tensor(np.stack([u[k], v[k], z["depth"][i][v[k], u[k]]], 1), dtype=torch.float64, device=DEV))
            dts.append(torch.tensor(cv2.distanceTransform((~z["mask"][i]).astype(np.uint8), cv2.DIST_L2, 3), dtype=torch.float64, device=DEV))
        T = np.array(dk["T_base_to_cam"])
        D[n] = dict(obs=obs, dts=dts, cx=dk["cx"], cy=dk["cy"], rgb=z["rgb"],
                    states=torch.tensor(z["state"], dtype=torch.float64, device=DEV),
                    pose=torch.tensor(np.concatenate([cv2.Rodrigues(T[:3, :3])[0][:, 0], T[:3, 3]]), dtype=torch.float64, device=DEV, requires_grad=True),
                    flog=torch.tensor(math.log(dk["f"]), dtype=torch.float64, device=DEV, requires_grad=True))
    off = torch.tensor(v1["joint_offset"][:5], dtype=torch.float64, device=DEV)
    # v1's wrist_flex / wrist_roll offsets belonged to wrong scales: start them at the value that keeps the v1 angle at
    # the median recorded value
    S_all = torch.cat([D[n]["states"] for n in names])
    for j in (3, 4):
        med = S_all[:, j].median()
        off[j] = v1["joint_scale"][j] * med + v1["joint_offset"][j] - DEG * med
    off.requires_grad_(True)

    def model_cam(n, idx):
        d = D[n]
        s, o = conv_tensors(off)
        P, pidx = kin.points(d["states"][idx] * s + o, MODEL)
        T = pose_from_vec(d["pose"])
        return torch.einsum("ij,bnj->bni", T[:3, :3], P) + T[:3, 3], pidx

    def loss_fn(n, idx):
        d = D[n]
        f = torch.exp(d["flog"])
        Pc, _ = model_cam(n, idx)
        l = 0.0
        for b, i in enumerate(idx.tolist()):
            uvz = d["obs"][i]
            z = uvz[:, 2]
            O = torch.stack([(uvz[:, 0] - d["cx"]) * z / f, (uvz[:, 1] - d["cy"]) * z / f, z], 1)
            l = l + huber(torch.cdist(O, Pc[b]).min(1).values, 0.01).mean()
            pz = Pc[b, :, 2].clamp(min=0.05)
            pu, pv = f * Pc[b, :, 0] / pz + d["cx"], f * Pc[b, :, 1] / pz + d["cy"]
            H, W = d["dts"][i].shape
            g = torch.stack([pu / (W - 1) * 2 - 1, pv / (H - 1) * 2 - 1], -1)[None, None]
            l = l + huber(torch.nn.functional.grid_sample(d["dts"][i][None, None], g, align_corners=True, padding_mode="border")[0, 0, 0] / f, 0.01).mean()
        return l / len(idx)

    params = [off] + [D[n][k] for n in names for k in ("pose", "flog")]
    opt = torch.optim.Adam(params, lr=0.002)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
    for it in range(args.iters):
        l = sum(loss_fn(n, torch.tensor(rng.choice(len(D[n]["obs"]), 6, replace=False), device=DEV)) for n in names) / len(names)
        opt.zero_grad(); l.backward(); opt.step(); sched.step()
        if it % 200 == 0:
            print(f"iter {it}: loss {l.item():.5f}; offsets deg {np.round(off.detach().cpu().numpy() / DEG, 1).tolist()}", flush=True)
    s, o = conv_tensors(off.detach())
    out = dict(joint_scale=s.cpu().tolist(), joint_offset=o.cpu().tolist(), joints=v1["joints"], urdf=URDF,
               convention="lerobot so_follower: use_degrees arm joints, gripper RANGE_0_100 over the URDF jaw limits",
               model_links=list(MODEL), datasets={})
    with torch.no_grad():
        for n in names:
            d = D[n]
            f = float(torch.exp(d["flog"]))
            T = pose_from_vec(d["pose"])
            res, gres = [], []
            for i in range(len(d["obs"])):
                Pc, pidx = model_cam(n, torch.tensor([i], device=DEV))
                Pc = Pc[0]
                uvz = d["obs"][i]
                O = torch.stack([(uvz[:, 0] - d["cx"]) * uvz[:, 2] / f, (uvz[:, 1] - d["cy"]) * uvz[:, 2] / f, uvz[:, 2]], 1)
                res.append(torch.cdist(O, Pc).min(1).values.median().item() * 1000)
                gsel = torch.isin(pidx, torch.tensor([MODEL.index(l) for l in GRIPPER_LINKS], device=DEV))
                near = torch.cdist(O, Pc[gsel]).min(1).values < 0.06
                if near.sum() > 60:
                    gres.append(torch.cdist(O[near], Pc).min(1).values.median().item() * 1000)
                if i % 30 == 0:
                    im = d["rgb"][i].copy()
                    for sel, col in ((~gsel, (255, 0, 255)), (gsel, (255, 255, 0))):
                        Q = Pc[sel].cpu().numpy()
                        uu = (f * Q[:, 0] / Q[:, 2] + d["cx"]).round().astype(int)
                        vv = (f * Q[:, 1] / Q[:, 2] + d["cy"]).round().astype(int)
                        ok = (uu >= 0) & (uu < im.shape[1]) & (vv >= 0) & (vv < im.shape[0])
                        im[vv[ok], uu[ok]] = col
                    cv2.imwrite(f"{args.viz_dir}/{n[:30]}_{i:03d}_g{d['states'][i, 5].item():.0f}.jpg", im[..., ::-1])
            out["datasets"][n] = dict(T_base_to_cam=T.cpu().numpy().tolist(), f=f, cx=d["cx"], cy=d["cy"],
                                      residual_mm_median=float(np.median(res)), near_gripper_residual_mm_median=float(np.median(gres)))
            print(f"{n[:45]}: f {f:.1f} (v1 {v1['datasets'][n]['f']:.1f}); arm residual median {np.median(res):.1f} mm "
                  f"(p90 {np.percentile(res, 90):.1f}) [v1 {v1['datasets'][n]['residual_mm_median']:.1f}]; near-gripper "
                  f"{np.median(gres):.1f} mm (p90 {np.percentile(gres, 90):.1f})", flush=True)
    print("offsets (deg):", np.round(np.array(out["joint_offset"][:5]) / DEG, 2).tolist())
    json.dump(out, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    sys.exit(main())
