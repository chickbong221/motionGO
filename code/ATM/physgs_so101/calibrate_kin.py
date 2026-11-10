"""
calibrate_kin.py (2026-11-06): fit the SO-101 kinematic calibration to the top camera of each dataset.
  shared (one robot):  joint value -> angle  angle_j = s_j * v_j + o_j   (start: degrees, s = pi/180, o = 0)
  per dataset:         base -> top-camera pose (6), top focal length f (cx, cy stay at the image centre)
Data per dataset: ~150 frames over ~15 episodes: registered sensor depth (register_depth.py), SAM 3.1 mask of the teal
arm, recorded joint states (observation.state) of the same 30 fps frame.
Loss: (a) 3D: every observed arm point (mask x depth) to the nearest surface point of the posed arm model (Huber,
1 cm); (b) 2D: the posed model projected into the image must fall inside the arm mask (distance transform of the mask,
Huber, in pixels / f). Start: base standing on the sensor's table plane (z = plane normal) under the arm's static
region, 12 yaw starts per dataset (Adam per start, keep the best), then a joint refinement of everything.
Quality report: per dataset the median 3D residual (mm), the silhouette IoU (posed model vs SAM mask) and an overlay.
  python calibrate_kin.py --datasets a,b,c --calib_dir <dir> --sam3_ckpt <ckpt> --out kin.json
"""
import argparse
import glob
import json
import math
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lift_so101 as L  # noqa: E402
from so101_kin import ARM_LINKS, JOINTS, SO101, pose_from_vec  # noqa: E402

URDF = "/dev/shm/physgo/so101/robot/so101_new_calib_camera.urdf"
DEV = "cuda"


def huber(x, d):
    a = x.abs()
    return torch.where(a < d, 0.5 * a * a / d, a - 0.5 * d)


def collect(name, root, calib, seg, n_eps, per_ep, rng):
    """Frames of one dataset: observed arm points (camera frame, computed with the stored f later), masks, states."""
    info, eps, _ = L.load_meta(root)
    dinfo = info["features"]["observation.images.top_depth"]["info"]
    Kt = np.array(calib["K_top"], np.float64)
    ids = eps.index[np.linspace(0, len(eps) - 1, n_eps).round().astype(int)].tolist()
    data = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"])
                      for f in sorted(glob.glob(f"{root}/data/chunk-*/file-*.parquet"))])
    out = []
    for e in ids:
        row = eps.loc[e]
        n30 = int(row["length"])
        top = L.decode_episode(root, row, "observation.images.top", n30)
        q = L.decode_episode(root, row, "observation.images.top_depth", n30, depth=True)
        pick = np.linspace(2, len(top) - 3, per_ep).round().astype(int)
        mask = L.best_instance(seg.track(top[pick], L.ARM_PROMPTS[0]))
        dep = L.register_depth_gpu(np.stack([L.depth_from_codes(x, dinfo) for x in q[pick]]), calib["depth_reg"], Kt, top.shape[1:3])
        st = data[data["episode_index"] == e].sort_values("frame_index")
        S = np.stack(st["observation.state"].values)
        for j, t in enumerate(pick):
            if mask is None or mask[j].sum() < 500:
                continue
            out.append(dict(rgb=top[t], depth=dep[j], mask=mask[j], state=S[min(2 * t, len(S) - 1)].astype(np.float64)))
    return out


def init_pose(frames, Kt):
    """Base frame start: z = table normal (up), origin = the static arm region projected onto the table."""
    rng = np.random.default_rng(0)
    d0 = frames[0]["depth"]
    excl = np.zeros(d0.shape, bool)
    for f in frames[:10]:
        excl |= f["mask"]
    T_wc, _ = L.table_frame(d0, Kt, excl, rng)            # cam -> table frame (z up, origin under image centre)
    static = np.mean([f["mask"] for f in frames], 0) > 0.8  # arm pixels in >80% of frames: the base / shoulder
    P = L.unproject_np(np.where(static, frames[0]["depth"], np.nan), Kt)
    P = P[np.isfinite(P[..., 2])]
    Pw = P @ T_wc[:3, :3].T + T_wc[:3, 3]
    c = np.median(Pw, 0)
    c[2] = 0.0                                             # on the table
    return T_wc, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--data_root", default="/dev/shm/data/so101_v2")
    ap.add_argument("--calib_dir", required=True)
    ap.add_argument("--sam3_ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--per_ep", type=int, default=10)
    ap.add_argument("--viz_dir", default="/dev/shm/physgo/so101/kin_viz")
    args = ap.parse_args()
    os.makedirs(args.viz_dir, exist_ok=True)
    kin = SO101(URDF, device=DEV, points_per_link=1500)
    seg = L.Segmenter(args.sam3_ckpt)
    rng = np.random.default_rng(0)
    names = args.datasets.split(",")
    D = {}
    for n in names:
        calib = json.load(open(f"{args.calib_dir}/{n}.json"))
        fr = collect(n, f"{args.data_root}/{n}", calib, seg, args.episodes, args.per_ep, rng)
        Kt = np.array(calib["K_top"], np.float64)
        T_wc, c = init_pose(fr, Kt)
        # observed points per frame as (u, v, z) so the focal can be fitted; subsample to 1500 per frame
        obs, dts, masks = [], [], []
        for f in fr:
            v, u = np.nonzero(f["mask"] & np.isfinite(f["depth"]))
            k = rng.choice(len(u), min(len(u), 1500), replace=False)
            obs.append(torch.tensor(np.stack([u[k], v[k], f["depth"][v[k], u[k]]], 1), device=DEV))
            dt = cv2.distanceTransform((~f["mask"]).astype(np.uint8), cv2.DIST_L2, 3)
            dts.append(torch.tensor(dt, device=DEV, dtype=torch.float64))
        D[n] = dict(frames=fr, K=Kt, T_wc=T_wc, c=c, obs=obs, dts=dts,
                    states=torch.tensor(np.stack([f["state"] for f in fr]), device=DEV))
        print(f"{n[:45]}: {len(fr)} frames; base start {np.round(c, 3)} m in the table frame", flush=True)

    s = torch.full((6,), math.pi / 180, dtype=torch.float64, device=DEV, requires_grad=True)
    o = torch.zeros(6, dtype=torch.float64, device=DEV, requires_grad=True)

    def model_cam(n, pose_vec, f_log, idx=None):
        d = D[n]
        st = d["states"] if idx is None else d["states"][idx]
        ang = kin.angles(st, s, o)
        P, _ = kin.points(ang, ARM_LINKS)                  # (B, N, 3) base frame
        T = pose_from_vec(pose_vec)                        # base -> camera
        return P @ T[:3, :3].T + T[:3, 3]

    def loss_fn(n, pose_vec, f_log, idx):
        d = D[n]
        f = torch.exp(f_log)
        cx, cy = d["K"][0, 2], d["K"][1, 2]
        Pc = model_cam(n, pose_vec, f_log, idx)
        l3, l2 = 0.0, 0.0
        for b, i in enumerate(idx.tolist()):
            uvz = d["obs"][i]
            z = uvz[:, 2]
            O = torch.stack([(uvz[:, 0] - cx) * z / f, (uvz[:, 1] - cy) * z / f, z], 1)
            dist = torch.cdist(O, Pc[b]).min(1).values
            l3 = l3 + huber(dist, 0.01).mean()
            pz = Pc[b, :, 2].clamp(min=0.05)
            pu, pv = f * Pc[b, :, 0] / pz + cx, f * Pc[b, :, 1] / pz + cy
            H, W = d["dts"][i].shape
            gu = (pu / (W - 1)) * 2 - 1
            gv = (pv / (H - 1)) * 2 - 1
            dtv = torch.nn.functional.grid_sample(d["dts"][i][None, None], torch.stack([gu, gv], -1)[None, None],
                                                  align_corners=True, padding_mode="border")[0, 0, 0]
            l2 = l2 + huber(dtv / f, 0.01).mean()
        return (l3 + l2) / len(idx)

    # stage 1: per dataset, 12 yaw starts (joint conversion fixed at degrees)
    poses = {}
    for n in names:
        d = D[n]
        Rw = d["T_wc"][:3, :3]                             # cam -> table rotation (rows = table axes in cam)
        tw = d["T_wc"][:3, 3]
        best = None
        for yi in range(12):
            yaw = 2 * math.pi * yi / 12
            Rb = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])  # base -> table
            R_bc = Rw.T @ Rb                               # base -> camera
            t_bc = Rw.T @ (d["c"] - tw)
            rv = cv2.Rodrigues(R_bc)[0][:, 0]
            pv = torch.tensor(np.concatenate([rv, t_bc]), device=DEV, requires_grad=True)
            fl = torch.tensor(math.log(d["K"][0, 0]), dtype=torch.float64, device=DEV, requires_grad=True)
            opt = torch.optim.Adam([pv, fl], lr=0.01)
            for it in range(150):
                idx = torch.tensor(rng.choice(len(d["obs"]), 8, replace=False), device=DEV)
                l = loss_fn(n, pv, fl, idx)
                opt.zero_grad(); l.backward(); opt.step()
            with torch.no_grad():
                l = loss_fn(n, pv, fl, torch.arange(len(d["obs"]), device=DEV)).item()
            if best is None or l < best[0]:
                best = (l, pv.detach().clone(), fl.detach().clone(), yi)
        poses[n] = [best[1].requires_grad_(True), best[2].requires_grad_(True)]
        print(f"{n[:45]}: best yaw start {best[3] * 30} deg, loss {best[0]:.5f}", flush=True)

    # stage 2: everything together (joint conversion shared)
    params = [s, o] + [p for v in poses.values() for p in v]
    opt = torch.optim.Adam(params, lr=0.003)
    for it in range(600):
        l = 0.0
        for n in names:
            idx = torch.tensor(rng.choice(len(D[n]["obs"]), 6, replace=False), device=DEV)
            l = l + loss_fn(n, poses[n][0], poses[n][1], idx)
        opt.zero_grad(); l.backward(); opt.step()
        if it % 100 == 0:
            print(f"joint fit {it}: loss {l.item() / len(names):.5f}", flush=True)

    # quality: 3D residual, silhouette IoU, overlays
    out = dict(joint_scale=s.detach().cpu().tolist(), joint_offset=o.detach().cpu().tolist(), joints=list(JOINTS),
               urdf=URDF, datasets={})
    with torch.no_grad():
        for n in names:
            d = D[n]
            pv, fl = poses[n]
            f = float(torch.exp(fl))
            T = pose_from_vec(pv).cpu().numpy()
            res, ious = [], []
            Pc_all = model_cam(n, pv, fl, torch.arange(len(d["obs"]), device=DEV)).cpu().numpy()
            cx, cy = d["K"][0, 2], d["K"][1, 2]
            for i, fr in enumerate(d["frames"]):
                uvz = d["obs"][i].cpu().numpy()
                O = np.stack([(uvz[:, 0] - cx) * uvz[:, 2] / f, (uvz[:, 1] - cy) * uvz[:, 2] / f, uvz[:, 2]], 1)
                from scipy.spatial import cKDTree
                res.append(np.median(cKDTree(Pc_all[i]).query(O)[0]))
                P = Pc_all[i]
                u = (f * P[:, 0] / P[:, 2] + cx).round().astype(int)
                v = (f * P[:, 1] / P[:, 2] + cy).round().astype(int)
                sil = np.zeros(fr["mask"].shape, np.uint8)
                ok = (u >= 0) & (u < sil.shape[1]) & (v >= 0) & (v < sil.shape[0])
                sil[v[ok], u[ok]] = 1
                sil = cv2.dilate(sil, np.ones((5, 5), np.uint8)) > 0
                ious.append((sil & fr["mask"]).sum() / max(1, (sil | fr["mask"]).sum()))
                if i % max(1, len(d["frames"]) // 6) == 0:
                    im = fr["rgb"].copy()
                    im[sil] = (0.5 * im[sil] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
                    cv2.imwrite(f"{args.viz_dir}/{n[:30]}_{i:03d}.jpg", im[..., ::-1])
            out["datasets"][n] = dict(T_base_to_cam=T.tolist(), f=f, cx=cx, cy=cy,
                                      residual_mm_median=float(np.median(res) * 1000), silhouette_iou=float(np.median(ious)))
            print(f"{n[:45]}: f {f:.1f} (DA3 {d['K'][0, 0]:.1f}); 3D residual median {np.median(res) * 1000:.1f} mm "
                  f"(p90 {np.percentile(res, 90) * 1000:.1f}); silhouette IoU median {np.median(ious):.2f}", flush=True)
    print("joint scale (deg units = 0.01745):", np.round(out["joint_scale"], 5).tolist())
    print("joint offset (rad):", np.round(out["joint_offset"], 3).tolist())
    json.dump(out, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    sys.exit(main())
