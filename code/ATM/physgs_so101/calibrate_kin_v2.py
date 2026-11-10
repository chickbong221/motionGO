"""
calibrate_kin_v2.py (2026-11-06): refine the SO-101 kinematic calibration (calibrate_kin.py v1) where v1 was weak: the
wrist roll and the gripper jaw (v1: wrist_roll scale 1.5x degrees, gripper scale ~0, i.e. a jaw that never moves).
  - gripper evidence: observed points inside the SAM 3.1 gripper mask ("gripper of the teal robot arm") vs the model's
    gripper parts (fixed jaw gripper_link + moving jaw), 3D + projected silhouette, weight 3
  - joint scales positive (log-parameterised) with a weak pull to 1 deg / unit (the three large joints fitted 0.0174,
    0.0173, 0.0177 rad / unit = degrees), offsets free
  - starts from the v1 result (camera poses, focal, offsets)
Frames (top + wrist RGB, registered depth, arm + gripper masks, states) are cached per dataset for the wrist-mount fit.
  python calibrate_kin_v2.py --datasets a,b,c --calib_dir <dir> --sam3_ckpt <ckpt> --init kin.json --out kin_v2.json
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
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lift_so101 as L  # noqa: E402
from so101_kin import ARM_LINKS, GRIPPER_LINKS, JOINTS, SO101, pose_from_vec  # noqa: E402

URDF = "/dev/shm/physgo/so101/robot/so101_new_calib_camera.urdf"
DEV = "cuda"
DEG = math.pi / 180


def huber(x, d):
    a = x.abs()
    return torch.where(a < d, 0.5 * a * a / d, a - 0.5 * d)


def collect(name, root, calib, seg, n_eps, per_ep, cache):
    if os.path.exists(cache):
        z = np.load(cache)
        return [dict(rgb=z["rgb"][i], wrist=z["wrist"][i], depth=z["depth"][i], mask=z["mask"][i], grip=z["grip"][i],
                     state=z["state"][i], ep=int(z["ep"][i]), t=int(z["t"][i])) for i in range(len(z["state"]))]
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
        wr = L.decode_episode(root, row, "observation.images.wrist", n30)
        q = L.decode_episode(root, row, "observation.images.top_depth", n30, depth=True)
        pick = np.linspace(2, len(top) - 3, per_ep).round().astype(int)
        arm = L.best_instance(seg.track(top[pick], L.ARM_PROMPTS[0]))
        grip = L.best_instance(seg.track(top[pick], L.GRIPPER_PROMPT))
        dep = L.register_depth_gpu(np.stack([L.depth_from_codes(x, dinfo) for x in q[pick]]), calib["depth_reg"], Kt, top.shape[1:3])
        st = data[data["episode_index"] == e].sort_values("frame_index")
        S = np.stack(st["observation.state"].values)
        for j, t in enumerate(pick):
            if arm is None or arm[j].sum() < 500:
                continue
            g = grip[j] & arm[j] if grip is not None else np.zeros_like(arm[j])
            out.append(dict(rgb=top[t], wrist=wr[t], depth=dep[j], mask=arm[j], grip=g,
                            state=S[min(2 * t, len(S) - 1)].astype(np.float64), ep=e, t=int(t)))
    np.savez_compressed(cache, **{k: np.stack([f[k] for f in out]) for k in out[0]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--data_root", default="/dev/shm/data/so101_v2")
    ap.add_argument("--calib_dir", required=True)
    ap.add_argument("--sam3_ckpt", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--per_ep", type=int, default=10)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--viz_dir", default="/dev/shm/physgo/so101/kin_viz_v2")
    args = ap.parse_args()
    os.makedirs(args.viz_dir, exist_ok=True)
    os.makedirs("/dev/shm/physgo/so101/kin_cache", exist_ok=True)
    init = json.load(open(args.init))
    kin = SO101(URDF, device=DEV, points_per_link=1500)
    seg = None
    rng = np.random.default_rng(0)
    names = args.datasets.split(",")
    D = {}
    for n in names:
        calib = json.load(open(f"{args.calib_dir}/{n}.json"))
        cache = f"/dev/shm/physgo/so101/kin_cache/{n}.npz"
        if not os.path.exists(cache) and seg is None:
            seg = L.Segmenter(args.sam3_ckpt)
        fr = collect(n, f"{args.data_root}/{n}", calib, seg, args.episodes, args.per_ep, cache)
        cx, cy = calib["K_top"][0][2], calib["K_top"][1][2]
        obs, gobs, dts, gdts = [], [], [], []
        for f in fr:
            for m, lst, k in ((f["mask"], obs, 1500), (f["grip"], gobs, 600)):
                v, u = np.nonzero(m & np.isfinite(f["depth"]))
                if len(u) < 30:
                    lst.append(None)
                    continue
                kk = rng.choice(len(u), min(len(u), k), replace=False)
                lst.append(torch.tensor(np.stack([u[kk], v[kk], f["depth"][v[kk], u[kk]]], 1), device=DEV))
            dts.append(torch.tensor(cv2.distanceTransform((~f["mask"]).astype(np.uint8), cv2.DIST_L2, 3), device=DEV, dtype=torch.float64))
            gdts.append(torch.tensor(cv2.distanceTransform((~f["grip"]).astype(np.uint8), cv2.DIST_L2, 3), device=DEV, dtype=torch.float64)
                        if f["grip"].sum() > 30 else None)
        di = init["datasets"][n]
        D[n] = dict(frames=fr, cx=cx, cy=cy, obs=obs, gobs=gobs, dts=dts, gdts=gdts,
                    states=torch.tensor(np.stack([f["state"] for f in fr]), device=DEV),
                    pose=torch.tensor(np.concatenate([cv2.Rodrigues(np.array(di["T_base_to_cam"])[:3, :3])[0][:, 0],
                                                      np.array(di["T_base_to_cam"])[:3, 3]]), device=DEV, requires_grad=True),
                    flog=torch.tensor(math.log(di["f"]), dtype=torch.float64, device=DEV, requires_grad=True))
        print(f"{n[:45]}: {len(fr)} frames, {sum(g is not None for g in gobs)} with gripper points", flush=True)

    log_s = torch.tensor(np.log(np.clip(np.abs(init["joint_scale"]), 1e-3, None)), device=DEV)
    log_s[5] = math.log(DEG)                 # v1 collapsed the gripper scale: restart it at degrees
    log_s[4] = math.log(DEG)                 # and the wrist roll
    log_s.requires_grad_(True)
    o = torch.tensor(init["joint_offset"], dtype=torch.float64, device=DEV, requires_grad=True)
    gi = [ARM_LINKS.index(l) for l in GRIPPER_LINKS]

    def frame_loss(n, i, P, pidx, f):
        d = D[n]
        cx, cy = d["cx"], d["cy"]
        out = 0.0
        for obs, dt, sel, w in ((d["obs"][i], d["dts"][i], None, 1.0), (d["gobs"][i], d["gdts"][i], gi, 3.0)):
            if obs is None or dt is None:
                continue
            Q = P if sel is None else P[torch.isin(pidx, torch.tensor(sel, device=DEV))]
            z = obs[:, 2]
            O = torch.stack([(obs[:, 0] - cx) * z / f, (obs[:, 1] - cy) * z / f, z], 1)
            l3 = huber(torch.cdist(O, Q).min(1).values, 0.005 if sel else 0.01).mean()
            pz = Q[:, 2].clamp(min=0.05)
            pu, pv = f * Q[:, 0] / pz + cx, f * Q[:, 1] / pz + cy
            H, W = dt.shape
            g = torch.stack([pu / (W - 1) * 2 - 1, pv / (H - 1) * 2 - 1], -1)[None, None]
            l2 = huber(torch.nn.functional.grid_sample(dt[None, None], g, align_corners=True, padding_mode="border")[0, 0, 0] / f, 0.01).mean()
            out = out + w * (l3 + l2)
        return out

    def batch_loss(n, idx):
        d = D[n]
        f = torch.exp(d["flog"])
        ang = d["states"][idx] * torch.exp(log_s) + o
        P, pidx = kin.points(ang, ARM_LINKS)
        T = pose_from_vec(d["pose"])
        Pc = P @ T[:3, :3].T + T[:3, 3]
        return sum(frame_loss(n, int(i), Pc[b], pidx, f) for b, i in enumerate(idx.tolist())) / len(idx)

    params = [log_s, o] + [D[n][k] for n in names for k in ("pose", "flog")]
    opt = torch.optim.Adam(params, lr=0.003)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
    for it in range(args.iters):
        l = sum(batch_loss(n, torch.tensor(rng.choice(len(D[n]["obs"]), 6, replace=False), device=DEV)) for n in names) / len(names)
        l = l + 1e-3 * ((log_s - math.log(DEG)) ** 2).sum()
        opt.zero_grad(); l.backward(); opt.step(); sched.step()
        if it % 250 == 0:
            print(f"iter {it}: loss {l.item():.5f}; scale/deg {np.round((torch.exp(log_s) / DEG).detach().cpu().numpy(), 3).tolist()}", flush=True)

    out = dict(joint_scale=torch.exp(log_s).detach().cpu().tolist(), joint_offset=o.detach().cpu().tolist(),
               joints=list(JOINTS), urdf=URDF, datasets={})
    with torch.no_grad():
        for n in names:
            d = D[n]
            f = float(torch.exp(d["flog"]))
            T = pose_from_vec(d["pose"])
            res, gres, ious, gious = [], [], [], []
            for i, fr in enumerate(d["frames"]):
                ang = d["states"][i:i + 1] * torch.exp(log_s) + o
                P, pidx = kin.points(ang, ARM_LINKS)
                Pc = (P[0] @ T[:3, :3].T + T[:3, 3]).cpu().numpy()
                pid = pidx.cpu().numpy()
                for m, sel, r_, io_ in ((fr["mask"], None, res, ious), (fr["grip"], gi, gres, gious)):
                    if m.sum() < 30:
                        continue
                    Q = Pc if sel is None else Pc[np.isin(pid, sel)]
                    v, u = np.nonzero(m & np.isfinite(fr["depth"]))
                    z = fr["depth"][v, u]
                    O = np.stack([(u - d["cx"]) * z / f, (v - d["cy"]) * z / f, z], 1)
                    r_.append(np.median(cKDTree(Q).query(O)[0]))
                    uu = (f * Q[:, 0] / Q[:, 2] + d["cx"]).round().astype(int)
                    vv = (f * Q[:, 1] / Q[:, 2] + d["cy"]).round().astype(int)
                    sil = np.zeros(m.shape, np.uint8)
                    ok = (uu >= 0) & (uu < m.shape[1]) & (vv >= 0) & (vv < m.shape[0])
                    sil[vv[ok], uu[ok]] = 1
                    sil = cv2.dilate(sil, np.ones((5, 5), np.uint8)) > 0
                    io_.append((sil & m).sum() / max(1, (sil | m).sum()))
                if i % max(1, len(d["frames"]) // 6) == 0:
                    im = fr["rgb"].copy()
                    for sel, col in ((None, (255, 0, 255)), (gi, (255, 255, 0))):
                        Q = Pc if sel is None else Pc[np.isin(pid, sel)]
                        uu = (f * Q[:, 0] / Q[:, 2] + d["cx"]).round().astype(int)
                        vv = (f * Q[:, 1] / Q[:, 2] + d["cy"]).round().astype(int)
                        ok = (uu >= 0) & (uu < im.shape[1]) & (vv >= 0) & (vv < im.shape[0])
                        im[vv[ok], uu[ok]] = col
                    cv2.imwrite(f"{args.viz_dir}/{n[:30]}_{i:03d}.jpg", im[..., ::-1])
            out["datasets"][n] = dict(T_base_to_cam=T.cpu().numpy().tolist(), f=f, cx=d["cx"], cy=d["cy"],
                                      residual_mm_median=float(np.median(res) * 1000), gripper_residual_mm_median=float(np.median(gres) * 1000),
                                      silhouette_iou=float(np.median(ious)), gripper_silhouette_iou=float(np.median(gious)))
            print(f"{n[:45]}: f {f:.1f}; arm residual {np.median(res) * 1000:.1f} mm, IoU {np.median(ious):.2f}; gripper residual "
                  f"{np.median(gres) * 1000:.1f} mm, IoU {np.median(gious):.2f}", flush=True)
    print("joint scale / deg:", np.round(np.array(out["joint_scale"]) / DEG, 3).tolist())
    print("joint offset (rad):", np.round(out["joint_offset"], 3).tolist())
    json.dump(out, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    sys.exit(main())
