"""
calibrate_kin_v3.py (2026-11-06): the wrist roll and gripper-jaw conversions, with everything else frozen at the v1
fit (calibrate_kin.py: camera poses, focal lengths, shoulder_pan / shoulder_lift / elbow_flex / wrist_flex; 3-4 mm arm
residual, scales ~1 deg / unit). v2 (free joint fit with a gripper term) drifted (focal 828, pan scale 0.52, 18 mm).
The model now includes the wrist camera + mount (URDF so101_new_calib_camera), which SAM's gripper / arm masks contain.
  stage 1  wrist_roll scale x offset grid (5 x 24), model = gripper_link + camera parts, data = gripper-mask points
  stage 2  gripper (jaw) scale x offset grid (7 x 26), model = + moving jaw, same data
  stage 3  local Adam refinement of the four numbers
Each grid cell: mean Huber(5 mm) distance from the observed gripper points (registered sensor depth x SAM gripper mask)
to the posed model, over all cached frames. Reports the gripper residual before / after, the arm residual (unchanged
check), and how far the jaw opens over the recorded gripper range.
  python calibrate_kin_v3.py --datasets a,b,c --v1 kin.json --out kin_v3.json
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
from so101_kin import ARM_LINKS, GRIPPER_LINKS, JOINTS, SO101, pose_from_vec  # noqa: E402

URDF = "/dev/shm/physgo/so101/robot/so101_new_calib_camera.urdf"
DEV = "cuda"
DEG = math.pi / 180
CAM_LINKS = ("wrist_camera_mount_link", "wrist_camera_link")


def huber(x, d):
    a = x.abs()
    return torch.where(a < d, 0.5 * a * a / d, a - 0.5 * d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--v1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--viz_dir", default="/dev/shm/physgo/so101/kin_viz_v3")
    args = ap.parse_args()
    os.makedirs(args.viz_dir, exist_ok=True)
    v1 = json.load(open(args.v1))
    kin = SO101(URDF, device=DEV, points_per_link=800)
    rng = np.random.default_rng(0)
    frames = []
    for n in args.datasets.split(","):
        z = np.load(f"/dev/shm/physgo/so101/kin_cache/{n}.npz")
        dk = v1["datasets"][n]
        T = torch.tensor(dk["T_base_to_cam"], dtype=torch.float64, device=DEV)
        for i in range(len(z["state"])):
            for key, cap in (("grip", 800), ("mask", 1500)):
                m = z[key][i] & np.isfinite(z["depth"][i])
                v, u = np.nonzero(m)
                if len(u) < 30:
                    pts = None
                else:
                    k = rng.choice(len(u), min(len(u), cap), replace=False)
                    zz = z["depth"][i][v[k], u[k]]
                    pts = torch.tensor(np.stack([(u[k] - dk["cx"]) * zz / dk["f"], (v[k] - dk["cy"]) * zz / dk["f"], zz], 1), dtype=torch.float64, device=DEV)
                if key == "grip":
                    gp = pts
                else:
                    ap_ = pts
            frames.append(dict(ds=n, T=T, f=dk["f"], cx=dk["cx"], cy=dk["cy"], gp=gp, ap=ap_,
                               state=torch.tensor(z["state"][i], dtype=torch.float64, device=DEV), rgb=z["rgb"][i] if i % 25 == 0 else None))
    G = [fr for fr in frames if fr["gp"] is not None]
    print(f"{len(frames)} frames, {len(G)} with gripper points", flush=True)
    S = torch.stack([fr["state"] for fr in G])
    s = torch.tensor(v1["joint_scale"], dtype=torch.float64, device=DEV)
    o = torch.tensor(v1["joint_offset"], dtype=torch.float64, device=DEV)
    Ts = torch.stack([fr["T"] for fr in G])

    def gripper_loss(s_, o_, links, per_frame=False):
        ang = S * s_ + o_
        P, _ = kin.points(ang, links)                                     # (B, N, 3) base
        Pc = torch.einsum("bij,bnj->bni", Ts[:, :3, :3], P) + Ts[:, None, :3, 3]
        # trimmed mean of the closest 60% per frame: the thin gripper's mask holds depth pixels of the table behind it
        # (edge bleeding / the depth camera seeing past the fingers), 10-30 cm off, which a plain mean follows
        def trimmed(d):
            k = max(1, int(0.6 * d.numel()))
            return torch.topk(d, k, largest=False).values.mean()
        ls = torch.stack([trimmed(torch.cdist(fr["gp"], Pc[b]).min(1).values) for b, fr in enumerate(G)])
        return ls if per_frame else ls.mean()

    with torch.no_grad():
        base = gripper_loss(s, o, GRIPPER_LINKS + CAM_LINKS).item()
        print(f"v1 gripper loss {base:.5f}", flush=True)
        # stage 1: wrist roll (jaw at its v1 setting)
        best = None
        for sr in (0.7, 0.85, 1.0, 1.15, 1.3):
            for k in range(24):
                s_, o_ = s.clone(), o.clone()
                s_[4], o_[4] = sr * DEG, -math.pi + k * 2 * math.pi / 24
                l = gripper_loss(s_, o_, ("gripper_link",) + CAM_LINKS).item()
                if best is None or l < best[0]:
                    best = (l, sr, o_[4].item())
        s[4], o[4] = best[1] * DEG, best[2]
        print(f"stage 1 wrist_roll: scale {best[1]:.2f} deg, offset {math.degrees(best[2]):.0f} deg, loss {best[0]:.5f}", flush=True)
        # stage 2: gripper jaw
        best = None
        for sg in (0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.7):
            for k in range(26):
                s_, o_ = s.clone(), o.clone()
                s_[5], o_[5] = sg * DEG, -0.8 + k * 0.1
                l = gripper_loss(s_, o_, GRIPPER_LINKS + CAM_LINKS).item()
                if best is None or l < best[0]:
                    best = (l, sg, o_[5].item())
        s[5], o[5] = best[1] * DEG, best[2]
        print(f"stage 2 gripper: scale {best[1]:.2f} deg, offset {best[2]:.2f} rad, loss {best[0]:.5f}", flush=True)
    # stage 3: local refinement of the four numbers
    p = torch.tensor([s[4].item(), o[4].item(), s[5].item(), o[5].item()], dtype=torch.float64, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([p], lr=0.002)
    for it in range(150):
        s_, o_ = s.clone(), o.clone()
        s_[4], o_[4], s_[5], o_[5] = p[0], p[1], p[2], p[3]
        l = gripper_loss(s_, o_, GRIPPER_LINKS + CAM_LINKS)
        opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad():
        s[4], o[4], s[5], o[5] = p[0], p[1], p[2], p[3]
        after = gripper_loss(s, o, GRIPPER_LINKS + CAM_LINKS, per_frame=True)
        print(f"stage 3: wrist_roll scale {s[4].item() / DEG:.3f} deg offset {math.degrees(o[4].item()):.1f} deg; gripper scale "
              f"{s[5].item() / DEG:.3f} deg offset {o[5].item():.3f} rad; loss {after.mean().item():.5f} (v1 {base:.5f})", flush=True)
        # residuals in mm: gripper points to the gripper model, arm points to the whole model
        def med_res(fr, pts, links):
            ang = fr["state"][None] * s + o
            P, _ = kin.points(ang, links)
            Pc = P[0] @ fr["T"][:3, :3].T + fr["T"][:3, 3]
            return torch.cdist(pts, Pc).min(1).values.median().item() * 1000
        gr = [med_res(fr, fr["gp"], GRIPPER_LINKS + CAM_LINKS) for fr in G]
        ar = [med_res(fr, fr["ap"], ARM_LINKS + CAM_LINKS) for fr in frames if fr["ap"] is not None]
        print(f"gripper residual median {np.median(gr):.1f} mm (p90 {np.percentile(gr, 90):.1f}); arm residual median "
              f"{np.median(ar):.1f} mm (p90 {np.percentile(ar, 90):.1f})", flush=True)
        # jaw opening over the recorded range: distance between the jaw tips (farthest moving-jaw point from the fixed jaw)
        g = S[:, 5]
        lo, hi = torch.quantile(g, 0.02).item(), torch.quantile(g, 0.98).item()
        print(f"recorded gripper value 2-98%: {lo:.1f}..{hi:.1f} -> jaw angle {math.degrees(s[5].item() * lo + o[5].item()):.0f}.."
              f"{math.degrees(s[5].item() * hi + o[5].item()):.0f} deg (URDF limits -10..100 deg)", flush=True)
        # overlays: model gripper (yellow) and rest (magenta) on the top image
        for fr in frames:
            if fr["rgb"] is None:
                continue
            ang = fr["state"][None] * s + o
            im = fr["rgb"].copy()
            for links, col in ((ARM_LINKS + CAM_LINKS, (255, 0, 255)), (GRIPPER_LINKS + CAM_LINKS, (255, 255, 0))):
                P, _ = kin.points(ang, links)
                Pc = (P[0] @ fr["T"][:3, :3].T + fr["T"][:3, 3]).cpu().numpy()
                u = (fr["f"] * Pc[:, 0] / Pc[:, 2] + fr["cx"]).round().astype(int)
                v = (fr["f"] * Pc[:, 1] / Pc[:, 2] + fr["cy"]).round().astype(int)
                ok = (u >= 0) & (u < im.shape[1]) & (v >= 0) & (v < im.shape[0])
                im[v[ok], u[ok]] = col
            cv2.imwrite(f"{args.viz_dir}/{fr['ds'][:30]}_{id(fr) % 100000}.jpg", im[..., ::-1])
    out = dict(v1)
    out["joint_scale"], out["joint_offset"] = s.cpu().tolist(), o.cpu().tolist()
    out["model_links_extra"] = list(CAM_LINKS)
    out["fit"] = dict(gripper_residual_mm=float(np.median(gr)), arm_residual_mm=float(np.median(ar)), stage="v3")
    json.dump(out, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    sys.exit(main())
