"""
calibrate_kin_v4.py (2026-11-06): wrist roll + gripper-jaw conversion, fitted on the right evidence.
Diagnosis of v2 / v3: SAM 3.1's "gripper of the teal robot arm" mask mostly marks arm segments near the base, not the
gripper (tests: overlays /dev/shm/grip_diag.jpg), while the v1 kinematic gripper sits on the real gripper. So the
gripper evidence here is the teal-arm mask points (reliable) within 6 cm of the v1 model gripper; the arm itself is
placed to 3-4 mm by v1. Everything else stays at v1 (camera poses, focal lengths, joints 0-3).
  stage 1  wrist_roll: scale 0.7-1.3 deg / unit x offset -90..90 deg (model: wrist_link + gripper + camera parts)
  stage 2  jaw: scale 0.5-1.5 deg / unit x offset -0.4..0.6 rad, only settings that keep the recorded range inside the
           URDF jaw limits (-10..100 deg)
  stage 3  bounded local refinement
Loss per frame: trimmed mean (closest 60%) of the evidence points' distance to the posed model. Reports the residual
(median, mm) before / after, the arm residual (must stay ~v1), the jaw angle range, and overlays with the jaw open and
closed.
  python calibrate_kin_v4.py --datasets a,b,c --v1 kin.json --out kin_v4.json
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
from so101_kin import ARM_LINKS, GRIPPER_LINKS, SO101  # noqa: E402

URDF = "/dev/shm/physgo/so101/robot/so101_new_calib_camera.urdf"
DEV = "cuda"
DEG = math.pi / 180
CAM_LINKS = ("wrist_camera_mount_link", "wrist_camera_link")
HAND = ("wrist_link",) + GRIPPER_LINKS + CAM_LINKS
JAW_LIM = (-10 * DEG, 100 * DEG)


def trimmed(d, q=0.6):
    k = max(1, int(q * d.numel()))
    return torch.topk(d, k, largest=False).values.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--v1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius", type=float, default=0.06)
    ap.add_argument("--viz_dir", default="/dev/shm/physgo/so101/kin_viz_v4")
    args = ap.parse_args()
    os.makedirs(args.viz_dir, exist_ok=True)
    v1 = json.load(open(args.v1))
    kin = SO101(URDF, device=DEV, points_per_link=800)
    rng = np.random.default_rng(0)
    s = torch.tensor(v1["joint_scale"], dtype=torch.float64, device=DEV)
    o = torch.tensor(v1["joint_offset"], dtype=torch.float64, device=DEV)
    s0, o0 = s.clone(), o.clone()
    s0[5], o0[5] = DEG, 0.0                 # selection only: jaw at a mid opening
    frames = []
    for n in args.datasets.split(","):
        z = np.load(f"/dev/shm/physgo/so101/kin_cache/{n}.npz")
        dk = v1["datasets"][n]
        T = torch.tensor(dk["T_base_to_cam"], dtype=torch.float64, device=DEV)
        for i in range(len(z["state"])):
            m = z["mask"][i] & np.isfinite(z["depth"][i])
            v, u = np.nonzero(m)
            if len(u) < 100:
                continue
            k = rng.choice(len(u), min(len(u), 4000), replace=False)
            zz = z["depth"][i][v[k], u[k]]
            A = torch.tensor(np.stack([(u[k] - dk["cx"]) * zz / dk["f"], (v[k] - dk["cy"]) * zz / dk["f"], zz], 1),
                             dtype=torch.float64, device=DEV)
            st = torch.tensor(z["state"][i], dtype=torch.float64, device=DEV)
            with torch.no_grad():
                P, _ = kin.points((st * s0 + o0)[None], GRIPPER_LINKS + CAM_LINKS)
                Pc = P[0] @ T[:3, :3].T + T[:3, 3]
                near = torch.cdist(A, Pc).min(1).values < args.radius
            if near.sum() < 60:
                continue
            frames.append(dict(ds=n, T=T, f=dk["f"], cx=dk["cx"], cy=dk["cy"], gp=A[near], ap=A, state=st, rgb=z["rgb"][i]))
    print(f"{len(frames)} frames with >=60 arm points within {args.radius * 100:.0f} cm of the model gripper", flush=True)
    S = torch.stack([fr["state"] for fr in frames])
    Ts = torch.stack([fr["T"] for fr in frames])

    def loss(s_, o_, per_frame=False):
        P, _ = kin.points(S * s_ + o_, HAND)
        Pc = torch.einsum("bij,bnj->bni", Ts[:, :3, :3], P) + Ts[:, None, :3, 3]
        ls = torch.stack([trimmed(torch.cdist(fr["gp"], Pc[b]).min(1).values) for b, fr in enumerate(frames)])
        return ls if per_frame else ls.mean()

    def med_mm(s_, o_, key, links):
        out = []
        for fr in frames:
            P, _ = kin.points((fr["state"] * s_ + o_)[None], links)
            Pc = P[0] @ fr["T"][:3, :3].T + fr["T"][:3, 3]
            out.append(torch.cdist(fr[key], Pc).min(1).values.median().item() * 1000)
        return float(np.median(out)), float(np.percentile(out, 90))

    g = S[:, 5]
    g_lo, g_hi = torch.quantile(g, 0.02).item(), torch.quantile(g, 0.98).item()
    with torch.no_grad():
        before = med_mm(s, o, "gp", HAND)
        print(f"v1: near-gripper residual median {before[0]:.1f} mm (p90 {before[1]:.1f}), loss {loss(s, o).item() * 1000:.2f} mm", flush=True)
        best = None
        for sr in (0.7, 0.85, 1.0, 1.15, 1.3):
            for od in np.arange(-90, 90.1, 7.5):
                s_, o_ = s.clone(), o.clone()
                s_[4], o_[4] = sr * DEG, od * DEG
                s_[5], o_[5] = DEG, 0.0
                l = loss(s_, o_).item()
                if best is None or l < best[0]:
                    best = (l, sr, od)
        s[4], o[4] = best[1] * DEG, best[2] * DEG
        print(f"stage 1 wrist_roll: scale {best[1]:.2f} deg/unit, offset {best[2]:.1f} deg, loss {best[0] * 1000:.2f} mm", flush=True)
        best = None
        for sg in np.arange(0.5, 1.51, 0.1):
            for og in np.arange(-0.4, 0.601, 0.05):
                lo_a, hi_a = sg * DEG * g_lo + og, sg * DEG * g_hi + og
                if lo_a < JAW_LIM[0] or hi_a > JAW_LIM[1]:
                    continue
                s_, o_ = s.clone(), o.clone()
                s_[5], o_[5] = sg * DEG, og
                l = loss(s_, o_).item()
                if best is None or l < best[0]:
                    best = (l, sg, og)
        s[5], o[5] = best[1] * DEG, best[2]
        print(f"stage 2 jaw: scale {best[1]:.2f} deg/unit, offset {best[2]:.2f} rad, loss {best[0] * 1000:.2f} mm", flush=True)
    p = torch.tensor([s[4].item(), o[4].item(), s[5].item(), o[5].item()], dtype=torch.float64, device=DEV, requires_grad=True)
    p0 = p.detach().clone()
    opt = torch.optim.Adam([p], lr=0.001)
    for it in range(120):
        s_, o_ = s.clone(), o.clone()
        s_[4], o_[4], s_[5], o_[5] = p[0], p[1], p[2], p[3]
        l = loss(s_, o_) + 1e-2 * ((p - p0) ** 2).sum()          # stay near the grid optimum
        opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad():
        s[4], o[4], s[5], o[5] = p[0], p[1], p[2], p[3]
        after = med_mm(s, o, "gp", HAND)
        arm = med_mm(s, o, "ap", ARM_LINKS + CAM_LINKS)
        ja = (math.degrees(s[5].item() * g_lo + o[5].item()), math.degrees(s[5].item() * g_hi + o[5].item()))
        print(f"final: wrist_roll {s[4].item() / DEG:.3f} deg/unit + {math.degrees(o[4].item()):.1f} deg; jaw {s[5].item() / DEG:.3f} "
              f"deg/unit + {o[5].item():.3f} rad", flush=True)
        print(f"near-gripper residual median {after[0]:.1f} mm (p90 {after[1]:.1f}) [v1 {before[0]:.1f}]; arm residual median "
              f"{arm[0]:.1f} mm (p90 {arm[1]:.1f}); jaw angle over the recorded range {ja[0]:.0f}..{ja[1]:.0f} deg", flush=True)
        # overlays: the 3 most closed and 3 most open frames, jaw yellow, rest of the hand orange, evidence green
        order = torch.argsort(S[:, 5]).tolist()
        for tag, sel in (("closed", order[:3]), ("open", order[-3:])):
            for j in sel:
                fr = frames[j]
                im = fr["rgb"].copy()
                for links, col, src in ((HAND, (255, 128, 0), None), (("moving_jaw_so101_v1_link",), (255, 255, 0), None), (None, (0, 255, 0), "gp")):
                    if src:
                        Pc = fr[src].cpu().numpy()
                    else:
                        P, _ = kin.points((fr["state"] * s + o)[None], links)
                        Pc = (P[0] @ fr["T"][:3, :3].T + fr["T"][:3, 3]).cpu().numpy()
                    u = (fr["f"] * Pc[:, 0] / Pc[:, 2] + fr["cx"]).round().astype(int)
                    v = (fr["f"] * Pc[:, 1] / Pc[:, 2] + fr["cy"]).round().astype(int)
                    ok = (u >= 0) & (u < im.shape[1]) & (v >= 0) & (v < im.shape[0])
                    im[v[ok], u[ok]] = col if src is None else (0.5 * im[v[ok], u[ok]] + 0.5 * np.array(col)).astype(np.uint8)
                cv2.imwrite(f"{args.viz_dir}/{tag}_{j:03d}_g{fr['state'][5].item():.0f}.jpg", im[..., ::-1])
    out = dict(v1)
    out["joint_scale"], out["joint_offset"] = s.cpu().tolist(), o.cpu().tolist()
    out["fit"] = dict(stage="v4", near_gripper_residual_mm=after[0], near_gripper_residual_mm_v1=before[0],
                      arm_residual_mm=arm[0], jaw_deg_range=ja, evidence="teal-arm mask points within 6 cm of the model gripper")
    json.dump(out, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    sys.exit(main())
