"""
gauss_budget.py: are 512 Gaussians per object enough for the SO-101 objects, and can SAM 3.1 separate the gripper?
For ~24 frames of one episode: per object, the observed points (registered sensor depth x SAM 3.1 mask, top view),
their count, and the detail lost by farthest-point subsampling to k = 256 / 512 / 1024 / 2048 points: distance from
every observed point to its nearest kept point (median and 95th percentile, mm). Also tries gripper phrases on the
top view: share of frames detected, mask size, and overlap with the teal arm (a gripper mask should lie inside it).
Saves a zoomed picture of the arm / gripper points at 512 vs 2048.
  python gauss_budget.py <dataset name> <episode> <sam3 ckpt>
"""
import json
import sys

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/dev/shm/physgo/code_so101/ATM/physgs_so101")
import lift_so101 as L  # noqa: E402
from physgs.data.lift import _object_points, resample  # noqa: E402

name, ep, ckpt = sys.argv[1], int(sys.argv[2]), sys.argv[3]
root = f"/dev/shm/data/so101_v2/{name}"
info, eps, _ = L.load_meta(root)
row = eps.loc[ep]
n30 = int(row["length"])
pick = slice(None, None, 6)   # every 6th 15-fps frame
top = L.decode_episode(root, row, "observation.images.top", n30)[pick]
q = L.decode_episode(root, row, "observation.images.top_depth", n30, depth=True)[pick]
calib = json.load(open(f"/dev/shm/physgo/so101/calib/{name}.json"))
Kt = np.array(calib["K_top"], np.float64)
dinfo = info["features"]["observation.images.top_depth"]["info"]
sensor = L.register_depth_gpu(np.stack([L.depth_from_codes(x, dinfo) for x in q]), calib["depth_reg"], Kt, top.shape[1:3])
seg = L.Segmenter(ckpt)
objs = {"left_arm": L.ARM_PROMPTS[0], "right_arm": L.ARM_PROMPTS[1]}
for n, p_top, _ in L.PROMPTS[name]:
    objs[n] = p_top
masks = {n: L.best_instance(seg.track(top, p)) for n, p in objs.items()}
T = len(top)
print(f"{T} frames")
# gripper phrases
for ph in ("robot gripper", "gripper", "teal gripper", "gripper of the teal robot arm", "robot hand"):
    tr = seg.track(top, ph)
    best = L.best_instance(tr)
    if best is None:
        print(f"  gripper phrase '{ph}': never detected"); continue
    det = best.any((1, 2))
    arm = masks["left_arm"]
    inside = [(best[t] & arm[t]).sum() / max(1, best[t].sum()) for t in range(T) if det[t]]
    area = [best[t].sum() for t in range(T) if det[t]]
    print(f"  gripper phrase '{ph}': detected {det.mean():.0%} of frames, area median {np.median(area):.0f} px, "
          f"share inside the teal-arm mask {np.median(inside):.0%}")
    masks[f"grip[{ph}]"] = best
dev = "cuda"
c2w = torch.eye(4, device=dev)[None]
K_t = torch.tensor(Kt, dtype=torch.float32, device=dev)[None]
stats = {}
keep = {}
for n, M in masks.items():
    if M is None:
        continue
    rows = []
    for t in range(T):
        if not M[t].any():
            continue
        segm = torch.from_numpy(M[t].astype(np.int64)).to(dev)[None]
        dep = torch.from_numpy(np.nan_to_num(sensor[t], nan=0.0)).to(dev)[None]
        rgb = torch.from_numpy(top[t]).to(dev)[None]
        sel, _ = _object_points(rgb, dep, segm, K_t, c2w, [1], 5.0)
        P = sel[0]
        if P.shape[0] < 16:
            continue
        r = [P.shape[0]]
        for k in (256, 512, 1024, 2048):
            S = resample(P, k)
            d = torch.cdist(P[:, :3], S[:, :3]).min(1).values * 1000
            r += [d.median().item(), torch.quantile(d, 0.95).item()]
        rows.append(r)
        if n in ("left_arm",) or n.startswith("grip"):
            keep.setdefault(n, (t, P))
    if rows:
        a = np.array(rows)
        stats[n] = a
        print(f"{n:38s} observed points median {np.median(a[:, 0]):6.0f} (max {a[:, 0].max():6.0f}); "
              f"NN to kept points median/95% mm: " + "  ".join(f"k={k}: {np.median(a[:, 1 + 2 * i]):.1f}/{np.median(a[:, 2 + 2 * i]):.1f}"
                                                            for i, k in enumerate((256, 512, 1024, 2048))))
if "left_arm" in keep:
    t, P = keep["left_arm"]
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    for j, k in enumerate((512, 2048, None)):
        S = (resample(P, k) if k else P).cpu().numpy()
        ax[j].scatter(S[:, 0], S[:, 1], s=2 if k else 0.2, c=S[:, 3:6].clip(0, 1))
        ax[j].set_aspect("equal"); ax[j].invert_yaxis(); ax[j].set_title(f"left arm, frame {t}: {k or 'all'} points ({P.shape[0]} observed)")
    plt.tight_layout(); plt.savefig("/dev/shm/gauss_budget_arm.jpg", dpi=60)
    print("saved /dev/shm/gauss_budget_arm.jpg")
