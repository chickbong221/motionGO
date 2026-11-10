"""
viz_lift.py: inspect one lifted SO-101 episode. For 5 instants: the top RGB with every object's Gaussians projected
back into it (colour per object; wrist-fused points included), plus top-down (x-y) and side (x-z) scatter of the
Gaussians in the world frame (table = z 0). Prints per-object visibility, size and height statistics.
  python viz_lift.py <episode.npz> <dataset dir> <out.jpg>
"""
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/dev/shm/physgo/code_so101/ATM/physgs_so101")
import lift_so101 as L  # noqa: E402

ep_path, root, out = sys.argv[1:4]
e = dict(np.load(ep_path))
cam = dict(np.load(ep_path.replace(".npz", "_cam.npz")))
names = [str(n) for n in e["obj_names"]]
g, npts = e["gauss"].astype(np.float32), e["npts"]
T, O = npts.shape
ep = int(ep_path.split("_")[-1].split(".")[0])
info, eps, _ = L.load_meta(root)
top = L.decode_episode(root, eps.loc[ep], "observation.images.top", int(eps.loc[ep, "length"]))
K, c2w = cam["K_top"], cam["c2w_top"]
w2c = np.linalg.inv(c2w)
print(f"T {T}, objects {names}; wrist frames used {cam['use_wrist'].mean():.0%}")
for k, n in enumerate(names):
    vis = npts[:, k] >= 16
    if vis.any():
        P = g[vis, k, :, :3]
        c = P.mean(1)
        rad = np.sqrt(((P - c[:, None]) ** 2).sum(-1).mean(1))
        print(f"  {n:12s} visible {vis.mean():.0%}, wrist-fused {cam['wrist_used'][:, k].mean():.0%}, radius median "
              f"{np.median(rad) * 100:.1f} cm, centroid z median {np.median(c[:, 2]) * 100:.1f} cm, z range "
              f"[{np.percentile(P[..., 2], 1) * 100:.1f}, {np.percentile(P[..., 2], 99) * 100:.1f}] cm")
    else:
        print(f"  {n:12s} never visible")
cols = plt.cm.tab10(np.arange(O))[:, :3]
ts = np.linspace(0, T - 1, 5).round().astype(int)
fig, ax = plt.subplots(4, 5, figsize=(25, 17))
gk = names.index("gripper") if "gripper" in names else None
for j, t in enumerate(ts):
    im = top[t].astype(np.float32) / 255
    for k in range(O):
        if npts[t, k] < 16:
            continue
        P = g[t, k, :, :3] @ w2c[:3, :3].T + w2c[:3, 3]
        u = (K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2]).round().astype(int)
        v = (K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]).round().astype(int)
        ok = (u >= 0) & (u < im.shape[1]) & (v >= 0) & (v < im.shape[0])
        im[v[ok], u[ok]] = cols[k]
        ax[1, j].scatter(g[t, k, :, 0], g[t, k, :, 1], s=1, color=cols[k], label=names[k])
        ax[2, j].scatter(g[t, k, :, 0], g[t, k, :, 2], s=1, color=cols[k])
    if gk is not None and npts[t, gk] >= 16:   # zoom: everything within 8 cm of the gripper, side view, true colours
        c = g[t, gk, :, :3].mean(0)
        for k in range(O):
            if npts[t, k] < 16:
                continue
            P = g[t, k]
            near = np.linalg.norm(P[:, :3] - c, axis=1) < 0.08
            ax[3, j].scatter(P[near, 0], P[near, 2], s=6, color=cols[k] if k == gk else P[near, 3:6].clip(0, 1),
                             edgecolors=cols[k], linewidths=0.3)
        ax[3, j].set_aspect("equal"); ax[3, j].set_title("zoom: gripper (blue edge) +-8 cm, x-z, RGB colours")
    ax[0, j].imshow(im); ax[0, j].set_title(f"t={t} (15 fps)"); ax[0, j].axis("off")
    ax[1, j].set_aspect("equal"); ax[1, j].set_title("top-down x-y (m)")
    ax[2, j].set_aspect("equal"); ax[2, j].set_title("side x-z (m)")
ax[1, 0].legend(markerscale=8, fontsize=8)
plt.tight_layout()
plt.savefig(out, dpi=55)
print("saved", out)
