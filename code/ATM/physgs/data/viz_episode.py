"""Quick look at an episode file: top-down Gaussians coloured by RGB with CoM labels, at a few frames.
    python -m physgs.data.viz_episode <episode.npz> <out.png>
"""
import sys

import numpy as np
from PIL import Image, ImageDraw


def main(path, out, frames=(0, 30, 60), size=480):
    e = np.load(path)
    g, npts, pos = e["gauss"].astype(np.float32), e["npts"], e["pos"]
    vis = npts >= 16
    xy = g[..., :2][vis[..., None, None].repeat(g.shape[2], 2).squeeze(-1)] if False else None
    allxy = g[:, :, :, :2][vis].reshape(-1, 2)
    lo, hi = np.percentile(allxy, 1, 0) - 0.2, np.percentile(allxy, 99, 0) + 0.2
    span = (hi - lo).max()
    to_px = lambda p: ((p - lo) / span * (size - 1)).astype(int)
    tiles = []
    for t in frames:
        t = min(t, g.shape[0] - 1)
        img = Image.new("RGB", (size, size), (30, 30, 30))
        d = ImageDraw.Draw(img)
        order = np.argsort(g[t, :, :, 2].mean(1))
        for b in order:
            if not vis[t, b]:
                continue
            pts = g[t, b]
            for (x, y), c in zip(to_px(pts[:, :2]), (pts[:, 3:6] * 255).astype(int)):
                d.point((x, size - 1 - y), fill=tuple(c))
            if np.isfinite(pos[t, b]).all():
                x, y = to_px(pos[t, b, :2])
                d.ellipse((x - 4, size - 1 - y - 4, x + 4, size - 1 - y + 4), outline=(255, 255, 0))
        d.text((5, 5), f"t={t}", fill=(255, 255, 255))
        tiles.append(np.array(img))
    Image.fromarray(np.concatenate(tiles, 1)).save(out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
