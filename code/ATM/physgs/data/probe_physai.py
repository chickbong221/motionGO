"""
Sanity checks for one downloaded PhysicalAI run (camera conventions, label association, depth scale).
    python -m physgs.data.probe_physai <run_dir> [camera]
"""
import json
import subprocess
import sys
from collections import Counter

import numpy as np
from PIL import Image

W, H = 1920, 1080


def decode(path, pix_fmt, channels, dtype):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"],
                         capture_output=True, check=True).stdout
    shape = (-1, H, W) if channels == 1 else (-1, H, W, channels)
    return np.frombuffer(out, dtype).reshape(shape)


def main(run_dir, cam="Corner"):
    com = np.load(f"{run_dir}/{cam}_com.npz")
    C = json.load(open(f"{run_dir}/{cam}.json"))
    raw = {tuple(o["color"][:3]): o["prim_path"].split("/")[2] for o in json.load(open(f"{run_dir}/{cam}_raw.json"))["objects"]}
    cols = [tuple(c[:3]) for c in com["segmentation_colors"]]
    D = decode(f"{run_dir}/{cam}.mkv", "gray16le", 1, np.uint16)
    hits = {i: Counter() for i in range(len(cols))}
    pairs = []
    for t in range(40, 150, 10):
        seg = np.array(Image.open(f"{run_dir}/{cam}_{t:04d}.png"))[..., :3]
        fl = C["camera"]["focal_length"][t]
        for i, c in enumerate(cols):
            x, y, z = com["data"][i, t]
            if z >= 0:
                continue
            u, v = int(W / 2 + fl[0] * W * x / (-z)), int(H / 2 - fl[0] * W * y / (-z))
            if 0 <= u < W and 0 <= v < H:
                body = raw.get(tuple(seg[v, u]), "?")
                hits[i][body] += 1
                if body == raw[c]:
                    pairs.append((-z, float(D[t, v, u])))
    ok = 0
    for i, c in enumerate(cols):
        if not hits[i]:
            continue
        top, n = hits[i].most_common(1)[0]
        same = top == raw[c]
        ok += same
        print("row %2d own=%-30s under_point=%-30s %d/%d %s" % (i, raw[c][:30], top[:30], n, sum(hits[i].values()), "OK" if same else ""))
    print("rows whose projected CoM lands on their own body:", ok)
    if pairs:
        p = np.array(pairs)
        print("depth raw / camera z:", (p[:, 1] / p[:, 0])[:12].round(1))
        print("camera z:", p[:12, 0].round(2), " raw:", p[:12, 1].astype(int))
    print("depth frames:", D.shape, "raw range", D.min(), D.max())


if __name__ == "__main__":
    main(*sys.argv[1:])


def fit_ground(run_dir, cam="Corner", t=0):
    """Fit fx, fy and depth scale so that ground pixels unproject onto the world plane z = 0."""
    C = json.load(open(f"{run_dir}/{cam}.json"))
    raw = {tuple(o["color"][:3]): o["prim_path"] for o in json.load(open(f"{run_dir}/{cam}_raw.json"))["objects"]}
    gcol = [c for c, p in raw.items() if p.endswith("/Ground")][0]
    seg = np.array(Image.open(f"{run_dir}/{cam}_{t:04d}.png"))[..., :3]
    D = decode(f"{run_dir}/{cam}.mkv", "gray16le", 1, np.uint16)[t].astype(np.float64)
    M = np.array(C["usd_transform"][t]).reshape(4, 4).T  # camera (USD/OpenGL) -> world
    R, o = M[:3, :3], M[:3, 3]
    v, u = np.nonzero((seg == gcol).all(-1))
    sel = np.random.default_rng(0).choice(len(u), 4000, replace=False)
    u, v, d = u[sel], v[sel], D[v[sel], u[sel]]
    fl = C["camera"]["focal_length"][t]
    print("camera height above z=0:", o[2].round(3), " fl:", fl)
    best = []
    for fx in np.linspace(1500, 4000, 51):
        for fy in np.linspace(1500, 4000, 51):
            dirs = np.stack([(u - W / 2) / fx, -(v - H / 2) / fy, -np.ones_like(u, float)], -1) @ R.T  # z-depth = 1
            lam = -o[2] / dirs[:, 2]                     # camera z-depth where the ray meets z = 0
            ray = lam * np.linalg.norm(dirs, axis=1)     # euclidean distance along the ray
            for kind, gt in (("zdepth", lam), ("raydist", ray)):
                s = np.median(d / gt)
                err = np.median(np.abs(d / s - gt))
                best.append((err, kind, fx, fy, s))
    best.sort(key=lambda x: x[0])
    for b in best[:5]:
        print("err %.4f m  %s  fx %.0f fy %.0f  raw-per-metre %.1f" % b)
    print("fl*W = %.0f, fl*H = %.0f" % (fl[0] * W, fl[1] * H))


def association_score(run_dir, cam="Corner", order_from="com"):
    """Fraction of rows whose projected CoM lands on its own body, taking the row colours from `order_from`."""
    com = np.load(f"{run_dir}/{cam}_com.npz")
    src = np.load(f"{run_dir}/{cam}_{order_from}.npz") if order_from != "static" else None
    if order_from == "static":
        cols = [tuple(o["segmentation_color"][:3]) for o in json.load(open(f"{run_dir}/{cam}_static.json"))["objects"]]
    else:
        cols = [tuple(c[:3]) for c in src["segmentation_colors"]]
    C = json.load(open(f"{run_dir}/{cam}.json"))
    raw = {tuple(o["color"][:3]): o["prim_path"].split("/")[2] for o in json.load(open(f"{run_dir}/{cam}_raw.json"))["objects"]}
    good = tot = 0
    for t in range(40, 150, 10):
        seg = np.array(Image.open(f"{run_dir}/{cam}_{t:04d}.png"))[..., :3]
        f = C["camera"]["focal_length"][t][0] * W
        for i, c in enumerate(cols[:com["data"].shape[0]]):
            x, y, z = com["data"][i, t]
            if z >= 0 or raw.get(c, "") == "Ground":
                continue
            u, v = int(W / 2 + f * x / (-z)), int(H / 2 - f * y / (-z))
            if 0 <= u < W and 0 <= v < H:
                tot += 1
                good += raw.get(tuple(seg[v, u]), "?") == raw.get(c, "!")
    print(f"{order_from:10s}: {good}/{tot} projected CoMs land on their own body")


def centroid_check(run_dir, cam="Corner", t=100):
    """World-frame centroid of each body's depth points vs the com label of the same colour."""
    C = json.load(open(f"{run_dir}/{cam}.json"))
    raw = {tuple(o["color"][:3]): o["prim_path"].split("/")[2] for o in json.load(open(f"{run_dir}/{cam}_raw.json"))["objects"]}
    com = np.load(f"{run_dir}/{cam}_com.npz")
    seg = np.array(Image.open(f"{run_dir}/{cam}_{t:04d}.png"))[..., :3]
    D = decode(f"{run_dir}/{cam}.mkv", "gray16le", 1, np.uint16)[t].astype(np.float64)
    M = np.array(C["usd_transform"][t]).reshape(4, 4).T
    f = C["camera"]["focal_length"][t][0] * W
    # depth scale from the ground plane
    gcol = [c for c, p in raw.items() if p == "Ground"][0]
    v, u = np.nonzero((seg == gcol).all(-1))
    dirs = np.stack([(u - W / 2) / f, -(v - H / 2) / f, -np.ones_like(u, float)], -1) @ M[:3, :3].T
    lam = -M[2, 3] / dirs[:, 2]
    s = np.median(D[v, u] / (lam * np.linalg.norm(dirs, axis=1)))
    print("depth raw per metre:", round(s, 1))
    for i, c in enumerate(com["segmentation_colors"]):
        c = tuple(c[:3])
        m = (seg == c).all(-1)
        if m.sum() < 300 or raw[c] == "Ground":
            continue
        v, u = np.nonzero(m)
        r = D[v, u] / s
        d = np.stack([(u - W / 2) / f, -(v - H / 2) / f, -np.ones_like(u, float)], -1)
        p_cam = d / np.linalg.norm(d, axis=1, keepdims=True) * r[:, None]
        p_w = p_cam @ M[:3, :3].T + M[:3, 3]
        lab = M[:3, :3] @ com["data"][i, t] + M[:3, 3]
        print("%-28s px %6d  depth-centroid %s  com %s  dist %.2f" % (raw[c][:28], m.sum(), p_w.mean(0).round(2), lab.round(2),
                                                                    np.linalg.norm(p_w.mean(0) - lab)))


def body_centroids(run_dir, cam, t, raw_full):
    """World centroid per prim root from calibrated depth + PNG colours."""
    C = json.load(open(f"{run_dir}/{cam}.json"))
    seg = np.array(Image.open(f"{run_dir}/{cam}_{t:04d}.png"))[..., :3]
    D = decode(f"{run_dir}/{cam}.mkv", "gray16le", 1, np.uint16)[t].astype(np.float64)
    M = np.array(C["usd_transform"][t]).reshape(4, 4).T
    f = C["camera"]["focal_length"][t][0] * W
    gcol = [c for c, p in raw_full.items() if p.endswith("/Ground")][0]
    v, u = np.nonzero((seg == gcol).all(-1))
    dirs = np.stack([(u - W / 2) / f, -(v - H / 2) / f, -np.ones_like(u, float)], -1) @ M[:3, :3].T
    s = np.median(D[v, u] / (-M[2, 3] / dirs[:, 2] * np.linalg.norm(dirs, axis=1)))
    out = {}
    for c, p in raw_full.items():
        m = (seg == c).all(-1)
        if m.sum() < 300 or p.endswith("/Ground"):
            continue
        v, u = np.nonzero(m)
        d = np.stack([(u - W / 2) / f, -(v - H / 2) / f, -np.ones_like(u, float)], -1)
        pw = (d / np.linalg.norm(d, axis=1, keepdims=True) * (D[v, u] / s)[:, None]) @ M[:3, :3].T + M[:3, 3]
        out.setdefault(p.split("/")[2], []).append(pw)
    return {k: np.concatenate(v).mean(0) for k, v in out.items()}, M


def order_test(run_dir, cam="Corner", t=100):
    raw_list = json.load(open(f"{run_dir}/{cam}_raw.json"))["objects"]
    raw_full = {tuple(o["color"][:3]): o["prim_path"] for o in raw_list}
    com = np.load(f"{run_dir}/{cam}_com.npz")
    cents, M = body_centroids(run_dir, cam, t, raw_full)
    rows_w = com["data"][:, t] @ M[:3, :3].T + M[:3, 3]
    orders = {
        "npz colours": [raw_full[tuple(c[:3])] for c in com["segmentation_colors"]],
        "raw json order": [o["prim_path"] for o in raw_list],
        "sorted prim paths": sorted(o["prim_path"] for o in raw_list),
        "sorted colours": [raw_full[c] for c in sorted(raw_full)],
    }
    for name, prims in orders.items():
        d = [np.linalg.norm(rows_w[i] - cents[p.split("/")[2]]) for i, p in enumerate(prims[:len(rows_w)])
             if p.split("/")[2] in cents]
        print("%-18s median dist %.2f m  (<0.3 m: %d/%d)" % (name, np.median(d), sum(x < 0.3 for x in d), len(d)))
    # nearest body for each row (geometric association)
    names = list(cents)
    P = np.stack([cents[n] for n in names])
    for i in range(min(8, len(rows_w))):
        j = np.linalg.norm(P - rows_w[i], axis=1).argmin()
        print("row %d (%s) nearest body %s at %.2f m" % (i, raw_full[tuple(com["segmentation_colors"][i][:3])].split("/")[2][:24],
                                                        names[j][:24], np.linalg.norm(P[j] - rows_w[i])))


def size_mass_check(run_dir, cam="Corner", t=100):
    """Does the visible size of each PNG body agree with (mass / density)^(1/3) from the static json?"""
    raw_full = {tuple(o["color"][:3]): o["prim_path"] for o in json.load(open(f"{run_dir}/{cam}_raw.json"))["objects"]}
    S = json.load(open(f"{run_dir}/{cam}_static.json"))["objects"]
    num = lambda x: isinstance(x, (int, float)) and x > 0
    vol = {o["prim_path"].split("/")[2]: (o["properties"]["mass"] / o["properties"]["density"]) ** (1 / 3)
           for o in S if num(o["properties"].get("mass")) and num(o["properties"].get("density"))}
    C = json.load(open(f"{run_dir}/{cam}.json"))
    seg = np.array(Image.open(f"{run_dir}/{cam}_{t:04d}.png"))[..., :3]
    D = decode(f"{run_dir}/{cam}.mkv", "gray16le", 1, np.uint16)[t].astype(np.float64)
    f = C["camera"]["focal_length"][t][0] * W
    xs, ys = [], []
    for c, p in raw_full.items():
        k = p.split("/")[2]
        m = (seg == c).all(-1)
        if k not in vol or m.sum() < 300:
            continue
        dist = np.median(D[m]) / 3800.0  # rough metres; only the ranking matters
        xs.append(vol[k]); ys.append(np.sqrt(m.sum()) * dist / f)
    print("bodies:", len(xs), " spearman(size_from_mass, visible_size) = %.2f" % np.corrcoef(np.argsort(np.argsort(xs)), np.argsort(np.argsort(ys)))[0, 1])
