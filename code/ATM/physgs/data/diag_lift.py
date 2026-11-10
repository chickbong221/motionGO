"""
Where does the encoder's position error come from? Compares, for visible objects of the validation split:
  centroid of the lifted Gaussians, encoder token position (centroid + learned offset), and the simulator CoM,
broken down by number of cameras, observed pixels, object extent and object name, plus a time-alignment check
(CoM label at t-1 / t / t+1 against the centroid at t).

    python -m physgs.data.diag_lift --data data/packed/physai [--frames 200000]
"""
import argparse
import json
import re
from collections import defaultdict

import numpy as np

from physgs.data.format import MIN_POINTS, Packed
from physgs.engine.common import split_episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--frames", type=int, default=3000, help="validation frames sampled")
    a = ap.parse_args()
    d = Packed(a.data)
    meta = json.load(open(f"{a.data}/meta.json"))
    eps_meta = meta["episodes"]
    gauss = np.load(f"{a.data}/gauss.npy", mmap_mode="r")
    tokens = np.load(f"{a.data}/tokens.npy", mmap_mode="r")
    _, va = split_episodes(d.num_episodes)
    rng = np.random.default_rng(0)

    # number of cameras per episode (PhysicalAI episode files store the cameras that were kept)
    ncams = {}
    for e in va:
        f = eps_meta[e]["file"]
        try:
            with np.load(f) as z:
                ncams[e] = len(z["cams"]) if "cams" in z.files else -1
        except Exception:
            ncams[e] = -1
    print("cameras per validation episode:", dict(sorted(zip(*np.unique(list(ncams.values()), return_counts=True)))))

    rows = []
    frames = []
    for e in va:
        s, t = d.ep_start[e], d.ep_start[e + 1]
        frames += [(e, f) for f in range(s + 1, t - 1)]
    pick = rng.choice(len(frames), min(a.frames, len(frames)), replace=False)
    for i in pick:
        e, f = frames[i]
        names = eps_meta[e]["obj_names"]
        for o in range(len(names)):
            if d.npts[f, o] < MIN_POINTS or not np.isfinite(d.pos[f, o]).all():
                continue
            g = np.asarray(gauss[f, o, :, :3], np.float32)
            cen = g.mean(0)
            ext = float(np.linalg.norm(g.max(0) - g.min(0)))
            com = d.pos[f, o]
            tok = np.asarray(tokens[f, o, 0:3], np.float32)
            err_c = np.linalg.norm(cen - com)
            err_t = np.linalg.norm(tok - com)
            al = [np.linalg.norm(cen - d.pos[f + k, o]) if np.isfinite(d.pos[f + k, o]).all() else np.nan for k in (-1, 0, 1)]
            speed = np.linalg.norm(d.vel[f, o]) if np.isfinite(d.vel[f, o]).all() else np.nan
            rows.append(dict(ep=e, cams=ncams.get(e, -1), npts=int(d.npts[f, o]), ext=ext, name=names[o],
                             err_c=err_c, err_t=err_t, al_m1=al[0], al_0=al[1], al_p1=al[2], speed=speed))
    print(f"visible object-frames analysed: {len(rows)}")

    def summ(sel, label):
        if not sel:
            return
        c = np.array([r["err_c"] for r in sel]); t = np.array([r["err_t"] for r in sel])
        print(f"  {label:34s} n={len(sel):6d}  centroid err med/mean/p90 {np.median(c):.3f}/{c.mean():.3f}/{np.percentile(c, 90):.3f}"
              f"   token err med/mean/p90 {np.median(t):.3f}/{t.mean():.3f}/{np.percentile(t, 90):.3f}")

    print("\n== overall"); summ(rows, "all")
    print("\n== by cameras in the episode")
    for k in sorted({r["cams"] for r in rows}):
        summ([r for r in rows if r["cams"] == k], f"{k} cameras")
    print("\n== by observed pixels")
    for lo, hi in ((16, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 1e9)):
        summ([r for r in rows if lo <= r["npts"] < hi], f"pixels [{lo}, {hi})")
    print("\n== by object extent (max - min of Gaussians, m)")
    for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 1.0), (1.0, 1e9)):
        summ([r for r in rows if lo <= r["ext"] < hi], f"extent [{lo}, {hi})")
    print("\n== by object type (name prefix)")
    kind = lambda n: re.sub(r"[_\d]+$", "", re.sub(r"^S\d+_", "", n)).split("_")[-1] if n else "?"
    groups = defaultdict(list)
    for r in rows:
        groups[kind(r["name"])].append(r)
    for k, sel in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:14]:
        summ(sel, k)
    print("\n== time alignment: centroid at t vs CoM label at t-1 / t / t+1 (moving objects, > 0.3 m/s)")
    mv = [r for r in rows if np.isfinite(r["speed"]) and r["speed"] > 0.3]
    for k in ("al_m1", "al_0", "al_p1"):
        v = np.array([r[k] for r in mv]); v = v[np.isfinite(v)]
        if len(v):
            print(f"  {k:6s} n={len(v)} median {np.median(v):.3f} mean {v.mean():.3f}")


if __name__ == "__main__":
    main()
