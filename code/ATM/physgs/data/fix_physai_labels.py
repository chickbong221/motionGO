"""
Corrected copy of a packed PhysicalAI dataset (large arrays are symlinked, labels are rewritten):

1. Time alignment. The physics labels lead the rendered frames: the Gaussian centroid at stored frame t matches
   the CoM label at t - 0.75 frames best (physgs.data.diag_align). CoM, velocity and angular-velocity labels share
   one time base (velocity == central difference of CoM), so all three are shifted by the same amount with linear
   interpolation inside each episode. The first frame of each episode has no earlier label and becomes NaN.
2. Large bodies (median extent of the visible Gaussians > --max_extent m: walls, crates, chairs, ramps) keep their
   tokens as scene context but lose their state labels: their CoM cannot be recovered from a partial view, and they
   dominated the position error. NaN labels are already excluded by every loss and by the evaluation.
3. The kinematic paddle of ball_mixer is flagged as the effector (is_robot), like the gripper in LIBERO.

    python -m physgs.data.fix_physai_labels --src data/packed/physai --dst data/packed/physai_v2
"""
import argparse
import json
import os

import numpy as np

from physgs.data.format import MIN_POINTS

LINKED = ("gauss.npy", "npts.npy", "contact.npy", "props.npy", "present.npy", "task_emb.npy", "center.npy", "ep_start.npy",
          "extra.jsonl")
OPTIONAL = ("extra.jsonl", "rot.npy")  # only in packs made with captions / orientation labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--shift", type=float, default=-0.75, help="label shift in stored frames (negative = earlier)")
    ap.add_argument("--max_extent", type=float, default=0.8)
    ap.add_argument("--extent_frames", type=int, default=12, help="visible frames sampled per object for the extent")
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    for f in LINKED:
        dst = os.path.join(a.dst, f)
        if not os.path.exists(dst) and (f not in OPTIONAL or os.path.exists(os.path.join(a.src, f))):
            os.symlink(os.path.abspath(os.path.join(a.src, f)), dst)
    meta = json.load(open(os.path.join(a.src, "meta.json")))
    ep_start = np.load(os.path.join(a.src, "ep_start.npy"))
    npts = np.load(os.path.join(a.src, "npts.npy"), mmap_mode="r")
    gauss = np.load(os.path.join(a.src, "gauss.npy"), mmap_mode="r")
    is_robot = np.load(os.path.join(a.src, "is_robot.npy")).copy()
    rng = np.random.default_rng(0)

    assert a.shift <= 0 and a.shift > -1, "implemented for shifts in (-1, 0]"
    w_prev = -a.shift                      # weight of label[t-1]
    stats = dict(large=0, objects=0, paddles=0)
    LABELS = ("pos", "vel", "angvel") + (("rot",) if os.path.exists(os.path.join(a.src, "rot.npy")) else ())
    for name in LABELS:
        src = np.load(os.path.join(a.src, f"{name}.npy"), mmap_mode="r")
        out = np.lib.format.open_memmap(os.path.join(a.dst, f"{name}.npy"), mode="w+", dtype=np.float32, shape=src.shape)
        for e in range(len(ep_start) - 1):
            s, t = ep_start[e], ep_start[e + 1]
            x = np.asarray(src[s:t], np.float32)
            y = np.full_like(x, np.nan)
            prev = x[:-1]
            if name == "rot":  # quaternions: blend on the same hemisphere, then renormalise
                prev = prev * np.where((prev * x[1:]).sum(-1, keepdims=True) < 0, -1.0, 1.0).astype(np.float32)
            y[1:] = w_prev * prev + (1 - w_prev) * x[1:]
            if name == "rot":
                y[1:] /= np.linalg.norm(y[1:], axis=-1, keepdims=True)
            out[s:t] = y
        out.flush()
        del out

    pos = np.load(os.path.join(a.dst, "pos.npy"), mmap_mode="r+")
    vel = np.load(os.path.join(a.dst, "vel.npy"), mmap_mode="r+")
    ang = np.load(os.path.join(a.dst, "angvel.npy"), mmap_mode="r+")
    rot = np.load(os.path.join(a.dst, "rot.npy"), mmap_mode="r+") if "rot" in LABELS else None
    for e in range(len(ep_start) - 1):
        s, t = ep_start[e], ep_start[e + 1]
        names = meta["episodes"][e]["obj_names"]
        for o, n in enumerate(names):
            stats["objects"] += 1
            if n.lower().startswith("paddle"):
                is_robot[e, o] = True
                stats["paddles"] += 1
            vis = np.nonzero(np.asarray(npts[s:t, o]) >= MIN_POINTS)[0]
            if len(vis) == 0:
                continue
            fr = rng.choice(vis, min(a.extent_frames, len(vis)), replace=False) + s
            g = np.asarray(gauss[np.sort(fr), o, :, :3], np.float32)
            ext = np.median(np.linalg.norm(g.max(1) - g.min(1), axis=-1))
            if ext > a.max_extent:
                pos[s:t, o] = np.nan
                vel[s:t, o] = np.nan
                ang[s:t, o] = np.nan
                if rot is not None:
                    rot[s:t, o] = np.nan
                stats["large"] += 1
                meta["episodes"][e].setdefault("unlabelled_large", []).append(n)
    pos.flush(); vel.flush(); ang.flush()
    if rot is not None:
        rot.flush()
    np.save(os.path.join(a.dst, "is_robot.npy"), is_robot)
    meta["label_fix"] = dict(shift_frames=a.shift, max_extent=a.max_extent, **stats)
    json.dump(meta, open(os.path.join(a.dst, "meta.json"), "w"))
    print(json.dumps(meta["label_fix"]))


if __name__ == "__main__":
    main()
