"""
Shared on-disk formats.

Episode file (one .npz per PhysicalAI run or LIBERO demo), written by the processors:
    gauss     f16 (T, O, P, 7)   per-object Gaussians (f32 if lifted with --gauss_dtype float32; pack keeps the dtype) [xyz world (m), rgb [0,1], scale (m)]; gravity is -z
    npts      i32 (T, O)         observed pixels per object (0 = not visible / padding)
    pos       f32 (T, O, 3)      centre-of-mass position, world frame (NaN = unknown)
    vel       f32 (T, O, 3)      linear velocity (m/s)
    angvel    f32 (T, O, 3)      angular velocity (rad/s)
    rot       f32 (T, O, 4)      PhysicalAI only: orientation quaternion (w, x, y, z), world frame (NaN = unknown)
    cam_K     f32 (C, 3, 3)      PhysicalAI only: intrinsics of the lifted cameras (at the processing size)
    cam_c2w   f32 (C, 4, 4)      PhysicalAI only: OpenCV camera -> world of the lifted cameras (fixed cameras)
    contact   f32 (T, O)         in contact with another body (0/1, NaN = unknown)
    props     f32 (O, 3)         [log mass (kg), static friction, restitution] (NaN = unknown)
    is_robot  bool (O,)          robot / effector token
    obj_names str (O,)
    body_info str (O,)           PhysicalAI only: JSON list of the body's records in <cam>_static.json (all properties)
    caption   str ()             PhysicalAI only: JSON {caption file: text} of the reference camera
    task_emb  f32 (768,)         BERT task embedding (zeros for PhysicalAI until computed from the caption)
    fps       f32 ()             frame rate of the stored sequence
    source    str ()             "physai" or "libero"
    actions   f32 (T, 7)         LIBERO only (kept so the same file serves the policy)

Packed dataset (a directory, written by pack.py) concatenates episodes, pads objects to O_max,
and stores arrays as .npy so training can memory-map them.
"""
import json
import os

import numpy as np

PROP_NAMES = ("log_mass", "friction", "restitution")
STATE_KEYS = ("pos", "vel", "angvel")
MIN_POINTS = 16


def save_episode(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_episode(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def pack(episode_paths, out_dir, max_objects, sort_by_visibility=True, num_points=None):
    """
    Concatenate episodes into memory-mappable arrays. Robot tokens always come first. With
    sort_by_visibility, objects are ordered by how often they are visible and the least visible beyond
    max_objects are dropped (PhysicalAI); otherwise the native order is kept (LIBERO), which an online
    rollout can reproduce.
    """
    os.makedirs(out_dir, exist_ok=True)
    lengths = []
    for p in episode_paths:  # first pass reads only the small arrays
        with np.load(p) as f:
            lengths.append(f["npts"].shape[0])
    with np.load(episode_paths[0]) as f:
        P = min(num_points or f["gauss"].shape[2], f["gauss"].shape[2])  # stored points are in FPS order: a prefix is an even subsample
        g_dtype = f["gauss"].dtype  # packed as stored (LIBERO float32, PhysicalAI float16)
    T_total = sum(lengths)
    O = max_objects
    n_eps = len(episode_paths)

    def mm(name, shape, dtype, fill=0):
        a = np.lib.format.open_memmap(os.path.join(out_dir, f"{name}.npy"), mode="w+", dtype=dtype, shape=shape)
        a[:] = fill
        return a

    gauss = mm("gauss", (T_total, O, P, 7), g_dtype)
    npts = mm("npts", (T_total, O), np.int32)
    pos = mm("pos", (T_total, O, 3), np.float32, np.nan)
    vel = mm("vel", (T_total, O, 3), np.float32, np.nan)
    angvel = mm("angvel", (T_total, O, 3), np.float32, np.nan)
    contact = mm("contact", (T_total, O), np.float32, np.nan)
    with np.load(episode_paths[0]) as f:
        rot_dim = f["rot"].shape[-1] if "rot" in f.files else 0
    rot = mm("rot", (T_total, O, rot_dim), np.float32, np.nan) if rot_dim else None
    props = np.full((n_eps, O, 3), np.nan, np.float32)
    is_robot = np.zeros((n_eps, O), bool)
    present = np.zeros((n_eps, O), bool)
    task_emb = np.zeros((n_eps, 768), np.float32)
    center = np.zeros((n_eps, 3), np.float32)
    ep_start = np.zeros(n_eps + 1, np.int64)
    meta, extra = [], []

    t0 = 0
    for i, path in enumerate(episode_paths):
        e = load_episode(path)
        T, O_e = e["npts"].shape
        if sort_by_visibility:
            order = np.argsort(-(e["npts"] > 0).sum(0) - 1e-6 * e["npts"].sum(0), kind="stable")
        else:
            order = np.arange(O_e)
        order = np.concatenate([order[e["is_robot"][order]], order[~e["is_robot"][order]]])
        keep = order[:O]
        k = len(keep)
        sl = slice(t0, t0 + T)
        gauss[sl, :k] = e["gauss"][:, keep, :P]
        npts[sl, :k] = e["npts"][:, keep]
        for name, arr in (("pos", pos), ("vel", vel), ("angvel", angvel)):
            arr[sl, :k] = e[name][:, keep]
        contact[sl, :k] = e["contact"][:, keep]
        if rot is not None and "rot" in e:
            rot[sl, :k] = e["rot"][:, keep]
        props[i, :k] = e["props"][keep]
        is_robot[i, :k] = e["is_robot"][keep]
        present[i, :k] = True
        task_emb[i] = e["task_emb"]
        # Scene centre: mean centroid of the objects visible in the first frame (or the episode's settle frame, after
        # objects that topple / drop at the start came to rest: physgs.data.libero_process).
        c0 = int(e["settle_frame"]) if "settle_frame" in e else 0
        vis = e["npts"][c0, keep] >= MIN_POINTS
        cen = e["gauss"][c0, keep, :P, :3].astype(np.float32).mean(1)
        center[i] = cen[vis].mean(0) if vis.any() else 0.0
        ep_start[i + 1] = t0 + T
        meta.append(dict(file=os.path.abspath(episode_paths[i]), source=str(e["source"]), fps=float(e["fps"]),
                         obj_names=[str(n) for n in e["obj_names"][keep]]))
        if "body_info" in e or "caption" in e:  # large text: its own file, so meta.json stays quick to load
            extra.append(dict(body_info=[str(s) for s in e["body_info"][keep]] if "body_info" in e else [],
                              caption=str(e["caption"]) if "caption" in e else ""))
        t0 += T

    for name, arr in (("props", props), ("is_robot", is_robot), ("present", present), ("task_emb", task_emb),
                      ("center", center), ("ep_start", ep_start)):
        np.save(os.path.join(out_dir, f"{name}.npy"), arr)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(dict(episodes=meta, max_objects=O, num_points=P), f)
    if extra:
        assert len(extra) == n_eps, "body_info / caption must be in every episode or none"
        with open(os.path.join(out_dir, "extra.jsonl"), "w") as f:  # line i = episode i
            f.writelines(json.dumps(x) + "\n" for x in extra)
    for a in (gauss, npts, pos, vel, angvel, contact) + ((rot,) if rot is not None else ()):
        a.flush()
    return out_dir


class Packed:
    """Read-only, memory-mapped view of a packed dataset directory."""

    def __init__(self, root):
        self.root = root
        load = lambda n, mmap=True: np.load(os.path.join(root, f"{n}.npy"), mmap_mode="r" if mmap else None)
        for n in ("gauss", "npts", "pos", "vel", "angvel", "contact"):
            setattr(self, n, load(n))
        for n in ("props", "is_robot", "present", "task_emb", "center", "ep_start"):
            setattr(self, n, load(n, mmap=False))
        # optional object-major copy of gauss (physgs.data.make_object_major) for contiguous per-object reads
        self.gauss_obj = load("gauss_obj") if os.path.exists(os.path.join(root, "gauss_obj.npy")) else None
        self.track_start = load("track_start", mmap=False) if self.gauss_obj is not None else None
        with open(os.path.join(root, "meta.json")) as f:
            self.meta = json.load(f)
        self.num_episodes = len(self.ep_start) - 1
        self.frame_episode = np.repeat(np.arange(self.num_episodes), np.diff(self.ep_start))

    def episode_slice(self, i):
        return slice(self.ep_start[i], self.ep_start[i + 1])
