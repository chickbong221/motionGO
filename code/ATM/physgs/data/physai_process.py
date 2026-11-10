"""
Convert one downloaded PhysicalAI run into an episode file (physgs.data.format).

Conventions established with physgs/data/probe_physai.py on objects_falling runs:
  * intrinsics: square pixels, fx = fy = focal_length[0] * W, principal point at the image centre;
  * extrinsics: `usd_transform` (row-major, row-vector convention) transposed is camera-to-world
    with an OpenGL camera (x right, y up, looking along -z); the world is z-up;
  * depth MKV (gray16le) stores the Euclidean ray distance, with a scale that differs per camera
    and run, so it is calibrated by fitting the ground plane z = 0;
  * physics npz files are per camera, in that camera's frame (com = CoM position, velocity in m/s,
    spin in deg/s). Their rows are NOT reliably in the order of their `segmentation_colors`, so rows
    are attached to bodies geometrically (Hungarian matching of row CoMs to lifted body centroids);
  * the PNG colours map to prim paths through `<cam>_raw.json`; a body is the prim root
    (/World/<name>), and its mass / friction / restitution come from `<cam>_static.json`.
"""
import glob
import json
import os
import re
import subprocess

import numba
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from physgs.data.format import MIN_POINTS, save_episode

W0, H0 = 1920, 1080
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
GROUND_RE = re.compile(r"ground|floor|plane|terrain", re.I)  # static support surfaces (not bodies)


def prim_root(path):
    parts = path.split("/")
    return parts[2] if len(parts) > 2 else path


def decode_video(path, pix_fmt, size, stride, channels, dtype, neighbor=False):
    w, h = size
    flags = ":flags=neighbor" if neighbor else ":flags=area"
    vf = f"select=not(mod(n\\,{stride})),scale={w}:{h}{flags}"
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-vsync", "0", "-f", "rawvideo",
                          "-pix_fmt", pix_fmt, "-"], capture_output=True, check=True).stdout
    shape = (-1, h, w) if channels == 1 else (-1, h, w, channels)
    return np.frombuffer(out, dtype).reshape(shape)


def ray_norm(K, w, h):
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    return np.sqrt(((u - K[0, 2]) / K[0, 0]) ** 2 + ((v - K[1, 2]) / K[1, 1]) ** 2 + 1.0)


def fit_depth_scale(raw_depth, ground_mask, K, c2w_cv):
    """Raw units per metre of ray distance, from pixels that see the plane z = 0. Returns (scale, residual_m)."""
    v, u = np.nonzero(ground_mask)
    if len(u) < 200:
        return None, None
    if len(u) > 5000:
        sel = np.random.default_rng(0).choice(len(u), 5000, replace=False)
        u, v = u[sel], v[sel]
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)  # OpenCV, z = 1
    # Note: from one camera the depth scale and the ground height are not separately identifiable (scaling all
    # rays maps z = 0 to another horizontal plane), so the ground is assumed at z = 0. Runs where that does
    # not hold give a large residual and are rejected by the caller.
    dw = d @ c2w_cv[:3, :3].T
    lam = -c2w_cv[2, 3] / dw[:, 2]
    ok = lam > 0
    if ok.sum() < 200:
        return None, None
    ray = lam[ok] * np.linalg.norm(d[ok], axis=1)
    raw = raw_depth[v[ok], u[ok]].astype(np.float64)
    scale = np.median(raw / ray)
    return scale, float(np.median(np.abs(raw / scale - ray)))


def seg_lookup(color_to_body, body_id):
    """Table over all 2^24 colours -> body index (-2 = ground, -1 = unknown colour), for one-gather mask decoding."""
    lut = np.full(1 << 24, -1, np.int16)
    for col, body in color_to_body.items():
        lut[(col[0] << 16) | (col[1] << 8) | col[2]] = -2 if GROUND_RE.search(body) else body_id[body]
    return lut


def decode_seg(im, lut):
    """RGB mask (H, W, 3) -> per-pixel value from the table."""
    return lut[(im[..., 0].astype(np.int32) << 16) | (im[..., 1].astype(np.int32) << 8) | im[..., 2]]


@numba.njit(parallel=True)
def _fps_sets(points, starts, n):
    """Farthest-point sampling of each set points[starts[s]:starts[s + 1]] (float32 xyz): local indices (S, n).
    Same picks as physgs.model.layers.farthest_point_sample (start at 0, d = dx*dx + dy*dy + dz*dz in float32,
    first maximum), without its per-iteration tensor passes."""
    S = len(starts) - 1
    idx = np.zeros((S, n), np.int64)
    for s in numba.prange(S):
        a, N = starts[s], starts[s + 1] - starts[s]
        dist = np.full(N, np.inf, np.float32)
        far = 0
        for i in range(n):
            idx[s, i] = far
            px, py, pz = points[a + far, 0], points[a + far, 1], points[a + far, 2]
            best, bi = np.float32(-1.0), 0
            for j in range(N):
                dx, dy, dz = points[a + j, 0] - px, points[a + j, 1] - py, points[a + j, 2] - pz
                d = dx * dx + dy * dy + dz * dz
                if d < dist[j]:
                    dist[j] = d
                if dist[j] > best:
                    best, bi = dist[j], j
            far = bi
    return idx


def resample_fast(point_sets, n):
    """lift.resample_many(point_sets, n, [None] * len) with the same random draws in the same order (global torch
    RNG), but the farthest-point sampling runs in _fps_sets."""
    out, fps = [None] * len(point_sets), []
    for i, p in enumerate(point_sets):
        N = p.shape[0]
        if N >= n:
            if N > 8 * n:
                p = p[torch.randperm(N)[:8 * n]]
            fps.append((i, p))
        else:
            extra = torch.randint(0, N, (n - N,))
            out[i] = torch.cat([p, p[extra]], dim=0)
    if fps:
        pts = np.ascontiguousarray(np.concatenate([p[:, :3].numpy() for _, p in fps]), dtype=np.float32)
        starts = np.cumsum([0] + [p.shape[0] for _, p in fps]).astype(np.int64)
        idx = _fps_sets(pts, starts, n)
        for k, (i, p) in enumerate(fps):
            out[i] = p[torch.from_numpy(idx[k])]
    return out


def lift_frame(rgb, depth, seg, K, c2w, num_bodies, num_points, min_points=16, max_depth=50.0):
    """
    The Gaussians of lift.lift_objects for one frame (same pixels in the same order, same resampling), but only
    object pixels are unprojected and they are grouped by one stable sort instead of masking the whole image once
    per body (at 1920x1080 x 4 views that masking dominated the run time).
    rgb (V, H, W, 3) uint8, depth (V, H, W) metres, seg (V, H, W) body index (-1 = none), K (V, 3, 3),
    c2w (V, 4, 4) OpenCV camera -> world. Returns gauss (B, num_points, 7) float32 and npts (B,) int.
    """
    feats, ids = [], []
    for v in range(depth.shape[0]):
        d = depth[v]
        m = (seg[v] >= 0) & (d > 0) & (d < max_depth) & np.isfinite(d)
        y, x = np.nonzero(m)
        z = d[m]
        fx, fy, cx, cy = K[v, 0, 0], K[v, 1, 1], K[v, 0, 2], K[v, 1, 2]
        p = np.stack([(x - cx) * z / fx, (y - cy) * z / fy, z], -1).astype(np.float32)
        p = p @ c2w[v, :3, :3].T.astype(np.float32) + c2w[v, :3, 3].astype(np.float32)
        feats.append(np.concatenate([p, rgb[v][m].astype(np.float32) / 255.0, (z / fx)[:, None].astype(np.float32)], 1))
        ids.append(seg[v][m])
    feats, ids = np.concatenate(feats), np.concatenate(ids).astype(np.int64)
    order = np.argsort(ids, kind="stable")
    counts = np.bincount(ids, minlength=num_bodies)[:num_bodies]
    groups = np.split(feats[order], np.cumsum(counts)[:-1])
    gauss = torch.zeros(num_bodies, num_points, 7)
    owners = [b for b in range(num_bodies) if counts[b] >= min_points]
    if owners:
        for b, g in zip(owners, resample_fast([torch.from_numpy(groups[b]) for b in owners], num_points)):
            gauss[b] = g
    return gauss, torch.from_numpy(counts)


def read_text(path):
    return open(path).read() if os.path.exists(path) else ""


def process_run(run_dir, out_path, size=(320, 180), stride=2, num_points=512, max_bodies=32, min_cams=2, device="cpu",
                caption_dir=None):
    run = os.path.basename(run_dir.rstrip("/"))
    cams = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{run_dir}/*.mkv"))
    cams = [c for c in cams if os.path.exists(f"{run_dir}/{c}.mp4") and os.path.exists(f"{run_dir}/{c}.json")]
    if not cams:
        return dict(run=run, ok=False, reason="no complete camera")
    w, h = size
    sx, sy = w / W0, h / H0

    # Segmentation colours are assigned per camera, so each camera gets its own colour -> body map.
    color_maps = {c: {tuple(o["color"][:3]): prim_root(o["prim_path"]) for o in json.load(open(f"{run_dir}/{c}_raw.json"))["objects"]}
                  for c in cams if os.path.exists(f"{run_dir}/{c}_raw.json")}
    cams = [c for c in cams if c in color_maps]
    if not cams:
        return dict(run=run, ok=False, reason="no segmentation metadata")
    static = json.load(open(f"{run_dir}/{cams[0]}_static.json"))
    props_by_body, info_by_body = {}, {}
    for o in static["objects"]:
        p = o["properties"]
        num = lambda x: float(x) if isinstance(x, (int, float)) else np.nan
        props_by_body[prim_root(o["prim_path"])] = [np.log(max(num(p.get("mass")), 1e-6)) if np.isfinite(num(p.get("mass"))) else np.nan,
                                                    num(p.get("static_friction")), num(p.get("restitution"))]
        # every record of the body (all its prims, all properties), kept verbatim for later use
        info_by_body.setdefault(prim_root(o["prim_path"]), []).append({k: v for k, v in o.items() if k != "color"})
    bodies = sorted({b for m in color_maps.values() for b in m.values() if not GROUND_RE.search(b)})
    body_id = {b: i for i, b in enumerate(bodies)}

    # Per camera only the decoded RGB and the raw 16-bit depth are kept; metric depth and the segmentation map are
    # made frame by frame just before lifting (same values as computing them for the whole video first, a third of the
    # memory: ~2.6 GB per 75-frame run of 4 cameras at 1920x1080 instead of ~7).
    def seg_frame(cam, k):
        png = f"{run_dir}/{cam['name']}_{cam['frames'][k]:04d}.png"
        if not os.path.exists(png):
            return None
        im = Image.open(png).convert("RGB")
        if im.size != (w, h):
            im = im.resize((w, h), Image.NEAREST)
        return decode_seg(np.asarray(im), cam["lut"])

    views, used, qa = [], [], {}
    T = None
    for cam in cams:
        color_to_body = color_maps[cam]
        C = json.load(open(f"{run_dir}/{cam}.json"))
        rgb = decode_video(f"{run_dir}/{cam}.mp4", "rgb24", size, stride, 3, np.uint8)
        dep = decode_video(f"{run_dir}/{cam}.mkv", "gray16le", size, stride, 1, np.uint16, neighbor=True)
        expected = len(range(0, C["frame_count"], stride))
        qa[f"{cam}_frames"] = [len(rgb), len(dep)]
        if min(len(rgb), len(dep)) < 0.95 * expected:
            continue  # some depth videos in the dataset are truncated to a few frames; drop that camera
        n = min(len(rgb), len(dep))
        frames = list(range(0, C["frame_count"], stride))[:n]
        view = dict(name=cam, frames=frames, lut=seg_lookup(color_to_body, body_id))
        v0 = seg_frame(view, 0)
        gmask0 = v0 == -2 if v0 is not None and (v0 == -2).any() else None
        f = C["camera"]["focal_length"][0][0] * W0
        K = np.array([[f * sx, 0, w / 2], [0, f * sy, h / 2], [0, 0, 1.0]])
        c2w = np.stack([np.array(M).reshape(4, 4).T @ GL_TO_CV for M in C["usd_transform"]])[frames]
        scale, resid = fit_depth_scale(dep[0], gmask0 if gmask0 is not None else np.zeros((h, w), bool), K, c2w[0])
        qa[f"{cam}_depth_residual_m"] = resid
        if scale is None or resid > 0.05:
            continue  # cannot calibrate this camera
        view.update(rgb=rgb[:n], dep=dep[:n], scale=scale, ray=ray_norm(K, w, h).astype(np.float32), K=K, c2w=c2w)
        views.append(view); used.append(cam)
        T = n if T is None else min(T, n)
    if len(views) < min_cams:
        return dict(run=run, ok=False, reason=f"only {len(views)} complete calibrated cameras", **qa)
    Ks = [v["K"] for v in views]
    c2ws = [v["c2w"] for v in views]

    def seg_or_empty(v, t):
        s = seg_frame(v, t)
        return np.full((h, w), -1, np.int16) if s is None else np.maximum(s, -1)

    # ---- lift every frame into per-body Gaussians
    gauss = np.zeros((T, len(bodies), num_points, 7), np.float16)
    npts = np.zeros((T, len(bodies)), np.int32)
    Kn = np.stack(Ks)
    if device != "cpu":
        # GPU: lift.lift_objects_batch on `chunk` frames at once (one FPS loop for all bodies of those frames). Same
        # pixels, order, counts and FPS as lift_frame; only the random pre-subsample of large bodies draws differently.
        from physgs.data.lift import lift_objects_batch
        Kt = torch.tensor(Kn, dtype=torch.float32, device=device)
        ids, chunk = list(range(len(bodies))), 8
        for t0 in range(0, T, chunk):
            scenes = []
            for t in range(t0, min(T, t0 + chunk)):
                scenes.append((torch.from_numpy(np.stack([v["rgb"][t] for v in views])).to(device),
                               torch.from_numpy(np.stack([v["dep"][t].astype(np.float32) / v["scale"] / v["ray"]
                                                          for v in views])).to(device),
                               torch.from_numpy(np.stack([seg_or_empty(v, t) for v in views]).astype(np.int64)).to(device),
                               Kt, torch.tensor(np.stack([c[t] for c in c2ws]), dtype=torch.float32, device=device),
                               ids, None))
            for k, (g, nn_) in enumerate(lift_objects_batch(scenes, num_points)):
                gauss[t0 + k] = g.cpu().numpy().astype(np.float16)
                npts[t0 + k] = nn_.numpy()
    else:
        for t in range(T):
            g, nn_ = lift_frame(np.stack([v["rgb"][t] for v in views]),
                                np.stack([v["dep"][t].astype(np.float32) / v["scale"] / v["ray"] for v in views]),
                                np.stack([seg_or_empty(v, t) for v in views]), Kn, np.stack([c[t] for c in c2ws]),
                                len(bodies), num_points)
            gauss[t] = g.numpy().astype(np.float16)
            npts[t] = nn_.numpy()

    # ---- physics labels: rows of the reference camera, attached to bodies geometrically
    ref = cams[0]
    C = json.load(open(f"{run_dir}/{ref}.json"))
    frames = list(range(0, C["frame_count"], stride))[:T]
    R_all = np.stack([np.array(M).reshape(4, 4).T for M in C["usd_transform"]])[frames]  # GL camera -> world
    com = np.load(f"{run_dir}/{ref}_com.npz")["data"][:, frames]
    vel = np.load(f"{run_dir}/{ref}_velocity.npz")["data"][:, frames]
    spin = np.load(f"{run_dir}/{ref}_spin.npz")["data"][:, frames]
    pos_w = np.einsum("tij,rtj->rti", R_all[:, :3, :3], com) + R_all[None, :, :3, 3]
    vel_w = np.einsum("tij,rtj->rti", R_all[:, :3, :3], vel)
    # The spin rows are NOT in the camera frame like com / velocity: world = R^T spin (not R spin). Checked on
    # objects_falling_c8917ca2_0: with R^T the 4 cameras agree exactly (median difference 0) and the spin matches
    # the rate from successive orientations (corr 0.97); with R they disagree by up to 93 rad/s (corr 0.60).
    spin_w = np.einsum("tji,rtj->rti", R_all[:, :3, :3], np.deg2rad(spin))
    # orientation: quaternions (w, x, y, z) in the GL camera frame; world = R q (identical across the 4 cameras)
    rot_w = None
    if os.path.exists(f"{run_dir}/{ref}_rot.npz"):
        q = np.load(f"{run_dir}/{ref}_rot.npz")["data"][:, frames].astype(np.float64)
        if q.shape[:2] == com.shape[:2] and q.shape[-1] == 4:
            Rq = Rotation.from_quat(q[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(q.shape[:2] + (3, 3))
            xyzw = Rotation.from_matrix(np.einsum("tij,rtjk->rtik", R_all[:, :3, :3], Rq).reshape(-1, 3, 3)).as_quat()
            rot_w = xyzw[:, [3, 0, 1, 2]].reshape(q.shape).astype(np.float32)

    vis = npts >= MIN_POINTS
    cen = gauss[..., :3].astype(np.float32).mean(2)                                   # (T, B, 3)
    radius = np.linalg.norm(gauss[..., :3].astype(np.float32) - cen[:, :, None], axis=-1).max(-1)
    cost = np.full((pos_w.shape[0], len(bodies)), 1e3)
    for b in range(len(bodies)):
        tv = np.nonzero(vis[:, b])[0]
        if len(tv):
            cost[:, b] = np.median(np.linalg.norm(pos_w[:, tv] - cen[None, tv, b], axis=-1), axis=1)
    rows, cols = linear_sum_assignment(cost)
    pos = np.full((T, len(bodies), 3), np.nan, np.float32)
    v_ = np.full_like(pos, np.nan)
    w_ = np.full_like(pos, np.nan)
    r_ = np.full((T, len(bodies), 4), np.nan, np.float32)
    matched = 0
    for r, b in zip(rows, cols):
        rb = np.nanmedian(np.where(vis[:, b], radius[:, b], np.nan)) if vis[:, b].any() else 0
        if cost[r, b] < max(0.15, 0.75 * rb):
            pos[:, b], v_[:, b], w_[:, b] = pos_w[r], vel_w[r], spin_w[r]
            if rot_w is not None:
                r_[:, b] = rot_w[r]
            matched += 1
    props = np.array([props_by_body.get(b, [np.nan] * 3) for b in bodies], np.float32)
    # captions of the reference camera ({"<cam>": base caption, "<cam>_physics": ..., ...}), raw JSON text
    caps = sorted(glob.glob(os.path.join(caption_dir, run, f"{ref}*.json"))) if caption_dir else []
    caption = json.dumps({os.path.basename(f)[:-5]: read_text(f) for f in caps})

    # keep the most visible bodies
    keep = np.argsort(-(vis.sum(0)))[:max_bodies]
    keep = keep[vis[:, keep].any(0)]
    save_episode(out_path, gauss=gauss[:, keep], npts=npts[:, keep], pos=pos[:, keep], vel=v_[:, keep],
                 angvel=w_[:, keep], rot=r_[:, keep], contact=np.full((T, len(keep)), np.nan, np.float32), props=props[keep],
                 body_info=np.array([json.dumps(info_by_body.get(bodies[i], [])) for i in keep]), caption=np.array(caption),
                 is_robot=np.zeros(len(keep), bool), obj_names=np.array([bodies[i] for i in keep]),
                 task_emb=np.zeros(768, np.float32), fps=np.float32(30.0 / stride), source=np.array("physai"),
                 cams=np.array(used), scene=np.array(run.rsplit("_", 2)[0]),  # cameras actually lifted (K, c2w: OpenCV)
                 cam_K=np.stack(Ks).astype(np.float32), cam_c2w=np.stack([c[0] for c in c2ws]).astype(np.float32))
    return dict(run=run, ok=True, T=T, bodies=int(len(keep)), matched=matched, total_bodies=len(bodies),
                cams_used=len(views), **qa)


if __name__ == "__main__":
    import sys
    print(process_run(sys.argv[1], sys.argv[2]))
