"""
Lift calibrated multi-view RGB-D + instance segmentation into per-object 3D Gaussian sets.

No learning: every masked pixel becomes an isotropic Gaussian at its unprojected 3D point,
coloured by the pixel and sized by its footprint (depth / focal length). All views are fused
and each object is resampled to a fixed number of Gaussians with farthest-point sampling.

Conventions: OpenCV cameras (x right, y down, z forward), K in pixels of the given image,
cam2world maps camera to world. Gaussian layout: [x, y, z, r, g, b, scale] (metres, [0,1], metres).
"""
import numpy as np
import torch

from physgs.model.layers import farthest_point_sample, gather

GAUSS_DIM = 7


def unproject(depth, K, cam2world):
    """depth: (V, H, W), K: (V, 3, 3), cam2world: (V, 4, 4) -> world points (V, H, W, 3)."""
    V, H, W = depth.shape
    v, u = torch.meshgrid(torch.arange(H, device=depth.device, dtype=depth.dtype),
                          torch.arange(W, device=depth.device, dtype=depth.dtype), indexing="ij")
    fx, fy = K[:, 0, 0, None, None], K[:, 1, 1, None, None]
    cx, cy = K[:, 0, 2, None, None], K[:, 1, 2, None, None]
    p = torch.stack([(u - cx) * depth / fx, (v - cy) * depth / fy, depth], dim=-1)
    return torch.einsum("vij,vhwj->vhwi", cam2world[:, :3, :3], p) + cam2world[:, None, None, :3, 3]


def resample(points, n, generator=None):
    """points: (N, D) -> (n, D) by FPS on xyz (random pre-subsample for large N, replacement for small N)."""
    N = points.shape[0]
    if N >= n:
        if N > 8 * n:
            points = points[torch.randperm(N, device=points.device, generator=generator)[:8 * n]]
        idx = farthest_point_sample(points[None, :, :3], n)[0]
        return points[idx]
    extra = torch.randint(0, N, (n - N,), device=points.device, generator=generator)
    return torch.cat([points, points[extra]], dim=0)


def resample_many(point_sets, n, generators):
    """
    Same result as [resample(p, n, g) for p, g in zip(point_sets, generators)] (same random draws in the same
    order), but every set that needs FPS goes through one batched FPS loop. Sets are padded with copies of their
    first point: FPS starts at that point, so the copies are at distance 0 and are never picked.
    """
    out, fps = [None] * len(point_sets), []
    for i, (p, g) in enumerate(zip(point_sets, generators)):
        N = p.shape[0]
        if N >= n:
            if N > 8 * n:
                p = p[torch.randperm(N, device=p.device, generator=g)[:8 * n]]
            fps.append((i, p))
        else:
            extra = torch.randint(0, N, (n - N,), device=p.device, generator=g)
            out[i] = torch.cat([p, p[extra]], dim=0)
    if fps:
        M = max(p.shape[0] for _, p in fps)
        pad = torch.stack([torch.cat([p, p[:1].expand(M - p.shape[0], -1)], dim=0) for _, p in fps])
        idx = farthest_point_sample(pad[..., :3], n)
        for k, (i, _) in enumerate(fps):
            out[i] = pad[k, idx[k]]
    return out


def _object_points(rgb, depth, seg, K, cam2world, obj_ids, max_depth):
    """Fused masked pixels of each object as Gaussians (N_i, 7), and their counts."""
    pts = unproject(depth, K, cam2world)                                   # (V, H, W, 3)
    col = rgb.float() / 255.0
    scale = (depth / K[:, 0, 0, None, None])[..., None]                   # pixel footprint
    valid = (depth > 0) & (depth < max_depth) & torch.isfinite(depth)
    feats = torch.cat([pts, col, scale], dim=-1)[valid]                    # (N, 7)
    ids = seg[valid]
    sels = [feats[ids == oid] for oid in obj_ids]
    return sels, torch.tensor([s.shape[0] for s in sels], dtype=torch.long)


@torch.no_grad()
def lift_objects_batch(scenes, num_points=512, min_points=16, max_depth=50.0):
    """
    lift_objects for several scenes at once: scenes = [(rgb, depth, seg, K, cam2world, obj_ids, generator), ...].
    Results equal calling lift_objects on each scene in turn; all objects of all scenes share one FPS loop.
    Returns a list of (gauss, npts) per scene.
    """
    sets, gens, owner, results = [], [], [], []
    for s, (rgb, depth, seg, K, c2w, obj_ids, gen) in enumerate(scenes):
        sels, npts = _object_points(rgb, depth, seg, K, c2w, obj_ids, max_depth)
        results.append((torch.zeros(len(obj_ids), num_points, GAUSS_DIM, device=depth.device), npts))
        for i, sel in enumerate(sels):
            if sel.shape[0] >= min_points:
                sets.append(sel); gens.append(gen); owner.append((s, i))
    for (s, i), g in zip(owner, resample_many(sets, num_points, gens)):
        results[s][0][i] = g
    return results


@torch.no_grad()
def lift_objects(rgb, depth, seg, K, cam2world, obj_ids, num_points=512, min_points=16, max_depth=50.0,
                 generator=None):
    """
    rgb: (V, H, W, 3) uint8, depth: (V, H, W) metres, seg: (V, H, W) int ids, K / cam2world per view.
    obj_ids: list of instance ids to extract (one entry per output object; the same id may not repeat).
    generator: torch.Generator for the random point subsampling (evaluation passes one seeded per episode so an
    episode is reproducible however it is run); None = the global RNG.
    Returns gauss (O, num_points, 7) float32 and npts (O,) int (number of observed pixels before resampling).
    """
    return lift_objects_batch([(rgb, depth, seg, K, cam2world, obj_ids, generator)], num_points, min_points,
                              max_depth)[0]


def centroid(gauss, npts, min_points=16):
    """Mean xyz of each object's Gaussians; NaN for objects that were not seen."""
    c = gauss[..., :3].mean(dim=-2)
    c[npts < min_points] = float("nan")
    return c


def to_numpy_f16(x):
    return x.detach().cpu().numpy().astype(np.float16)
