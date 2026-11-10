"""
Object encoder / decoder over per-object Gaussian sets.

Input: one object's Gaussians at t-1 and t (from physgs.data.lift), centred on the object's
observed centroid at t. Output token:
    state   : CoM offset from the centroid (3), linear velocity (3), angular velocity (3)
    contact : logit (1)
    belief  : mean and log-std of [log mass, static friction, restitution]
    z       : shape / appearance latent
A decoder maps z back to a Gaussian set (object frame), trained with Chamfer + colour + scale.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from physgs.model.layers import CrossAttnBlock, SelfAttnBlock, farthest_point_sample, fourier_dim, fourier_embed, gather, mlp

STATE_DIM = 9
PROP_DIM = 3
POS_SCALE = 4.0  # object-frame coordinates are multiplied by this before the Fourier embedding


def object_frame(g_prev, g_cur):
    """Centre both frames on the current centroid. g: (B, P, 7). Returns rel tensors and the centroid (B, 3)."""
    c = g_cur[..., :3].mean(dim=1)
    rel_prev = torch.cat([g_prev[..., :3] - c[:, None], g_prev[..., 3:]], dim=-1)
    rel_cur = torch.cat([g_cur[..., :3] - c[:, None], g_cur[..., 3:]], dim=-1)
    return rel_prev, rel_cur, c


def point_features(rel, frame_flag):
    xyz = rel[..., :3] * POS_SCALE
    log_scale = torch.log(rel[..., 6:7].clamp(min=1e-5))
    flag = torch.full_like(log_scale, frame_flag)
    return torch.cat([fourier_embed(xyz), rel[..., 3:6], log_scale, flag], dim=-1)


class ObjectEncoder(nn.Module):
    """
    Input: the object's Gaussians over a window of `num_frames` frames ending at t, centred on the centroid at t.
    The current frame keeps all points; older frames keep their first `hist_points` (the stored Gaussians are in
    farthest-point order, so this is an even subsample). Each frame gets a learned time embedding; with
    num_frames = 2 the module is exactly the original two-frame encoder.
    """

    def __init__(self, dim=384, depth=6, heads=8, num_queries=64, latent_dim=128, num_frames=2, hist_points=512,
                 dropout=0.0):
        super().__init__()
        self.num_queries = num_queries
        self.num_frames, self.hist_points = num_frames, hist_points
        self.in_proj = nn.Linear(fourier_dim() + 3 + 1 + 1, dim)
        if num_frames > 2:
            self.time_embed = nn.Parameter(torch.zeros(num_frames, dim))
        self.cross = CrossAttnBlock(dim, heads, dropout=dropout)
        self.blocks = nn.ModuleList([SelfAttnBlock(dim, heads, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.state_head = mlp(dim, dim, STATE_DIM + 1)
        self.prop_head = mlp(dim, dim, 2 * PROP_DIM)
        self.z_head = mlp(dim, dim, latent_dim)

    def forward(self, g_hist):
        """g_hist: (B, K, P, 7) world-frame Gaussians of the same objects, oldest first, last = current frame."""
        K = g_hist.shape[1]
        assert K == self.num_frames, f"encoder expects {self.num_frames} frames, got {K}"
        c = g_hist[:, -1, :, :3].mean(dim=1)
        feats = []
        for k in range(K):
            g = g_hist[:, k] if k == K - 1 else g_hist[:, k, :self.hist_points]
            rel = torch.cat([g[..., :3] - c[:, None], g[..., 3:]], dim=-1)
            x = self.in_proj(point_features(rel, 1.0 if k == K - 1 else -1.0))
            if K > 2:
                x = x + self.time_embed[k]
            feats.append(x)
        rel_cur, x_cur = rel, feats[-1]
        x = torch.cat(feats, dim=1)
        idx = farthest_point_sample(rel_cur[..., :3].detach(), self.num_queries)
        q = self.cross(gather(x_cur, idx), x)
        for blk in self.blocks:
            q = blk(q)
        h = self.norm(q.mean(dim=1))
        state = self.state_head(h)
        prop = self.prop_head(h)
        return dict(
            centroid=c,
            offset=state[:, :3] / POS_SCALE,
            vel=state[:, 3:6],
            angvel=state[:, 6:9],
            contact_logit=state[:, 9],
            prop_mu=prop[:, :PROP_DIM],
            prop_logstd=prop[:, PROP_DIM:].clamp(-5, 3),
            z=self.z_head(h),
        )


class ObjectDecoder(nn.Module):
    """z -> Gaussian set in the object frame (relative xyz, rgb, scale)."""

    def __init__(self, latent_dim=128, dim=384, depth=4, heads=8, num_tokens=64, points_per_token=8):
        super().__init__()
        self.num_tokens, self.ppt = num_tokens, points_per_token
        self.queries = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        self.z_proj = nn.Linear(latent_dim, dim)
        self.cross = CrossAttnBlock(dim, heads)
        self.blocks = nn.ModuleList([SelfAttnBlock(dim, heads) for _ in range(depth)])
        self.out = nn.Linear(dim, points_per_token * 7)

    def forward(self, z):
        B = z.shape[0]
        q = self.queries[None].expand(B, -1, -1)
        q = self.cross(q, self.z_proj(z)[:, None])
        for blk in self.blocks:
            q = blk(q)
        out = self.out(q).view(B, self.num_tokens * self.ppt, 7)
        xyz = out[..., :3] / POS_SCALE
        rgb = torch.sigmoid(out[..., 3:6])
        scale = torch.exp(out[..., 6:7].clamp(-12, 1)) * 0.01
        return torch.cat([xyz, rgb, scale], dim=-1)


def gaussian_nll(x, mu, logstd):
    return (logstd + 0.5 * ((x - mu) / logstd.exp()) ** 2 + 0.5 * math.log(2 * math.pi))


def masked_mean(x, mask):
    mask = mask.to(x.dtype)
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(x)
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


class ObjectAutoencoder(nn.Module):
    def __init__(self, dim=384, enc_depth=6, dec_depth=4, heads=8, latent_dim=128, w_color=0.5, w_scale=0.1,
                 w_offset=10.0, w_vel=1.0, w_angvel=0.1, w_contact=0.2, w_prop=0.1, w_z=1e-4,
                 num_frames=2, hist_points=512, dropout=0.0):
        super().__init__()
        self.num_frames = num_frames
        self.encoder = ObjectEncoder(dim, enc_depth, heads, latent_dim=latent_dim, num_frames=num_frames,
                                     hist_points=hist_points, dropout=dropout)
        self.decoder = ObjectDecoder(latent_dim, dim, dec_depth, heads)
        self.w = dict(color=w_color, scale=w_scale, offset=w_offset, vel=w_vel, angvel=w_angvel,
                      contact=w_contact, prop=w_prop, z=w_z)

    def recon_loss(self, pred, target):
        """Chamfer on xyz (object frame, scaled) plus nearest-neighbour colour and log-scale."""
        d = torch.cdist(pred[..., :3] * POS_SCALE, target[..., :3] * POS_SCALE)
        i_pt = d.argmin(dim=2)  # nearest target for each predicted point
        i_tp = d.argmin(dim=1)  # nearest prediction for each target point
        cham = d.min(dim=2).values.mean() + d.min(dim=1).values.mean()
        color = (pred[..., 3:6] - gather(target[..., 3:6], i_pt)).abs().mean() + \
                (target[..., 3:6] - gather(pred[..., 3:6], i_tp)).abs().mean()
        ls = lambda g: torch.log(g[..., 6:7].clamp(min=1e-5))
        scale = (ls(target) - gather(ls(pred), i_tp)).abs().mean()
        return cham, color, scale

    def forward(self, batch):
        """
        batch: g_hist (B, K, P, 7) oldest first (or g_prev, g_cur (B, P, 7)); labels pos (B, 3) [world CoM], vel, angvel (B, 3), contact (B,),
        props (B, 3); NaN marks unknown labels.
        """
        g_hist = batch["g_hist"] if "g_hist" in batch else torch.stack([batch["g_prev"], batch["g_cur"]], 1)
        out = self.encoder(g_hist)
        rel_cur = g_hist[:, -1].clone()
        rel_cur[..., :3] = rel_cur[..., :3] - out["centroid"][:, None]
        recon = self.decoder(out["z"])
        cham, color, scale = self.recon_loss(recon, rel_cur)
        losses = dict(chamfer=cham, color=color, scale=scale, z=(out["z"] ** 2).mean())

        offset_tgt = batch["pos"] - out["centroid"]
        for name, pred, tgt in (("offset", out["offset"], offset_tgt), ("vel", out["vel"], batch["vel"]),
                                ("angvel", out["angvel"], batch["angvel"])):
            m = torch.isfinite(tgt).all(-1)
            losses[name] = masked_mean(F.smooth_l1_loss(pred, torch.nan_to_num(tgt), reduction="none", beta=0.05), m)
        m = torch.isfinite(batch["contact"])
        losses["contact"] = masked_mean(F.binary_cross_entropy_with_logits(
            out["contact_logit"], torch.nan_to_num(batch["contact"]), reduction="none"), m)
        m = torch.isfinite(batch["props"])
        losses["prop"] = masked_mean(gaussian_nll(torch.nan_to_num(batch["props"]), out["prop_mu"], out["prop_logstd"]), m)

        total = cham + sum(self.w[k] * v for k, v in losses.items() if k in self.w)
        return total, {k: float(v) for k, v in losses.items()}, out

    @torch.no_grad()
    def tokens(self, *gs):
        """
        Token for the dynamics model: [pos(3), vel(3), angvel(3), contact(1), prop_mu(3), prop_logstd(3), z].
        Call as tokens(g_hist) with g_hist (B, K, P, 7), or tokens(g_prev, g_cur) for a two-frame encoder.
        """
        g_hist = gs[0] if len(gs) == 1 else torch.stack(gs, 1)
        o = self.encoder(g_hist)
        pos = o["centroid"] + o["offset"]
        return torch.cat([pos, o["vel"], o["angvel"], torch.sigmoid(o["contact_logit"])[:, None],
                          o["prop_mu"], o["prop_logstd"], o["z"]], dim=-1)


TOKEN_SLICES = dict(pos=slice(0, 3), vel=slice(3, 6), angvel=slice(6, 9), contact=slice(9, 10),
                    prop_mu=slice(10, 13), prop_logstd=slice(13, 16), z=slice(16, None))


if __name__ == "__main__":
    # python -m physgs.model.object_encoder
    B, P = 8, 512
    for K in (2, 8):
        m = ObjectAutoencoder(num_frames=K, hist_points=128 if K > 2 else 512)
        g = torch.rand(B, K, P, 7)
        batch = dict(g_hist=g, pos=torch.rand(B, 3), vel=torch.randn(B, 3), angvel=torch.randn(B, 3),
                     contact=torch.tensor([0., 1, float("nan"), 0, 1, 0, 1, 0]), props=torch.randn(B, 3))
        batch["props"][0] = float("nan")
        loss, logs, _ = m(batch)
        loss.backward()
        print(K, "frames", {k: round(v, 4) for k, v in logs.items()})
        print("token", tuple(m.tokens(g).shape), "params (M):", sum(p.numel() for p in m.parameters()) / 1e6)
