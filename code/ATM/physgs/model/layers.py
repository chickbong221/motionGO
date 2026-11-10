import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def fourier_embed(x, num_freqs=8, max_freq=32.0):
    """NeRF-style positional encoding of 3D points. x: (..., 3) -> (..., 3 + 6 * num_freqs)."""
    freqs = torch.exp(torch.linspace(0.0, math.log(max_freq), num_freqs, device=x.device, dtype=x.dtype))
    xf = x[..., None] * freqs * math.pi  # (..., 3, F)
    return torch.cat([x, torch.sin(xf).flatten(-2), torch.cos(xf).flatten(-2)], dim=-1)


def fourier_dim(num_freqs=8):
    return 3 + 6 * num_freqs


def mlp(in_dim, hidden, out_dim, layers=2):
    mods, d = [], in_dim
    for _ in range(layers - 1):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class SelfAttnBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))
        self.drop = nn.Dropout(dropout)  # feed-forward dropout (no parameters: existing checkpoints still load)

    def forward(self, x, key_padding_mask=None):
        """key_padding_mask: (B, N) bool, True = ignore that element as a key."""
        h = self.norm1(x)
        x = x + self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return x + self.drop(self.mlp(self.norm2(x)))


class CrossAttnBlock(nn.Module):
    """Queries attend to a (large) context set, as in 3DShape2VecSet / Perceiver."""

    def __init__(self, dim, heads, context_dim=None, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        context_dim = context_dim or dim
        self.norm_q = nn.LayerNorm(dim)
        self.norm_c = nn.LayerNorm(context_dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, kdim=context_dim, vdim=context_dim,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))
        self.drop = nn.Dropout(dropout)  # feed-forward dropout (no parameters: existing checkpoints still load)

    def forward(self, q, context):
        c = self.norm_c(context)
        q = q + self.attn(self.norm_q(q), c, c, need_weights=False)[0]
        return q + self.drop(self.mlp(self.norm2(q)))


class SlotReadout(nn.Module):
    """
    Slot-attention-style read of an unordered set: the softmax runs over the slots,
    so slots compete for set elements and keep their identity across time.
    """

    def __init__(self, slot_dim, set_dim, dim):
        super().__init__()
        self.norm_s = nn.LayerNorm(slot_dim)
        self.norm_x = nn.LayerNorm(set_dim)
        self.q = nn.Linear(slot_dim, dim, bias=False)
        self.k = nn.Linear(set_dim, dim, bias=False)
        self.v = nn.Linear(set_dim, dim, bias=False)
        self.scale = dim ** -0.5

    def forward(self, slots, x):
        q, k, v = self.q(self.norm_s(slots)), self.k(self.norm_x(x)), self.v(self.norm_x(x))
        logits = torch.einsum("bnd,bmd->bnm", q, k) * self.scale  # (B, slots, set)
        attn = logits.softmax(dim=1) + 1e-8                         # compete over slots
        attn = attn / attn.sum(dim=-1, keepdim=True)                # weighted mean over set
        return torch.einsum("bnm,bmd->bnd", attn, v), attn


def farthest_point_sample(points, n):
    """points: (B, P, 3) -> indices (B, n). Deterministic start at index 0."""
    B, P, _ = points.shape
    idx = torch.zeros(B, n, dtype=torch.long, device=points.device)
    dist = torch.full((B, P), float("inf"), device=points.device)
    farthest = torch.zeros(B, dtype=torch.long, device=points.device)
    batch = torch.arange(B, device=points.device)
    for i in range(n):
        idx[:, i] = farthest
        d = ((points - points[batch, farthest][:, None]) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        farthest = dist.argmax(-1)
    return idx


def gather(x, idx):
    """x: (B, P, ...), idx: (B, n) -> (B, n, ...)."""
    return x[torch.arange(x.shape[0], device=x.device)[:, None], idx]


def chamfer(a, b):
    """Symmetric Chamfer distance between point sets a: (B, N, D) and b: (B, M, D)."""
    d = torch.cdist(a, b)
    return d.min(dim=2).values.mean() + d.min(dim=1).values.mean()


def gaussian_kl(mu_q, logstd_q, mu_p, logstd_p):
    """KL(q || p) for diagonal Gaussians, summed over the last dim."""
    var_q, var_p = (2 * logstd_q).exp(), (2 * logstd_p).exp()
    return (logstd_p - logstd_q + (var_q + (mu_q - mu_p) ** 2) / (2 * var_p) - 0.5).sum(-1)


def split_gaussian(x, min_std=0.1):
    mu, std = x.chunk(2, dim=-1)
    std = F.softplus(std) + min_std
    return mu, std.log()
