"""
Gaussian-splatting autoencoder with physical heads.

    RGB-D frames (t-1, t) from V calibrated views
        -> PixelGaussianHead : per-pixel 3D Gaussians + features (depth is given, Splatter-Image style)
        -> GaussianSetEncoder: N latent tokens (FPS anchors + cross-attention, as in GWM / 3DShape2VecSet)
        -> GaussianSetDecoder: M Gaussians per token, rendered with gsplat
        -> PhysicsHead       : per-token velocity, mass, friction

A token is (anchor xyz, latent z). The anchor keeps the token's 3D position explicit,
so a dynamics model can predict motion as anchor displacement.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from physgs.model.layers import (CrossAttnBlock, SelfAttnBlock, chamfer, farthest_point_sample, fourier_dim,
                                 fourier_embed, gather, mlp)

GAUSS_PARAM_DIM = 3 + 4 + 1 + 3  # scale, quat, opacity, colour


def unproject(u, v, z, K, cam2world):
    """Pixel coords (u, v) and depth z, all (B, P), -> world points (B, P, 3). OpenCV camera convention."""
    fx, fy, cx, cy = K[:, 0, 0, None], K[:, 1, 1, None], K[:, 0, 2, None], K[:, 1, 2, None]
    p = torch.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], dim=-1)
    return p @ cam2world[:, :3, :3].transpose(1, 2) + cam2world[:, None, :3, 3]


class PixelGaussianHead(nn.Module):
    """One Gaussian per `stride` x `stride` pixel cell, from two consecutive RGB-D frames of one view."""

    def __init__(self, feat_dim=32, hidden=64, stride=4):
        super().__init__()
        assert stride in (1, 2, 4, 8)
        self.stride, self.feat_dim = stride, feat_dim
        layers, ch = [nn.Conv2d(8, hidden, 3, padding=1), nn.GELU()], hidden
        for _ in range(int(torch.log2(torch.tensor(stride)).item())):
            layers += [nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.GroupNorm(8, ch), nn.GELU()]
        layers += [nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(), nn.Conv2d(ch, 1 + GAUSS_PARAM_DIM + feat_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, rgb, depth, K, cam2world):
        """
        rgb: (B, 2, 3, H, W) in [0, 1], frames (t-1, t); depth: (B, 2, H, W) metres;
        K: (B, 3, 3); cam2world: (B, 4, 4) at time t.
        Returns a dict of per-Gaussian tensors with P = (H / stride) * (W / stride).
        """
        B, _, _, H, W = rgb.shape
        s = self.stride
        x = torch.cat([rgb.flatten(1, 2), depth], dim=1)  # (B, 8, H, W)
        out = self.net(x).flatten(2).transpose(1, 2)     # (B, P, C)
        d_off, scale, quat, opa, col, feat = out.split([1, 3, 4, 1, 3, self.feat_dim], dim=-1)

        rows = torch.arange(s // 2, H, s, device=rgb.device)
        cols = torch.arange(s // 2, W, s, device=rgb.device)
        vv, uu = torch.meshgrid(rows, cols, indexing="ij")
        z = depth[:, 1, s // 2::s, s // 2::s].flatten(1)
        z = z * (1 + 0.05 * torch.tanh(d_off[..., 0]))
        u = uu.flatten()[None].float().expand(B, -1)
        v = vv.flatten()[None].float().expand(B, -1)
        means = unproject(u, v, z, K, cam2world)

        base_col = rgb[:, 1, :, s // 2::s, s // 2::s].flatten(2).transpose(1, 2)
        footprint = (z * s / K[:, 0, 0, None])[..., None]  # world size of one cell
        return dict(
            means=means,
            scales=footprint * torch.exp(scale.clamp(-3, 3)) * 0.5,
            quats=F.normalize(quat + torch.tensor([1.0, 0, 0, 0], device=quat.device), dim=-1),
            opacities=torch.sigmoid(opa[..., 0] + 2.0),
            colors=(base_col + 0.2 * torch.tanh(col)).clamp(0, 1),
            feats=feat,
            pix=torch.stack([v, u], dim=-1).long(),  # (row, col) for label lookup
        )


class GaussianSetEncoder(nn.Module):
    def __init__(self, feat_dim=32, num_tokens=64, dim=192, latent_dim=64, depth=4, heads=6):
        super().__init__()
        self.num_tokens = num_tokens
        in_dim = fourier_dim() + GAUSS_PARAM_DIM + feat_dim
        self.in_proj = nn.Linear(in_dim, dim)
        self.cross = CrossAttnBlock(dim, heads)
        self.blocks = nn.ModuleList([SelfAttnBlock(dim, heads) for _ in range(depth)])
        self.to_latent = nn.Linear(dim, 2 * latent_dim)

    def forward(self, g):
        x = torch.cat([fourier_embed(g["means"]), g["scales"].log(), g["quats"], g["opacities"][..., None],
                       g["colors"], g["feats"]], dim=-1)
        x = self.in_proj(x)
        idx = farthest_point_sample(g["means"].detach(), self.num_tokens)
        q = self.cross(gather(x, idx), x)
        for blk in self.blocks:
            q = blk(q)
        mu, logvar = self.to_latent(q).chunk(2, dim=-1)
        return dict(anchor=gather(g["means"], idx), mu=mu, logvar=logvar.clamp(-10, 5), idx=idx)


class GaussianSetDecoder(nn.Module):
    def __init__(self, latent_dim=64, dim=192, depth=4, heads=6, gaussians_per_token=64, radius=0.15):
        super().__init__()
        self.M, self.radius = gaussians_per_token, radius
        self.in_proj = nn.Linear(fourier_dim() + latent_dim, dim)
        self.blocks = nn.ModuleList([SelfAttnBlock(dim, heads) for _ in range(depth)])
        self.out = nn.Linear(dim, self.M * (3 + GAUSS_PARAM_DIM))

    def forward(self, anchor, z):
        """anchor: (B, N, 3), z: (B, N, latent) -> Gaussians with N * M elements."""
        B, N, _ = anchor.shape
        h = self.in_proj(torch.cat([fourier_embed(anchor), z], dim=-1))
        for blk in self.blocks:
            h = blk(h)
        out = self.out(h).view(B, N, self.M, 3 + GAUSS_PARAM_DIM)
        off, scale, quat, opa, col = out.split([3, 3, 4, 1, 3], dim=-1)
        means = anchor[:, :, None] + self.radius * torch.tanh(off)
        return dict(
            means=means.flatten(1, 2),
            scales=(0.01 * torch.exp(scale.clamp(-4, 3))).flatten(1, 2),
            quats=F.normalize(quat + torch.tensor([1.0, 0, 0, 0], device=quat.device), dim=-1).flatten(1, 2),
            opacities=torch.sigmoid(opa[..., 0]).flatten(1, 2),
            colors=torch.sigmoid(col).flatten(1, 2),
        )


class PhysicsHead(nn.Module):
    """Per-token physical state: velocity (3), log-mass (1), friction (1), is-object logit (1)."""

    def __init__(self, latent_dim=64, hidden=256):
        super().__init__()
        self.net = mlp(fourier_dim() + latent_dim, hidden, 6, layers=3)

    def forward(self, anchor, z):
        vel, log_mass, friction, obj = self.net(torch.cat([fourier_embed(anchor), z], dim=-1)).split([3, 1, 1, 1], -1)
        return dict(vel=vel, log_mass=log_mass[..., 0], friction=F.softplus(friction[..., 0]), obj_logit=obj[..., 0])


def render_gaussians(g, K, cam2world, H, W, background=None):
    """Render a batch of Gaussian sets into V views with gsplat. Returns rgb (B, V, H, W, 3), depth (B, V, H, W)."""
    from gsplat import rasterization  # optional dependency; only needed for training the autoencoder

    rgbs, depths = [], []
    for b in range(g["means"].shape[0]):
        bg = background if background is not None else torch.zeros(K.shape[1], 3, device=K.device)
        out, _, _ = rasterization(
            g["means"][b], g["quats"][b], g["scales"][b], g["opacities"][b], g["colors"][b],
            torch.linalg.inv(cam2world[b]), K[b], W, H, render_mode="RGB+ED", backgrounds=bg,
        )
        rgbs.append(out[..., :3])
        depths.append(out[..., 3])
    return torch.stack(rgbs), torch.stack(depths)


class GSAutoencoder(nn.Module):
    """
    Encodes V views of (t-1, t) RGB-D into N physical Gaussian tokens and decodes them back.
    Novel-view supervision: encode from `input_views`, render and supervise all views.
    """

    def __init__(self, feat_dim=32, stride=4, num_tokens=64, dim=192, latent_dim=64, depth=4, heads=6,
                 gaussians_per_token=64, kl_weight=1e-4, chamfer_weight=1.0, depth_weight=0.5, phys_weight=1.0):
        super().__init__()
        self.pixel_head = PixelGaussianHead(feat_dim=feat_dim, stride=stride)
        self.encoder = GaussianSetEncoder(feat_dim, num_tokens, dim, latent_dim, depth, heads)
        self.decoder = GaussianSetDecoder(latent_dim, dim, depth, heads, gaussians_per_token)
        self.physics = PhysicsHead(latent_dim)
        self.w = dict(kl=kl_weight, chamfer=chamfer_weight, depth=depth_weight, phys=phys_weight)

    def lift(self, rgb, depth, K, cam2world):
        """rgb: (B, V, 2, 3, H, W); depth: (B, V, 2, H, W); K, cam2world: (B, V, ...). Concatenates views."""
        B, V = rgb.shape[:2]
        g = self.pixel_head(rgb.flatten(0, 1), depth.flatten(0, 1), K.flatten(0, 1), cam2world.flatten(0, 1))
        P = g["means"].shape[1]
        out = {k: v.reshape(B, V * P, *v.shape[2:]) for k, v in g.items()}
        out["view"] = torch.arange(V, device=rgb.device).repeat_interleave(P)[None].expand(B, -1)
        return out

    def encode(self, rgb, depth, K, cam2world, sample=False):
        g = self.lift(rgb, depth, K, cam2world)
        e = self.encoder(g)
        z = e["mu"] + torch.randn_like(e["mu"]) * (0.5 * e["logvar"]).exp() if sample else e["mu"]
        return dict(anchor=e["anchor"], z=z, mu=e["mu"], logvar=e["logvar"], idx=e["idx"], gaussians=g)

    def tokens(self, rgb, depth, K, cam2world):
        """Deterministic tokens (B, N, 3 + latent) for the dynamics model."""
        e = self.encode(rgb, depth, K, cam2world)
        return torch.cat([e["anchor"], e["z"]], dim=-1)

    def decode(self, tokens):
        return self.decoder(tokens[..., :3], tokens[..., 3:])

    def predict_physics(self, tokens):
        return self.physics(tokens[..., :3], tokens[..., 3:])

    def loss(self, batch, input_views=None):
        """
        batch: rgb (B, V, 2, 3, H, W) in [0, 1], depth (B, V, 2, H, W), K (B, V, 3, 3), cam2world (B, V, 4, 4),
        optional physics labels per pixel of frame t: obj_id (B, V, H, W) long (-1 = none),
        obj_vel (B, O, 3), obj_log_mass (B, O), obj_friction (B, O), robot_mask (B, V, H, W) bool.
        """
        rgb, depth, K, c2w = batch["rgb"], batch["depth"], batch["K"], batch["cam2world"]
        iv = slice(None) if input_views is None else input_views
        e = self.encode(rgb[:, iv], depth[:, iv], K[:, iv], c2w[:, iv], sample=True)
        dec = self.decoder(e["anchor"], e["z"])
        H, W = rgb.shape[-2:]
        r_rgb, r_depth = render_gaussians(dec, K, c2w, H, W)
        tgt_rgb = rgb[:, :, 1].permute(0, 1, 3, 4, 2)
        tgt_depth = depth[:, :, 1]
        losses = dict(
            rgb=(r_rgb - tgt_rgb).abs().mean(),
            depth=(r_depth - tgt_depth).abs().mean(),
            chamfer=chamfer(dec["means"], e["gaussians"]["means"].detach()),
            kl=(-0.5 * (1 + e["logvar"] - e["mu"] ** 2 - e["logvar"].exp())).sum(-1).mean(),
        )
        if "obj_id" in batch:
            losses.update(self.physics_loss(e, batch, iv))
        total = losses["rgb"] + self.w["depth"] * losses["depth"] + self.w["chamfer"] * losses["chamfer"] \
            + self.w["kl"] * losses["kl"] + self.w["phys"] * sum(v for k, v in losses.items() if k.startswith("phys_"))
        return total, {k: v.item() for k, v in losses.items()}

    def physics_loss(self, e, batch, iv):
        """Label each token by the object under its anchor pixel; robot tokens are ignored."""
        g, idx = e["gaussians"], e["idx"]
        view = gather(g["view"], idx)                 # (B, N)
        pix = gather(g["pix"], idx)                   # (B, N, 2)
        b = torch.arange(view.shape[0], device=view.device)[:, None]
        obj = batch["obj_id"][:, iv][b, view, pix[..., 0], pix[..., 1]]  # (B, N)
        if "robot_mask" in batch:
            robot = batch["robot_mask"][:, iv][b, view, pix[..., 0], pix[..., 1]]
        else:
            robot = torch.zeros_like(obj, dtype=torch.bool)
        pred = self.physics(e["anchor"], e["z"])

        is_obj = obj >= 0
        valid = ~robot
        out = dict(phys_obj=F.binary_cross_entropy_with_logits(pred["obj_logit"][valid], is_obj[valid].float()))
        o = obj.clamp(min=0)
        vel_tgt = torch.where(is_obj[..., None], gather(batch["obj_vel"], o), torch.zeros_like(pred["vel"]))
        out["phys_vel"] = F.smooth_l1_loss(pred["vel"][valid], vel_tgt[valid])
        if is_obj.any():
            out["phys_mass"] = F.mse_loss(pred["log_mass"][is_obj], gather(batch["obj_log_mass"], o)[is_obj])
            out["phys_friction"] = F.mse_loss(pred["friction"][is_obj], gather(batch["obj_friction"], o)[is_obj])
        return out


if __name__ == "__main__":
    # Shape smoke test (no gsplat needed): python -m physgs.model.gs_autoencoder
    B, V, H, W = 2, 4, 128, 128
    model = GSAutoencoder()
    rgb = torch.rand(B, V, 2, 3, H, W)
    depth = 0.8 + 0.4 * torch.rand(B, V, 2, H, W)
    K = torch.tensor([[150.0, 0, 64], [0, 150.0, 64], [0, 0, 1]]).expand(B, V, 3, 3)
    c2w = torch.eye(4).expand(B, V, 4, 4)
    tok = model.tokens(rgb, depth, K, c2w)
    dec = model.decode(tok)
    phys = model.predict_physics(tok)
    print("tokens", tuple(tok.shape), "decoded gaussians", tuple(dec["means"].shape), "vel", tuple(phys["vel"].shape))
    print("params (M):", sum(p.numel() for p in model.parameters()) / 1e6)
