"""
Jointly trained object-centric physical world model (2026-10-31 revision of the 10-29 model, the paper's method).
Replaces the separately trained object autoencoder (object_encoder.py) + token RSSM (object_rssm.py), which stay
unchanged.

Units: no metric constant anywhere. Lengths are in units of the episode's length scale L (the median RMS radius of the
objects visible in the episode's first frame, the frame that also fixes the scene centre), velocities in L per second,
the frame interval dt in seconds. Each object's shape is encoded and decoded in units of its own RMS radius rho, so a
2 m prop and a 4 cm bowl share one shape range. log L (the absolute scale, which mass depends on) and log dt condition
the sequence model. Property beliefs live in a space normalized by running statistics of the property labels (as GNS,
Sanchez-Gonzalez et al., ICML 2020), updated only while those labels are trained on.

One RSSM, one slot per object (identity from the instance id). Per step t and slot k:
    z_t ~ q(z | h_t, G_t, c_t)  when observed (set encoder of the Gaussians + centroid), else the prior p(z | h_t)
    r_t = [h_t, z_t]                                     object representation
    p^_t = p_{t-1} + dt v_{t-1}                          predicted position
    v_t = v_{t-1} + g_v(r_t, iota_t)                     velocity: persists, changed by the interaction (through
                                                         h_t) and corrected by the position error iota_t =
                                                         (c_t - p^_t) / rho_t when observed (0 in imagination), in
                                                         units of the typical per-frame motion
    p_t = c_t when observed, else p^_t                   (so the velocity is what moves the object)
    b_t = g_b(r_t)                                       belief over log mass, friction, restitution (mean, log-std);
                                                         kept fixed during imagination
    u_t = E(r_t, p_t, v_t, rho_t, b_t)                   interaction token
    m_t = RelAttn(u_t, {u_j}, e(p, v, rho))              attention over objects; messages see the pair's relative
                                                         position and velocity in units of the pair's size
    h_{t+1} = f(r_t, m_t; task, log L, log dt)           GRU, FiLM by the task embedding
    G^_t = p_t + rho_t D(E [r_t, m_t])                   decoded Gaussians
Losses, in three groups: reconstruction + KL (Chamfer, colour, log scale in object radii; KL balancing, free nats),
prediction (one-step position consistency of the velocity, reconstruction of imagined Gaussians), physics (velocity
labels of filtered and imagined states, imagined displacement p_{t0+k} - p_{t0} against the labelled centre-of-mass
displacement, property NLL; only where labels exist). No absolute position label: the position is the visible centroid,
which differs from the labelled point by a roughly constant offset (LIBERO median 2.7 cm objects, 6.4 cm gripper) that
the displacement cancels. No angular velocity, orientation or hand-built measurement.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from physgs.model.layers import CrossAttnBlock, SelfAttnBlock, fourier_dim, fourier_embed, gather, gaussian_kl, mlp, split_gaussian
from physgs.model.object_encoder import PROP_DIM, gaussian_nll, masked_mean

DYN_DIM = 6        # forecast features 0:6 = position, velocity (what the policy's action head reads)
EDGE_DIM = 8       # relative position (3), distance (1), relative motion per frame (3), approach per frame (1)
EPS = 1e-6         # numerical floor of ratios (unit-free)


def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def forecast_dim(embed_dim=128):
    """pos, vel (DYN_DIM), e (embed_dim), belief mean / log-std, log radius, is_robot, present."""
    return DYN_DIM + embed_dim + 2 * PROP_DIM + 3


def rms_radius(xyz):
    """(..., P, 3) -> (...): RMS distance of the points from their centroid."""
    x = xyz.float()
    return ((x - x.mean(-2, keepdim=True)) ** 2).sum(-1).mean(-1).sqrt()


@torch.no_grad()
def length_scale(gauss, vis, is_robot):
    """
    Episode length scale L from its first frame: gauss (B, O, P, 7), vis / is_robot (B, O) -> (B,), the median RMS
    radius of the visible objects other than the robot (of every visible object if there is none; 1 if nothing is
    visible).
    """
    r = rms_radius(gauss[..., :3])
    out = []
    for i in range(r.shape[0]):
        m = vis[i] & ~is_robot[i]
        m = m if m.any() else vis[i]
        out.append(r[i][m].median().clamp(min=EPS) if m.any() else r.new_tensor(1.0))
    return torch.stack(out)


class RunningNorm(nn.Module):
    """Per-dimension mean / std of every finite value seen by update() (Welford); stored with the checkpoint."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("count", torch.zeros(dim))
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("m2", torch.zeros(dim))

    @torch.no_grad()
    def reset(self):
        for b in (self.count, self.mean, self.m2):
            b.zero_()

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, self.mean.shape[0]).float()
        ok = torch.isfinite(x)
        n = ok.sum(0).float()
        if n.sum() == 0:
            return
        bm = torch.where(ok, x, 0.0).sum(0) / n.clamp(min=1)
        bm2 = torch.where(ok, (x - bm) ** 2, 0.0).sum(0)
        tot = self.count + n
        delta = bm - self.mean
        self.mean += delta * n / tot.clamp(min=1)
        self.m2 += bm2 + delta ** 2 * self.count * n / tot.clamp(min=1)
        self.count.copy_(tot)

    @property
    def std(self):
        return torch.where(self.count > 1, (self.m2 / (self.count - 1).clamp(min=1)).sqrt(), torch.ones_like(self.m2)).clamp(min=EPS)

    def forward(self, x):
        return (x - self.mean) / self.std


class GaussianSetEncoder(nn.Module):
    """Learned queries cross-attend to one object's Gaussians, centred on their centroid and in units of their RMS
    radius (positions and scales); mean-pooled feature."""

    def __init__(self, dim, depth, heads, num_queries=32, dropout=0.0):
        super().__init__()
        self.in_proj = nn.Linear(fourier_dim() + 3 + 1, dim)
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.cross = CrossAttnBlock(dim, heads, dropout=dropout)
        self.blocks = nn.ModuleList([SelfAttnBlock(dim, heads, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, g):
        """g: (n, P, 7) Gaussians. Returns features (n, dim), centroids (n, 3) and RMS radii (n,)."""
        c = g[..., :3].mean(1)
        rho = rms_radius(g[..., :3]).clamp(min=EPS).to(g.dtype)
        k = 1.0 / rho[:, None, None]
        x = self.in_proj(torch.cat([fourier_embed((g[..., :3] - c[:, None]) * k), g[..., 3:6],
                                    torch.log((g[..., 6:7] * k).clamp(min=EPS))], -1))
        q = self.cross(self.queries[None].expand(g.shape[0], -1, -1), x)
        for blk in self.blocks:
            q = blk(q)
        return self.norm(q.mean(1)), c, rho


class ShapeDecoder(nn.Module):
    """Embedding -> Gaussian set in the object's normalized frame: xyz and log scale in units of its RMS radius, rgb."""

    def __init__(self, latent_dim=128, dim=384, depth=4, heads=8, num_tokens=64, points_per_token=8):
        super().__init__()
        self.num_tokens, self.ppt = num_tokens, points_per_token
        self.queries = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        self.z_proj = nn.Linear(latent_dim, dim)
        self.cross = CrossAttnBlock(dim, heads)
        self.blocks = nn.ModuleList([SelfAttnBlock(dim, heads) for _ in range(depth)])
        self.out = nn.Linear(dim, points_per_token * 7)

    def forward(self, e):
        q = self.cross(self.queries[None].expand(e.shape[0], -1, -1), self.z_proj(e)[:, None])
        for blk in self.blocks:
            q = blk(q)
        out = self.out(q).view(e.shape[0], self.num_tokens * self.ppt, 7)
        return out[..., :3], torch.sigmoid(out[..., 3:6]), out[..., 6:7]


class RelationBlock(nn.Module):
    """
    Attention over objects in which the message from object j to object k is an MLP of both tokens and of their
    relative geometry (an interaction network, Battaglia et al., NeurIPS 2016, with attention weights): products such
    as approach speed x restitution can be formed per pair, which a dot-product over per-object tokens cannot.
    """

    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.q, self.k = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.dst, self.src = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.edge = mlp(EDGE_DIM, dim, dim)
        self.edge_bias = nn.Linear(dim, heads)
        self.msg = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge, attend):
        """x (N, O, D), edge (N, O, O, EDGE_DIM) receiver k x sender j, attend (N, O, O) bool."""
        N, O, D = x.shape
        H, dh = self.heads, D // self.heads
        h = self.norm1(x)
        e = self.edge(edge)
        logits = torch.einsum("nkhd,njhd->nkjh", self.q(h).view(N, O, H, dh), self.k(h).view(N, O, H, dh)) / math.sqrt(dh)
        logits = (logits + self.edge_bias(e)).float().masked_fill(~attend[..., None], float("-inf"))
        a = logits.softmax(2).to(h.dtype)
        m = self.msg(F.gelu(self.dst(h)[:, :, None] + self.src(h)[:, None] + e))                 # (N, O, O, D)
        x = x + torch.einsum("nkjh,nkjhd->nkhd", a, m.view(N, O, O, H, dh)).reshape(N, O, D)
        return x + self.drop(self.ff(self.norm2(x)))


class JointWorldModel(nn.Module):
    def __init__(self, deter=512, stoch=32, dim=384, heads=8, enc_depth=4, enc_queries=32, inter_layers=2,
                 task_dim=768, embed_dim=128, dec_depth=3, dec_tokens=64, dec_ppt=8, free_nats=1.0, kl_dyn=0.5,
                 kl_rep=0.1, feedback=True, feedback_detach=True, dropout=0.0, w=None):
        super().__init__()
        self.deter, self.stoch, self.dim, self.embed_dim = deter, stoch, dim, embed_dim
        self.free_nats, self.kl_dyn, self.kl_rep = free_nats, kl_dyn, kl_rep
        self.feedback, self.feedback_detach = feedback, feedback_detach
        self.enc_ckpt = 1.0               # fraction of the set-encoder chunks recomputed in the backward pass (memory)
        # running statistics (property labels, per-frame motion) are those of the data the model is first trained on;
        # train_joint freezes them when it starts from a checkpoint, so adaptation does not move the normalized spaces
        self.update_norms = True
        self.w = dict(chamfer=1.0, color=0.5, scale=0.1, kl=1.0,                             # reconstruction, KL
                      pos1=10.0, img_chamfer=1.0, img_color=0.5, img_scale=0.1,                # prediction
                      vel=1.0, img_vel=1.0, img_disp=10.0, prop=0.1)                           # physics labels
        self.w.update(w or {})
        self.r_dim = deter + stoch
        self.s_dim = deter + stoch + dim
        self.prop_norm = RunningNorm(PROP_DIM)
        # per-frame motion of visible objects in units of their radius: the scale of the position error iota
        self.motion_norm = RunningNorm(3)

        self.encoder = GaussianSetEncoder(dim, enc_depth, heads, enc_queries, dropout)
        # observed centroid, its offset from the predicted position (in object radii), log radius
        self.geo_mlp = mlp(7, dim, dim)
        self.h0 = nn.Parameter(torch.zeros(deter))
        self.type_embed = nn.Embedding(2, dim)                       # 0 = object, 1 = robot / effector
        self.task_proj = nn.Linear(task_dim, 2 * dim)
        self.cond_proj = nn.Linear(2, dim)                           # log L, log dt
        self.in_proj = nn.Linear(self.s_dim, dim)
        self.gru = nn.GRUCell(dim, deter)
        self.prior = mlp(deter, dim, 2 * stoch)
        self.post = mlp(deter + dim, dim, 2 * stoch)
        # physical heads on r = [h, z]; the velocity change (also reads the position error) starts at 0. The position
        # error reaches it directly, not only through the KL-taxed z: the decoder works around the observed centroid
        # and never needs the motion, so through z alone the velocity was not learned (synthetic check, 10-31)
        self.vel_head = mlp(self.r_dim + 3, dim, 3)
        nn.init.zeros_(self.vel_head[-1].weight)
        nn.init.zeros_(self.vel_head[-1].bias)
        self.belief_head = mlp(self.r_dim, dim, 2 * PROP_DIM)
        # interaction token: representation + position, velocity, log radius, belief
        self.token = nn.Sequential(nn.Linear(self.r_dim + 7 + 2 * PROP_DIM, dim), nn.LayerNorm(dim))
        self.interact = nn.ModuleList([RelationBlock(dim, heads, dropout=dropout) for _ in range(inter_layers)])
        self.embed = nn.Linear(self.s_dim, embed_dim)                # decoder input, policy forecast
        self.decoder = ShapeDecoder(embed_dim, dim, dec_depth, heads, dec_tokens, dec_ppt)

    # ------------------------------------------------------------------------------------------------ units
    @staticmethod
    def to_units(gauss, L):
        """Gaussians (B, ..., 7) in metres -> units of L (xyz and scale)."""
        k = L.view(-1, *[1] * (gauss.dim() - 1)).to(gauss.dtype)
        return torch.cat([gauss[..., :3] / k, gauss[..., 3:6], gauss[..., 6:7] / k], -1)

    # ------------------------------------------------------------------------------------------------ one step
    def context(self, is_robot, present, task, fps=None, L=None):
        gamma, beta = self.task_proj(task)[:, None].chunk(2, -1)
        B = task.shape[0]
        fps = torch.full((B,), 20.0, device=task.device) if fps is None else fps.float()
        L = torch.ones(B, device=task.device) if L is None else L.float()
        dt = 1.0 / fps
        cond = self.cond_proj(torch.stack([L.log(), dt.log()], -1))[:, None]
        return dict(type=self.type_embed(is_robot.long()), present=present, gamma=gamma, beta=beta,
                    dt=dt.view(B, 1, 1), cond=cond)

    def initial(self, B, O, device):
        z = lambda *s: torch.zeros(B, O, *s, device=device)
        return dict(h=self.h0.expand(B, O, -1).contiguous(), z=z(self.stoch), m=z(self.dim), pos=z(3), vel=z(3),
                    rho=torch.ones(B, O, device=device), belief_mu=z(PROP_DIM), belief_logstd=z(PROP_DIM),
                    seen=torch.zeros(B, O, dtype=torch.bool, device=device))

    def _sequence(self, st, ctx):
        B, O, _ = st["h"].shape
        x = self.in_proj(torch.cat([st["h"], st["z"], st["m"]], -1)) + ctx["cond"] + ctx["type"]
        x = x * (1 + ctx["gamma"]) + ctx["beta"]
        with torch.autocast(device_type=x.device.type, enabled=False):  # GRUCell has no bf16 kernel in torch 2.0
            h = self.gru(x.float().flatten(0, 1), st["h"].float().flatten(0, 1))
        return h.view(B, O, self.deter)

    @staticmethod
    def _edges(pos, vel, rho, dt):
        """Pair features, receiver k (dim 1) x sender j (dim 2), in units of the pair's size rho_k + rho_j: relative
        position, distance (about 1 at contact), relative motion per frame, approach per frame (> 0: closing)."""
        d = pos[:, None] - pos[:, :, None]                                                  # p_j - p_k
        dv = vel[:, None] - vel[:, :, None]
        s = (rho[:, None] + rho[:, :, None])[..., None].clamp(min=EPS)
        dist = ((d ** 2).sum(-1, keepdim=True) + (EPS * s) ** 2).sqrt()                    # finite gradient at k = j
        dt = dt[..., None]
        approach = -(dv * d).sum(-1, keepdim=True) / dist * dt / s
        return symlog(torch.cat([d / s, dist / s, dv * dt / s, approach], -1))

    def step(self, st, ctx, obs=None, first=False, sample=True, keep_belief=False):
        """
        st: previous state; obs: dict(feat (B, O, dim), centroid (B, O, 3), rho (B, O), visible (B, O)) or None
        (imagination). keep_belief: the belief is copied from st (imagination). Returns (state, prior stats).
        """
        h = st["h"] if first else self._sequence(st, ctx)
        p_mu, p_ls = split_gaussian(self.prior(h))
        seen_prev = st["seen"]
        sp = seen_prev[..., None]
        pred_pos = torch.where(sp, st["pos"] + ctx["dt"] * st["vel"], st["pos"])            # p_{t-1} + dt v_{t-1}
        if obs is not None:
            vis = obs["visible"]
            rho_o = obs["rho"].clamp(min=EPS)
            anchor = torch.where(sp, pred_pos, obs["centroid"])                               # first sighting: itself
            geo = torch.cat([symlog(obs["centroid"]), symlog((obs["centroid"] - anchor) / rho_o[..., None]),
                             rho_o.log()[..., None]], -1)
            q_mu, q_ls = split_gaussian(self.post(torch.cat([h, obs["feat"] + self.geo_mlp(geo)], -1)))
            q_mu, q_ls = torch.where(vis[..., None], q_mu, p_mu), torch.where(vis[..., None], q_ls, p_ls)
            seen = seen_prev | vis
            rho = torch.where(vis, rho_o, st["rho"])
        else:
            vis, seen, rho = torch.zeros_like(seen_prev), seen_prev, st["rho"]
            q_mu, q_ls = p_mu, p_ls
        z = q_mu + torch.randn_like(q_mu) * q_ls.exp() if sample else q_mu
        r = torch.cat([h, z], -1)

        # physical state: the velocity persists and changes by g_v; position = the observed centroid when visible,
        # else integrated
        if obs is not None:
            err = (obs["centroid"] - pred_pos) / rho_o[..., None] / self.motion_norm.std
            iota = torch.where((vis & seen_prev)[..., None], err, torch.zeros_like(err))
        else:
            iota = torch.zeros_like(pred_pos)
        vel = torch.where(sp, st["vel"], torch.zeros_like(st["vel"])) + self.vel_head(torch.cat([r, symlog(iota)], -1))
        pos = torch.where(vis[..., None], obs["centroid"], pred_pos) if obs is not None else pred_pos
        sn = seen[..., None]
        pos, vel = torch.where(sn, pos, torch.zeros_like(pos)), torch.where(sn, vel, torch.zeros_like(vel))
        if keep_belief:
            bmu, bls = st["belief_mu"], st["belief_logstd"]
        else:
            b = self.belief_head(r)
            bmu, bls = b[..., :PROP_DIM], b[..., PROP_DIM:].clamp(-5, 3)

        # interaction: one token per object, relational attention over the present objects seen so far
        phys = torch.cat([symlog(pos), symlog(vel), rho.log()[..., None], bmu, bls], -1)
        edge = self._edges(pos, vel, rho, ctx["dt"])
        if self.feedback_detach:
            phys, edge = phys.detach(), edge.detach()
        if not self.feedback:
            phys, edge = torch.zeros_like(phys), torch.zeros_like(edge)
        x = self.token(torch.cat([r, phys], -1)) + ctx["type"]
        O = x.shape[1]
        attend = (ctx["present"] & seen)[:, None, :] | torch.eye(O, dtype=torch.bool, device=x.device)
        for blk in self.interact:
            x = blk(x, edge, attend)
        s = torch.cat([r, x], -1)
        out = dict(h=h, z=z, m=x, s=s, pos=pos, vel=vel, rho=rho, pred_pos=pred_pos, belief_mu=bmu,
                   belief_logstd=bls, seen=seen, seen_prev=seen_prev, mu=q_mu, logstd=q_ls)
        return out, dict(mu=p_mu, logstd=p_ls)

    def decode(self, s, pos, rho):
        """s (n, s_dim), pos (n, 3), rho (n,) -> (n, M, 7) Gaussians in the scene frame (units of L)."""
        xyz, rgb, ls = self.decoder(self.embed(s))
        k = rho[:, None, None]
        return torch.cat([pos[:, None] + xyz * k, rgb, ls.clamp(max=10).exp() * k], -1)

    def encode_obs(self, gauss, visible, chunk=2048):
        """gauss: (..., O, P, 7) in units of L, visible: (..., O). Set-encoder features, centroids and RMS radii of
        the visible objects, in chunks of sets; the first enc_ckpt fraction of the chunks is recomputed in the backward
        pass (same gradients, less memory), the rest keeps its activations (faster)."""
        lead = visible.shape
        feat = torch.zeros(*lead, self.dim, device=gauss.device, dtype=torch.float32)
        cen = torch.zeros(*lead, 3, device=gauss.device, dtype=torch.float32)
        rho = torch.ones(*lead, device=gauss.device, dtype=torch.float32)
        if visible.any():
            g = gauss[visible]
            grad = self.training and torch.is_grad_enabled()
            starts = list(range(0, g.shape[0], chunk))
            n_ckpt = round(float(self.enc_ckpt) * len(starts)) if grad else 0
            fs, cs, rs = [], [], []
            for i, s in enumerate(starts):
                x = g[s:s + chunk]
                f, c, r = torch.utils.checkpoint.checkpoint(self.encoder, x, use_reentrant=False) if i < n_ckpt else self.encoder(x)
                fs.append(f.float()); cs.append(c.float()); rs.append(r.float())
            feat[visible], cen[visible], rho[visible] = torch.cat(fs), torch.cat(cs), torch.cat(rs)
        return feat, cen, rho

    def observations(self, gauss, visible):
        feat, cen, rho = self.encode_obs(gauss, visible)
        return dict(feat=feat, centroid=cen, rho=rho, visible=visible)

    # ------------------------------------------------------------------------------------------------ sequences
    def observe(self, obs, ctx, sample=True):
        B, T, O = obs["visible"].shape
        st = self.initial(B, O, obs["visible"].device)
        posts, priors = [], []
        for t in range(T):
            st, pr = self.step(st, ctx, {k: v[:, t] for k, v in obs.items()}, first=(t == 0), sample=sample)
            posts.append(st)
            priors.append(pr)
        return posts, priors

    def imagine(self, st, ctx, horizon, sample=True):
        out = []
        for _ in range(horizon):
            st, _ = self.step(st, ctx, None, sample=sample, keep_belief=True)
            out.append(st)
        return out

    # ------------------------------------------------------------------------------------------------ losses
    @staticmethod
    def recon_terms(pred, target, pos, rho):
        """Permutation-invariant Gaussian distance in units of the object's radius: Chamfer on xyz, nearest-neighbour
        colour, log scale. pred = (xyz, rgb, log scale) in the normalized object frame, target (n, P, 7) scene frame."""
        xyz, rgb, ls = pred
        k = rho[:, None, None]
        tgt = (target[..., :3] - pos[:, None]) / k
        d = torch.cdist(xyz, tgt)
        i_pt, i_tp = d.argmin(2), d.argmin(1)
        cham = d.min(2).values.mean() + d.min(1).values.mean()
        color = (rgb - gather(target[..., 3:6], i_pt)).abs().mean() + (target[..., 3:6] - gather(rgb, i_tp)).abs().mean()
        scale = (torch.log((target[..., 6:7] / k).clamp(min=EPS)) - gather(ls, i_tp)).abs().mean()
        return cham, color, scale

    def _recon(self, states, idx, gauss, mask, rec_points):
        """Decode the states at time indices idx and compare with the observed Gaussians."""
        S, P_, R, G = [], [], [], []
        for st, t in zip(states, idx):
            m = mask[:, t]
            if m.any():
                S.append(st["s"][m]); P_.append(st["pos"][m]); R.append(st["rho"][m]); G.append(gauss[:, t][m][:, :rec_points])
        if not S:
            z = gauss.sum() * 0
            return z, z, z
        return self.recon_terms(self.decoder(self.embed(torch.cat(S))), torch.cat(G).float(), torch.cat(P_), torch.cat(R))

    @staticmethod
    def _sl1(pred, tgt, mask):
        return masked_mean(F.smooth_l1_loss(pred, tgt, reduction="none", beta=0.05), mask)

    def loss(self, batch, physics=True, overshoot=15, rec_steps=8, img_rec_steps=4, rec_points=256):
        """
        batch: gauss (B, T, O, P, 7) scene frame in metres, visible (B, T, O), present / is_robot (B, O), task (B, 768),
        fps (B,), L (B,) episode length scale, labels pos / vel (B, T, O, 3) scene frame, props (B, O, 3); NaN =
        unknown. physics=False: video objective only (no physical label used). Errors against the labels are always
        logged (err_*, metres and m/s, not trained on).
        """
        vis, present = batch["visible"], batch["present"]
        B, T, O = vis.shape
        L = batch["L"].float()
        g = self.to_units(batch["gauss"].float(), L)                 # the loader sends float16 (half the transfer)
        lab = {k: batch[k] / L.view(B, 1, 1, 1) for k in ("pos", "vel")}
        vis = vis & present[:, None]
        ctx = self.context(batch["is_robot"], present, batch["task"], batch.get("fps"), L)
        obs = self.observations(g, vis)
        if self.training and self.update_norms:  # per-frame motion of objects visible at t - 1 and t, in their radius
            both = vis[:, 1:] & vis[:, :-1]
            self.motion_norm.update(((obs["centroid"][:, 1:] - obs["centroid"][:, :-1]) / obs["rho"][:, 1:, ..., None])[both])
        posts, priors = self.observe(obs, ctx)
        logs, total = {}, 0.0
        stack = lambda sts, k: torch.stack([s[k] for s in sts], 1)

        def add(name, v):
            nonlocal total
            logs[name] = v
            total = total + self.w[name] * v

        # video: reconstruction of the observed Gaussians at a random subset of steps, KL
        ts = torch.randperm(T)[:min(rec_steps, T)].tolist()
        cham, color, scale = self._recon([posts[t] for t in ts], ts, g, vis, rec_points)
        add("chamfer", cham); add("color", color); add("scale", scale)
        kl = []
        for t in range(1, T):
            q, p = posts[t], priors[t]
            dyn = gaussian_kl(q["mu"].detach(), q["logstd"].detach(), p["mu"], p["logstd"]).clamp(min=self.free_nats)
            rep = gaussian_kl(q["mu"], q["logstd"], p["mu"].detach(), p["logstd"].detach()).clamp(min=self.free_nats)
            kl.append(masked_mean(self.kl_dyn * dyn + self.kl_rep * rep, vis[:, t]))
        add("kl", torch.stack(kl).mean())
        # prediction: p_{t-1} + dt v_{t-1} against the observed centroid p_t
        m1 = vis[:, 1:] & stack(posts[1:], "seen_prev")
        add("pos1", self._sl1(stack(posts[1:], "pred_pos"), stack(posts[1:], "pos"), m1))
        # physics: filtered velocities and beliefs against the labels
        seen = stack(posts, "seen") & present[:, None]
        if physics:
            add("vel", self._sl1(stack(posts, "vel"), torch.nan_to_num(lab["vel"]), seen & torch.isfinite(lab["vel"]).all(-1)))
            props = batch["props"]
            if self.training and self.update_norms:
                self.prop_norm.update(props[present])
            props = self.prop_norm(props)[:, None].expand(B, T, O, PROP_DIM)
            m = torch.isfinite(props) & seen[..., None]
            add("prop", masked_mean(gaussian_nll(torch.nan_to_num(props), stack(posts, "belief_mu"),
                                                 stack(posts, "belief_logstd")), m))
        # open-loop imagination from a random filtered state
        img, t0, H = None, 0, 0
        if overshoot > 0 and T > 1:
            t0 = int(torch.randint(0, max(1, T - overshoot), ()))
            H = min(overshoot, T - 1 - t0)
            img = self.imagine(dict(posts[t0]), ctx, H)
            known = posts[t0]["seen"]
            js = torch.randperm(H)[:min(img_rec_steps, H)].tolist()
            cham, color, scale = self._recon([img[j] for j in js], [t0 + 1 + j for j in js], g, vis & known[:, None],
                                             rec_points)
            add("img_chamfer", cham); add("img_color", color); add("img_scale", scale)
            if physics:
                m = (present & known)[:, None].expand(B, H, O)
                lk = lab["vel"][:, t0 + 1:t0 + 1 + H]
                add("img_vel", self._sl1(stack(img, "vel"), torch.nan_to_num(lk), m & torch.isfinite(lk).all(-1)))
                # displacement from the start of the rollout (cancels the centroid / labelled-point offset)
                dl = lab["pos"][:, t0 + 1:t0 + 1 + H] - lab["pos"][:, t0:t0 + 1]
                dp = stack(img, "pos") - posts[t0]["pos"][:, None]
                add("img_disp", self._sl1(dp, torch.nan_to_num(dl), m & torch.isfinite(dl).all(-1)))
        # readable errors against the labels (logs only): metres (centre of mass, so err_pos includes the offset from
        # the visible centroid), m/s
        with torch.no_grad():
            Lm = L.view(B, 1, 1)

            def err(pred, lb, m):
                m = m & torch.isfinite(lb).all(-1)
                return ((pred - torch.nan_to_num(lb)).norm(dim=-1) * Lm * m).sum() / m.sum().clamp(min=1)
            for k in ("pos", "vel"):
                logs[f"err_{k}"] = err(stack(posts, k), lab[k], seen & vis)
                if img is not None:
                    logs[f"err_img_{k}"] = err(stack(img, k), lab[k][:, t0 + 1:t0 + 1 + H],
                                               (present & posts[t0]["seen"])[:, None].expand(B, H, O))
            if img is not None:                                              # reference: every object stays still
                logs["err_img_pos_still"] = err(posts[t0]["pos"][:, None].expand(B, H, O, 3), lab["pos"][:, t0 + 1:t0 + 1 + H],
                                                (present & posts[t0]["seen"])[:, None].expand(B, H, O))
        logs["total"] = total
        vals = torch.stack([v.detach().float() for v in logs.values()]).tolist()
        return total, dict(zip(logs, vals))

    # ------------------------------------------------------------------------------------------------ forecast
    def forecast_features(self, st, belief, is_robot, present):
        """Policy features per object (forecast_dim): position (L), velocity (L / s) = the first DYN_DIM, e = E s,
        belief mean / log-std of the filtered state (normalized), log radius (L), is_robot, present & seen (last, as
        the policy reads it)."""
        return torch.cat([st["pos"], st["vel"], self.embed(st["s"]), belief["belief_mu"], belief["belief_logstd"],
                          st["rho"].log()[..., None], is_robot[..., None].float(),
                          (present & st["seen"])[..., None].float()], -1)


if __name__ == "__main__":
    # python -m physgs.model.joint_wm
    torch.manual_seed(0)
    rn = RunningNorm(3)
    x = torch.randn(1000, 3) * torch.tensor([2.0, 0.1, 5.0]) + torch.tensor([1.0, -3.0, 0.0])
    x[::7, 1] = float("nan")
    for c in x.split(97):
        rn.update(c)
    ok = torch.isfinite(x)
    print("running norm mean", rn.mean.tolist(), "std", rn.std.tolist(),
          "\n   direct  mean", [x[ok[:, i], i].mean().item() for i in range(3)], "std", [x[ok[:, i], i].std().item() for i in range(3)])

    B, T, O, P = 2, 20, 6, 128
    nan = float("nan")
    size = torch.tensor([1.0, 0.05, 1.0, 0.03, 2.0, 0.5]).view(1, 1, O, 1, 1)               # mixed object sizes
    gauss = torch.rand(B, T, O, P, 7)
    gauss[..., :3] = (gauss[..., :3] - 0.5) * size + torch.randn(B, 1, O, 1, 3) + torch.linspace(0, 1, T).view(1, T, 1, 1, 1)
    gauss[..., 6] *= 0.1 * size[..., 0]
    is_robot = torch.arange(O)[None].expand(B, O) == 0
    present = torch.arange(O)[None].expand(B, O) < 5
    vis = torch.rand(B, T, O) > 0.2
    vis[:, 0] = True
    L = length_scale(gauss[:, 0], vis[:, 0] & present, is_robot)
    print("length scale", L.tolist())
    batch = dict(gauss=gauss, visible=vis, present=present, is_robot=is_robot, task=torch.randn(B, 768),
                 fps=torch.tensor([20.0, 15.0]), L=L, pos=gauss[..., :3].mean(-2), vel=torch.randn(B, T, O, 3),
                 props=torch.randn(B, O, 3))
    batch["props"][0, 2] = nan
    m = JointWorldModel(deter=128, dim=96, heads=4, enc_depth=1, inter_layers=1, dec_depth=1)
    for phys in (True, False):
        loss, logs = m.loss(batch, physics=phys, overshoot=8)
        loss.backward()
        bad = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        print("physics" if phys else "video", {k: round(v, 3) for k, v in logs.items()}, "non-finite grads:", bad)
    print("params (M):", sum(p.numel() for p in m.parameters()) / 1e6)

    # unit check: the scene scaled by k (Gaussians, labels, L) gives the same normalized model, so the same losses
    # once the absolute-scale input log L is switched off
    m.eval()
    with torch.no_grad():
        m.cond_proj.weight[:, 0] = 0
        outs = []
        for k in (1.0, 37.0):
            b = dict(batch, gauss=torch.cat([gauss[..., :3] * k, gauss[..., 3:6], gauss[..., 6:7] * k], -1), L=L * k,
                     pos=batch["pos"] * k, vel=batch["vel"] * k)
            torch.manual_seed(1)
            outs.append(m.loss(b, physics=True, overshoot=8)[1])
        diff = max(abs(outs[0][k] - outs[1][k]) / max(1.0, abs(outs[0][k])) for k in outs[0] if not k.startswith("err_"))
        print("scale x37: max relative difference of the normalized losses", f"{diff:.2e}",
              "| metric err_img_pos ratio", round(outs[1]["err_img_pos"] / outs[0]["err_img_pos"], 3))
