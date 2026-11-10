"""
Action-free object RSSM.

One slot per object; identity comes from the instance id, so slot k always tracks object k.
Each slot has a deterministic GRU state h and a Gaussian stochastic state s. Before every
GRU update, slots exchange information through masked self-attention (object interactions),
conditioned on the task embedding (FiLM). Invisible objects fall back to the prior; padded
objects are masked out everywhere.

Heads on [h, s]:  next physical state (pos, vel, angvel, contact) and shape latent z.
Head on h only:   property belief (mean, log-std of log-mass, friction, restitution),
                  so the estimate can sharpen as interaction history accumulates.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from physgs.model.layers import SelfAttnBlock, gaussian_kl, mlp, split_gaussian
from physgs.model.object_encoder import PROP_DIM, TOKEN_SLICES, gaussian_nll, masked_mean

PRED_KEYS = ("pos", "vel", "angvel", "contact", "z")


class ObjectRSSM(nn.Module):
    def __init__(self, token_dim=16 + 128, latent_dim=128, deter=512, stoch=32, dim=384, heads=8, layers=3,
                 task_dim=768, free_nats=1.0, kl_balance=0.8, w=None, dropout=0.0):
        super().__init__()
        self.deter, self.stoch, self.latent_dim = deter, stoch, latent_dim
        self.free_nats, self.kl_balance = free_nats, kl_balance
        self.w = dict(pos=10.0, vel=1.0, angvel=0.1, contact=0.2, z=1.0, prop=0.1, kl=1.0)
        self.w.update(w or {})

        self.h0 = nn.Parameter(torch.zeros(deter))
        self.type_embed = nn.Embedding(2, dim)            # 0 = object, 1 = robot / effector
        self.obs_embed = mlp(token_dim, dim, dim)
        self.in_proj = nn.Linear(deter + stoch, dim)
        self.task_proj = nn.Linear(task_dim, 2 * dim)
        self.interact = nn.ModuleList([SelfAttnBlock(dim, heads, dropout=dropout) for _ in range(layers)])
        self.gru = nn.GRUCell(dim, deter)
        self.prior = mlp(deter, dim, 2 * stoch)
        self.post = mlp(deter + dim, dim, 2 * stoch)
        self.state_head = mlp(deter + stoch, dim, 3 + 3 + 1)      # vel, angvel, contact
        self.delta_head = mlp(deter, dim, 3)                        # position change over one step
        self.z_head = mlp(deter + stoch, dim, latent_dim)
        self.belief_head = mlp(deter, dim, 2 * PROP_DIM)

    # ---------------------------------------------------------------- core steps
    # Positions are carried explicitly in the state: each step predicts a displacement that is added to the
    # previous position (a physics prior; regressing absolute positions from h was much worse at short horizons).
    # When an object is observed, the posterior position is the observed one.
    # `seen` marks objects observed at least once: before its first observation an object has no position (it stays
    # at 0 and is excluded from the position losses, the evaluation and the policy's forecast).
    def initial(self, B, O, device):
        return dict(h=self.h0.expand(B, O, -1).contiguous(), s=torch.zeros(B, O, self.stoch, device=device),
                    pos=torch.zeros(B, O, 3, device=device), seen=torch.zeros(B, O, dtype=torch.bool, device=device))

    def _transition(self, state, ctx):
        B, O, _ = state["h"].shape
        x = self.in_proj(torch.cat([state["h"], state["s"]], -1)) + ctx["type"]
        x = x * (1 + ctx["gamma"]) + ctx["beta"]
        for blk in self.interact:
            x = blk(x, key_padding_mask=~ctx["present"])
        with torch.autocast(device_type=x.device.type, enabled=False):  # GRUCell has no bf16 kernel in torch 2.0
            h = self.gru(x.float().flatten(0, 1), state["h"].float().flatten(0, 1))
        return h.view(B, O, self.deter)

    def _sample(self, mu, logstd, sample):
        return mu + torch.randn_like(mu) * logstd.exp() if sample else mu

    def obs_step(self, state, token, visible, ctx, first=False, sample=True):
        h = state["h"] if first else self._transition(state, ctx)
        p_mu, p_ls = split_gaussian(self.prior(h))
        q_mu, q_ls = split_gaussian(self.post(torch.cat([h, self.obs_embed(token)], -1)))
        v = visible[..., None]
        q_mu, q_ls = torch.where(v, q_mu, p_mu), torch.where(v, q_ls, p_ls)  # no observation -> prior
        s = self._sample(q_mu, q_ls, sample)
        obs_pos = token[..., TOKEN_SLICES["pos"]]
        seen_prev = torch.zeros_like(visible) if first else state.get("seen", torch.ones_like(visible))
        pos_prior = torch.zeros_like(obs_pos) if first else state["pos"] + self.delta_head(h)   # one-step prediction
        pos_prior = torch.where(seen_prev[..., None], pos_prior, torch.zeros_like(pos_prior))  # unseen: no position
        pos = torch.where(v, obs_pos, pos_prior)
        return (dict(h=h, s=s, mu=q_mu, logstd=q_ls, pos=pos, pos_prior=pos_prior, seen=seen_prev | visible,
                     seen_prev=seen_prev), dict(mu=p_mu, logstd=p_ls))

    def img_step(self, state, ctx, sample=True):
        h = self._transition(state, ctx)
        mu, ls = split_gaussian(self.prior(h))
        out = dict(h=h, s=self._sample(mu, ls, sample), mu=mu, logstd=ls, pos=state["pos"] + self.delta_head(h))
        if "seen" in state:
            out["seen"] = state["seen"]
            out["pos"] = torch.where(state["seen"][..., None], out["pos"], torch.zeros_like(out["pos"]))
        return out

    def context(self, is_robot, present, task):
        gamma, beta = self.task_proj(task)[:, None].chunk(2, -1)
        return dict(type=self.type_embed(is_robot.long()), present=present, gamma=gamma, beta=beta)

    def heads(self, state):
        feat = torch.cat([state["h"], state["s"]], -1)
        st = self.state_head(feat)
        belief = self.belief_head(state["h"])
        return dict(pos=state["pos"], vel=st[..., 0:3], angvel=st[..., 3:6], contact=st[..., 6],
                    z=self.z_head(feat), belief_mu=belief[..., :PROP_DIM],
                    belief_logstd=belief[..., PROP_DIM:].clamp(-5, 3))

    # ---------------------------------------------------------------- sequences
    def observe(self, tokens, visible, ctx, sample=True):
        """tokens: (B, T, O, D), visible: (B, T, O). Returns per-step posterior and prior lists."""
        B, T, O, _ = tokens.shape
        state = self.initial(B, O, tokens.device)
        posts, priors = [], []
        for t in range(T):
            state, prior = self.obs_step(state, tokens[:, t], visible[:, t], ctx, first=(t == 0), sample=sample)
            posts.append(state)
            priors.append(prior)
        return posts, priors

    def imagine(self, state, ctx, horizon, sample=True):
        states = []
        for _ in range(horizon):
            state = self.img_step(state, ctx, sample=sample)
            states.append(state)
        return states

    # ---------------------------------------------------------------- losses
    def target(self, tokens, labels):
        """Supervision per step: simulator labels where known, otherwise the encoder's token."""
        tgt = {}
        for k in ("pos", "vel", "angvel"):
            lab = labels[k]
            tgt[k] = torch.where(torch.isfinite(lab), lab, tokens[..., TOKEN_SLICES[k]])
        tgt["contact"] = labels["contact"]  # only supervised where the simulator provides it (LIBERO)
        tgt["z"] = tokens[..., TOKEN_SLICES["z"]]
        return tgt

    def pred_loss(self, pred, tgt, mask):
        l = {}
        for k in ("pos", "vel", "angvel"):
            l[k] = masked_mean(F.smooth_l1_loss(pred[k], tgt[k], reduction="none", beta=0.05), mask)
        has_c = torch.isfinite(tgt["contact"])
        l["contact"] = masked_mean(F.binary_cross_entropy_with_logits(
            pred["contact"], torch.nan_to_num(tgt["contact"]).clamp(0, 1), reduction="none"), mask & has_c)
        l["z"] = masked_mean((pred["z"] - tgt["z"]) ** 2, mask)
        return l

    def kl_loss(self, post, prior, mask):
        dyn = gaussian_kl(post["mu"].detach(), post["logstd"].detach(), prior["mu"], prior["logstd"])
        rep = gaussian_kl(post["mu"], post["logstd"], prior["mu"].detach(), prior["logstd"].detach())
        dyn, rep = dyn.clamp(min=self.free_nats), rep.clamp(min=self.free_nats)
        return masked_mean(self.kl_balance * dyn + (1 - self.kl_balance) * rep, mask)

    def loss(self, batch, overshoot=16):
        """
        batch: tokens (B, T, O, D), visible / present (B, T, O) / (B, O), is_robot (B, O), task (B, 768),
        labels pos/vel/angvel (B, T, O, 3), contact (B, T, O), props (B, O, 3); positions are scene-centred.
        """
        tokens, visible, present = batch["tokens"], batch["visible"], batch["present"]
        B, T, O, _ = tokens.shape
        ctx = self.context(batch["is_robot"], present, batch["task"])
        posts, priors = self.observe(tokens, visible, ctx)
        tgt = self.target(tokens, batch)
        pres = present[:, None].expand(B, T, O)
        vis_mask = visible & pres

        logs, total = {}, 0.0  # logs hold tensors until the end: one host sync per step instead of one per term
        # filtered (posterior) reconstruction; the heads run once per step, not once per predicted quantity
        hp = [self.heads(p) for p in posts]
        pred = {k: torch.stack([h[k] for h in hp], 1) for k in PRED_KEYS + ("belief_mu", "belief_logstd")}
        pred["pos"] = torch.stack([p["pos_prior"] for p in posts], 1)  # posterior pos is the observation itself
        rec = self.pred_loss(pred, tgt, vis_mask)
        # the one-step position prediction exists only for objects seen before this step
        seen_prev = torch.stack([p["seen_prev"] for p in posts], 1)
        rec["pos"] = masked_mean(F.smooth_l1_loss(pred["pos"], tgt["pos"], reduction="none", beta=0.05), vis_mask & seen_prev)
        for k, v in rec.items():
            logs[f"rec_{k}"] = v
            total = total + self.w[k] * v
        kl = torch.stack([self.kl_loss(p, q, pres[:, t]) for t, (p, q) in enumerate(zip(posts[1:], priors[1:]), 1)]).mean()
        logs["kl"] = kl
        total = total + self.w["kl"] * kl
        # property belief from history (time-invariant labels)
        props = batch["props"][:, None].expand(B, T, O, PROP_DIM)
        m = torch.isfinite(props) & pres[..., None]
        prop = masked_mean(gaussian_nll(torch.nan_to_num(props), pred["belief_mu"], pred["belief_logstd"]), m)
        logs["belief_nll"] = prop
        total = total + self.w["prop"] * prop
        # open-loop imagination from a random filtered state
        if overshoot > 0 and T > 1:
            t0 = int(torch.randint(0, max(1, T - overshoot), ()))
            H = min(overshoot, T - 1 - t0)
            imag = self.imagine({k: posts[t0][k] for k in ("h", "s", "pos", "seen")}, ctx, H)
            hi = [self.heads(s) for s in imag]
            ipred = {k: torch.stack([h[k] for h in hi], 1) for k in PRED_KEYS}
            itgt = {k: v[:, t0 + 1:t0 + 1 + H] for k, v in tgt.items()}
            # only objects already seen at t0 can be forecast (no jump from the origin to a first observation)
            ol = self.pred_loss(ipred, itgt, vis_mask[:, t0 + 1:t0 + 1 + H] & posts[t0]["seen"][:, None])
            for k, v in ol.items():
                logs[f"ol_{k}"] = v
                total = total + self.w[k] * v
        logs["total"] = total
        vals = torch.stack([v.detach().float() for v in logs.values()]).tolist()
        return total, dict(zip(logs, vals))

    @torch.no_grad()
    def forecast(self, tokens, visible, is_robot, present, task, horizon=16, sample=False):
        """
        Filter the history and imagine `horizon` steps.
        Returns (B, horizon, O, 3+3+3+1+latent+6) = [pos, vel, angvel, contact prob, z, belief mu, belief log-std].
        """
        ctx = self.context(is_robot, present, task)
        posts, _ = self.observe(tokens, visible, ctx, sample=sample)
        states = self.imagine({k: posts[-1][k] for k in ("h", "s", "pos", "seen")}, ctx, horizon, sample=sample)
        belief = self.heads(posts[-1])
        out = []
        for s in states:
            p = self.heads(s)
            out.append(torch.cat([p["pos"], p["vel"], p["angvel"], torch.sigmoid(p["contact"])[..., None], p["z"],
                                  belief["belief_mu"], belief["belief_logstd"]], -1))
        return torch.stack(out, 1)


FORECAST_DIM = lambda latent_dim=128: 10 + latent_dim + 2 * PROP_DIM


if __name__ == "__main__":
    # python -m physgs.model.object_rssm
    B, T, O, D = 4, 24, 10, 16 + 128
    m = ObjectRSSM()
    nan = float("nan")
    batch = dict(tokens=torch.randn(B, T, O, D), visible=torch.rand(B, T, O) > 0.1,
                 present=torch.arange(O)[None].expand(B, O) < 8, is_robot=torch.arange(O)[None].expand(B, O) == 0,
                 task=torch.randn(B, 768), pos=torch.randn(B, T, O, 3), vel=torch.randn(B, T, O, 3),
                 angvel=torch.full((B, T, O, 3), nan), contact=torch.full((B, T, O), nan), props=torch.randn(B, O, 3))
    loss, logs = m.loss(batch)
    loss.backward()
    print({k: round(v, 3) for k, v in logs.items()})
    f = m.forecast(batch["tokens"][:, :8], batch["visible"][:, :8], batch["is_robot"], batch["present"], batch["task"])
    print("forecast", tuple(f.shape), "params (M):", sum(p.numel() for p in m.parameters()) / 1e6)
