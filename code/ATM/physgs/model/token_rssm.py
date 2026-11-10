"""
Action-free RSSM over a set of Gaussian tokens.

Each of the K slots has a deterministic state h (GRU) and a stochastic state s. Slots
interact through self-attention before the GRU update (object-object dynamics) and are
conditioned on the task embedding. The encoder's tokens at each frame are an unordered
set from FPS, so the posterior reads them with slot-attention (softmax over slots):
token identity lives in the recurrent state, not in the encoder's ordering.

No action enters the model: it is trained on action-free videos and, at policy time,
imagines how the scene's physical tokens should evolve for the task.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from physgs.model.layers import SelfAttnBlock, SlotReadout, chamfer, gaussian_kl, mlp, split_gaussian


class TokenRSSM(nn.Module):
    def __init__(self, token_dim=3 + 64, num_slots=64, deter=256, stoch=32, dim=192, heads=6, task_dim=768,
                 free_nats=1.0, kl_balance=0.8, anchor_weight=10.0):
        super().__init__()
        self.K, self.deter, self.stoch = num_slots, deter, stoch
        self.free_nats, self.kl_balance, self.anchor_weight = free_nats, kl_balance, anchor_weight

        self.init_h = nn.Parameter(torch.randn(num_slots, deter) * 0.1)  # distinct slots break symmetry
        self.task_proj = nn.Linear(task_dim, 2 * dim)                     # FiLM
        self.in_proj = nn.Linear(deter + stoch, dim)
        self.interact = nn.ModuleList([SelfAttnBlock(dim, heads) for _ in range(2)])
        self.gru = nn.GRUCell(dim, deter)
        self.prior = mlp(deter, 2 * dim, 2 * stoch)
        self.read = SlotReadout(deter, token_dim, dim)
        self.post = mlp(deter + dim, 2 * dim, 2 * stoch)
        self.to_token = mlp(deter + stoch, 2 * dim, token_dim)

    def initial(self, B):
        dev = self.init_h.device
        return dict(h=self.init_h[None].expand(B, -1, -1).contiguous(),
                    s=torch.zeros(B, self.K, self.stoch, device=dev))

    def _transition(self, state, task):
        x = self.in_proj(torch.cat([state["h"], state["s"]], dim=-1))
        gamma, beta = self.task_proj(task)[:, None].chunk(2, dim=-1)
        x = x * (1 + gamma) + beta
        for blk in self.interact:
            x = blk(x)
        B = x.shape[0]
        h = self.gru(x.flatten(0, 1), state["h"].flatten(0, 1)).view(B, self.K, self.deter)
        return h

    def img_step(self, state, task, sample=True):
        h = self._transition(state, task)
        mu, logstd = split_gaussian(self.prior(h))
        s = mu + torch.randn_like(mu) * logstd.exp() if sample else mu
        return dict(h=h, s=s, mu=mu, logstd=logstd)

    def obs_step(self, state, obs_tokens, task, first=False):
        """obs_tokens: (B, N, token_dim). On the first frame the prior transition is skipped."""
        h = state["h"] if first else self._transition(state, task)
        p_mu, p_logstd = split_gaussian(self.prior(h))
        read, attn = self.read(h, obs_tokens)
        q_mu, q_logstd = split_gaussian(self.post(torch.cat([h, read], dim=-1)))
        s = q_mu + torch.randn_like(q_mu) * q_logstd.exp()
        post = dict(h=h, s=s, mu=q_mu, logstd=q_logstd, attn=attn)
        prior = dict(h=h, mu=p_mu, logstd=p_logstd)
        return post, prior

    def observe(self, obs_tokens, task):
        """obs_tokens: (B, T, N, token_dim), task: (B, task_dim). Returns lists of posterior / prior states."""
        state = self.initial(obs_tokens.shape[0])
        posts, priors = [], []
        for t in range(obs_tokens.shape[1]):
            state, prior = self.obs_step(state, obs_tokens[:, t], task, first=(t == 0))
            posts.append(state)
            priors.append(prior)
        return posts, priors

    def imagine(self, state, task, horizon, sample=True):
        """Roll the prior forward `horizon` steps from a (posterior) state; returns predicted tokens (B, H, K, token_dim)."""
        tokens = []
        for _ in range(horizon):
            state = self.img_step(state, task, sample=sample)
            tokens.append(self.token(state))
        return torch.stack(tokens, dim=1), state

    def token(self, state):
        return self.to_token(torch.cat([state["h"], state["s"]], dim=-1))

    def token_loss(self, pred, target):
        """Set loss between predicted slots and observed (unordered) tokens; anchors are weighted up."""
        w = torch.ones(pred.shape[-1], device=pred.device)
        w[:3] = self.anchor_weight
        return chamfer(pred * w.sqrt(), target * w.sqrt())

    def kl_loss(self, post, prior):
        kl_dyn = gaussian_kl(post["mu"].detach(), post["logstd"].detach(), prior["mu"], prior["logstd"])
        kl_rep = gaussian_kl(post["mu"], post["logstd"], prior["mu"].detach(), prior["logstd"].detach())
        kl_dyn = kl_dyn.clamp(min=self.free_nats).mean()
        kl_rep = kl_rep.clamp(min=self.free_nats).mean()
        return self.kl_balance * kl_dyn + (1 - self.kl_balance) * kl_rep

    def loss(self, obs_tokens, task, overshoot=8, kl_weight=1.0):
        """
        obs_tokens: (B, T, N, token_dim) from the frozen GS autoencoder; task: (B, task_dim).
        Reconstruction from the posterior, KL, and an open-loop term: from a random filtered state,
        imagine `overshoot` steps and match the observed future (this is what the policy consumes).
        """
        posts, priors = self.observe(obs_tokens, task)
        T = obs_tokens.shape[1]
        rec = torch.stack([self.token_loss(self.token(p), obs_tokens[:, t]) for t, p in enumerate(posts)]).mean()
        kl = torch.stack([self.kl_loss(p, q) for p, q in zip(posts[1:], priors[1:])]).mean()

        open_loop = torch.zeros((), device=obs_tokens.device)
        if overshoot > 0 and T > 1:
            t0 = torch.randint(0, max(1, T - overshoot), ()).item()
            H = min(overshoot, T - 1 - t0)
            pred, _ = self.imagine({k: posts[t0][k] for k in ("h", "s")}, task, H)
            open_loop = torch.stack([self.token_loss(pred[:, i], obs_tokens[:, t0 + 1 + i]) for i in range(H)]).mean()

        total = rec + kl_weight * kl + open_loop
        return total, dict(rec=rec.item(), kl=kl.item(), open_loop=open_loop.item())

    @torch.no_grad()
    def forecast(self, obs_tokens, task, horizon, num_samples=1):
        """
        Policy-time hint: filter the observed history, then imagine `horizon` future token sets.
        obs_tokens: (B, T_hist, N, token_dim) -> (B, num_samples, horizon, K, token_dim).
        """
        posts, _ = self.observe(obs_tokens, task)
        last = {k: posts[-1][k] for k in ("h", "s")}
        outs = []
        for i in range(num_samples):
            pred, _ = self.imagine(last, task, horizon, sample=num_samples > 1)
            outs.append(pred)
        return torch.stack(outs, dim=1)


if __name__ == "__main__":
    # Shape smoke test: python -m physgs.model.token_rssm
    B, T, N, D = 2, 10, 64, 3 + 64
    rssm = TokenRSSM(token_dim=D, num_slots=N)
    tokens, task = torch.randn(B, T, N, D), torch.randn(B, 768)
    total, logs = rssm.loss(tokens, task)
    total.backward()
    print("loss", logs)
    print("forecast", tuple(rssm.forecast(tokens[:, :3], task, horizon=16, num_samples=2).shape))
    print("params (M):", sum(p.numel() for p in rssm.parameters()) / 1e6)
