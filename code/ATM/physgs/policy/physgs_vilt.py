"""
ATM's ViLT policy with the track forecast replaced by the object-RSSM forecast.

Everything except the guidance input is ATM's BCViLTPolicy (image patches, extra states, spatial /
temporal transformers, policy head), so the comparison with ATM isolates the representation:
    ATM   : predicted 2D point tracks  -> 64 track tokens (32 points x 2 views, 16 steps each)
            + the flattened tracks into the action head (2 x 16 x 32 x 2 = 2,048 numbers)
    PhysGS: predicted object states    -> one token per object per forecast step (head_input="flat":
            every step, 6 objects x 16 steps = 96 tokens on LIBERO-Spatial)
            + the whole flattened forecast into the action head (6 x 16 x 146 = 14,016 numbers)
            (head_input="pool", horizon_stride=4 is the v1 design: 4 steps, a 128-d pooled summary to the head)
Forecast input: (b, t, H, O, F) where F = [pos, vel, angvel, contact, z, belief mu, belief logstd,
is_robot, present] for each of O objects at H future steps (from physgs.engine.precompute_forecast).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange

from atm.policy.vilt import BCViLTPolicy
from atm.policy.vilt_modules.policy_head import *  # noqa: F401,F403 (policy head classes are looked up by name)

DYN_DIM = 10  # default: forecast features 0:10 = pos (3), vel (3), angvel (3), contact (1); joint WM of 10-31: 6 (pos, vel)


class GripperClsHead(DeterministicHead):  # noqa: F405 (from ATM's policy_head)
    """
    DeterministicHead whose last output is a gripper logit: the arm (all but the last dim) is regressed with MSE as
    before, the gripper command (+1 close / -1 open) is classified with binary cross-entropy, and the executed command
    is its sign. Regressing a binary command averages the two values near a switch (hesitant, late or early flips).
    Padding frames (action 0) get the target 0.5, as the MSE head is pulled to 0 there.
    """

    def get_action(self, x):
        y = self.forward(x)
        return torch.cat([y[..., :-1], torch.where(y[..., -1:] > 0, 1.0, -1.0).to(y.dtype)], dim=-1)

    def loss_fn(self, act, target, reduction="mean"):
        arm = F.mse_loss(act[..., :-1], target[..., :-1], reduction="none")
        grip = F.binary_cross_entropy_with_logits(act[..., -1:], (target[..., -1:] + 1) / 2, reduction="none")
        loss = torch.cat([arm, grip], dim=-1)
        return (loss.mean() if reduction == "mean" else loss) * self.loss_coef


class PhysGSViLTPolicy(BCViLTPolicy):
    def __init__(self, obs_cfg, img_encoder_cfg, language_encoder_cfg, extra_state_encoder_cfg, forecast_cfg,
                 spatial_transformer_cfg, temporal_transformer_cfg, policy_head_cfg, load_path=None):
        # BCViLTPolicy.__init__ calls self._setup_track(**track_cfg); route our forecast config through it.
        super().__init__(obs_cfg, img_encoder_cfg, language_encoder_cfg, extra_state_encoder_cfg, forecast_cfg,
                         spatial_transformer_cfg, temporal_transformer_cfg, policy_head_cfg, load_path=None)
        if load_path is not None:
            self.load(load_path)

    # ------------------------------------------------------------------ setup
    def _process_obs_shapes(self, obs_shapes, num_views, extra_states, img_mean, img_std, max_seq_len):
        obs_shapes = dict(obs_shapes)
        obs_shapes.setdefault("tracks", [1, 1, 2])  # unused by this policy; keeps the parent happy
        super()._process_obs_shapes(obs_shapes, num_views, extra_states, img_mean, img_std, max_seq_len)

    def _setup_track(self, num_objects, horizon, feat_dim, horizon_stride=4, pool_dim=128, use_zero_forecast=False,
                     head_input="pool", object_slots=None, grad_checkpoint=False, forecast_dropout=0.0,
                     include_now=False, token="step", head_norm=True, dyn_dim=DYN_DIM):
        """
        num_objects: object slots in the forecast arrays; object_slots: how many of them to use (the first ones;
        padding comes last), None = all. horizon: imagined steps; include_now: the arrays also hold the current state
        as step 0 (physgs.engine.precompute_forecast --include_now), as ATM's tracks start at the current point
        positions. horizon_stride: use every k-th forecast step.
        token: "step" (one token per object and step) or "trajectory" (ATM's track tokens: one token per object, a
        single linear map of its whole forecast trajectory, as ATM's TrackPatchEmbed with the patch = the whole track).
        head_input for the action head: "pool" (a pool_dim attention summary of the forecast tokens), "flat" (the whole
        forecast) or "dyn" (the forecast dynamics pos, vel, angvel, contact of every object and step, as ATM feeds its
        flattened tracks; shape latents and beliefs reach the policy only through the forecast tokens); head_norm:
        layer-normalise that flat input (ATM feeds its tracks raw).
        forecast_dropout: in training, the whole forecast of a sample is zeroed with this probability, so the policy
        also learns from the images and cannot hinge on the forecast alone.
        grad_checkpoint: recompute the spatial transformer's layers in the backward pass instead of storing their
        activations (memory only; the computed values are identical).
        """
        self.track = nn.Identity()  # the parent's train()/eval() touch self.track
        self.grad_checkpoint = grad_checkpoint
        self.object_slots = object_slots or num_objects
        self.num_objects, self.horizon, self.feat_dim = self.object_slots, horizon, feat_dim
        self.include_now = include_now
        steps = horizon + int(include_now)
        self.h_idx = list(range(horizon_stride - 1, steps, horizon_stride))  # stride 4: steps 4, 8, 12, 16
        self.use_zero_forecast = use_zero_forecast
        self.forecast_dropout = forecast_dropout
        self.head_input, self.token, self.head_norm = head_input, token, head_norm
        d = self.spatial_embed_size
        if token == "trajectory":
            assert head_input in ("flat", "dyn"), "trajectory tokens are used with a flat / dyn head input (as ATM)"
            self.forecast_traj = nn.Linear(len(self.h_idx) * feat_dim, d)
            self.num_forecast_tokens = self.num_objects
        elif token == "step":
            self.forecast_proj = nn.Sequential(nn.Linear(feat_dim, d), nn.GELU(), nn.Linear(d, d))
            self.forecast_step_embed = nn.Parameter(torch.randn(1, len(self.h_idx), 1, d) * 0.02)
            self.num_forecast_tokens = len(self.h_idx) * self.num_objects
        else:
            raise ValueError(token)
        if head_input == "pool":
            self.forecast_pool_q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            self.forecast_pool = nn.MultiheadAttention(d, 4, batch_first=True)
            self.forecast_pool_out = nn.Linear(d, pool_dim)
            self.head_dim = pool_dim
        elif head_input in ("flat", "dyn"):
            self.dyn_dim = dyn_dim  # leading forecast features read by head_input="dyn"
            self.head_dim = len(self.h_idx) * self.num_objects * (feat_dim if head_input == "flat" else dyn_dim)
            self.forecast_flat_norm = nn.LayerNorm(self.head_dim) if head_norm else nn.Identity()
        else:
            raise ValueError(head_input)

    def _setup_spatial_positional_embeddings(self):
        d = self.spatial_embed_size
        self.register_parameter("spatial_token", nn.Parameter(torch.randn(1, 1, d)))
        self.register_parameter("img_patch_pos_embed", nn.Parameter(torch.randn(1, self.img_num_patches, d)))
        # image views, forecast, sentence
        self.register_parameter("modality_embed", nn.Parameter(torch.randn(1, len(self.image_encoders) + 2, d)))
        modality_idx = []
        for i, enc in enumerate(self.image_encoders):
            modality_idx += [i] * enc.num_patches
        modality_idx += [len(self.image_encoders)] * self.num_forecast_tokens
        modality_idx += [len(self.image_encoders) + 1]
        self.modality_idx = torch.LongTensor(modality_idx)

    def _setup_policy_head(self, network_name, **policy_head_kwargs):
        policy_head_kwargs["input_size"] = self.temporal_embed_size + self.head_dim
        self.act_shape = policy_head_kwargs["output_size"]
        self.out_shape = np.prod(self.act_shape)
        policy_head_kwargs["output_size"] = self.out_shape
        self.policy_head = eval(network_name)(**policy_head_kwargs)

    # ------------------------------------------------------------------ encoding
    def forecast_encode(self, forecast):
        """forecast: (b, t, H, O, F) -> tokens (b, t, H'*O', d) and the action-head input (b, t, head_dim)."""
        f = torch.nan_to_num(forecast[:, :, self.h_idx, :self.object_slots])
        if self.use_zero_forecast:
            f = torch.zeros_like(f)
        elif self.training and self.forecast_dropout > 0:  # whole forecast of a sample, all its frames
            keep = (torch.rand(f.shape[0], device=f.device) >= self.forecast_dropout).to(f.dtype)
            f = f * keep.view(-1, 1, 1, 1, 1)
        present = f[..., -1:] > 0.5
        b, t = f.shape[:2]
        if self.token == "trajectory":
            # one token per object from its whole trajectory; objects absent at every step give a zero token
            tok = self.forecast_traj(rearrange(f * present, "b t h o f -> b t o (h f)")) * present.any(2)
        else:
            tok = self.forecast_proj(f) * present + self.forecast_step_embed[:, None]
            tok = rearrange(tok, "b t h o d -> b t (h o) d")
        if self.head_input in ("flat", "dyn"):
            g = f if self.head_input == "flat" else f[..., :self.dyn_dim]
            head = self.forecast_flat_norm(rearrange(g * present, "b t h o f -> b t (h o f)"))
            return tok, head
        flat = rearrange(tok, "b t n d -> (b t) n d")
        mask = rearrange(~present[..., 0], "b t h o -> (b t) (h o)")
        mask[:, 0] = False  # never mask every key (robot token comes first)
        pooled, _ = self.forecast_pool(self.forecast_pool_q.expand(flat.shape[0], -1, -1), flat, flat,
                                       key_padding_mask=mask, need_weights=False)
        pooled = self.forecast_pool_out(pooled[:, 0]).view(b, t, -1)
        return tok, pooled

    def _spatial_transformer(self, x):
        """ATM's spatial transformer (unmasked), optionally recomputing each layer in the backward pass."""
        st = self.spatial_transformer
        if not (self.grad_checkpoint and self.training and torch.is_grad_enabled()) or st.mask is not None:
            return st(x)

        def layer(x, att_norm, att, ff_norm, ff):
            x = x + att(att_norm(x))
            return x + st.drop_path(ff(ff_norm(x)))

        for mods in st.layers:
            x = torch.utils.checkpoint.checkpoint(layer, x, *mods, use_reentrant=False)
        return x

    def spatial_encode(self, obs, forecast, task_emb, extra_states, return_recon=False):
        img_encoded = []
        for view_idx in range(self.num_views):
            x = obs[:, view_idx]
            b, t = x.shape[:2]
            e = self.image_encoders[view_idx](rearrange(x, "b t c h w -> (b t) c h w"))
            img_encoded.append(rearrange(e, "(b t) c h w -> b t (h w) c", b=b, t=t))
        img_encoded = torch.cat(img_encoded, -2) + self.img_patch_pos_embed.unsqueeze(0)
        B, T = img_encoded.shape[:2]

        text_encoded = self.language_encoder_spatial(task_emb).view(B, 1, 1, -1).expand(-1, T, -1, -1)
        fc_tok, fc_pool = self.forecast_encode(forecast)

        if self.spatial_transformer_use_text:
            tokens = torch.cat([img_encoded, fc_tok, text_encoded], -2)
            tokens = tokens + self.modality_embed[None, :, self.modality_idx, :]
        else:
            tokens = torch.cat([img_encoded, fc_tok], -2)
            tokens = tokens + self.modality_embed[None, :, self.modality_idx[:-1], :]
        encoded = torch.cat([self.spatial_token.unsqueeze(0).expand(B, T, -1, -1), tokens], -2)
        out = self._spatial_transformer(rearrange(encoded, "b t n c -> (b t) n c"))[:, 0]
        out = self.spatial_downsample(out).view(B, T, 1, -1)

        extra = None if self.extra_encoder is None else self.extra_encoder(extra_states)
        text_t = self.language_encoder_temporal(task_emb).view(B, 1, 1, -1).expand(-1, T, -1, -1)
        cls = self.action_cls_token.unsqueeze(0).expand(B, T, -1, -1)
        seq = [cls, text_t, out] if self.temporal_transformer_use_text else [cls, out]
        if extra is not None:
            seq.append(extra)
        output = torch.cat(seq, -2)
        return (output, fc_pool) if return_recon else output

    def forward(self, obs, forecast, track, task_emb, extra_states):
        x, fc_pool = self.spatial_encode(obs, forecast, task_emb, extra_states, return_recon=True)
        x = self.temporal_encode(x)
        return self.policy_head(torch.cat([x, fc_pool], dim=-1))

    def forward_vis(self, *args, **kwargs):
        return None, {}

    # ------------------------------------------------------------------ rollout
    def act(self, obs, task_emb, extra_states, forecast):
        """obs: (b, v, h, w, c) uint8; forecast: (b, H, O, F) for the current step."""
        self.eval()
        B = obs.shape[0]
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        obs = torch.as_tensor(rearrange(obs, "b v h w c -> b v 1 c h w").copy(), device=device, dtype=dtype)
        if obs.shape[-2:] != tuple(self.obs_shapes["rgb"][-2:]):
            obs = rearrange(obs, "b v t c h w -> (b v t) c h w")
            obs = F.interpolate(obs, size=self.obs_shapes["rgb"][-2:], mode="bilinear", align_corners=False)
            obs = rearrange(obs, "(b v t) c h w -> b v t c h w", b=B, v=self.num_views)
        task_emb = torch.as_tensor(task_emb, device=device, dtype=dtype)
        extra_states = {k: torch.as_tensor(v, device=device, dtype=dtype)[:, None] for k, v in extra_states.items()}
        forecast = torch.as_tensor(forecast, device=device, dtype=dtype)[:, None]
        obs = self._preprocess_rgb(obs)
        with torch.no_grad():
            x, fc_pool = self.spatial_encode(obs, forecast, task_emb, extra_states, return_recon=True)
            self.latent_queue.append((x, fc_pool))
            xs = torch.cat([a for a, _ in self.latent_queue], dim=1)
            h = self.temporal_encode(xs)
            action = self.policy_head.get_action(torch.cat([h[:, -1], fc_pool[:, -1]], dim=-1))
        action = torch.clamp(action.reshape(-1, *self.act_shape), -1, 1)
        return action.float().cpu().numpy(), None
