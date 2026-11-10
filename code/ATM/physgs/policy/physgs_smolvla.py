"""
smolVLA as the backbone of the PhysGO policy (2026-11-07: the whole policy network, with the
imagination entering the policy the same way). Option of physgs.engine.train_policy (backbone=smolvla); the default
backbone stays PhysGSViLTPolicy (physgs/policy/physgs_vilt.py), unchanged.

The network is lerobot/smolvla_base (SmolVLM2 vision-language model + action expert, flow matching, action chunks), with
the forecast entering through the same two paths as in PhysGSViLTPolicy (forecast_cfg token=trajectory, head_input=dyn):
  1. forecast tokens in the perception stream: one token per object, a single linear map of its whole forecast
     trajectory (steps 0..15 x 143 features), zero for an object absent at every step, plus a modality embedding. In the
     ViLT they join the image patches and the sentence token in the (unmasked) spatial transformer; here they join the
     image and language tokens in the VLM prefix, in the same attention block (images, language and forecast attend to
     each other; the state token and the action expert attend to all of them). Scaled by sqrt(width) as smolVLA scales
     its image and language embeddings.
  2. the forecast dynamics straight into the action head: pos and vel (dyn_dim 6) of every object and step, raw
     (head_norm false), flattened. In the ViLT they are concatenated to the policy head input; here the head is the
     action expert, so a linear map of the same vector is added to every action token (zero-initialized: at the start
     the network is exactly smolvla_base).
  forecast_dropout (training: a sample's whole forecast zeroed), include_now, horizon, object_slots: as the ViLT.
forecast_cfg.enabled=false gives plain smolVLA (no forecast input), the comparison without imagination.
Inputs come from physgs.policy.dataset.ForecastBCDataset as for the ViLT: images (b, v, t, c, h, w) uint8 (top, wrist:
camera1, camera2; smolVLA letterboxes to 512 itself), extra_states joint_states (state, already in [-1, 1]),
lang_tokens / lang_mask (the task sentence tokenized with smolVLA's tokenizer, so101_policy_h5.py --tokenize), and with
dataset action_chunk the next chunk_size actions + action_is_pad. smolVLA's own recipe parts: frozen vision encoder,
expert-only training (train_expert_only), 10 flow steps at inference.
"""
import math

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


class PhysGOSmolVLAPolicy(nn.Module):
    def __init__(self, smolvla_cfg, forecast_cfg, act_dim, state_dim, num_views=2, load_path=None, **unused):
        super().__init__()
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        base = SmolVLAPolicy.from_pretrained(smolvla_cfg["pretrained"])
        c = base.config
        c.chunk_size = c.n_action_steps = int(smolvla_cfg["chunk_size"])
        img = tuple(smolvla_cfg.get("image_shape", [3, 256, 256]))
        c.input_features = {f"observation.images.camera{i + 1}": PolicyFeature(type=FeatureType.VISUAL, shape=img)
                            for i in range(num_views)}
        c.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))
        c.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(act_dim,))}
        for k in ("num_steps", "freeze_vision_encoder", "train_expert_only"):
            if k in smolvla_cfg:
                setattr(c, k, smolvla_cfg[k])
        self.policy, self.vla = base, base.model
        # (2026-11-08) freezing is applied once when the model is built from the checkpoint's config (expert only,
        # frozen vision encoder): re-apply smolVLA's own set_requires_grad with the requested settings, so the VLM
        # (and optionally the vision encoder) can be fine-tuned too; vlm_lr_scale gives the pretrained VLM weights a
        # smaller learning rate (optim_groups, used by train_policy)
        ve = self.vla.vlm_with_expert
        if ve.train_expert_only != c.train_expert_only or ve.freeze_vision_encoder != c.freeze_vision_encoder:
            ve.train_expert_only, ve.freeze_vision_encoder = c.train_expert_only, c.freeze_vision_encoder
            for p in ve.vlm.parameters():
                p.requires_grad = True
            ve.set_requires_grad()
        self.vlm_lr_scale = float(smolvla_cfg.get("vlm_lr_scale", 1.0))
        sm, ss = smolvla_cfg.get("state_mean"), smolvla_cfg.get("state_std")
        self.state_mean = torch.tensor(sm, dtype=torch.float32) if sm is not None else None
        self.state_std = torch.tensor(ss, dtype=torch.float32) if ss is not None else None
        self.num_views, self.act_dim, self.chunk = num_views, act_dim, c.chunk_size
        self.fc = dict(forecast_cfg)
        self.enabled = self.fc.get("enabled", True)
        if self.enabled:
            self._setup_forecast(**{k: v for k, v in self.fc.items() if k != "enabled"})
            self._patch()
        if load_path is not None:
            self.load(load_path)

    # ------------------------------------------------------------------ forecast (same interface as the ViLT)
    def _setup_forecast(self, num_objects, horizon, feat_dim, horizon_stride=1, include_now=True, token="trajectory",
                        head_input="dyn", head_norm=False, dyn_dim=6, forecast_dropout=0.0, object_slots=None,
                        use_zero_forecast=False, **unused):
        assert token == "trajectory" and head_input == "dyn", "smolvla backbone: the default interface (trajectory, dyn)"
        self.object_slots = object_slots or num_objects
        steps = horizon + int(include_now)
        self.h_idx = list(range(horizon_stride - 1, steps, horizon_stride))
        self.dyn_dim, self.forecast_dropout, self.use_zero_forecast = dyn_dim, forecast_dropout, use_zero_forecast
        width = self.vla.vlm_with_expert.config.text_config.hidden_size
        expert = self.vla.vlm_with_expert.expert_hidden_size
        self.width = width
        self.forecast_traj = nn.Linear(len(self.h_idx) * feat_dim, width)
        self.forecast_modality = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        head_dim = len(self.h_idx) * self.object_slots * dyn_dim
        self.forecast_flat_norm = nn.LayerNorm(head_dim) if head_norm else nn.Identity()
        self.forecast_head = nn.Linear(head_dim, expert)
        nn.init.zeros_(self.forecast_head.weight)
        nn.init.zeros_(self.forecast_head.bias)

    def forecast_encode(self, forecast):
        """forecast (b, H, O, F) -> tokens (b, O', width) and the flat dynamics (b, head_dim); as PhysGSViLTPolicy."""
        f = torch.nan_to_num(forecast[:, self.h_idx, :self.object_slots]).float()
        if self.use_zero_forecast:
            f = torch.zeros_like(f)
        elif self.training and self.forecast_dropout > 0:
            keep = (torch.rand(f.shape[0], device=f.device) >= self.forecast_dropout).to(f.dtype)
            f = f * keep.view(-1, 1, 1, 1)
        present = f[..., -1:] > 0.5
        tok = self.forecast_traj(rearrange(f * present, "b h o f -> b o (h f)")) * present.any(1)
        tok = (tok + self.forecast_modality) * math.sqrt(self.width)
        head = self.forecast_flat_norm(rearrange((f * present)[..., :self.dyn_dim], "b h o f -> b (h o f)"))
        return tok, head

    def _patch(self):
        vla, orig_prefix, orig_suffix = self.vla, self.vla.embed_prefix, self.vla.embed_suffix
        assert vla.prefix_length <= 0, "forecast tokens are inserted before the state token (no prefix padding)"

        def embed_prefix(images, img_masks, lang_tokens, lang_masks, state=None):
            embs, pad, att = orig_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
            tok = self._cur[0].to(embs.dtype)
            b, n = tok.shape[:2]
            one = torch.ones(b, n, dtype=pad.dtype, device=pad.device)
            zero = torch.zeros(b, n, dtype=att.dtype, device=att.device)   # same block as images and language
            return (torch.cat([embs[:, :-1], tok, embs[:, -1:]], 1), torch.cat([pad[:, :-1], one, pad[:, -1:]], 1),
                    torch.cat([att[:, :-1], zero, att[:, -1:]], 1))

        def embed_suffix(noisy_actions, timestep):
            embs, pad, att = orig_suffix(noisy_actions, timestep)
            return embs + self.forecast_head(self._cur[1]).to(embs.dtype)[:, None], pad, att

        vla.embed_prefix, vla.embed_suffix = embed_prefix, embed_suffix

    # ------------------------------------------------------------------ batches
    def _batch(self, obs, extra_states, forecast):
        """obs (b, v, t, c, h, w) uint8 / float in 0..255; extra (b, t, .); forecast (b, t, H, O, F) -> smolVLA batch."""
        batch = {f"observation.images.camera{i + 1}": obs[:, i, -1].float() / 255.0 for i in range(self.num_views)}
        # the 6-D SO-101 state from ATM's extra-state layout: 5 arm joints (joint_states, padded to 7) + gripper
        batch["observation.state"] = torch.cat([extra_states["joint_states"][:, -1, :5],
                                                extra_states["gripper_states"][:, -1, :1]], -1).float()
        if self.state_mean is not None:   # LeRobot's MEAN_STD normalization of the state (repo setting, 2026-11-08)
            batch["observation.state"] = (batch["observation.state"] - self.state_mean.to(batch["observation.state"].device)) \
                / self.state_std.to(batch["observation.state"].device)
        batch["observation.language.tokens"] = extra_states["lang_tokens"][:, -1].long()
        batch["observation.language.attention_mask"] = extra_states["lang_mask"][:, -1].bool()
        if self.enabled:
            self._cur = self.forecast_encode(forecast[:, -1])
        return batch

    def forward_loss(self, obs, forecast, track, task_emb, extra_states, action):
        batch = self._batch(obs, extra_states, forecast)
        batch["action"] = action.float()
        if "action_is_pad" in extra_states:
            batch["action_is_pad"] = extra_states["action_is_pad"].bool()
        loss, logs = self.policy.forward(batch)
        return loss, {"bc_loss": loss.item(), "loss": loss.item()}

    @torch.no_grad()
    def predict_chunk(self, obs, extra_states, forecast, noise=None):
        """obs (b, v, h, w, c) uint8 (as PhysGSViLTPolicy.act); extra_states {k: (b, d)}; forecast (b, H, O, F)
        -> (b, chunk, act_dim) actions in the dataset's scale."""
        self.eval()
        dev = next(self.parameters()).device
        o = torch.as_tensor(np.ascontiguousarray(obs), device=dev)
        o = rearrange(o, "b v h w c -> b v 1 c h w")
        ex = {k: torch.as_tensor(v, device=dev)[:, None] for k, v in extra_states.items()}
        fc = None if forecast is None else torch.as_tensor(forecast, device=dev)[:, None]
        batch = self._batch(o, ex, fc)
        return self.policy.predict_action_chunk(batch, noise=noise).float()

    def act(self, obs, task_emb, extra_states, forecast):
        """Same call as PhysGSViLTPolicy.act; returns the first action of a fresh chunk."""
        a = self.predict_chunk(obs, extra_states, forecast)[:, 0]
        return torch.clamp(a, -1, 1).cpu().numpy(), None

    def optim_groups(self, lr):
        """AdamW parameter groups: the pretrained VLM's trainable weights at lr x vlm_lr_scale, the rest at lr."""
        vlm = {id(p) for p in self.vla.vlm_with_expert.vlm.parameters()}
        train = [p for p in self.parameters() if p.requires_grad]
        g_vlm = [p for p in train if id(p) in vlm]
        g_rest = [p for p in train if id(p) not in vlm]
        return [{"params": g_rest, "lr": lr}] + ([{"params": g_vlm, "lr": lr * self.vlm_lr_scale}] if g_vlm else [])

    def reset(self):
        self.policy.reset()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))
