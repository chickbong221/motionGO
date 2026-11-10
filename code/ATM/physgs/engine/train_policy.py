"""
Train PhysGSViLTPolicy with BC (single GPU). Same optimiser / schedule / epochs as ATM's train_bc.

    python -m physgs.engine.train_policy train_dataset="[...]" val_dataset="[...]" forecast_dir=... \
        model_cfg.forecast_cfg.num_objects=12 model_cfg.forecast_cfg.feat_dim=146 seed=0
"""
import math
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from physgs.engine.common import Logger, h100_attention_fix
from physgs.policy.dataset import ForecastBCDataset
# SO-101 version (2026-11-07): backbone option (default vilt = the original PhysGSViLTPolicy, imported when used, so the
# smolvla backbone runs in an env without ATM's robomimic dependency); optional per-step lr warm-up and AdamW betas
# (defaults: none / torch's, i.e. the original optimiser).


class EMA:
    """Exponential moving average of the weights (float parameters and buffers; integer buffers are copied)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.state[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.state[k].copy_(v)

    def swap(self, model):  # model <-> averaged weights (call twice to restore)
        cur = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.state)
        self.state = cur


def run_epoch(model, loader, opt=None, clip=100.0, bf16=True, grad_accum=1, ema=None, gpu_aug=None, on_step=None):
    """grad_accum > 1 sums gradients over that many batches per update, e.g. 4 x 128 to match ATM's 4-GPU batch.
    gpu_aug: ATM's image augmentation applied here on the GPU (physgs.policy.gpu_aug) instead of in the loader."""
    train = opt is not None
    model.train(train)
    if train:  # an epoch's last micro-batches that do not fill an update are dropped (as drop_last does for items)
        opt.zero_grad(set_to_none=True)
    tot, n = {}, 0
    for obs, forecast, track, task_emb, action, extra in tqdm(loader, leave=False):
        obs, forecast, track, task_emb, action = (x.cuda(non_blocking=True) for x in (obs, forecast, track, task_emb, action))
        extra = {k: v.cuda(non_blocking=True) for k, v in extra.items()}
        if train and gpu_aug is not None:
            obs = gpu_aug(obs)
        with torch.set_grad_enabled(train), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            loss, logs = model.forward_loss(obs, forecast, track, task_emb, extra, action)
        if train:
            (loss / grad_accum).backward()
            if (n + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                if on_step is not None:
                    on_step()
        for k, v in logs.items():
            tot[k] = tot.get(k, 0.0) + v
        n += 1
    return {k: v / max(n, 1) for k, v in tot.items()}


def main():
    cfg = OmegaConf.merge(OmegaConf.load(os.path.join(os.path.dirname(__file__), "..", "conf", "policy.yaml")),
                          OmegaConf.from_cli(sys.argv[1:]))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    h100_attention_fix()
    if cfg.get("tf32", False):  # TF32 tensor cores for fp32 matmuls / convolutions (H100: ~2x, negligible numerics)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(cfg.out_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.out_dir, "config.yaml"))

    dcfg = OmegaConf.to_container(cfg.dataset_cfg, resolve=True)
    dcfg["object_slots"] = cfg.model_cfg.forecast_cfg.get("object_slots")  # the policy reads only these slots
    img = [cfg.img_size] * 2 if isinstance(cfg.img_size, int) else list(cfg.img_size)
    gpu_aug = None
    if cfg.get("gpu_aug", False):
        from physgs.policy.gpu_aug import GPUAugment
        # ATM shifts its 128-px images by up to 8 px: the same fraction of the image at other sizes (256 px: 16 px)
        shift = cfg.get("aug_translation") or max(1, round(8 * img[0] / 128))
        print(f"GPU augmentation: colour jitter 0.3, image shift {shift} px at {img[0]} px", flush=True)
        gpu_aug = GPUAugment(img, cfg.aug_prob, translation=shift)
    train = ForecastBCDataset(cfg.forecast_dir, dataset_dir=list(cfg.train_dataset),
                              aug_prob=0.0 if gpu_aug else cfg.aug_prob, **dcfg)
    val = ForecastBCDataset(cfg.forecast_dir, dataset_dir=list(cfg.val_dataset), aug_prob=0.0, **dcfg)
    tl = DataLoader(train, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True,
                    pin_memory=True, persistent_workers=cfg.num_workers > 0)
    vl = DataLoader(val, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    backbone = cfg.get("backbone", "vilt")
    if backbone == "vilt":
        from physgs.policy.physgs_vilt import PhysGSViLTPolicy
        model = PhysGSViLTPolicy(**OmegaConf.to_container(cfg.model_cfg, resolve=True)).cuda()
    elif backbone == "smolvla":
        from physgs.policy.physgs_smolvla import PhysGOSmolVLAPolicy
        mc = OmegaConf.to_container(cfg.model_cfg, resolve=True)
        model = PhysGOSmolVLAPolicy(OmegaConf.to_container(cfg.smolvla_cfg, resolve=True), mc["forecast_cfg"],
                                    act_dim=mc["policy_head_cfg"]["output_size"][0], state_dim=cfg.smolvla_cfg.state_dim,
                                    num_views=mc["obs_cfg"]["num_views"], load_path=mc.get("load_path")).cuda()
    else:
        raise ValueError(f"backbone {backbone}")
    print(f"policy ({backbone}) params (M): {sum(p.numel() for p in model.parameters()) / 1e6:.1f}, trainable "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}", flush=True)
    betas = tuple(cfg.betas) if cfg.get("betas") else (0.9, 0.999)
    # a backbone may ask for parameter groups (smolvla: a smaller lr for the pretrained VLM); the default is the original
    groups = model.optim_groups(cfg.lr) if hasattr(model, "optim_groups") else [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(groups, lr=cfg.lr, betas=betas, weight_decay=cfg.get("weight_decay", 1e-4))
    opt.zero_grad(set_to_none=True)
    warm = int(cfg.get("warmup_steps", 0))
    if cfg.get("sched") == "smolvla":
        # LeRobot's CosineDecayWithWarmupSchedulerConfig, the smolVLA preset (2026-11-08, the repo's setting):
        # linear warm-up from lr / (warmup + 1) to lr over warmup_steps, then cosine decay to decay_lr over decay_steps
        dec, alpha = int(cfg.decay_steps), float(cfg.decay_lr) / cfg.lr

        def lam(u):
            if u < warm:
                return (1 / (warm + 1) - 1) * (1 - u / warm) + 1 if u > 0 else 1 / (warm + 1)
            return (1 - alpha) * 0.5 * (1 + math.cos(math.pi * min(u, dec) / dec)) + alpha
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
        on_step = sched.step
        warm = max(warm, 1)   # per-update schedule: skip the epoch-level step below
    elif warm:
        # one explicit per-update schedule: linear warm-up x the same per-epoch cosine (to 0 over cfg.epochs) as the
        # original CosineAnnealingLR. (2026-11-07: overwriting the lr under CosineAnnealingLR collapsed it, because its
        # epoch update is computed from the current lr.)
        spe = max(1, len(tl) // cfg.grad_accum)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda u: min(1.0, (u + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(u // spe, cfg.epochs) / cfg.epochs)))
        on_step = sched.step
    else:  # the original schedule
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=0.0)
        on_step = None
    # ema > 0: validation and every saved checkpoint use the averaged weights (so evaluation needs no change)
    ema = EMA(model, cfg.ema) if cfg.get("ema", 0) else None
    logger = Logger(cfg.out_dir)
    best = float("inf")
    for epoch in range(cfg.epochs):
        tr = run_epoch(model, tl, opt, cfg.clip_grad, cfg.mix_precision, cfg.grad_accum, ema, gpu_aug, on_step)
        if not warm:
            sched.step()
        logger.log(epoch, **{f"train/{k}": v for k, v in tr.items()}, lr=opt.param_groups[0]["lr"])
        if ema is not None:
            ema.swap(model)
        if epoch % cfg.val_freq == 0:
            with torch.no_grad():
                va = run_epoch(model, vl, None, bf16=cfg.mix_precision)
            logger.log(epoch, **{f"val/{k}": v for k, v in va.items()})
            print(f"epoch {epoch}: train {tr['loss']:.4f} val {va['loss']:.4f}", flush=True)
            if va["loss"] < best:
                best = va["loss"]
                model.save(os.path.join(cfg.out_dir, "model_best.ckpt"))
        keep = cfg.get("keep_every", 0)  # extra checkpoints for selecting one by rollout success
        if epoch % cfg.save_freq == 0 or (keep and epoch % keep == 0):
            model.save(os.path.join(cfg.out_dir, f"model_{epoch}.ckpt"))
        if ema is not None:
            ema.swap(model)  # back to the training weights
    if ema is not None:
        ema.swap(model)
    model.save(os.path.join(cfg.out_dir, "model_final.ckpt"))


if __name__ == "__main__":
    main()
