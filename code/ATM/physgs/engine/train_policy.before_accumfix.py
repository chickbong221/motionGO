"""
Train PhysGSViLTPolicy with BC (single GPU). Same optimiser / schedule / epochs as ATM's train_bc.

    python -m physgs.engine.train_policy train_dataset="[...]" val_dataset="[...]" forecast_dir=... \
        model_cfg.forecast_cfg.num_objects=12 model_cfg.forecast_cfg.feat_dim=146 seed=0
"""
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from physgs.engine.common import Logger, h100_attention_fix
from physgs.policy.dataset import ForecastBCDataset
from physgs.policy.physgs_vilt import PhysGSViLTPolicy


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


def run_epoch(model, loader, opt=None, clip=100.0, bf16=True, grad_accum=1, ema=None):
    """grad_accum > 1 sums gradients over that many batches per update, e.g. 4 x 128 to match ATM's 4-GPU batch."""
    train = opt is not None
    model.train(train)
    tot, n = {}, 0
    for obs, forecast, track, task_emb, action, extra in tqdm(loader, leave=False):
        obs, forecast, track, task_emb, action = (x.cuda(non_blocking=True) for x in (obs, forecast, track, task_emb, action))
        extra = {k: v.cuda(non_blocking=True) for k, v in extra.items()}
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
    os.makedirs(cfg.out_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.out_dir, "config.yaml"))

    dcfg = OmegaConf.to_container(cfg.dataset_cfg, resolve=True)
    dcfg["object_slots"] = cfg.model_cfg.forecast_cfg.get("object_slots")  # the policy reads only these slots
    train = ForecastBCDataset(cfg.forecast_dir, dataset_dir=list(cfg.train_dataset), aug_prob=cfg.aug_prob, **dcfg)
    val = ForecastBCDataset(cfg.forecast_dir, dataset_dir=list(cfg.val_dataset), aug_prob=0.0, **dcfg)
    tl = DataLoader(train, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True,
                    pin_memory=True, persistent_workers=cfg.num_workers > 0)
    vl = DataLoader(val, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = PhysGSViLTPolicy(**OmegaConf.to_container(cfg.model_cfg, resolve=True)).cuda()
    print("policy params (M):", sum(p.numel() for p in model.parameters()) / 1e6)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.get("weight_decay", 1e-4))
    opt.zero_grad(set_to_none=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=0.0)
    # ema > 0: validation and every saved checkpoint use the averaged weights (so evaluation needs no change)
    ema = EMA(model, cfg.ema) if cfg.get("ema", 0) else None
    logger = Logger(cfg.out_dir)
    best = float("inf")
    for epoch in range(cfg.epochs):
        tr = run_epoch(model, tl, opt, cfg.clip_grad, cfg.mix_precision, cfg.grad_accum, ema)
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
