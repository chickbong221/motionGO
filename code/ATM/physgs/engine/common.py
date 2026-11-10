import json
import math
import os
import time

import numpy as np
import torch


def h100_attention_fix():
    """torch 2.0.1's memory-efficient attention kernel (used by nn.MultiheadAttention) fails in the backward pass on
    H100 / sm_90 ("illegal instruction", CUBLAS_STATUS_EXECUTION_FAILED); attention then uses the other kernels.
    Other GPUs (the 4090s, sm_89) are left as they are."""
    if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0):
        torch.backends.cuda.enable_mem_efficient_sdp(False)


def split_episodes(n, val_frac=0.05, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def dataset_split(root, n):
    """Train / val episodes of a packed dataset: its fixed split.json if it has one (physgs.data.make_split), else random."""
    path = os.path.join(root, "split.json")
    if not os.path.exists(path):
        return split_episodes(n)
    with open(path) as f:
        s = json.load(f)
    return np.array(s["train"], dtype=np.int64), np.array(s["val"], dtype=np.int64)


def cosine_lr(step, total, base, warmup=1000, min_ratio=0.05):
    if step < warmup:
        return base * step / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


class WSDSchedule:
    """
    Warmup-stable-decay learning rate (Hagele et al., NeurIPS 2024; the WSD style of NVIDIA Megatron-Core): linear
    warm-up to `base`, constant, then a linear cooldown to min_ratio * base over `cool_frac` of the steps done so far.
    The cooldown starts when the held-out score has not improved by more than `rel_tol` (relative) over `patience`
    checks, but never before `min_frac` of max_steps (no stop during warm-up or early noisy phases); at the latest it
    starts at max_steps / (1 + cool_frac), so a run always ends cooled down and never after max_steps.
    """

    def __init__(self, base, max_steps, warmup, min_ratio, cool_frac=0.2, patience=4, rel_tol=0.005, min_frac=0.4):
        self.base, self.max_steps, self.warmup, self.min_ratio = base, max_steps, warmup, min_ratio
        self.cool_frac, self.patience, self.rel_tol, self.min_frac = cool_frac, patience, rel_tol, min_frac
        self.latest = int(max_steps / (1 + cool_frac))
        self.cool_start, self.end = None, max_steps
        self.best, self.stale = float("inf"), 0

    def lr(self, step):
        if step < self.warmup:
            return self.base * step / self.warmup
        if self.cool_start is None or step < self.cool_start:
            return self.base
        p = min(1.0, (step - self.cool_start) / max(1, self.end - self.cool_start))
        return self.base * (1 - (1 - self.min_ratio) * p)

    def check(self, step, score):
        """Called at each held-out check; returns a message when the cooldown starts."""
        if self.cool_start is not None:
            return None
        if score < self.best * (1 - self.rel_tol):
            self.best, self.stale = score, 0
        else:
            self.stale += 1
        why = None
        if step >= self.min_frac * self.max_steps and self.stale >= self.patience:
            why = f"no >{self.rel_tol:.1%} improvement in {self.stale} checks"
        elif step >= self.latest:
            why = "latest start for the step cap"
        if why:
            self.cool_start, self.end = step, min(self.max_steps, step + max(1, int(self.cool_frac * step)))
            return f"cooldown {self.cool_start} -> {self.end} ({why})"
        return None


def add_schedule_args(ap):
    ap.add_argument("--schedule", default="cosine", choices=("cosine", "wsd", "decay"),
                    help="cosine over --steps, warmup-stable-decay with a plateau-triggered cooldown (see WSDSchedule), "
                         "or decay = warm-up then a linear cooldown to min_lr_ratio over --steps (the cooldown of a "
                         "stopped wsd run continued from its checkpoint)")
    ap.add_argument("--cool_frac", type=float, default=0.2, help="wsd: cooldown length as a fraction of the steps done")
    ap.add_argument("--patience", type=int, default=4, help="wsd: checks without improvement before the cooldown")
    ap.add_argument("--rel_tol", type=float, default=0.005, help="wsd: relative improvement that counts")
    ap.add_argument("--min_frac", type=float, default=0.4, help="wsd: no cooldown before this fraction of --steps")


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


class Logger:
    def __init__(self, out_dir, use_wandb=False, project="physgs", name=None, config=None):
        os.makedirs(out_dir, exist_ok=True)
        self.f = open(os.path.join(out_dir, "log.jsonl"), "a")
        self.wandb = None
        if use_wandb:
            import wandb
            self.wandb = wandb.init(project=project, name=name, config=config, dir=out_dir)
        self.t0 = time.time()

    def log(self, step, **kv):
        kv = {k: float(v) for k, v in kv.items()}
        kv.update(step=step, time=time.time() - self.t0)
        self.f.write(json.dumps(kv) + "\n")
        self.f.flush()
        if self.wandb:
            self.wandb.log(kv, step=step)


def average(logs):
    keys = logs[0].keys()
    return {k: float(np.mean([l[k] for l in logs])) for k in keys}


def save_ckpt(path, model, cfg, step):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(model=model.state_dict(), cfg=cfg, step=step), path + ".tmp")
    os.replace(path + ".tmp", path)
