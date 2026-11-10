"""
Train the action-free object RSSM on precomputed tokens.

    python -m physgs.engine.train_rssm --data data/packed/physai --out runs/rssm_physai
    python -m physgs.engine.train_rssm --data data/packed/libero_spatial --init runs/rssm_physai/last.pt \
        --out runs/rssm_libero_spatial --steps 20000 --lr 1e-4
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from physgs.data.dataset import TokenSeqDataset
from physgs.data.format import Packed
from physgs.engine.common import (Logger, WSDSchedule, add_schedule_args, average, cosine_lr, dataset_split,
                                  h100_attention_fix, save_ckpt, to_device)
from physgs.model.object_rssm import PRED_KEYS, ObjectRSSM


@torch.no_grad()
def forecast_score(model, loader, context, horizon, device):
    """
    Held-out checkpoint score, after ATM (which keeps the checkpoint with the lowest validation training loss): the
    open-loop part of the training objective (pos, vel, angvel, contact, z with the training weights), forecasting
    `horizon` steps after `context` filtered steps from every window of the loader, without sampling (deterministic).
    Also returns the CoM error in metres (mean over the horizon and at its last step) for reading.
    """
    sums, counts = {k: 0.0 for k in PRED_KEYS}, {k: 0.0 for k in PRED_KEYS}
    pos_sum, pos_n, last_sum, last_n = 0.0, 0, 0.0, 0
    for b in loader:
        b = to_device(b, device)
        ctx = model.context(b["is_robot"], b["present"], b["task"])
        posts, _ = model.observe(b["tokens"][:, :context], b["visible"][:, :context], ctx, sample=False)
        imag = model.imagine({k: posts[-1][k] for k in ("h", "s", "pos", "seen")}, ctx, horizon, sample=False)
        hi = [model.heads(s) for s in imag]
        pred = {k: torch.stack([h[k] for h in hi], 1).float() for k in PRED_KEYS}
        tgt = {k: v[:, context:context + horizon] for k, v in model.target(b["tokens"], b).items()}
        mask = (b["visible"] & b["present"][:, None])[:, context:context + horizon] & posts[-1]["seen"][:, None]
        for k in ("pos", "vel", "angvel"):
            l = F.smooth_l1_loss(pred[k], tgt[k], reduction="none", beta=0.05)
            sums[k] += float(l[mask].sum()); counts[k] += float(mask.sum()) * l.shape[-1]
        has_c = mask & torch.isfinite(tgt["contact"])
        l = F.binary_cross_entropy_with_logits(pred["contact"], torch.nan_to_num(tgt["contact"]).clamp(0, 1), reduction="none")
        sums["contact"] += float(l[has_c].sum()); counts["contact"] += float(has_c.sum())
        l = (pred["z"] - tgt["z"]) ** 2
        sums["z"] += float(l[mask].sum()); counts["z"] += float(mask.sum()) * l.shape[-1]
        lab = mask & torch.isfinite(b["pos"][:, context:context + horizon]).all(-1)
        e = (pred["pos"] - torch.nan_to_num(b["pos"][:, context:context + horizon])).norm(dim=-1)
        pos_sum += float(e[lab].sum()); pos_n += int(lab.sum())
        last_sum += float(e[:, -1][lab[:, -1]].sum()); last_n += int(lab[:, -1].sum())
    terms = {k: sums[k] / counts[k] for k in PRED_KEYS if counts[k] > 0}
    score = sum(model.w[k] * v for k, v in terms.items())
    return dict(score=score, **{f"fc_{k}": v for k, v in terms.items()},
                fc_pos_m=pos_sum / max(pos_n, 1), fc_last_m=last_sum / max(last_n, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="packed dataset dir(s) with tokens.npy, comma separated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--grad_accum", type=int, default=1, help="micro-batches per update (effective batch = batch x this)")
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--overshoot", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--deter", type=int, default=512)
    ap.add_argument("--stoch", type=int, default=32)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval_every", type=int, default=2000, help="steps between validation checks (if --check_epochs 0)")
    ap.add_argument("--check_epochs", type=float, default=0, help="validation check every this many epochs (ATM: 5)")
    ap.add_argument("--warmup_epochs", type=float, default=0, help="linear warm-up epochs (ATM: 5); 0 = 1000 steps")
    ap.add_argument("--min_lr_ratio", type=float, default=0.05, help="cosine floor as a fraction of --lr (ATM: 0)")
    ap.add_argument("--dropout", type=float, default=0.0, help="attention and feed-forward dropout (ATM: 0.2)")
    ap.add_argument("--aug_shift", type=float, default=0.0, help="random horizontal scene shift, metres (0 = off)")
    ap.add_argument("--aug_prob", type=float, default=0.9, help="probability of the shift (ATM: 0.9)")
    ap.add_argument("--fc_context", type=int, default=8, help="checkpoint score: filtered steps before the forecast")
    ap.add_argument("--fc_horizon", type=int, default=16, help="checkpoint score: forecast steps")
    ap.add_argument("--fc_stride", type=int, default=4, help="checkpoint score: a forecast starts every this many frames")
    ap.add_argument("--wandb", action="store_true")
    add_schedule_args(ap)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    h100_attention_fix()

    train_sets, val_sets, fc_sets = [], [], []
    token_dim = None
    fc_len = args.fc_context + args.fc_horizon
    for root in args.data.split(","):
        tr, va = dataset_split(root, Packed(root).num_episodes)
        train_sets.append(TokenSeqDataset(root, args.seq_len, step=4, episodes=tr, aug_shift=args.aug_shift,
                                          aug_prob=args.aug_prob))
        val_sets.append(TokenSeqDataset(root, args.seq_len, step=args.seq_len, episodes=va))
        fc = TokenSeqDataset(root, fc_len, step=args.fc_stride, episodes=va)
        fc.items = [it for it in fc.items if it[2] - it[1] == fc_len]  # full-length windows only
        fc_sets.append(fc)
        token_dim = np.load(f"{root}/tokens.npy", mmap_mode="r").shape[-1]
    train, val = torch.utils.data.ConcatDataset(train_sets), torch.utils.data.ConcatDataset(val_sets)
    fc_val = torch.utils.data.ConcatDataset(fc_sets)
    # all datasets must share the padded object count for batching
    tl = DataLoader(train, args.batch, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True,
                    persistent_workers=True)
    vl = DataLoader(val, args.batch, shuffle=False, num_workers=2, drop_last=False)
    fl = DataLoader(fc_val, 64, shuffle=False, num_workers=4, drop_last=False)
    per_epoch = len(tl) // args.grad_accum  # updates per epoch
    check_every = max(1, round(args.check_epochs * per_epoch)) if args.check_epochs else args.eval_every
    warmup = round(args.warmup_epochs * per_epoch) if args.warmup_epochs else 1000
    print(f"train windows {len(train):,} ({per_epoch} steps per epoch of {args.batch} x {args.grad_accum})  val "
          f"{len(val):,}  forecast-check windows {len(fc_val):,}  token_dim {token_dim}  check every {check_every} "
          f"steps  warm-up {warmup} steps")

    cfg = dict(token_dim=token_dim, latent_dim=token_dim - 16, deter=args.deter, stoch=args.stoch, dim=args.dim,
               layers=args.layers, dropout=args.dropout)
    model = ObjectRSSM(**cfg).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device)["model"])
    print("params (M):", sum(p.numel() for p in model.parameters()) / 1e6)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    logger = Logger(args.out, args.wandb, name=os.path.basename(args.out), config=vars(args))

    step, buf, micro = 0, [], 0
    opt.zero_grad(set_to_none=True)
    best = float("inf")  # best.pt keeps the checkpoint with the lowest held-out forecast score (ATM: model_best)
    wsd = WSDSchedule(args.lr, args.steps, warmup, args.min_lr_ratio, args.cool_frac, args.patience, args.rel_tol,
                      args.min_frac) if args.schedule == "wsd" else None
    end = lambda: wsd.end if wsd else args.steps
    while step < end():
        for batch in tl:
            lr = wsd.lr(step) if wsd else cosine_lr(step, args.steps, args.lr, warmup=warmup, min_ratio=args.min_lr_ratio)
            for g in opt.param_groups:
                g["lr"] = lr
            batch = to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss, logs = model.loss(batch, overshoot=args.overshoot)
            (loss / args.grad_accum).backward()
            buf.append(logs)
            micro += 1
            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                logger.log(step, **{f"train/{k}": v for k, v in average(buf).items()}, lr=opt.param_groups[0]["lr"])
                buf = []
            if step % check_every == 0 or step == end():
                model.eval()
                vbuf = []
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    for i, vb in enumerate(vl):
                        if i == 30:
                            break
                        _, vlogs = model.loss(to_device(vb, device), overshoot=args.overshoot)
                        vbuf.append(vlogs)
                fc = forecast_score(model, fl, args.fc_context, args.fc_horizon, device)
                v = dict(average(vbuf) if vbuf else {}, **fc)
                logger.log(step, **{f"val/{k}": x for k, x in v.items()})
                print(f"step {step}: " + " ".join(f"{k}={x:.4f}" for k, x in v.items()), flush=True)
                model.train()
                save_ckpt(os.path.join(args.out, "last.pt"), model, cfg, step)
                if fc["score"] < best:
                    best = fc["score"]
                    save_ckpt(os.path.join(args.out, "best.pt"), model, cfg, step)
                    with open(os.path.join(args.out, "best_step.txt"), "w") as f:
                        f.write(f"best step {step}: forecast score {best:.4f}, CoM error {fc['fc_pos_m']:.4f} m\n")
                    print(f"best.pt <- step {step} (forecast score {best:.4f})", flush=True)
                msg = wsd.check(step, fc["score"]) if wsd else None
                if msg:
                    print(f"wsd: {msg}", flush=True)
                    with open(os.path.join(args.out, "schedule.txt"), "w") as f:
                        f.write(msg + "\n")
            if step >= end():
                break


if __name__ == "__main__":
    main()
