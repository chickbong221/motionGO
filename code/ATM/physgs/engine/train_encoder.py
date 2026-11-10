"""
Train the object encoder / decoder on packed episodes.

    python -m physgs.engine.train_encoder --data data/packed/physai --out runs/encoder_physai
    # fine-tune on LIBERO from the PhysicalAI checkpoint
    python -m physgs.engine.train_encoder --data data/packed/libero_spatial --init runs/encoder_physai/last.pt \
        --out runs/encoder_libero_spatial --steps 20000 --lr 1e-4
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from physgs.data.dataset import ObjectFrameDataset
from physgs.data.format import Packed
from physgs.engine.common import (Logger, WSDSchedule, add_schedule_args, average, cosine_lr, dataset_split,
                                  h100_attention_fix, save_ckpt, to_device)
from physgs.model.object_encoder import ObjectAutoencoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="packed dataset dir(s), comma separated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None)
    ap.add_argument("--resume", default=None, help="checkpoint to continue from at its step, on the --steps schedule "
                    "(no optimizer state is saved, so the learning rate re-warms over --rewarm steps)")
    ap.add_argument("--rewarm", type=int, default=500)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--enc_depth", type=int, default=6)
    ap.add_argument("--dec_depth", type=int, default=4)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--frames", type=int, default=2, help="window of frames per encoder input (t-K+1 .. t)")
    ap.add_argument("--hist_points", type=int, default=512, help="Gaussians kept for each older frame")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--val_batches", type=int, default=20,
                    help="validation batches per check (random); 0 = the whole validation set in order, like ATM")
    ap.add_argument("--val_items", type=int, default=0,
                    help="with --val_batches 0: a fixed random subset of this many validation items (seed 0), the "
                         "same items at every check; 0 = all of them")
    ap.add_argument("--dropout", type=float, default=0.0, help="encoder attention and feed-forward dropout (ATM: 0.2)")
    ap.add_argument("--warmup", type=int, default=1000, help="linear warm-up steps")
    ap.add_argument("--min_lr_ratio", type=float, default=0.05, help="cosine floor as a fraction of --lr (ATM: 0)")
    ap.add_argument("--select", default="loss",
                    help="best.pt criterion: 'loss' (total validation loss) or comma-separated loss terms, scored as the "
                         "mean of each term relative to its value at the first check (e.g. chamfer,offset,vel,angvel: "
                         "shape and motion only; the property term overfits early and otherwise decides best.pt)")
    ap.add_argument("--wandb", action="store_true")
    add_schedule_args(ap)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    h100_attention_fix()

    train_sets, val_sets = [], []
    for root in args.data.split(","):
        tr, va = dataset_split(root, Packed(root).num_episodes)
        train_sets.append(ObjectFrameDataset(root, tr, jitter=0.002, frames=args.frames))
        val_sets.append(ObjectFrameDataset(root, va, frames=args.frames))
    train = torch.utils.data.ConcatDataset(train_sets)
    val = torch.utils.data.ConcatDataset(val_sets)
    if args.val_items and args.val_items < len(val):
        keep = np.sort(np.random.default_rng(0).choice(len(val), args.val_items, replace=False))
        print(f"validation: fixed subset of {args.val_items:,} of {len(val):,} items")
        val = torch.utils.data.Subset(val, keep.tolist())
    print(f"train objects-frames {len(train):,}  val {len(val):,}")
    tl = DataLoader(train, args.batch, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True,
                    persistent_workers=True)
    if args.val_batches:
        vl = DataLoader(val, args.batch, shuffle=True, num_workers=2, drop_last=True)
    else:
        vl = DataLoader(val, args.batch, shuffle=False, num_workers=4, drop_last=False)

    cfg = dict(dim=args.dim, enc_depth=args.enc_depth, dec_depth=args.dec_depth, latent_dim=args.latent,
               num_frames=args.frames, hist_points=args.hist_points)
    if args.dropout:
        cfg["dropout"] = args.dropout
    model = ObjectAutoencoder(**cfg).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device)["model"])
    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        assert ck["cfg"] == cfg, f"resume config {ck['cfg']} != {cfg}"
        model.load_state_dict(ck["model"])
        start = ck["step"]
        print(f"resumed from {args.resume} at step {start}")
    print("params (M):", sum(p.numel() for p in model.parameters()) / 1e6)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    logger = Logger(args.out, args.wandb, name=os.path.basename(args.out), config=vars(args))

    step, buf = start, []
    best, ref = float("inf"), None  # best.pt keeps the checkpoint with the lowest --select score
    wsd = WSDSchedule(args.lr, args.steps, args.warmup, args.min_lr_ratio, args.cool_frac, args.patience, args.rel_tol,
                      args.min_frac) if args.schedule == "wsd" else None
    end = lambda: wsd.end if wsd else args.steps
    while step < end():
        for batch in tl:
            rewarm = min(1.0, (step - start + 1) / args.rewarm) if start else 1.0
            lr = wsd.lr(step) if wsd else cosine_lr(step, args.steps, args.lr, warmup=args.warmup, min_ratio=args.min_lr_ratio)
            for g in opt.param_groups:
                g["lr"] = lr * rewarm
            batch = to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss, logs, _ = model(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            buf.append(dict(logs, loss=float(loss)))
            step += 1
            if step % 100 == 0:
                logger.log(step, **{f"train/{k}": v for k, v in average(buf).items()}, lr=opt.param_groups[0]["lr"])
                buf = []
            if step % args.eval_every == 0 or step == end():
                model.eval()
                vbuf = []
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    for i, vb in enumerate(vl):
                        if args.val_batches and i == args.val_batches:
                            break
                        vloss, vlogs, _ = model(to_device(vb, device))
                        vbuf.append(dict(vlogs, loss=float(vloss)))
                if vbuf:
                    v = average(vbuf)
                    logger.log(step, **{f"val/{k}": x for k, x in v.items()})
                    print(f"step {step}: " + " ".join(f"{k}={x:.4f}" for k, x in v.items()), flush=True)
                model.train()
                save_ckpt(os.path.join(args.out, "last.pt"), model, cfg, step)
                if vbuf:
                    if args.select == "loss":
                        score, what = v["loss"], "validation loss"
                    else:
                        terms = args.select.split(",")
                        ref = ref or {t: max(v[t], 1e-8) for t in terms}
                        score, what = float(np.mean([v[t] / ref[t] for t in terms])), f"relative {args.select}"
                    if score < best:
                        best = score
                        save_ckpt(os.path.join(args.out, "best.pt"), model, cfg, step)
                        with open(os.path.join(args.out, "best_step.txt"), "w") as f:
                            f.write(f"best step {step}: {what} {best:.4f} (validation loss {v['loss']:.4f})\n")
                        print(f"best.pt <- step {step} ({what} {best:.4f})", flush=True)
                    msg = wsd.check(step, score) if wsd else None
                    if msg:
                        print(f"wsd: {msg}", flush=True)
                        with open(os.path.join(args.out, "schedule.txt"), "w") as f:
                            f.write(msg + "\n")
            if step >= end():
                break


if __name__ == "__main__":
    main()
