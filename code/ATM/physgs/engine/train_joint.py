"""
Train the joint object-centric physical world model (physgs.model.joint_wm) on packed Gaussian sequences.

Pre-training (synthetic data with physical labels: video + physics objectives):
    python -m physgs.engine.train_joint --data data/packed/physai_v2 --physics 1 --out runs/joint/pretrain
Task adaptation with replay (task videos: video objective only; replayed pre-training data: both objectives):
    python -m physgs.engine.train_joint --data data/packed/libero_spatial --physics 0 \
        --replay data/packed/physai_v2 --lambda_replay 1.0 --init runs/joint/pretrain/best.pt --out runs/joint/libero
--physics 1 on the task data is the additional-supervision setting (simulator labels).
Several GPUs (data parallel, 2026-11-05): --batch is per GPU, each GPU reads its own shard of the training windows,
the gradients are averaged before the clip and the step, GPU 0 runs the held-out check and sends its score so every
GPU follows the same schedule:
    torchrun --standalone --nproc_per_node 2 -m physgs.engine.train_joint --data ... --batch 16
"""
import argparse
import json
import os

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from physgs.data.format import Packed
from physgs.data.joint_dataset import GaussSeqDataset
from physgs.engine.common import (Logger, WSDSchedule, add_schedule_args, average, cosine_lr, dataset_split,
                                  h100_attention_fix, save_ckpt, to_device)
from physgs.model.joint_wm import JointWorldModel

CFG_KEYS = ("deter", "stoch", "dim", "heads", "enc_depth", "enc_queries", "inter_layers", "embed_dim", "dec_depth",
            "dec_tokens", "dec_ppt", "feedback", "feedback_detach", "dropout")
# best.pt scores (held-out): the video terms, plus the labelled velocity / displacement errors when the data is trained
# with physics.
# Not "total": the property NLL overfits PhysicalAI's few episodes and swings by +-2 on the held-out set (10-30 run
# picked step 4000 of 10,800 and cooled down early)
VIDEO_SCORE = ("chamfer", "img_chamfer", "pos1")
PHYSICS_SCORE = ("vel", "img_vel", "img_disp")


def loaders(roots, args, batch, shuffle_train=True, rank=0, world=1):
    tr_sets, va_sets = [], []
    for root in roots.split(","):
        tr, va = dataset_split(root, Packed(root).num_episodes)
        tr_sets.append(GaussSeqDataset(root, args.seq_len, step=4, episodes=tr, num_points=args.num_points,
                                       aug_shift=args.aug_shift, aug_prob=args.aug_prob, jitter=args.jitter))
        # held-out windows every seq_len / 4 frames: ~4x more windows than non-overlapping ones (PhysicalAI ball_mixer
        # has only 12 val episodes), so best.pt and the plateau-triggered schedule react to a less noisy score
        va_sets.append(GaussSeqDataset(root, args.seq_len, step=max(1, args.seq_len // 4), episodes=va,
                                       num_points=args.num_points))
    tr, va = torch.utils.data.ConcatDataset(tr_sets), torch.utils.data.ConcatDataset(va_sets)
    if world > 1:  # disjoint shards of the training windows, reshuffled every pass (forever sets the epoch)
        sampler = DistributedSampler(tr, num_replicas=world, rank=rank, shuffle=shuffle_train, drop_last=True)
        tl = DataLoader(tr, batch, sampler=sampler, num_workers=args.workers, drop_last=True, pin_memory=True,
                        persistent_workers=True)
    else:
        tl = DataLoader(tr, batch, shuffle=shuffle_train, num_workers=args.workers, drop_last=True, pin_memory=True,
                        persistent_workers=True)
    # the same held-out windows at every check: the generator is re-seeded before each pass (main), nothing dropped
    vl = DataLoader(va, batch, shuffle=True, num_workers=2, drop_last=False, generator=torch.Generator())
    return tl, vl, len(tr), len(va)


def forever(loader):
    epoch = 0
    while True:
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        for b in loader:
            yield b
        epoch += 1


def dist_setup():
    """(rank, world size) from torchrun's environment; (0, 1) without it (one GPU, unchanged behaviour)."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def sync_grads(params, bad, world, device):
    """Average the gradients over the GPUs. Every GPU calls this once per step, also after an out-of-memory batch
    (bad = 1), so the collectives never get out of step. Returns True when any GPU had a bad batch: then every GPU
    skips the update. A parameter without a gradient on every GPU keeps grad None (AdamW leaves it untouched, as on
    one GPU)."""
    flags = torch.tensor([float(bad)] + [float(p.grad is not None) for p in params], device=device)
    dist.all_reduce(flags, op=dist.ReduceOp.MAX)
    if flags[0] > 0:
        return True
    used = [p for p, f in zip(params, flags[1:].tolist()) if f > 0]
    for p in used:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    grads = [p.grad for p in used]
    flat = _flatten_dense_tensors(grads)
    dist.all_reduce(flat)
    flat /= world
    for g, s in zip(grads, _unflatten_dense_tensors(flat, grads)):
        g.copy_(s)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="task packed dataset dir(s), comma separated")
    ap.add_argument("--physics", type=int, default=1, help="physics objective on --data (labels must exist)")
    ap.add_argument("--replay", default=None, help="packed dataset dir(s) replayed with both objectives")
    ap.add_argument("--lambda_replay", type=float, default=1.0)
    ap.add_argument("--replay_batch", type=int, default=0, help="0 = --batch")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--overshoot", type=int, default=15, help="imagined steps per window (ATM: the current + 15 future)")
    ap.add_argument("--num_points", type=int, default=512, help="Gaussians per object read by the posterior")
    ap.add_argument("--rec_points", type=int, default=256, help="observed Gaussians per object in the Chamfer target")
    ap.add_argument("--rec_steps", type=int, default=8, help="filtered steps per window decoded for the reconstruction")
    ap.add_argument("--img_rec_steps", type=int, default=4, help="imagined steps per window decoded")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--min_lr_ratio", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0, help="gradient-norm clip")
    ap.add_argument("--precision", default="fp32", choices=("fp32", "bf16"),
                    help="fp32: float32 without TF32, as ATM; bf16: autocast + TF32 (v5 encoder / RSSM)")
    ap.add_argument("--deter", type=int, default=512)
    ap.add_argument("--stoch", type=int, default=32)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--enc_depth", type=int, default=4)
    ap.add_argument("--enc_queries", type=int, default=32)
    ap.add_argument("--inter_layers", type=int, default=2)
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--dec_depth", type=int, default=3)
    ap.add_argument("--dec_tokens", type=int, default=64)
    ap.add_argument("--dec_ppt", type=int, default=8)
    ap.add_argument("--feedback", type=int, default=1, help="0: the predicted physics does not enter the recurrence")
    ap.add_argument("--feedback_detach", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--aug_shift", type=float, default=0.0, help="horizontal window shift, in units of the length scale L")
    ap.add_argument("--aug_prob", type=float, default=0.9)
    ap.add_argument("--jitter", type=float, default=0.0, help="xyz noise on the input Gaussians, in units of L")
    ap.add_argument("--prop_stats", default="init", choices=("init", "task"),
                    help="property-label statistics of the belief: init = those of the --init checkpoint (pre-training "
                         "data); task = recomputed from the --data training labels and frozen (use when the task data "
                         "is trained with physics: LIBERO friction is 0.95 everywhere, 20 std from PhysicalAI's 0.50 +- 0.02)")
    ap.add_argument("--enc_ckpt", type=float, default=0.8,
                    help="fraction of the set-encoder chunks recomputed in the backward pass: 1 = all (least memory), "
                         "0 = none (fastest); same gradients. A100 80 GB, 10-31: LIBERO batch 64 + 8 PhysicalAI replay "
                         "0.8 = 62 GB 2.98 s/step (1.0: 45 GB 3.10 s); LIBERO 64 alone 0.4 = 67 GB 1.71 s (1.0: 34 GB 1.86 s)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--val_batches", type=int, default=20)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--resume", default=None,
                    help="continue a stopped run from its <out>/last.pt (2026-11-06, move to another shell): weights and "
                         "step from the checkpoint, best score and wsd state replayed from <out>/log.jsonl, AdamW state "
                         "from <out>/last_opt.pt when it holds the same step")
    ap.add_argument("--rewarm", type=int, default=100,
                    help="--resume without a saved optimizer state: linear lr re-warm-up over this many steps while "
                         "AdamW rebuilds its moments (a fresh AdamW at the full lr makes sign-like first updates)")
    add_schedule_args(ap)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = args.precision == "bf16" and device == "cuda"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = amp
    h100_attention_fix()
    rank, world = dist_setup()
    is0 = rank == 0
    say = print if is0 else (lambda *a, **k: None)
    if world > 1:
        say(f"data parallel on {world} GPUs: batch {args.batch} per GPU, {args.batch * world} per step")

    tl, vl, n_tr, n_va = loaders(args.data, args, args.batch, rank=rank, world=world)
    rl = None
    if args.replay:
        rl, _, n_rp, _ = loaders(args.replay, args, args.replay_batch or args.batch, rank=rank, world=world)
        say(f"replay windows {n_rp:,}")
    say(f"task windows {n_tr:,} (physics objective: {bool(args.physics)})  val {n_va:,}")

    cfg = {k: getattr(args, k) for k in CFG_KEYS}
    model = JointWorldModel(**cfg).to(device)
    model.enc_ckpt = args.enc_ckpt
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device)["model"])
        model.update_norms = False  # keep the pre-training statistics (property labels, per-frame motion)
    if args.prop_stats == "task":
        model.prop_norm.reset()
        for root in args.data.split(","):
            d = Packed(root)
            tr, _ = dataset_split(root, d.num_episodes)
            model.prop_norm.update(torch.from_numpy(d.props[tr][d.present[tr]].astype("float32")).to(device))
        say(f"property statistics from the task labels: mean {model.prop_norm.mean.tolist()} std {model.prop_norm.std.tolist()}")
    step0, opt_state = 0, None
    if args.resume:  # after --init / --prop_stats, so the checkpoint's weights and statistics are the ones used
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        step0 = int(ck["step"])
        p = os.path.join(os.path.dirname(args.resume), "last_opt.pt")
        if os.path.exists(p):
            o = torch.load(p, map_location=device)
            opt_state = o["opt"] if int(o.get("step", -1)) == step0 else None
        say(f"resumed from {args.resume} at step {step0}: optimizer state "
            f"{'restored' if opt_state else f'reset, lr re-warm-up over {args.rewarm} steps'}")
    if world > 1:  # every GPU starts from GPU 0's weights and statistics
        for t in model.state_dict().values():
            dist.broadcast(t, 0)
    say("params (M):", round(sum(p.numel() for p in model.parameters()) / 1e6, 2))
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    if opt_state is not None:
        opt.load_state_dict(opt_state)
    logger = Logger(args.out, args.wandb, name=os.path.basename(args.out), config=vars(args)) if is0 else None
    kw = dict(overshoot=args.overshoot, rec_steps=args.rec_steps, img_rec_steps=args.img_rec_steps,
              rec_points=args.rec_points)
    wsd = WSDSchedule(args.lr, args.steps, args.warmup, args.min_lr_ratio, args.cool_frac, args.patience, args.rel_tol,
                      args.min_frac) if args.schedule == "wsd" else None
    end = lambda: wsd.end if wsd else args.steps
    score_keys = VIDEO_SCORE + (PHYSICS_SCORE if args.physics else ())
    task_it, rep_it = forever(tl), (forever(rl) if rl is not None else None)
    step, buf, best, skipped, oom = step0, [], float("inf"), 0, 0
    if args.resume:  # the held-out scores up to the checkpoint give back best and the wsd state (cooldown start)
        lp = os.path.join(args.out, "log.jsonl")
        rows = [json.loads(l) for l in open(lp) if l.strip()] if os.path.exists(lp) else []
        for r in rows:
            if "score" in r and int(r["step"]) <= step0:
                best = min(best, r["score"])
                msg = wsd.check(int(r["step"]), r["score"]) if wsd else None
                if msg:
                    say(f"wsd (replayed): {msg}")
        if is0:  # drop the entries the stopped run logged after its checkpoint (they are trained again)
            with open(lp + ".tmp", "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows if int(r["step"]) <= step0)
            os.replace(lp + ".tmp", lp)
            logger.f.close()  # the logger's handle points at the replaced file: append to the new one
            logger.f = open(lp, "a")
        say(f"resume: best score {best:.4f}, schedule end {end()}, lr at step {step0}: "
            f"{wsd.lr(step0) if wsd else float('nan'):.2e}")
    while step < end():
        if wsd:
            lr = wsd.lr(step)
        elif args.schedule == "decay":
            lr = args.lr * step / args.warmup if step < args.warmup else                 args.lr * (1 - (1 - args.min_lr_ratio) * (step - args.warmup) / max(1, args.steps - args.warmup))
        else:
            lr = cosine_lr(step, args.steps, args.lr, warmup=args.warmup, min_ratio=args.min_lr_ratio)
        if args.resume and opt_state is None and step - step0 < args.rewarm:
            lr *= (step - step0 + 1) / args.rewarm
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss, logs = model.loss(to_device(next(task_it), device), physics=bool(args.physics), **kw)
                if rep_it is not None:
                    rloss, rlogs = model.loss(to_device(next(rep_it), device), physics=True, **kw)
                    loss = loss + args.lambda_replay * rloss
                    logs.update({f"replay_{k}": v for k, v in rlogs.items()})
            loss.backward()
        except torch.cuda.OutOfMemoryError:  # a rare batch with many visible objects (memory is sized near the limit)
            loss = logs = None
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            oom += 1
            print(f"[gpu {rank}] step {step}: out of memory, batch skipped ({oom} so far)", flush=True)
        gnorm = None
        if world > 1:  # all GPUs skip together when any had an out-of-memory or non-finite batch
            bad_here = loss is None or not bool(torch.isfinite(loss))
            if sync_grads(params, bad_here, world, device):
                if loss is not None and not bad_here:
                    loss = logs = None  # another GPU's batch was bad: drop this one too
                    opt.zero_grad(set_to_none=True)
        if loss is not None:
            gnorm = torch.nn.utils.clip_grad_norm_(params, args.clip)
        if loss is None:
            pass
        elif not (torch.isfinite(loss) and torch.isfinite(gnorm)):  # one non-finite step would make every weight NaN
            opt.zero_grad(set_to_none=True)
            skipped += 1
            bad = [k for k, v in logs.items() if v != v or abs(v) == float("inf")]
            print(f"step {step}: non-finite loss / gradient, update skipped ({skipped} so far); non-finite terms "
                  f"{bad or 'none (gradient only)'}", flush=True)
        else:
            opt.step()
            buf.append(logs)
        if world > 1:  # running statistics (property labels, per-frame motion) follow GPU 0, as DDP's buffers do
            for b in model.buffers():
                dist.broadcast(b, 0)
        step += 1
        if step % 100 == 0 and buf and is0:
            a = average(buf)
            logger.log(step, **{f"train/{k}": v for k, v in a.items()}, lr=lr)
            print(f"step {step} lr {lr:.2e} " + " ".join(f"{k}={v:.4f}" for k, v in a.items() if not k.startswith("replay_")),
                  flush=True)
        if step % 100 == 0:
            buf = []
        if step % args.eval_every == 0 or step == end():
            score, cool_req = 0.0, 0.0
            if is0:  # the held-out check runs on GPU 0; the others wait for its score below
                model.eval()
                vbuf = []
                vl.generator.manual_seed(0)
                # fork_rng: the seed below fixes the decoded / imagined steps of every check without resetting the
                # training RNG (before 10-30 every check restarted the training noise from seed 0)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp),                         torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == "cuda" else []):
                    torch.manual_seed(0)
                    for i, vb in enumerate(vl):
                        if i == args.val_batches:
                            break
                        # labels are read for the report even when they are not trained on (physics=True), the score
                        # uses only the trained terms
                        _, vlogs = model.loss(to_device(vb, device), physics=True, **kw)
                        vbuf.append(vlogs)
                model.train()
                v = average(vbuf)
                score = sum(v[k] for k in score_keys)
                logger.log(step, **{f"val/{k}": x for k, x in v.items()}, score=score)
                print(f"val {step}: score={score:.4f} " + " ".join(f"{k}={x:.4f}" for k, x in v.items()), flush=True)
                save_ckpt(os.path.join(args.out, "last.pt"), model, cfg, step)
                # the optimizer next to last.pt, so a stopped run can continue exactly (--resume)
                torch.save(dict(opt=opt.state_dict(), step=step), os.path.join(args.out, "last_opt.pt.tmp"))
                os.replace(os.path.join(args.out, "last_opt.pt.tmp"), os.path.join(args.out, "last_opt.pt"))
                if score < best:
                    best = score
                    save_ckpt(os.path.join(args.out, "best.pt"), model, cfg, step)
                    with open(os.path.join(args.out, "best_step.txt"), "w") as f:
                        f.write(f"best step {step}: score {best:.4f} ({'+'.join(score_keys)}); val err_pos "
                                f"{v.get('err_pos', float('nan')):.4f} m, err_img_pos {v.get('err_img_pos', float('nan')):.4f} m "
                                f"(still {v.get('err_img_pos_still', float('nan')):.4f} m)\n")
                    print(f"best.pt <- step {step} (score {best:.4f})", flush=True)
                cool_req = float(os.path.exists(os.path.join(args.out, "COOLDOWN")))
            if world > 1:  # one score and one cooldown request for every GPU -> the same schedule everywhere
                t = torch.tensor([score, cool_req], device=device, dtype=torch.float64)
                dist.broadcast(t, 0)
                score, cool_req = float(t[0]), float(t[1])
            msg = wsd.check(step, score) if wsd else None
            if wsd and wsd.cool_start is None and cool_req > 0:
                # cooldown on request (touch <out>/COOLDOWN): same length rule as the plateau trigger
                wsd.cool_start, wsd.end = step, min(wsd.max_steps, step + max(1, int(wsd.cool_frac * step)))
                msg = f"cooldown {wsd.cool_start} -> {wsd.end} (requested: {args.out}/COOLDOWN)"
            if msg and is0:
                print(f"wsd: {msg}", flush=True)
                with open(os.path.join(args.out, "schedule.txt"), "w") as f:
                    f.write(msg + "\n")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
