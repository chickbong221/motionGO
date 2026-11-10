"""
eval_policy_so101.py (2026-11-07): offline evaluation of a PhysGO policy (either backbone) on the SO-101 episodes that
were NOT policy demos (evaluation on the training videos; there is no robot in the loop). Every episode is run
frame by frame through the policy's own act() with the inputs it gets online (top + wrist image, robot state, task,
the PhysGO forecast of that frame), episodes batched in parallel; the ViLT keeps its frame history (latent queue) over
the episode as in its rollouts. The predicted action is compared with the recorded one:
  arm_deg      mean |error| of the 5 arm joints, degrees (actions are absolute joint targets)
  grip         mean |error| of the gripper (LeRobot 0..100 units)
  chunk_arm_deg (smolvla) the same over the whole predicted chunk vs the recorded next steps
per task, over all frames of all evaluated episodes.
  python eval_policy_so101.py <run dir (holds policy/config.yaml, policy/model_final.ckpt)> [--episodes demo|eval]
Run with the backbone's env (vilt: the original env; smolvla: env_vla).
"""
import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/dev/shm/physgo/code_so101/ATM")
PG = "/dev/shm/physgo/so101/policy_h5"
TASK = lambda d: "A" if "pot" in d else ("B" if "open_drawer" in d else "C")  # noqa: E731


def load_policy(run, ckpt="model_final"):
    cfg = OmegaConf.load(f"{run}/policy/config.yaml")
    mc = OmegaConf.to_container(cfg.model_cfg, resolve=True)
    if cfg.get("backbone", "vilt") == "smolvla":
        from physgs.policy.physgs_smolvla import PhysGOSmolVLAPolicy
        p = PhysGOSmolVLAPolicy(OmegaConf.to_container(cfg.smolvla_cfg, resolve=True), mc["forecast_cfg"],
                                act_dim=mc["policy_head_cfg"]["output_size"][0], state_dim=cfg.smolvla_cfg.state_dim,
                                num_views=mc["obs_cfg"]["num_views"])
    else:
        from physgs.policy.physgs_vilt import PhysGSViLTPolicy
        p = PhysGSViLTPolicy(**mc)
    p.load(f"{run}/policy/{ckpt}.ckpt")
    p = p.cuda()
    p.eval()   # ATM's train() override returns None: no chaining
    return p, cfg


def chunk_exec_eval(policy, cfg, files, keys, half, sd, args):
    """LeRobot's recommended smolVLA setting, evaluated as it runs on the robot: every exec/2 frames (15 fps) a chunk of
    50 actions at 30 fps is predicted from ONE random noise sample (the repo samples), un-normalized (MEAN_STD, + state
    when relative), and its first `exec` actions are executed; the executed 30 fps stream is scored against the recorded
    30 fps actions (all steps, and the 15 fps frames only for comparison with one-action policies), plus smoothness:
    step-to-step change of the commanded arm target vs the recorded one (chunk boundaries included)."""
    dc = cfg.dataset_cfg
    tm, ts = np.array(dc.target_mean, np.float32), np.array(dc.target_std, np.float32)
    relf = bool(dc.get("relative_actions", False))
    rate = int(dc.action_rate)
    ex = int(args.exec_steps or policy.chunk)
    plan_every = max(1, ex // rate)
    res = {}
    for s in range(0, len(files), args.batch):
        group = files[s:s + args.batch]
        eps = []
        for f in group:
            with h5py.File(f, "r") as h:
                r = h["root"]
                eps.append(dict(task=TASK(os.path.basename(os.path.dirname(f))), name=f,
                                top=r["top/video"][0].transpose(0, 2, 3, 1), wrist=r["wrist/video"][0].transpose(0, 2, 3, 1),
                                a30=r["actions30"][:], extra={k: r["extra_states"][k][:] for k in keys},
                                fc=np.load(f"{cfg.forecast_dir}/{os.path.basename(os.path.dirname(f))}/{os.path.basename(f)[:-5]}.npy")))
        B = len(eps)
        T15 = [len(e["top"]) for e in eps]
        pred = [np.full_like(e["a30"], np.nan) for e in eps]
        fut = [np.full_like(e["a30"], np.nan) for e in eps]   # the latest chunk's planned actions (async blending)
        for t in range(0, max(T15), plan_every):
            tt = [min(t, n - 1) for n in T15]
            obs = np.stack([np.stack([e["top"][i], e["wrist"][i]]) for e, i in zip(eps, tt)])
            extra = {k: np.stack([e["extra"][k][i] for e, i in zip(eps, tt)]).astype(np.float32) for k in keys}
            fc = np.stack([e["fc"][i] for e, i in zip(eps, tt)]).astype(np.float32)
            shape = (B, policy.chunk, policy.policy.config.max_action_dim)
            if args.noise == "zero":
                noise = torch.zeros(shape, device="cuda")
            else:   # one random sample per chunk (the repo), seeded per episode and planning step
                noise = torch.stack([torch.randn(shape[1:], device="cuda", generator=torch.Generator(device="cuda").manual_seed(
                    int(abs(hash(e["name"])) % 2 ** 31) + t)) for e in eps])
            with torch.no_grad():
                ch = policy.predict_chunk(obs, extra, fc, noise=noise).float().cpu().numpy() * ts + tm   # (B, chunk, 6)
            if relf:
                st = np.concatenate([extra["joint_states"][:, :5], extra["gripper_states"][:, :1]], 1)
                ch = ch + st[:, None]
            for b, e in enumerate(eps):
                if t >= T15[b]:
                    continue
                i0 = t * rate
                m = min(policy.chunk, len(e["a30"]) - i0)
                if m <= 0:
                    continue
                new = np.clip(ch[b, :m], -1.5, 1.5)
                if args.aggregate == "weighted":
                    # LeRobot async inference (aggregate_fn "weighted_average"): where the new chunk overlaps the
                    # previous chunk's not yet executed actions, execute 0.3 x old + 0.7 x new
                    old = fut[b][i0:i0 + m]
                    have = np.isfinite(old).all(1)
                    new = np.where(have[:, None], 0.3 * old + 0.7 * new, new)
                fut[b][i0:i0 + m] = new
                n = min(ex, m)
                pred[b][i0:i0 + n] = new[:n]
        for b, e in enumerate(eps):
            p, a = pred[b], e["a30"]
            ok = np.isfinite(p).all(1)
            d = (p - a)[ok]
            r = res.setdefault(e["task"], {})
            err = np.abs(d) * half
            m = dict(arm_deg=err[:, :5].mean(1), grip=err[:, 5], mse=(d ** 2).mean(1), mse_z=((d / sd) ** 2).mean(1))
            even = np.zeros(len(a), bool)
            even[::rate] = True
            m.update(arm_deg_15=err[even[ok], :5].mean(1), mse_z_15=((d[even[ok]] / sd) ** 2).mean(1))
            pv, av = p[ok], a[ok]
            dp, dr = np.diff(pv, axis=0) * half, np.diff(av, axis=0) * half
            m.update(jitter_deg=np.abs(dp[:, :5]).mean(1), rec_jitter_deg=np.abs(dr[:, :5]).mean(1),
                     dstep_err_deg=np.abs(dp[:, :5] - dr[:, :5]).mean(1))
            for k, v in m.items():
                r.setdefault(k, []).extend(np.asarray(v, float).tolist())
        print(f"{s + len(group)}/{len(files)} episodes", flush=True)
    out = {t: dict(frames=len(r["arm_deg"]), **{k: float(np.mean(v)) for k, v in r.items()}) for t, r in sorted(res.items())}
    keys_m = list(next(iter(res.values())).keys())
    out["all"] = dict(frames=sum(len(r["arm_deg"]) for r in res.values()),
                      **{k: float(np.mean([x for r in res.values() for x in r[k]])) for k in keys_m})
    out["setting"] = dict(exec_steps=ex, replan_every_frames_15fps=plan_every, noise=args.noise, relative=relf,
                          aggregate=args.aggregate)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    # test (default): the non-demo episodes minus the 6 validation ones (137), the same for every run; val: those 6 (the
    # offline checkpoint selection of the full recipe, in place of joint_full.sh's simulator selection); eval: all 143
    # non-demo episodes; demo: the 15 policy demos
    ap.add_argument("--episodes", default="test", choices=("test", "val", "eval", "demo"))
    ap.add_argument("--ckpt", default="model_final")
    ap.add_argument("--num_steps", type=int, default=0, help="smolvla: flow-matching steps at inference (0: the config's, 10)")
    ap.add_argument("--bf16", action="store_true", help="policy inference under bf16 autocast (speed test)")
    # default zero (2026-11-08): on validation zero start noise beat one random sample (arm 1.222 vs 1.243 deg, mse_z
    # 0.0064 vs 0.0067) and an average of 4 samples; deterministic and the fastest
    ap.add_argument("--exec_steps", type=int, default=0,
                    help="chunked smolVLA (repo setting): 30 fps actions executed per chunk before re-planning (0: the whole chunk)")
    ap.add_argument("--ensemble", default="", help="chunked smolVLA: comma list run_dir:ckpt of more models whose chunks are averaged")
    ap.add_argument("--aggregate", default="none", choices=("none", "weighted"),
                    help="chunked smolVLA: blend overlapping chunks as LeRobot's async inference (0.3 old + 0.7 new)")
    ap.add_argument("--noise", default=None, choices=("random", "zero", "episode"),
                    help="smolvla: flow-matching start noise (zero: one deterministic, mean-like action)")
    ap.add_argument("--samples", type=int, default=1, help="smolvla: average of this many noise samples per step")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="first N episodes only (smoke test)")
    args = ap.parse_args()
    nrm = json.load(open(f"{PG}/norm.json"))
    _cfg = OmegaConf.load(f"{args.run}/policy/config.yaml")
    if args.noise is None:   # chunked repo setting: one random sample per chunk (the repo); one-action models: zero
        args.noise = "random" if int(_cfg.dataset_cfg.get("action_rate", 1)) > 1 else "zero"
    half = (np.array(nrm["max"]) - np.array(nrm["min"])) / 2          # [-1, 1] -> joint units
    policy, cfg = load_policy(args.run, args.ckpt)
    if args.ensemble:
        # (2026-11-08) an ensemble of independently trained runs of the same setting: chunks averaged at each re-plan
        members = [policy] + [load_policy(*x.split(":"))[0] for x in args.ensemble.split(",")]

        class _Ens:
            chunk, policy = members[0].chunk, members[0].policy

            @staticmethod
            def predict_chunk(obs, extra, fc, noise=None):
                return torch.stack([m.predict_chunk(obs, extra, fc, noise=noise).float() for m in members]).mean(0)
        policy = _Ens()
    if args.num_steps and cfg.get("backbone", "vilt") == "smolvla":
        policy.policy.config.num_steps = args.num_steps
    variant = (f"_ns{args.num_steps}" if args.num_steps else "") + ("_bf16" if args.bf16 else "") +         ("" if (args.noise == "random") == (int(_cfg.dataset_cfg.get("action_rate", 1)) > 1) else "_" + args.noise) +         (f"_ex{args.exec_steps}" if args.exec_steps else "") + ("_agg" if args.aggregate == "weighted" else "") + (f"_ens{1 + len(args.ensemble.split(','))}" if args.ensemble else "") + (f"_avg{args.samples}" if args.samples > 1 else "")
    smol = cfg.get("backbone", "vilt") == "smolvla"
    rel = np.array(cfg.dataset_cfg.rel_scale, np.float32)         if cfg.dataset_cfg.get("relative_actions", False) and cfg.dataset_cfg.get("rel_scale") is not None else None
    fdir = cfg.forecast_dir
    keys = list(cfg.extra_state_keys)
    files = []
    for d in sorted(glob.glob(f"{PG}/*_demo")):
        demo = {os.path.basename(os.path.realpath(f)) for f in glob.glob(f"{d}/bc_train_5/*.hdf5")}
        val = {os.path.basename(os.path.realpath(f)) for f in glob.glob(f"{d}/val/*.hdf5")}
        for f in sorted(glob.glob(f"{d}/demo_*.hdf5")):
            b = os.path.basename(f)
            kind = "demo" if b in demo else ("val" if b in val else "test")
            if args.episodes == kind or (args.episodes == "eval" and kind != "demo"):
                files.append(f)
    if args.limit:
        files = files[:args.limit]
    # MSE as usually reported for offline policy tests: on the normalized actions. Two normalizations: the policy's own
    # [-1, 1] scale (mse) and LeRobot's usual z-score with the per-joint mean / std of the 15 demo actions (mse_z)
    demo_act = np.concatenate([h5py.File(f, "r")["root/actions"][:] for d in sorted(glob.glob(f"{PG}/*_demo"))
                               for f in sorted(glob.glob(f"{d}/bc_train_5/*.hdf5"))])
    mu, sd = demo_act.mean(0), demo_act.std(0) + 1e-8
    if smol and int(cfg.dataset_cfg.get("action_rate", 1)) > 1:
        out = chunk_exec_eval(policy, cfg, files, keys, half, sd, args)
        if not args.limit:
            json.dump(out, open(f"{args.run}/eval_{args.episodes}_{args.ckpt}{variant}.json", "w"), indent=1)
        for t, r in out.items():
            print(f"  {t:4s} " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()))
        return
    res = {}
    for s in range(0, len(files), args.batch):
        group = files[s:s + args.batch]
        eps = []
        for f in group:
            with h5py.File(f, "r") as h:
                r = h["root"]
                eps.append(dict(task=TASK(os.path.basename(os.path.dirname(f))),
                                top=r["top/video"][0].transpose(0, 2, 3, 1), wrist=r["wrist/video"][0].transpose(0, 2, 3, 1),
                                act=r["actions"][:], emb=r["task_emb_bert"][:],
                                extra={k: r["extra_states"][k][:] for k in keys},
                                fc=np.load(f"{fdir}/{os.path.basename(os.path.dirname(f))}/{os.path.basename(f)[:-5]}.npy")))
        B, Tm = len(eps), max(len(e["act"]) for e in eps)
        if hasattr(policy, "reset"):
            policy.reset()
        prev = [None] * B                     # previous predicted absolute action per episode (smoothness)
        if smol and args.noise == "episode":  # one in-distribution noise sample per episode, held for all its steps
            ep_noise = torch.stack([torch.randn((policy.chunk, policy.policy.config.max_action_dim), device="cuda",
                                                generator=torch.Generator(device="cuda").manual_seed(
                                                    int(abs(hash(os.path.basename(f) + os.path.dirname(f))) % 2 ** 31)))
                                    for f in group])
        for t in range(Tm):
            tt = [min(t, len(e["act"]) - 1) for e in eps]
            obs = np.stack([np.stack([e["top"][i], e["wrist"][i]]) for e, i in zip(eps, tt)])       # b v h w c
            extra = {k: np.stack([e["extra"][k][i] for e, i in zip(eps, tt)]).astype(np.float32) for k in keys}
            fc = np.stack([e["fc"][i] for e, i in zip(eps, tt)]).astype(np.float32)
            emb = np.stack([e["emb"] for e in eps]).astype(np.float32)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                if smol:
                    g = torch.Generator(device="cuda").manual_seed(t)
                    shape = (B, policy.chunk, policy.policy.config.max_action_dim)
                    chunks = []
                    for _k in range(args.samples):
                        if args.noise == "zero":
                            noise = torch.zeros(shape, device="cuda")
                        elif args.noise == "episode":
                            noise = ep_noise
                        else:
                            noise = torch.randn(shape, device="cuda", generator=g)
                        chunks.append(policy.predict_chunk(obs, extra, fc, noise=noise).clamp(-1, 1).cpu().numpy())
                    chunk = np.mean(chunks, 0)
                    a = chunk[:, 0]
                else:
                    a, _ = policy.act(obs, emb, extra, fc)
            if rel is not None:   # relative-action run: back to absolute targets (state of the issuing frame added)
                st = np.concatenate([extra["joint_states"][:, :5], extra["gripper_states"][:, :1]], 1)
                a = a * rel + st
                if smol:
                    chunk = chunk * rel + st[:, None]
            for b, (e, i) in enumerate(zip(eps, tt)):
                if t >= len(e["act"]):
                    continue
                d = a[b] - e["act"][i]
                err = np.abs(d) * half
                r = res.setdefault(e["task"], {})
                m = dict(arm_deg=err[:5].mean(), grip=err[5], mse=np.mean(d ** 2), mse_z=np.mean((d / sd) ** 2))
                if i > 0 and prev[b] is not None:
                    # smoothness on the robot: step-to-step change of the commanded arm target, predicted vs recorded
                    dp, dr = (a[b] - prev[b]) * half, (e["act"][i] - e["act"][i - 1]) * half
                    m.update(jitter_deg=np.abs(dp[:5]).mean(), rec_jitter_deg=np.abs(dr[:5]).mean(),
                             dstep_err_deg=np.abs(dp[:5] - dr[:5]).mean(), grip_jitter=abs(dp[5]))
                prev[b] = a[b].copy()
                if smol:
                    n = min(policy.chunk, len(e["act"]) - i)
                    dc = chunk[b, :n] - e["act"][i:i + n]
                    m.update(chunk_arm_deg=(np.abs(dc) * half)[:, :5].mean(), chunk_mse=np.mean(dc ** 2),
                             chunk_mse_z=np.mean((dc / sd) ** 2))
                for k, v in m.items():
                    r.setdefault(k, []).append(float(v))
        print(f"{s + len(group)}/{len(files)} episodes", flush=True)
    out = {t: dict(frames=len(r["arm_deg"]), **{k: float(np.mean(v)) for k, v in r.items()}) for t, r in sorted(res.items())}
    keys_m = list(next(iter(res.values())).keys())
    out["all"] = dict(frames=sum(len(r["arm_deg"]) for r in res.values()),
                      **{k: float(np.mean([x for r in res.values() for x in r[k]])) for k in keys_m})
    if not args.limit:
        json.dump(out, open(f"{args.run}/eval_{args.episodes}_{args.ckpt}{variant}.json", "w"), indent=1)
    for t, r in out.items():
        print(f"  {t:4s} " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()))


if __name__ == "__main__":
    main()
