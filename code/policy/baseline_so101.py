"""
baseline_so101.py (2026-11-07): trivial baselines for the offline SO-101 action metrics (eval_policy_so101.py), so the
policy numbers can be read: how much of the score does "command the current joint position" already get?
  hold      action(t) = state(t)                 (first action)
  hold_chunk  actions(t..t+24) = state(t)         (the whole 25-step chunk)
  replay    action(t) = action(t-1)              (the previous recorded command: a strong smoothness baseline)
Same metrics as the evaluation: arm_deg, grip, mse ([-1, 1] scale), mse_z (z-score with the 15 demo actions' mean/std).
  python baseline_so101.py [--episodes test|val]
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np

PG = "/dev/shm/physgo/so101/policy_h5"
TASK = lambda d: "A" if "pot" in d else ("B" if "open_drawer" in d else "C")  # noqa: E731


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="test", choices=("test", "val"))
    args = ap.parse_args()
    nrm = json.load(open(f"{PG}/norm.json"))
    half = (np.array(nrm["max"]) - np.array(nrm["min"])) / 2
    demo = np.concatenate([h5py.File(f, "r")["root/actions"][:] for f in sorted(glob.glob(f"{PG}/*_demo/bc_train_5/*.hdf5"))])
    sd = demo.std(0) + 1e-8
    res = {}
    for d in sorted(glob.glob(f"{PG}/*_demo")):
        dem = {os.path.basename(os.path.realpath(f)) for f in glob.glob(f"{d}/bc_train_5/*.hdf5")}
        val = {os.path.basename(os.path.realpath(f)) for f in glob.glob(f"{d}/val/*.hdf5")}
        for f in sorted(glob.glob(f"{d}/demo_*.hdf5")):
            b = os.path.basename(f)
            kind = "demo" if b in dem else ("val" if b in val else "test")
            if kind != args.episodes:
                continue
            with h5py.File(f, "r") as h:
                a = h["root/actions"][:]
                ex = h["root/extra_states"]
                s = np.concatenate([ex["joint_states"][:, :5], ex["gripper_states"][:, :1]], 1)
            T = len(a)
            for name, pred in (("hold", s), ("replay", np.concatenate([a[:1], a[:-1]]))):
                dd = pred - a
                r = res.setdefault(name, {})
                for k, v in dict(arm_deg=(np.abs(dd) * half)[:, :5].mean(1), grip=(np.abs(dd) * half)[:, 5],
                                 mse=(dd ** 2).mean(1), mse_z=((dd / sd) ** 2).mean(1)).items():
                    r.setdefault(k, []).extend(v.tolist())
            r = res.setdefault("hold_chunk", {})
            for t in range(T):
                n = min(25, T - t)
                dc = s[t][None] - a[t:t + n]
                r.setdefault("chunk_arm_deg", []).append(float((np.abs(dc) * half)[:, :5].mean()))
                r.setdefault("chunk_mse", []).append(float((dc ** 2).mean()))
                r.setdefault("chunk_mse_z", []).append(float(((dc / sd) ** 2).mean()))
    for name, r in res.items():
        print(f"{args.episodes:4s} {name:10s} " + "  ".join(f"{k} {np.mean(v):.4f}" for k, v in r.items()))


if __name__ == "__main__":
    main()
