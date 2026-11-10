"""
so101_policy_h5.py (2026-11-07): the SO-101 episodes in the demo format of the PhysGO policy (physgs.policy.dataset.
ForecastBCDataset, the files scripts/render_libero_multiview.py writes for LIBERO), so physgs.engine.train_policy runs
on them unchanged (the original model and settings; the backbone is an option).
  <out>/<task>_demo/demo_<ep>.hdf5      every episode (the evaluation reads the non-demo ones)
  <out>/<task>_demo/bc_train_5/         links to the 5 policy demos per task (demos.json)
  <out>/<task>_demo/val/                links to 2 non-demo episodes per task (validation loss only)
  <out>/norm.json                       per-joint min / max used to map state and action to [-1, 1]
Each file, as ForecastBCDataset reads it:
  root/top/video, root/wrist/video   uint8 (1, T, 3, 256, 256): the frames the world model lifted (30 fps frames 2k,
                                     T = 15 fps steps, so forecast row t is frame t), letterboxed to 256 x 256 with the
                                     aspect kept (smolVLA's pre-training size; the original policy resizes to 128)
  root/actions                       (T, 6) recorded action at 30 fps frame 2k, scaled to [-1, 1] per joint
  root/task_emb_bert                 (768,) BERT task embedding (prep_so101.py task_emb, as LIBERO's)
  root/extra_states/joint_states     (T, 7) the 5 arm joints of the recorded state at frame 2k, scaled like the actions,
                                     zero-padded to LIBERO's 7 (ATM's extra-state encoder has joint_states 7 built in)
  root/extra_states/gripper_states   (T, 2) the gripper of that state, twice (LIBERO: the two finger positions)
  attrs: extra_states_source="render" (state taken at the frame instant: never shifted), task, dataset, episode
Scaling: one min / max per joint over state and action of all episodes (the joints' physical range).
Run with env_vision (PyAV decode):  python so101_policy_h5.py --out <dir> [--worker i --nworkers n]
then python so101_policy_h5.py --out <dir> --links   (after all workers: norm check + bc_train_5 / val links)
"""
import argparse
import glob
import json
import os
import sys

import cv2
import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, "/dev/shm/physgo/code_so101/ATM/physgs_so101")
import lift_so101 as L  # noqa: E402

DATA = "/dev/shm/data/so101_v2"
W = "/dev/shm/physgo/so101"
DATASETS = ["so101-left_arm_task_red_cube_out_of_pot_into_the_orange_hollow_cube",
            "so101_left_arm_open_drawer_pick_red_cube_stack_on_blue_cube",
            "so101-left_arm_task_drawer_placing_red_squre_block"]
EXCLUDE = {("so101-left_arm_task_drawer_placing_red_squre_block", 5)}   # no drawer, no block in the scene
SIZE = 256


def letterbox(frames, size=SIZE):
    """(T, H, W, 3) -> (T, 3, size, size): resize to fit, aspect kept, zero padding centred."""
    T, H, Wd = frames.shape[:3]
    s = size / max(H, Wd)
    h, w = round(H * s), round(Wd * s)
    out = np.zeros((T, size, size, 3), np.uint8)
    y0, x0 = (size - h) // 2, (size - w) // 2
    for i in range(T):
        out[i, y0:y0 + h, x0:x0 + w] = cv2.resize(frames[i], (w, h), interpolation=cv2.INTER_AREA)
    return out.transpose(0, 3, 1, 2)


def tables(name):
    root = f"{DATA}/{name}"
    return pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state", "action"])
                      for f in sorted(glob.glob(f"{root}/data/chunk-*/file-*.parquet"))])


def norm_stats():
    lo, hi = np.full(6, np.inf), np.full(6, -np.inf)
    for n in DATASETS:
        d = tables(n)
        for k in ("observation.state", "action"):
            a = np.stack(d[k].values)
            lo, hi = np.minimum(lo, a.min(0)), np.maximum(hi, a.max(0))
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--links", action="store_true")
    args = ap.parse_args()
    demos = json.load(open(f"{W}/demos.json"))
    os.makedirs(args.out, exist_ok=True)
    nf = f"{args.out}/norm.json"
    if not os.path.exists(nf) and args.worker == 0 and not args.links:
        lo, hi = norm_stats()
        json.dump(dict(min=lo.tolist(), max=hi.tolist(), joints=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
                                                                 "wrist_roll", "gripper"],
                       note="x_scaled = 2 (x - min) / (max - min) - 1, over state and action of all episodes"), open(nf, "w"), indent=1)
    while not os.path.exists(nf):
        import time
        time.sleep(5)
    nrm = json.load(open(nf))
    lo, hi = np.array(nrm["min"]), np.array(nrm["max"])
    scale = lambda x: (2 * (x - lo) / (hi - lo) - 1).astype(np.float32)  # noqa: E731
    if args.links:
        rng = np.random.default_rng(0)
        for n in DATASETS:
            task = os.path.basename(n)
            d = f"{args.out}/{task}_demo"
            eps = sorted(int(f.split("_")[-1][:-5]) for f in glob.glob(f"{d}/demo_*.hdf5"))
            for sub, ids in (("bc_train_5", demos[n]), ("val", list(rng.choice([e for e in eps if e not in demos[n]], 2, replace=False)))):
                os.makedirs(f"{d}/{sub}", exist_ok=True)
                for e in ids:
                    lk = f"{d}/{sub}/demo_{e}.hdf5"
                    if not os.path.lexists(lk):
                        os.symlink(f"../demo_{e}.hdf5", lk)
            print(task[:40], len(eps), "episodes; bc_train_5", sorted(demos[n]), "val", sorted(os.listdir(f"{d}/val")))
        return
    emb = {n: np.load(f"{W}/task_emb/{n}.npy").astype(np.float32).reshape(-1) for n in DATASETS}
    packed = json.load(open(f"{W}/packed/so101_3task/split.json"))["files"]
    pk_T = {}
    ep_start = np.load(f"{W}/packed/so101_3task/ep_start.npy")
    for i, f in enumerate(packed):
        pk_T[f] = int(ep_start[i + 1] - ep_start[i])
    jobs = [(n, int(e)) for n in DATASETS for e in L.load_meta(f"{DATA}/{n}")[1].index.tolist() if (n, int(e)) not in EXCLUDE]
    tab = {}
    for n, e in jobs[args.worker::args.nworkers]:
        out = f"{args.out}/{n}_demo/demo_{e}.hdf5"
        if os.path.exists(out):
            continue
        root = f"{DATA}/{n}"
        info, eps, tasks = L.load_meta(root)
        row = eps.loc[e]
        n30 = int(row["length"])
        top = L.decode_episode(root, row, "observation.images.top", n30)
        wrist = L.decode_episode(root, row, "observation.images.wrist", n30)
        T = pk_T[f"{n}/episode_{e:04d}.npz"]
        assert len(top) >= T and len(wrist) >= T, (n, e, len(top), len(wrist), T)
        if n not in tab:
            tab[n] = tables(n)
        st = tab[n][tab[n]["episode_index"] == e].sort_values("frame_index")
        state = np.stack(st["observation.state"].values)[::2][:T]
        action = np.stack(st["action"].values)[::2][:T]
        assert len(state) == T
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".tmp"
        with h5py.File(tmp, "w") as f:
            r = f.create_group("root")
            for v, fr in (("top", top[:T]), ("wrist", wrist[:T])):
                r.create_group(v).create_dataset("video", data=letterbox(fr)[None], compression="lzf", chunks=(1, 1, 3, SIZE, SIZE))
            r.create_dataset("actions", data=scale(action))
            r.create_dataset("task_emb_bert", data=emb[n])
            ex = r.create_group("extra_states")
            ss = scale(state)
            ex.create_dataset("joint_states", data=np.concatenate([ss[:, :5], np.zeros((T, 2), np.float32)], 1))
            ex.create_dataset("gripper_states", data=np.repeat(ss[:, 5:6], 2, 1))
            f.attrs["extra_states_source"] = "render"
            f.attrs["task"] = tasks[0]
            f.attrs["dataset"] = n
            f.attrs["episode"] = e
        os.replace(tmp, out)
        print(n[:40], e, "T", T, "demo" if e in demos[n] else "", flush=True)


if __name__ == "__main__":
    main()
