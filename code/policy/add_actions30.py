"""
add_actions30.py (2026-11-08): LeRobot's recommended smolVLA setting (the setting the original repo
recommends) predicts 50-action chunks at the dataset's 30 fps. The policy files hold frames and actions at the world
model's 15 fps (30 fps frames 2k); this adds the whole 30 fps action sequence as root/actions30 (T30, 6), scaled like
root/actions ([-1, 1] per joint, norm.json), so a chunk at frame k is actions30[2k : 2k + 50]. Then adds to norm.json
the z-score statistics (LeRobot's MEAN_STD normalization) over the 15 policy demos:
  zs_state           mean / std of the 6-D state (joint_states[:5] + gripper_states[:1])
  zs_abs_chunk       mean / std of absolute chunk targets actions30[2k + j], j < 50
  zs_rel_chunk       mean / std of relative chunk targets actions30[2k + j] - state(k), j < 50
Run with env_vla:  python add_actions30.py
"""
import glob
import json

import h5py
import numpy as np
import pandas as pd

PG = "/dev/shm/physgo/so101/policy_h5"
DATA = "/dev/shm/data/so101_v2"
CHUNK = 50


def main():
    nrm = json.load(open(f"{PG}/norm.json"))
    lo, hi = np.array(nrm["min"]), np.array(nrm["max"])
    scale = lambda x: (2 * (x - lo) / (hi - lo) - 1).astype(np.float32)  # noqa: E731
    tabs = {}
    files = sorted(glob.glob(f"{PG}/*_demo/demo_*.hdf5"))
    for fn in files:
        with h5py.File(fn, "r+") as f:
            n, e = str(f.attrs["dataset"]), int(f.attrs["episode"])
            if n not in tabs:
                tabs[n] = pd.concat([pd.read_parquet(p, columns=["episode_index", "frame_index", "action"])
                                     for p in sorted(glob.glob(f"{DATA}/{n}/data/chunk-*/file-*.parquet"))])
            st = tabs[n][tabs[n]["episode_index"] == e].sort_values("frame_index")
            a30 = scale(np.stack(st["action"].values))
            assert np.allclose(a30[::2][:f["root/actions"].shape[0]], f["root/actions"][:], atol=1e-5), fn
            if "actions30" in f["root"]:
                del f["root/actions30"]
            f["root"].create_dataset("actions30", data=a30)
    demos = sorted(glob.glob(f"{PG}/*_demo/bc_train_5/*.hdf5"))
    S, A, Rl = [], [], []
    for fn in demos:
        with h5py.File(fn, "r") as f:
            ex = f["root/extra_states"]
            s = np.concatenate([ex["joint_states"][:, :5], ex["gripper_states"][:, :1]], 1)
            a30 = f["root/actions30"][:]
        S.append(s)
        for k in range(len(s)):
            c = a30[2 * k: 2 * k + CHUNK]
            A.append(c)
            Rl.append(c - s[k])
    S, A, Rl = np.concatenate(S), np.concatenate(A), np.concatenate(Rl)
    for key, x in (("zs_state", S), ("zs_abs_chunk", A), ("zs_rel_chunk", Rl)):
        nrm[key] = dict(mean=x.mean(0).tolist(), std=(x.std(0) + 1e-6).tolist())
        print(key, "mean", np.round(x.mean(0), 4).tolist(), "std", np.round(x.std(0), 4).tolist())
    nrm["zs_note"] = ("z-score (LeRobot MEAN_STD) statistics over the 15 policy demos, in the [-1, 1] units: state; chunk "
                      "targets actions30[2k + j], j < 50 (absolute) and minus state(k) (relative)")
    json.dump(nrm, open(f"{PG}/norm.json", "w"), indent=1)
    print(len(files), "files got actions30")


if __name__ == "__main__":
    main()
