"""
make_meta.py (2026-11-10): per-task constants of the live pipeline -> <PHYSGO_ASSETS>/so101/live/meta.json, from the
recorded datasets and the offline lift (run once on hessian-p2, env_vla + /tmp/livepkgs):
  task            the task sentence (meta/tasks.parquet, as so101_policy_h5.py / tokenize_h5.py used it)
  depth_info      the top_depth codec of the recordings (LeRobot log quantization), so live sensor depth is coded and
                  decoded exactly as the recorded depth was
  gripper_colour  median over the dataset's episodes of the lift's per-episode gripper colour (lift_so101_v2 measures
                  it over 20 frames of the whole episode, which a live run does not have yet)
  prompts         the SAM 3.1 phrases (lift_so101.PROMPTS)
  start_state     median joint state at the first frame of the episodes (run_live.py --go_home)
  python make_meta.py [--data /dev/shm/data/so101_v2]
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

import live_common as C
import lift_so101 as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/dev/shm/data/so101_v2")
    args = ap.parse_args()
    meta = {}
    for task, name in C.NAMES.items():
        info, eps, tasks = L.load_meta(f"{args.data}/{name}")
        di = info["features"]["observation.images.top_depth"]["info"]
        cols = np.array([np.load(f)["gripper_colour"] for f in sorted(glob.glob(f"{C.ASSETS}/so101/lifted_v2/{name}/*_cam.npz"))])
        ft = info["features"]
        d = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"])
                       for f in sorted(glob.glob(f"{args.data}/{name}/data/chunk-*/file-*.parquet"))])
        s0 = np.stack(d[d["frame_index"] == 0].sort_values("episode_index")["observation.state"].values)
        meta[name] = dict(
            start_state=np.median(s0, 0).tolist(), start_state_std=s0.std(0).tolist(),
            letter=task, task=tasks[0], fps=int(info["fps"]),
            depth_info={k: di[k] for k in ("depth_unit", "video.depth_min", "video.depth_max", "video.shift", "video.use_log")},
            gripper_colour=np.median(cols, 0).tolist(), gripper_colour_std=cols.std(0).tolist(), gripper_colour_n=len(cols),
            prompts=[list(p) for p in L.PROMPTS[name]],
            top_hw=ft["observation.images.top"]["shape"][:2], wrist_hw=ft["observation.images.wrist"]["shape"][:2],
            depth_hw=ft["observation.images.top_depth"]["shape"][:2])
        print(task, json.dumps(meta[name])[:400])
    os.makedirs(C.asset("so101/live"), exist_ok=True)
    json.dump(meta, open(C.asset("so101/live/meta.json"), "w"), indent=1)
    print("wrote", C.asset("so101/live/meta.json"))


if __name__ == "__main__":
    main()
