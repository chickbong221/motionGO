"""
Fixed per-task train / val split of a packed LIBERO dataset by demo index, written to <data>/split.json and used by
every training and evaluation script in place of the random 5% split.

    python -m physgs.data.make_split --data data/packed/libero_spatial_v2 --train_demos 47
(demos 0..46 of every task train, 47..49 validate)
"""
import argparse
import collections
import json
import os
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--train_demos", type=int, required=True, help="demos with index below this train, the rest validate")
    a = ap.parse_args()
    with open(os.path.join(a.data, "meta.json")) as f:
        episodes = json.load(f)["episodes"]
    train, val = [], []
    count = collections.Counter()
    for i, e in enumerate(episodes):
        demo = int(re.search(r"demo_(\d+)\.npz$", e["file"]).group(1))
        task = os.path.basename(os.path.dirname(e["file"]))
        (train if demo < a.train_demos else val).append(i)
        count[task, demo < a.train_demos] += 1
    tasks = sorted({t for t, _ in count})
    with open(os.path.join(a.data, "split.json"), "w") as f:
        json.dump(dict(train=train, val=val, train_demos=a.train_demos), f)
    print(f"{len(tasks)} tasks: train {len(train)}, val {len(val)} episodes; per task (train, val):",
          sorted({(count[t, True], count[t, False]) for t in tasks}))


if __name__ == "__main__":
    main()
