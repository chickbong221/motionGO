"""
prep_so101.py (2026-11-06): the small steps around lift_so101.py.
  task_emb  BERT embedding of each dataset's instruction, computed exactly as LIBERO's (scripts/preprocess_libero.
            get_task_embs: bert-base-cased, max_length 25 with padding, pooler_output) -> <out>/<dataset>.npy
  demos     the policy demos: 5 episodes per dataset spread evenly over its episodes -> demos.json
  pack      physgs.data.format.pack of all lifted episodes of the given datasets (native object order, as LIBERO,
            robots first; object slots = the most tokens in any episode, --max_objects 0, i.e. gripper + 3 for the
            SO-101 tasks: padding slots cost memory and time only) + split.json with every episode in train AND val (all
            episodes are training video; the held-out check runs on the training videos)
  python prep_so101.py task_emb --datasets <dir>,<dir> --out <dir>
  python prep_so101.py demos --datasets <dir>,<dir> --per_task 5 --out demos.json
  python prep_so101.py pack --episodes_dir <dir> --datasets <name>,<name> --out <packed dir>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd


def task_emb(args):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tz = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModel.from_pretrained("bert-base-cased").eval()
    os.makedirs(args.out, exist_ok=True)
    for root in args.datasets.split(","):
        root = root.rstrip("/")
        text = pd.read_parquet(f"{root}/meta/tasks.parquet").index.tolist()
        assert len(text) == 1, f"{root}: expected one task, got {text}"
        tok = tz(text=text, add_special_tokens=True, max_length=25, padding="max_length", return_attention_mask=True,
                 return_tensors="pt")
        with torch.no_grad():
            e = model(tok["input_ids"], tok["attention_mask"])["pooler_output"][0].numpy().astype(np.float32)
        np.save(f"{args.out}/{os.path.basename(root)}.npy", e)
        print(os.path.basename(root), "|", text[0], "| tokens", int(tok["attention_mask"].sum()), "| norm", float(np.linalg.norm(e)))


def demos(args):
    out = {}
    for root in args.datasets.split(","):
        root = root.rstrip("/")
        eps = pd.concat([pd.read_parquet(f, columns=["episode_index", "length"])
                         for f in sorted(glob.glob(f"{root}/meta/episodes/chunk-*/file-*.parquet"))])
        ids = sorted(eps["episode_index"].tolist())
        pick = [ids[int(i)] for i in np.linspace(0, len(ids) - 1, args.per_task).round()]
        out[os.path.basename(root)] = pick
        print(os.path.basename(root), len(ids), "episodes -> demos", pick)
    json.dump(out, open(args.out, "w"), indent=1)


def pack(args):
    from physgs.data.format import pack as pack_eps
    paths = []
    for name in args.datasets.split(","):
        p = sorted(glob.glob(f"{args.episodes_dir}/{name}/episode_*.npz"))
        p = [x for x in p if not x.endswith("_cam.npz")]
        # --exclude name:ep,...  (2026-11-07: task C episode 5 has neither the drawer unit nor the red block in the
        # scene, the arm moves a white stick: not a task episode)
        bad = {int(t.split(":")[1]) for t in args.exclude.split(",") if t and name.startswith(t.split(":")[0])}
        p = [x for x in p if int(x.split("_")[-1].split(".")[0]) not in bad]
        print(name, len(p), "episodes")
        paths += p
    mo = args.max_objects or max(int(np.load(x)["npts"].shape[1]) for x in paths)
    print("object slots", mo)
    pack_eps(paths, args.out, max_objects=mo, sort_by_visibility=False)
    n = len(paths)
    json.dump(dict(train=list(range(n)), val=list(range(n)), files=[os.path.basename(os.path.dirname(x)) + "/" + os.path.basename(x) for x in paths],
                   excluded=args.exclude, note="all episodes train and val"),
              open(f"{args.out}/split.json", "w"))
    print("packed", n, "episodes ->", args.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("task_emb"); a.add_argument("--datasets", required=True); a.add_argument("--out", required=True)
    b = sub.add_parser("demos"); b.add_argument("--datasets", required=True); b.add_argument("--per_task", type=int, default=5)
    b.add_argument("--out", required=True)
    c = sub.add_parser("pack"); c.add_argument("--episodes_dir", required=True); c.add_argument("--datasets", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--max_objects", type=int, default=0, help="0: the most tokens in any episode")
    c.add_argument("--exclude", default="", help="dataset-name-prefix:episode,... left out of the packed data")
    args = ap.parse_args()
    dict(task_emb=task_emb, demos=demos, pack=pack)[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
