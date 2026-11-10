"""
Success rates of the sweep: per variant and checkpoint (best = lowest val loss, final), the suite average for
each seed and mean +- std over seeds.

    python physgs/sweep/summarize.py /mnt/data/duongnm2/physgs/runs/sweep
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/duongnm2/physgs/runs/sweep"
rows = defaultdict(lambda: defaultdict(dict))  # (variant, ckpt) -> seed -> task -> rate
for f in glob.glob(os.path.join(root, "*", "eval_*", "s*_*.json")):
    variant, ckpt = f.split(os.sep)[-3], f.split(os.sep)[-2][5:]
    seed, task = os.path.basename(f)[:-5].split("_", 1)
    rows[(variant, ckpt)][seed][task] = json.load(open(f))["success_rate"]

print(f"{'variant':8s} {'ckpt':6s} {'seeds':>5s} {'tasks/seed':>12s}  mean +- std over seeds   per seed")
for v, ck in sorted(rows):
    per_seed = {s: np.mean(list(t.values())) for s, t in sorted(rows[(v, ck)].items())}
    n_tasks = [len(t) for _, t in sorted(rows[(v, ck)].items())]
    m = np.array(list(per_seed.values()))
    print(f"{v:8s} {ck:6s} {len(m):5d} {str(n_tasks):>12s}  {m.mean():.3f} +- {m.std():.3f}         "
          + " ".join(f"{s}={x:.2f}" for s, x in per_seed.items()))
