"""
check_sam_memory.py (2026-11-10): which part of a live SAM 3.1 session grows with the episode? Runs LiveSegmenter on the
first --frames frames of a recorded episode and prints, every --every frames, the allocated GPU memory and the GPU bytes
held under each top-level key of the first session's inference state (and of its tracker states).
  python check_sam_memory.py --task A --episode 1 --frames 150
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import live_common as C  # noqa: E402
import lift_so101 as L  # noqa: E402
from live_sam import LiveSegmenter  # noqa: E402


def gpu_bytes(x, seen):
    if id(x) in seen:
        return 0
    seen.add(id(x))
    if isinstance(x, torch.Tensor):
        return x.untyped_storage().nbytes() if x.is_cuda else 0
    if isinstance(x, dict):
        return sum(gpu_bytes(v, seen) for v in x.values())
    if isinstance(x, (list, tuple, set)):
        return sum(gpu_bytes(v, seen) for v in x)
    if hasattr(x, "__dict__") and not isinstance(x, type):
        return sum(gpu_bytes(v, seen) for v in vars(x).values())
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="A")
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--every", type=int, default=30)
    ap.add_argument("--data", default="/dev/shm/data/so101_v2")
    args = ap.parse_args()
    name = C.dataset_name(args.task)
    info, eps, _ = L.load_meta(f"{args.data}/{name}")
    row = eps.loc[args.episode]
    top = L.decode_episode(f"{args.data}/{name}", row, "observation.images.top", int(row["length"]))[:args.frames]
    seg = LiveSegmenter([p[1] for p in L.PROMPTS[name]], max_frames=args.frames + 1)
    import collections
    import gc

    def census():
        c = collections.Counter()
        for o in gc.get_objects():
            try:
                if torch.is_tensor(o) and o.is_cuda:
                    c[(tuple(o.shape), str(o.dtype))] += 1
            except Exception:  # noqa: BLE001
                pass
        return c

    first = None
    for t in range(len(top)):
        seg.start(top[t]) if t == 0 else seg.step(top[t])
        if t % args.every == 0 or t == len(top) - 1:
            st = seg.p._all_inference_states[seg.sids[0]]["state"]
            keys = {k: round(gpu_bytes(v, set()) / 1e6, 1) for k, v in st.items()}
            trk = {}
            for s2 in st.get("sam2_inference_states", []):
                for k, v in s2.items():
                    trk[k] = round(trk.get(k, 0) + gpu_bytes(v, set()) / 1e6, 1)
            print(json.dumps(dict(t=t, alloc_gb=round(torch.cuda.memory_allocated() / 1e9, 2),
                                  session0_mb={k: v for k, v in keys.items() if v > 0.5},
                                  tracker_mb={k: v for k, v in trk.items() if v > 0.5},
                                  sam2_states=[type(x).__name__ for x in st.get("sam2_inference_states", [])][:4])), flush=True)
            c = census()
            if first is None:
                first = c
            else:
                grow = sorted(((c[k] - first.get(k, 0), k) for k in c if c[k] - first.get(k, 0) > 0), reverse=True)[:8]
                print("  growing tensors (count, (shape, dtype)):", grow, flush=True)


if __name__ == "__main__":
    main()
