"""
check_live_sam.py (2026-11-10): does live_sam.LiveSegmenter (one frame at a time) reproduce SAM 3.1's offline pass?
On the first --frames 15 fps frames of a recorded episode, per object phrase, the most confident mask per frame of
  live      LiveSegmenter (hotstart 0, grounding batch 1, frame by frame)
  off0      the offline pass (lift_so101.Segmenter.track, whole clip at once) with the same settings: must equal live
  off15     the offline pass with SAM 3.1's defaults as the lift ran it (hotstart 15, grounding batch 16): the change
            from going causal
IoU per frame (both empty = 1) and the fraction of frames where an object is found, per phrase.
  python check_live_sam.py --task B --episode 1 [--frames 60]
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


def best(d, shape):
    return max(d.values(), key=lambda mp: mp[1])[0] if d else np.zeros(shape, bool)


def iou(a, b):
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="B")
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--data", default="/dev/shm/data/so101_v2")
    ap.add_argument("--no_share", action="store_true", help="live without the shared image encoder")
    ap.add_argument("--live_only", action="store_true", help="timing only, no offline passes")
    ap.add_argument("--trim_check", action="store_true", help="live with vs without trimming old tracker outputs")
    args = ap.parse_args()
    name = C.dataset_name(args.task)
    info, eps, _ = L.load_meta(f"{args.data}/{name}")
    row = eps.loc[args.episode]
    top = L.decode_episode(f"{args.data}/{name}", row, "observation.images.top", int(row["length"]))[:args.frames]
    phrases = [p[1] for p in L.PROMPTS[name]]
    if args.trim_check:   # live with vs without trimming old tracker outputs: masks must be identical
        out = {}
        for keep in (None, 32):
            seg = LiveSegmenter(phrases, max_frames=args.frames + 1, trim_keep=keep)
            torch.cuda.reset_peak_memory_stats()
            ms = [[] for _ in phrases]
            for t in range(len(top)):
                r = seg.start(top[t]) if t == 0 else seg.step(top[t])
                for k, d in enumerate(r):
                    ms[k].append(best(d, top.shape[1:3]))
            out[keep] = (ms, round(torch.cuda.memory_allocated() / 1e9, 2), round(torch.cuda.max_memory_allocated() / 1e9, 2))
            seg.close()
            del seg
            torch.cuda.empty_cache()
        res = {ph: dict(iou_trim_vs_full_min=round(float(min(iou(a, b) for a, b in zip(out[None][0][k], out[32][0][k]))), 4),
                        found=round(float(np.mean([x.any() for x in out[32][0][k]])), 3)) for k, ph in enumerate(phrases)}
        print(json.dumps(dict(task=name, episode=args.episode, frames=len(top), per_phrase=res,
                              alloc_end_gb_full=out[None][1], alloc_end_gb_trim=out[32][1],
                              peak_gb_full=out[None][2], peak_gb_trim=out[32][2]), indent=1))
        return
    seg = LiveSegmenter(phrases, max_frames=args.frames + 1, share_backbone=not args.no_share)
    live = [[] for _ in phrases]
    import time
    dt = []
    for t in range(len(top)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = seg.start(top[t]) if t == 0 else seg.step(top[t])
        torch.cuda.synchronize()
        dt.append(time.perf_counter() - t0)
        for k, d in enumerate(r):
            live[k].append(best(d, top.shape[1:3]))
    seg.close()
    seg.share_backbone = False      # the offline passes below run without the shared encoder
    timing = dict(live_step_ms_median=round(1000 * float(np.median(dt[1:])), 1),
                  live_step_ms_p90=round(1000 * float(np.percentile(dt[1:], 90)), 1),
                  first_step_ms=round(1000 * dt[0], 1), share_backbone=not args.no_share,
                  gpu_mem_peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    m = seg.p.model
    off = L.Segmenter.__new__(L.Segmenter)      # the offline segmenter on the same predictor
    off.p = seg.p
    res = {}
    passes = {} if args.live_only else {"off0": (0, 1, 1, 0), "off15": (15, 16, 16, 8)}
    for tag, (hs, pb, gb, th) in passes.items():
        m.hotstart_delay, m.postprocess_batch_size, m.batched_grounding_batch_size = hs, pb, gb
        m.hotstart_unmatch_thresh = m.hotstart_dup_thresh = th
        for k, ph in enumerate(phrases):
            with seg._ctx():
                tr = off.track(top, ph)
            mo = [best(d, top.shape[1:3]) for d in tr]
            ious = [iou(a, b) for a, b in zip(live[k], mo)]
            r = res.setdefault(ph, {})
            r[f"iou_live_vs_{tag}_mean"] = round(float(np.mean(ious)), 4)
            r[f"iou_live_vs_{tag}_min"] = round(float(np.min(ious)), 4)
            r[f"found_{tag}"] = round(float(np.mean([x.any() for x in mo])), 3)
            r["found_live"] = round(float(np.mean([x.any() for x in live[k]])), 3)
        torch.cuda.empty_cache()
    print(json.dumps(dict(task=name, episode=args.episode, frames=len(top), timing=timing, per_phrase=res), indent=1))


if __name__ == "__main__":
    main()
