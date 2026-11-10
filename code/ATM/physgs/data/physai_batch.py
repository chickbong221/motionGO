"""
Download + process PhysicalAI runs in batches that fit a small disk.

One batch = the runs of one video shard (about 100 runs). For each batch the downloader streams
the video shard (RGB mp4 only), the aligned segmentation shard (only the frames we keep), and the
depth / physics / camera shards covering those runs; then a process pool converts each run to an
episode file and the raw files are deleted. Downloading of batch k+1 overlaps processing of batch k.

    python -m physgs.data.physai_batch --scenes objects_falling,ball_mixer,rolling_ramp_objects,towers \
        --shards_per_scene 5 --out data/physai_episodes --raw data/physai_raw --workers 12
Re-running skips runs whose episode file already exists.
"""
import argparse
import bisect
import glob
import itertools
import json
import os
import queue
import re
import shutil
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from physgs.data import physai_hub as H

is_rgb = lambda n: n.endswith(".mp4") and not n.endswith("_per_clip.mp4")  # skip physics-colourised videos


def wanted_frame(name, stride):
    m = re.search(r"_(\d{4})\.png$", name)
    return m is None or int(m.group(1)) % stride == 0


def download_batch(scene, k, index, raw_dir, stride, done_runs, allowed):
    t0 = time.time()
    bdir = os.path.join(raw_dir, f"{scene}_{k:05d}")
    os.makedirs(bdir, exist_ok=True)
    tars = os.path.join(bdir, "_tars")

    def fetch_extract(path, keep, out=bdir):
        local = H.download_file(path, os.path.join(tars, os.path.basename(path)))
        names = H.extract_local(local, out, keep)
        os.remove(local)
        return names

    want = lambda n: is_rgb(n) and (allowed is None or H.run_of(n) in allowed) and H.run_of(n) not in done_runs
    names = fetch_extract(index["videos"][k][0], want)
    runs = sorted({H.run_of(n) for n in names})
    if not runs:
        shutil.rmtree(bdir, ignore_errors=True)
        return None
    rs = set(runs)
    keep_run = lambda n: H.run_of(n) in rs
    keep_seg = lambda n: keep_run(n) and (not n.endswith(".png") or wanted_frame(n, stride))
    todo = [(p, keep_seg) for p in H.shards_for_runs(index["segmentation"], runs)]
    for m in ("depths", "physics", "cameras"):
        todo += [(p, keep_run) for p in H.shards_for_runs(index[m], runs)]
    if "captions" in index:  # caption files share names with the camera files (<run>/<cam>.json): own folder
        todo += [(p, keep_run, os.path.join(bdir, "captions")) for p in H.shards_for_runs(index["captions"], runs)]
    with ThreadPoolExecutor(4) as pool:  # 4 files at a time, each over 8 connections
        list(pool.map(lambda a: fetch_extract(*a), todo))
    shutil.rmtree(tars, ignore_errors=True)
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(bdir) for f in fs) / 1e9
    print(f"[download] {scene} shard {k}: {len(runs)} runs, {len(todo) + 1} shards, {size:.1f} GB kept, "
          f"{time.time() - t0:.0f} s", flush=True)
    return dict(scene=scene, k=k, dir=bdir, runs=runs)


def run_gb(batch_dir, run, size, stride):
    """Peak memory of processing one run, estimated from its frames and cameras: physai_process keeps the decoded RGB
    (3 bytes per pixel and stored frame) and the raw depth (2 bytes) and makes the rest frame by frame; with decode
    transients ~5.5 bytes, +25% margin, plus ~0.8 GB per process (4.06 GB measured for 75 frames x 4 cameras at
    1920x1080, estimate 5.1)."""
    d = os.path.join(batch_dir, run)
    cams = [f for f in os.listdir(d) if f.endswith(".mkv")] if os.path.isdir(d) else []
    frames = 150
    for f in os.listdir(d) if os.path.isdir(d) else []:
        if f.endswith(".json") and not f.endswith(("_raw.json", "_static.json")):
            try:
                frames = int(json.load(open(os.path.join(d, f)))["frame_count"])
                break
            except Exception:
                continue
    return 1.25 * 5.5 * max(len(cams), 1) * size[0] * size[1] * ((frames + stride - 1) // stride) / 1e9 + 0.8


def _process(run_dir, out_path, stride, size=(320, 180), num_points=512, device="cpu"):
    from physgs.data.physai_process import process_run
    try:
        return process_run(run_dir, out_path, size=size, stride=stride, num_points=num_points, device=device,
                           caption_dir=os.path.join(os.path.dirname(run_dir), "captions"))
    except Exception as e:  # keep the batch going; the run is reported and skipped
        return dict(run=os.path.basename(run_dir), ok=False, reason=repr(e)[:300])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="objects_falling,ball_mixer,rolling_ramp_objects,towers")
    ap.add_argument("--shards_per_scene", type=int, default=5)
    ap.add_argument("--shard_offset", type=int, default=0)
    ap.add_argument("--out", default="data/physai_episodes")
    ap.add_argument("--raw", default="data/physai_raw")
    ap.add_argument("--index_dir", default="data/physai_index")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--prefetch", type=int, default=1, help="batches downloaded ahead of processing")
    ap.add_argument("--skip", default=None, help="file of 'scene:k' video shards already processed elsewhere")
    ap.add_argument("--shards_file", default=None,
                    help="file of 'scene:k' video shards to process (exactly these; overrides --shards_per_scene)")
    ap.add_argument("--size", default="320x180", help="decode size WxH of the 1920x1080 videos (e.g. 640x360, or native)")
    ap.add_argument("--num_points", type=int, default=512, help="Gaussians per object (stored in FPS order)")
    ap.add_argument("--captions", action="store_true", help="also download the captions and keep them in the episodes")
    ap.add_argument("--runs_file", default=None,
                    help="file of run names (one per line, e.g. objects_falling_c8917ca2_4539): process exactly these, "
                         "from the video shards that hold them")
    ap.add_argument("--queue_dir", default=None,
                    help="shared shard queue: take shards one at a time from <queue>/todo/<order>__<origin>__<scene>__<k> "
                         "(atomic rename into claimed/), so several streams balance themselves; --out may contain "
                         "{origin}, the stream whose list the shard came from")
    ap.add_argument("--stream", default="", help="name of this stream (queue claims, process log name)")
    ap.add_argument("--mem_budget_gb", type=float, default=1e9,
                    help="start runs only while their summed memory estimate (run_gb) fits; workers is the upper bound")
    ap.add_argument("--device", default="cpu", help="lifting device: cpu (numba) or cuda (lift.lift_objects_batch, "
                    "8 frames per FPS loop); each worker process initialises CUDA itself")
    args = ap.parse_args()
    skip = set(open(args.skip).read().split()) if args.skip else set()
    only = set(open(args.shards_file).read().split()) if args.shards_file else None
    size = (1920, 1080) if args.size == "native" else tuple(int(x) for x in args.size.lower().split("x"))
    runs_wanted = set(open(args.runs_file).read().split()) if args.runs_file else None
    if args.queue_dir is None:
        os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.raw, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    log_name = f"process_log_{args.stream}.jsonl" if args.stream else "process_log.jsonl"

    if args.queue_dir is not None:
        indexes = {s: H.build_index(s, modalities=H.MODALITIES + (("captions",) if args.captions else ()),
                                    cache_dir=args.index_dir) for s in args.scenes.split(",")}
        for sub in ("claimed", "done"):
            os.makedirs(os.path.join(args.queue_dir, sub), exist_ok=True)

        def claimed():  # lazily: the next shard is claimed only when the downloader is ready for it
            while True:
                if os.path.exists(os.path.join(args.queue_dir, f"STOP_{args.stream}")):  # finish the claimed shards, take no more
                    print(f"[queue] {args.stream}: STOP_{args.stream} found, no new shards", flush=True)
                    return
                todo = sorted(os.listdir(os.path.join(args.queue_dir, "todo")))
                if not todo:
                    return
                name = todo[0]
                try:  # rename is atomic on the NFS server: exactly one stream wins each shard
                    os.rename(os.path.join(args.queue_dir, "todo", name),
                              os.path.join(args.queue_dir, "claimed", f"{name}__{args.stream}"))
                except FileNotFoundError:
                    continue
                _, origin, scene, k = name.split("__")
                print(f"[queue] {args.stream} claimed {scene}:{k} (list {origin})", flush=True)
                yield scene, int(k), indexes[scene], None, args.out.format(origin=origin), name

        jobs = claimed()
    else:
        jobs = plan_jobs(args, skip, only, runs_wanted)
        print(f"[plan] {len(jobs)} video shards to process", flush=True)

    q = queue.Queue(maxsize=args.prefetch)

    def downloader():
        for scene, k, index, allowed, out, qname in jobs:
            done = {f[:-4] for f in os.listdir(os.path.join(out, scene))} if os.path.isdir(os.path.join(out, scene)) else set()
            for log_path in glob.glob(os.path.join(out, "process_log*.jsonl")):
                # runs that already failed (e.g. scrambled colour maps) are not retried
                done |= {json.loads(l)["run"] for l in open(log_path) if l.strip()}
            try:
                b = download_batch(scene, k, index, args.raw, args.stride, done, allowed)
            except Exception as e:
                print(f"[download] {scene} shard {k} failed: {e!r}", flush=True)
                if qname:  # left for a later pass (move it back into todo/ to retry)
                    os.makedirs(os.path.join(args.queue_dir, "failed"), exist_ok=True)
                    open(os.path.join(args.queue_dir, "failed", qname), "w").close()
                continue
            if b is not None:
                q.put(dict(b, out=out, qname=qname))
            elif qname:  # every run of the shard was already processed
                open(os.path.join(args.queue_dir, "done", qname), "w").close()
        q.put(None)

    threading.Thread(target=downloader, daemon=True).start()
    logs = {}
    with ProcessPoolExecutor(args.workers) as pool:
        while True:
            b = q.get()
            if b is None:
                break
            t0 = time.time()
            os.makedirs(b["out"], exist_ok=True)
            if b["out"] not in logs:
                logs[b["out"]] = open(os.path.join(b["out"], log_name), "a")
            log = logs[b["out"]]
            odir = os.path.join(b["out"], b["scene"])
            # Start runs while their estimated memory fits the budget (a ball_mixer run has 120 frames, the others 75,
            # so a fixed worker count either wastes memory or runs out of it).
            pending = list(b["runs"])
            running = {}

            def submit_fitting():
                while pending and len(running) < args.workers and (not running or sum(running.values()) + run_gb(b["dir"], pending[0], size, args.stride)
                                   <= args.mem_budget_gb):
                    r = pending.pop(0)
                    running[pool.submit(_process, os.path.join(b["dir"], r), os.path.join(odir, f"{r}.npz"), args.stride,
                                        size, args.num_points, args.device)] = run_gb(b["dir"], r, size, args.stride)

            ok = 0
            submit_fitting()
            while running:
                f = next(as_completed(list(running)))
                running.pop(f)
                submit_fitting()
                res = f.result()
                ok += bool(res.get("ok"))
                failed_dir = os.path.join(os.path.dirname(args.raw.rstrip("/")), "physai_failed")
                if not res.get("ok") and len(os.listdir(failed_dir) if os.path.isdir(failed_dir) else []) < 5:
                    shutil.move(os.path.join(b["dir"], res["run"]), os.path.join(failed_dir, res["run"]))  # keep for debugging
                log.write(json.dumps(res) + "\n")
            log.flush()
            shutil.rmtree(b["dir"], ignore_errors=True)
            if b["qname"]:
                open(os.path.join(args.queue_dir, "done", b["qname"]), "w").close()
            print(f"[process] {b['scene']} shard {b['k']}: {ok}/{len(b['runs'])} ok, {time.time() - t0:.0f} s", flush=True)


def plan_jobs(args, skip, only, runs_wanted):
    jobs = []
    for scene in args.scenes.split(","):
        index = H.build_index(scene, modalities=H.MODALITIES + (("captions",) if args.captions else ()),
                              cache_dir=args.index_dir)
        # Some regions of the dataset have truncated depth videos (a few frames, tiny files). Rank video
        # shards by the depth bytes that fall in their run range (shard sizes come from the repo listing,
        # no extra requests) and process the depth-rich ones first; truncated cameras are still dropped
        # per run during processing.
        firsts = [f for _, f in index["videos"]]
        depth_size = dict(H.list_shards("depths", scene))
        depth_bytes = [0] * len(firsts)
        for path, first in index["depths"]:
            k = max(0, bisect.bisect_right(firsts, first) - 1)
            depth_bytes[k] += depth_size.get(path, 0)
        order = sorted(range(len(firsts)), key=lambda k: -depth_bytes[k])
        print(f"[plan] {scene}: depth GB per video shard, best {[round(depth_bytes[k] / 1e9, 1) for k in order[:6]]}, "
              f"worst {[round(depth_bytes[k] / 1e9, 1) for k in order[-3:]]}", flush=True)
        order = [k for k in order if f"{scene}:{k}" not in skip]
        allowed = None
        if runs_wanted is not None:
            allowed = {r for r in runs_wanted if r.startswith(scene + "_")}
            paths = [p for p, _ in index["videos"]]
            holding = {paths.index(p) for p in H.shards_for_runs(index["videos"], sorted(allowed))}
            order = [k for k in order if k in holding]
        elif only is not None:
            order = [k for k in order if f"{scene}:{k}" in only]
        else:
            order = order[args.shard_offset:args.shard_offset + args.shards_per_scene]
        jobs.append([(scene, k, index, allowed, args.out, None) for k in order])
    # interleave scenes so that a partial download stays balanced across them
    return [j for group in itertools.zip_longest(*jobs) for j in group if j is not None]


if __name__ == "__main__":
    main()
