"""
hf_backup_run.py (2026-11-07): keep the SO-101 world-model run recoverable if shell p2 dies (p1 died 11-06 with its
runs): every --every seconds, if the run's checkpoint changed since the last upload, upload best.pt, best_step.txt,
last.pt, last_opt.pt (exact resume, train_joint --resume), log.jsonl, schedule.txt and the run's stdout log to the
public dataset repo under runs/<name>/. Stops after uploading once more when <run>/wm/done appears.
Token: HF_TOKEN_PATH (p2 RAM only).
  python hf_backup_run.py --run /dev/shm/physgo/runs/so101/so101_rel4_lab --repo TuanDepZai/physgs-so101
"""
import argparse
import os
import time

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--every", type=int, default=1800)
    args = ap.parse_args()
    api = HfApi()
    name = os.path.basename(args.run.rstrip("/"))
    wm = f"{args.run}/wm"
    files = ["best.pt", "best_step.txt", "last.pt", "last_opt.pt", "log.jsonl", "schedule.txt"]
    last_sig = None
    while True:
        done = os.path.exists(f"{wm}/done")
        sig = tuple(os.path.getmtime(f"{wm}/{f}") if os.path.exists(f"{wm}/{f}") else 0 for f in ("last.pt", "best.pt"))
        if sig != last_sig and sig[0]:
            time.sleep(60)   # let a checkpoint being written finish
            ops = []
            from huggingface_hub import CommitOperationAdd
            for f in files:
                if os.path.exists(f"{wm}/{f}"):
                    ops.append(CommitOperationAdd(path_in_repo=f"runs/{name}/wm/{f}", path_or_fileobj=f"{wm}/{f}"))
            for f in (f"{args.run}/wm.log", f"{args.run}/init_pretrain_best_step.txt"):
                if os.path.exists(f):
                    ops.append(CommitOperationAdd(path_in_repo=f"runs/{name}/{os.path.basename(f)}", path_or_fileobj=f))
            try:
                api.create_commit(args.repo, repo_type="dataset", operations=ops,
                                  commit_message=f"{name}: checkpoint backup ({time.strftime('%m-%d %H:%M', time.gmtime())} UTC)")
                step = open(f"{wm}/best_step.txt").read()[:60] if os.path.exists(f"{wm}/best_step.txt") else ""
                print(time.strftime("%m-%d_%H:%M", time.gmtime()), "uploaded", len(ops), "files;", step, flush=True)
                last_sig = sig
            except Exception as ex:  # noqa: BLE001  network hiccup: try again next round
                print(time.strftime("%m-%d_%H:%M", time.gmtime()), "upload failed:", ex, flush=True)
        if done and sig == last_sig:
            print("run done, final checkpoint uploaded", flush=True)
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
