"""
Upload processed episodes to a public Hugging Face dataset repo, robust to a flaky uplink:
files already on the Hub are skipped, the rest go up in small commits that are retried with backoff,
so one failed file costs one chunk instead of the whole upload.

    python -m physgs.data.hf_upload --repo <user>/physgs-physai-objects --folder data/physai_episodes
(the token is read from HF_TOKEN or ~/.cache/huggingface/token)
"""
import argparse
import glob
import os
import time

# Uploads through the Xet storage backend stall on wine4; the classic LFS path works.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import CommitOperationAdd, HfApi  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--chunk", type=int, default=40, help="files per commit")
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--public", action="store_true", help="ignored: repos are always public")
    ap.add_argument("--glob", default="*/*.npz", help="files to upload, relative to --folder")
    ap.add_argument("--prefix", default="", help="folder in the repo the files go to (default: its root)")
    ap.add_argument("--meta_suffix", default="",
                    help="upload the folder's top-level json/jsonl as <name>_<suffix>.<ext>, so several processing "
                         "streams can share one repo without overwriting each other's logs")
    args = ap.parse_args()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, repo_type="dataset", private=False, exist_ok=True)

    on_hub = set(api.list_repo_files(args.repo, repo_type="dataset"))
    files = sorted(f for f in glob.glob(os.path.join(args.folder, args.glob))
                   if not f.endswith(".tmp.npz"))  # save_episode's half-written temp files

    def repo_path(f):
        rel = os.path.relpath(f, args.folder).replace(os.sep, "/")
        if args.meta_suffix and "/" not in rel:
            stem, ext = os.path.splitext(rel)
            rel = f"{stem}_{args.meta_suffix}{ext}"
        return f"{args.prefix.strip('/')}/{rel}" if args.prefix else rel

    todo = [f for f in files if repo_path(f) not in on_hub]
    meta = [f for f in glob.glob(os.path.join(args.folder, "*.jsonl")) + glob.glob(os.path.join(args.folder, "*.json"))]
    print(f"{len(files)} files locally, {len(files) - len(todo)} already on the Hub, {len(todo)} to upload", flush=True)

    chunks = [todo[i:i + args.chunk] for i in range(0, len(todo), args.chunk)] + [meta]
    done = 0
    for chunk in chunks:
        if not chunk:
            continue
        ops = [CommitOperationAdd(path_in_repo=repo_path(f), path_or_fileobj=f) for f in chunk]
        for attempt in range(args.retries):
            try:
                api.create_commit(args.repo, ops, repo_type="dataset",
                                  commit_message=f"add {len(ops)} files ({os.path.basename(os.path.dirname(chunk[0]))})")
                done += len(ops)
                print(f"committed {done}/{len(todo) + len(meta)}", flush=True)
                break
            except Exception as e:  # network hiccup: wait and retry this chunk only
                print(f"chunk failed (attempt {attempt + 1}/{args.retries}): {e!r}"[:300], flush=True)
                time.sleep(30 * (attempt + 1))
    print("upload pass finished", flush=True)


if __name__ == "__main__":
    main()
