"""
Upload a packed dataset folder (physgs.data.pack_episodes output) to a Hugging Face dataset repo, one file per commit
with retries. Files larger than --max_gb (the Hub refuses files over 50 GB) go up as 10 GB parts
<name>.part-000, -001, ... cut into --stage; `cat <name>.part-* > <name>` restores them.

    python -m physgs.data.hf_upload_packed --repo <user>/<repo> --folder data/packed/libero_90256 --prefix packed \
        --stage /mnt/data/.../hf_stage
(the token is read from HF_TOKEN or ~/.cache/huggingface/token)
"""
import argparse
import glob
import os
import subprocess
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import CommitOperationAdd, HfApi  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--prefix", default="packed")
    ap.add_argument("--stage", required=True, help="where parts of large files are cut")
    ap.add_argument("--max_gb", type=float, default=40)
    ap.add_argument("--part_gb", type=int, default=10)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--public", action="store_true", help="ignored: repos are always public")
    args = ap.parse_args()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, repo_type="dataset", private=False, exist_ok=True)
    on_hub = set(api.list_repo_files(args.repo, repo_type="dataset"))
    os.makedirs(args.stage, exist_ok=True)
    uploads = []  # (local file, path in repo)
    for f in sorted(glob.glob(os.path.join(args.folder, "*"))):
        if not os.path.isfile(f):
            continue
        name = os.path.basename(f)
        if os.path.getsize(f) <= args.max_gb * 2 ** 30:
            uploads.append((f, f"{args.prefix}/{name}"))
            continue
        parts = sorted(glob.glob(os.path.join(args.stage, name + ".part-*")))
        if not parts or os.path.exists(os.path.join(args.stage, name + ".cutting")):
            open(os.path.join(args.stage, name + ".cutting"), "w").close()
            subprocess.run(["split", "-b", f"{args.part_gb}G", "-d", "-a", "3", f, os.path.join(args.stage, name + ".part-")], check=True)
            os.remove(os.path.join(args.stage, name + ".cutting"))
            parts = sorted(glob.glob(os.path.join(args.stage, name + ".part-*")))
        print(f"{name}: {os.path.getsize(f) / 2 ** 30:.1f} GB in {len(parts)} parts", flush=True)
        uploads += [(p, f"{args.prefix}/{os.path.basename(p)}") for p in parts]
    todo = [(f, r) for f, r in uploads if r not in on_hub]
    print(f"{len(uploads)} files, {len(uploads) - len(todo)} already on the Hub, {len(todo)} to upload", flush=True)
    for f, r in todo:
        for attempt in range(args.retries):
            try:
                t = time.time()
                api.create_commit(args.repo, [CommitOperationAdd(path_in_repo=r, path_or_fileobj=f)], repo_type="dataset",
                                  commit_message=f"add {r}")
                print(f"uploaded {r} ({os.path.getsize(f) / 2 ** 30:.2f} GB, {time.time() - t:.0f} s)", flush=True)
                break
            except Exception as e:  # network hiccup: wait and retry this file
                print(f"{r} failed (attempt {attempt + 1}/{args.retries}): {e!r}"[:300], flush=True)
                time.sleep(60 * (attempt + 1))
    print("packed upload pass finished", flush=True)


if __name__ == "__main__":
    main()
