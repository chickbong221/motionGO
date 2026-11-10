"""
Download the episodes of a Hugging Face dataset repo that are not in any of the given local folders (e.g. the share
processed on another machine), into --out with the repo's layout (<scene>/<run>.npz).

    python -m physgs.data.hf_fetch_missing --repo TuanDepZai/physgs-physai-objects-1080 \
        --have data/physai1080_A,data/physai1080_B,data/physai1080_W --out data/physai1080_hub
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--have", required=True, help="comma-separated local episode folders (<scene>/<run>.npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="download at most this many (0 = all; for tests)")
    a = ap.parse_args()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    remote = [f for f in api.list_repo_files(a.repo, repo_type="dataset") if f.endswith(".npz") and f.count("/") == 1]
    have = {f for d in a.have.split(",") + [a.out] if os.path.isdir(d)
            for f in (f"{s}/{n}" for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))
                      for n in os.listdir(os.path.join(d, s)) if n.endswith(".npz"))}
    todo = sorted(set(remote) - have)
    print(f"{len(remote)} episodes on the Hub, {len(set(remote) & have)} already local, {len(todo)} to download", flush=True)
    if a.limit:
        todo = todo[:a.limit]

    def get(f):
        for attempt in range(5):
            try:
                hf_hub_download(a.repo, f, repo_type="dataset", local_dir=a.out)
                return True
            except Exception as e:  # flaky network: back off and retry this file
                print(f"{f}: {e!r}"[:200], flush=True)
                time.sleep(20 * (attempt + 1))
        return False

    with ThreadPoolExecutor(a.threads) as pool:
        ok = sum(pool.map(get, todo))
    print(f"downloaded {ok}/{len(todo)}", flush=True)
    if ok < len(todo):  # fail loudly: training must not go on silently without part of the data
        raise SystemExit(f"{len(todo) - ok} episodes could not be downloaded")


if __name__ == "__main__":
    main()
