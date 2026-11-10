"""
hf_backup_so101.py (2026-11-07): back up the SO-101 lifting results from p2 to a PUBLIC Hugging Face dataset (after
packing and checking, so a dead shell never means re-extracting). Uploads only what cannot be
rebuilt in minutes:
  lifted_v2/<dataset>/episode_*.npz (+ _cam.npz)   the lifted episodes (SAM 3.1 + sensor depth + kinematics: ~hours)
  lifted_v2/logs/*.log                             per-episode summaries of the lift
  calib/*.json                                     top-camera, depth registration, kinematic calibrations
  task_emb/*.npy, demos.json                       BERT task embeddings, the 5 policy demos per task
  code/                                            the SO-101 code version that produced them
The packed folder is not uploaded: prep_so101.py pack rebuilds it from lifted_v2 in minutes.
Token: HF_TOKEN_PATH (p2 RAM only, /dev/shm/hf_auth/token).
  python hf_backup_so101.py --repo TuanDepZai/physgs-so101 [--dry]
"""
import argparse
import fnmatch
import os

from huggingface_hub import HfApi

W = "/dev/shm/physgo/so101"
CODE = "/dev/shm/physgo/code_so101/ATM/physgs_so101"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    api = HfApi()
    print("account", api.whoami()["name"])
    items = [(f"{W}/lifted_v2", "lifted_v2", ["*.npz", "logs/*.log", "logs/*.txt"]),
             (f"{W}/calib", "calib", ["*.json"]),
             (f"{W}/task_emb", "task_emb", ["*.npy"]),
             (CODE, "code/physgs_so101", ["*.py", "*.sh"]),
             ("/dev/shm/physgo/tools", "code/tools", ["lift_v2_run.sh", "so101_wm_run.sh"])]
    # count what upload_folder will send: allow_patterns are fnmatch patterns on the path relative to the folder, and
    # fnmatch's * also matches "/" (so "*.npz" covers lifted_v2/<dataset>/episode_*.npz)
    files = [os.path.join(dp, f) for src, _, pats in items for dp, _, fs in os.walk(src) for f in fs
             if any(fnmatch.fnmatch(os.path.relpath(os.path.join(dp, f), src), p) for p in pats)]
    n, size = len(files), sum(os.path.getsize(f) for f in files)
    print(f"{n} files, {size / 1e9:.2f} GB")
    if args.dry:
        return
    api.create_repo(args.repo, repo_type="dataset", private=False, exist_ok=True)
    for src, dst, pats in items:
        api.upload_folder(folder_path=src, path_in_repo=dst, repo_id=args.repo, repo_type="dataset", allow_patterns=pats,
                          commit_message=f"SO-101 lift v2: {dst}")
        print("uploaded", dst, flush=True)
    api.upload_file(path_or_fileobj=f"{W}/demos.json", path_in_repo="demos.json", repo_id=args.repo, repo_type="dataset")
    info = api.repo_info(args.repo, repo_type="dataset")
    print("done", args.repo, "private" if info.private else "public")


if __name__ == "__main__":
    main()
