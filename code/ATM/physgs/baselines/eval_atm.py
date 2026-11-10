"""
Evaluate an ATM policy checkpoint with exactly the loop of physgs.engine.eval_policy, for a fair comparison:
same environment construction (same cameras and render settings), same initial states (init_states[0..N-1]),
same 5 settle steps, same horizon, success as soon as the task is solved.

ATM's in-training rollouts instead give each vectorised env a block of initial states and advance a counter
across rollouts, so different checkpoints are scored on different scenes; this script removes that difference.

    python -m physgs.baselines.eval_atm --suite libero_spatial --task <task_name> \
        --policy results/policy/<run>/model_100.ckpt --episodes 20 --out runs/atm_eval/model_100.json
"""
import argparse
import json
import os

import numpy as np
import torch
from omegaconf import OmegaConf

from atm.policy import *  # noqa: F401,F403  (BCViLTPolicy and friends, as in ATM's own evaluator)
from libero import benchmark, get_libero_path
from libero.envs import OffScreenRenderEnv
from physgs.engine.eval_policy import OBS_KEYS, POLICY_VIEWS
from scripts.render_libero_multiview import add_placeholder_cameras, place_extra_cameras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--horizon_steps", type=int, default=600)
    ap.add_argument("--lift_views", default="agentview,mv_left,mv_right,mv_top",
                    help="rendered but unused, so the environment is built exactly as for PhysGS")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--libero_root", default="./data/libero/")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = "cuda"
    lift_views = args.lift_views.split(",")

    cfg = OmegaConf.load(os.path.join(os.path.dirname(args.policy), "config.yaml"))
    cfg.model_cfg.load_path = None
    policy = eval(cfg.model_name)(**OmegaConf.to_container(cfg.model_cfg, resolve=True)).to(device)
    policy.load(args.policy)
    policy.eval()

    add_placeholder_cameras()
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_id = suite.get_task_names().index(args.task)
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    views = list(dict.fromkeys(list(POLICY_VIEWS) + lift_views))
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_names=views, camera_heights=args.size,
                             camera_widths=args.size, camera_depths=True, camera_segmentations="instance")
    init_states = suite.get_task_init_states(task_id)
    from scripts.preprocess_libero import get_task_bert_embs
    task_emb = get_task_bert_embs(args.libero_root)[task.language]

    successes = []
    for ep in range(args.episodes):
        env.reset()
        obs = env.set_init_state(init_states[ep % len(init_states)])
        place_extra_cameras(env, lift_views)
        dummy = np.zeros(7)
        dummy[-1] = -1.0
        for _ in range(5):
            obs, _, _, _ = env.step(dummy)
        policy.reset()
        success = False
        for _ in range(args.horizon_steps):
            rgb = np.stack([obs[f"{v}_image"][::-1] for v in POLICY_VIEWS])[None]
            extra = {k: obs[OBS_KEYS[k]][None] for k in policy.extra_state_keys}
            with torch.no_grad():
                action, _ = policy.act(rgb, task_emb[None], extra)
            obs, _, done, _ = env.step(action[0])
            if env.check_success():
                success = True
                break
        successes.append(success)
        print(f"episode {ep}: {'success' if success else 'fail'}  (running {np.mean(successes):.2f})", flush=True)
    result = dict(suite=args.suite, task=args.task, policy=args.policy, success_rate=float(np.mean(successes)),
                  episodes=len(successes), successes=[bool(s) for s in successes])
    print(result)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=1)
    env.close()


if __name__ == "__main__":
    main()
