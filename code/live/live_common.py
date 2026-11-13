"""
live_common.py (2026-11-10): paths and per-task constants of the live SO-101 pipeline.
  PHYSGO_CODE    the SO-101 code tree (default /dev/shm/physgo/code_so101: ATM/physgs, ATM/physgs_so101, policy)
  PHYSGO_ASSETS  the assets tree (default /dev/shm/physgo), the same relative layout as on hessian-p2:
                   so101/calib/kin_v5.json, so101/calib/<dataset>.json      top camera pose / intrinsics, depth registration
                   so101/robot/so101_new_calib_camera.urdf (+ assets/)     SO-101 model (forward kinematics)
                   so101/task_emb/<dataset>.npy                            BERT task embedding (world model context)
                   so101/policy_h5/norm.json                               joint scaling of the policy
                   so101/live/meta.json                                    make_meta.py: task text, depth codec, gripper colour
                   runs/so101/so101_rel4_lab_80k/wm/best.pt                the world model
                   runs/<policy>/policy/{config.yaml,model.ckpt}           the policies (physgo, plain_smolVLA)
"""
import json
import os
import sys

CODE = os.environ.get("PHYSGO_CODE", "/dev/shm/physgo/code_so101")
ASSETS = os.environ.get("PHYSGO_ASSETS", "/dev/shm/physgo")
for _p in (f"{CODE}/policy", f"{CODE}/ATM/physgs_so101", f"{CODE}/ATM"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# eval_policy_so101 / infer_check_so101 / so101_policy_h5 put the hessian-p2 code paths first in sys.path when they
# are imported; import the packages they use now, so they bind to PHYSGO_CODE wherever the pipeline runs
import physgs  # noqa: E402,F401
import lift_so101  # noqa: E402,F401
import so101_kin  # noqa: E402,F401

NAMES = {"A": "so101-left_arm_task_red_cube_out_of_pot_into_the_orange_hollow_cube",
         "B": "so101_left_arm_open_drawer_pick_red_cube_stack_on_blue_cube",
         "C": "so101-left_arm_task_drawer_placing_red_squre_block"}
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FPS = 15                     # policy / world-model rate: every 2nd frame of the 30 fps recordings
WM80K = "runs/so101/so101_rel4_lab_80k/wm/best.pt"
# policy -> (run, checkpoint, world model its forecasts come from). Both policies are smolVLA's default chunked setting
# (512 px, 50 actions at 30 fps, MEAN_STD, relative targets); run_live executes those chunks at 30 Hz
POLICIES = {"physgo": ("runs/physgo", "model", WM80K),
            "plain": ("runs/plain_smolVLA", "model", None)}


def dataset_name(task):
    return NAMES.get(task, task)


def asset(rel):
    return os.path.join(ASSETS, rel)


def load_meta():
    return json.load(open(asset("so101/live/meta.json")))
