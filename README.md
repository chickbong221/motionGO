# motionGO: live inference on SO-101

## 1. Install

```bash
git clone https://github.com/chickbong221/motionGO.git
cd motionGO
bash install.sh
source ~/motiongo_env/bin/activate
```

Requirements: Linux, an NVIDIA GPU with a CUDA 12.8 driver, and Python 3.12.

## 2. Download models

```bash
hf auth login
bash download.sh
```

`hf auth login` needs an account with access to [facebook/sam3.1](https://huggingface.co/facebook/sam3.1).

| Model | Where it comes from |
|---|---|
| SAM 3.1 | `facebook/sam3.1`, downloaded on the first run |
| smolVLA base | `lerobot/smolvla_base`, downloaded on the first run |
| World model | `TuanDepZai/physgs-so101`: `runs/so101_rel4_lab/wm/best.pt` (`download.sh`) |
| PhysGO policy | `TuanDepZai/physgs-so101`: `runs/so101_policy/so101_smolvla1_rel/policy/model_160.ckpt` (`download.sh`) |
| Plain smolVLA policy | `TuanDepZai/physgs-so101`: `runs/so101_policy/so101_smolvla1_rel_plain/policy/model_50.ckpt` (`download.sh`) |
| Calibration, task embeddings, joint scaling | `TuanDepZai/physgs-so101`: `calib/`, `task_emb/`, `policy_h5/norm.json` (`download.sh`) |

## 3. Run

```bash
export PHYSGO_CODE=$PWD/code PHYSGO_ASSETS=$PWD/assets
cd code/live
```

Dry run (no motor commands):

```bash
python run_live.py --task B --policy physgo --port /dev/ttyACM0 --robot_id <calibration id> --top_serial <RealSense serial> --wrist_cam 0 --dry_run
```

Robot run:

```bash
python run_live.py --task B --policy physgo --port /dev/ttyACM0 --robot_id <calibration id> --top_serial <RealSense serial> --wrist_cam 0 --go_home
```

Plain smolVLA: the same command with `--policy plain`.

Press ENTER to start an episode and Ctrl-C to stop it. Before the first episode, open `scene_check.png` in the run
folder: the red dots must sit on the real gripper.

| Option | Value |
|---|---|
| `--task` | `A` red cube out of the pot, `B` open the drawer and stack the cubes, `C` block into the drawer |
| `--policy` | `physgo` or `plain` |
| `--robot_id` | LeRobot calibration id of the follower arm used for the recordings |
| `--top_serial` | RealSense serial number of the top camera |
| `--wrist_cam` | OpenCV index or path, or `realsense:<serial>` |
| `--steps` | steps per episode (default 600) |
| `--max_rel` | maximum joint change per step (default 20) |
| `--num_steps` | flow-matching steps (default 10, 5 is faster) |
| `--sam_ckpt` | local `sam3.1_multiplex.pt` instead of the Hugging Face download |
| `--keep_torque` | keep motor torque on at exit |

Output folder: `code/live/live_runs/<time>_<task>_<policy>_robot/`, holding `steps.npz`, `summary.json` and
`scene_check.png`.

Robot model in `assets/so101/robot`: TheRobotStudio/SO-ARM100 (Apache-2.0).
