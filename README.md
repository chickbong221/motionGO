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
| World model (PhysGO) | `TuanDepZai/physgs-so101`: `runs/so101_rel4_lab_80k/wm/best.pt` (`download.sh`) |
| PhysGO policy | `TuanDepZai/physgs-so101`: `placeholder/physgo/model.ckpt` and `config.yaml` (`download.sh`) |
| Plain smolVLA policy | `TuanDepZai/physgs-so101`: `placeholder/plain_smolVLA/model.ckpt` and `config.yaml` (`download.sh`) |
| Calibration, task embeddings, joint scaling | `TuanDepZai/physgs-so101`: `calib/`, `task_emb/`, `so101_v3/norm_all.json` (`download.sh`) |

`download.sh` puts everything under `assets/` and checks it against `assets.sha256`. `--policy physgo` loads the PhysGO
policy and the world model, and `--policy plain` loads the plain smolVLA policy.

## 3. Run

Connect the follower arm (USB), the RealSense top camera and the wrist camera. Leave the leader arm unplugged.

```bash
export PHYSGO_CODE=$PWD/code PHYSGO_ASSETS=$PWD/assets
cd code/live
python devices.py
```

`python devices.py` must print a port, a calibration file, a RealSense serial and a wrist camera, with no
`NOT FOUND`.

Dry run (no motor commands):

```bash
python run_live.py --task A --policy physgo --dry_run
python run_live.py --task B --policy physgo --dry_run
python run_live.py --task C --policy physgo --dry_run
```

PhysGO on the robot:

```bash
python run_live.py --task A --policy physgo --go_home
python run_live.py --task B --policy physgo --go_home
python run_live.py --task C --policy physgo --go_home
```

Plain smolVLA on the robot:

```bash
python run_live.py --task A --policy plain --go_home
python run_live.py --task B --policy plain --go_home
python run_live.py --task C --policy plain --go_home
```

| Task | Scene |
|---|---|
| `A` | red cube in the steel pot, orange hollow cube |
| `B` | plastic drawer unit, red cube, blue cube |
| `C` | plastic drawer unit, red square block |

When the run asks:

- **ENTER to use the calibration file:** press ENTER.
- **Ready:** check `scene_check.png` in the printed run folder (the red dots must be on the real gripper), then press
  ENTER to start.
- **Stop:** press Ctrl-C.

If `devices.py` reports a missing or ambiguous device, add the matching option:

| Option | Example |
|---|---|
| `--port` | `--port /dev/ttyACM0` |
| `--robot_id` | `--robot_id my_so101_follower` (the name of the `.json` file) |
| `--top_serial` | `--top_serial 135522071234` |
| `--wrist_cam` | `--wrist_cam /dev/video6` |

Output folder: `code/live/live_runs/<time>_<task>_<policy>_robot/`, holding `steps.npz`, `summary.json` and
`scene_check.png`.

Robot model in `assets/so101/robot`: TheRobotStudio/SO-ARM100 (Apache-2.0).
