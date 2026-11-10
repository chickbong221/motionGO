---
task_categories:
- robotics
tags:
- maniskill
- physics
- manipulation
- gaussians
- object-centric
- world-model
pretty_name: PhysGS ManiSkill3 physics-variation experts
---

# PhysGS ManiSkill3: physics-variation expert episodes as object Gaussians

2,950 successful expert episodes of five ManiSkill3 Panda tasks. Each task was run under 3 x 3 physics settings:
mass x{0.5, 1, 4} of the manipulated object, and object-table friction {0.1, 0.3, 1.0}. Episodes are stored in the
object-centric Gaussian format of PhysGS (the same as our LIBERO data).

## Experts
| Task | Expert | Settings | Episodes |
|---|---|---|---|
| PushCube | privileged-state PPO (100% at every setting) | 9 | 450 |
| PushCube | motion planner | 9 | 450 |
| PullCube | privileged-state PPO | 9 | 450 |
| PullCube | motion planner | 9 | 450 |
| PokeCube | privileged-state PPO | 9 | 450 |
| PullCubeTool | motion planner | 9 | 450 |
| LiftPegUpright | motion planner | 5* | 250 |

- **Episode lengths:** episodes end at the first success.
- **Action space:** all episodes use `pd_ee_delta_pose`. The motion-planner episodes were planned in joint space at
  2x speed and then converted to `pd_ee_delta_pose`.
- **Physics and simulation:** ManiSkill 3.0.1 and SAPIEN 3.0.1, controlled at 20 Hz.
- **\*LiftPegUpright:** no expert solves it at mass x0.5 or at (x1, friction 1.0), so those 4 settings are absent.

## How it was made
1. **Rendering.** Each episode is replayed from its simulator states and rendered at 256 x 256 from 5 cameras.
   - Four lifting views (`agentview`, `mv_left`, `mv_right`, `mv_top`) around the workspace.
   - A wrist camera `robot0_eye_in_hand` (the panda_wristcam RealSense pose).
   - Each view stores RGB, metric depth and segmentation.
2. **Lifting.** Tokens are the task objects (cube / peg / L-shaped tool) and the robot (`gripper`, `arm`).
   - Each token becomes up to 512 world-frame Gaussians per frame.
   - Labels per token: pose, velocity, angular velocity, contact, mass and friction of the episode's setting.
   - The red/white goal target is visual only: it appears in the images but is not a token.

## Layout
- `episodes/<source>_<task>/<setting>__<episode>/demo_0.npz`: per-episode token arrays.
  - Same fields as PhysGS LIBERO: `gauss`, `npts`, `pos`, `vel`, `angvel`, `contact`, `props`, `is_robot`,
    `obj_names`, `task_emb`, `actions`.
  - `fps` is 20 and `source` is "maniskill".
  - `task_emb` is the BERT embedding of the task instruction (LIBERO recipe).
- `packed/`: all episodes in one memory-mappable folder, with:
  - `meta.json`;
  - `split.json`: the last 10% of every (source_task, setting) validate;
  - `manifest.json`: episode counts.
- `expert_states/<source>/<task>/m<mass>_f<friction>/*.npz`: the original expert episodes in their simulator form,
  enough to re-render with any camera. Fields: actions, full simulator states, TCP, success, metadata.
