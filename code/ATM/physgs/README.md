# PhysGS: object-centric physical Gaussians + action-free RSSM for few-label imitation

Built on ATM (branch `physgs`). ATM's own code is untouched; everything new lives in `physgs/`
and `scripts/render_libero_multiview.py`.

## Pipeline
| Step | Code | Output |
|---|---|---|
| PhysicalAI download + processing (batched, small disk) | `physgs/data/physai_batch.py` (uses `physai_hub.py`, `physai_process.py`) | `data/physai_episodes/<scene>/<run>.npz` |
| LIBERO multi-view re-render (RGB-D, segmentation, cameras, sim physics, contacts) | `scripts/render_libero_multiview.py` | `data/physgs_libero/<suite>/<task>/demo_k.hdf5` |
| LIBERO → episodes (object Gaussians + labels) | `physgs/data/libero_process.py` | `data/libero_episodes/<suite>/<task>/demo_k.npz` |
| Pack episodes for training | `physgs/data/pack_episodes.py` (`--native_order` for LIBERO) | `data/packed/<name>/` |
| Object encoder / decoder | `physgs/engine/train_encoder.py`, model `physgs/model/object_encoder.py` (≈22M) | `runs/encoder_*/last.pt` |
| Tokens | `physgs/engine/precompute_tokens.py` | `data/packed/<name>/tokens.npy` |
| Action-free object RSSM | `physgs/engine/train_rssm.py`, model `physgs/model/object_rssm.py` (≈9M) | `runs/rssm_*/last.pt` |
| Policy forecasts | `physgs/engine/precompute_forecast.py` | `data/forecast/<suite>/<task>/demo_k.npy` |
| Policy (ATM ViLT with forecast tokens) | `physgs/engine/train_policy.py`, `physgs/policy/` , config `physgs/conf/policy.yaml` | `runs/policy/*/model_best.ckpt` |
| Online evaluation | `physgs/engine/eval_policy.py` | success rate json |
| Upload processed data | `physgs/data/hf_upload.py` | private HF dataset |

Episode format and packed format: `physgs/data/format.py`.

## PhysicalAI conventions (checked with `physgs/data/probe_physai.py`)
- fx = fy = `focal_length[0]` × width, principal point at the centre (square pixels).
- `usd_transform` transposed = camera-to-world, OpenGL camera; world is z-up.
- Depth MKV = Euclidean ray distance; the scale differs per camera/run, so it is fitted to the ground plane z = 0
  (median residual 2–8 mm).
- Some depth videos are truncated to a few frames; such cameras are dropped and runs need ≥ 2 complete cameras.
  `physai_hub.complete_depth_runs` pre-selects runs from tar-header sizes.
- Physics npz rows are not in the order of their `segmentation_colors`; rows are attached to bodies by Hungarian
  matching of CoM tracks to lifted body centroids. Mass / friction / restitution come from `<cam>_static.json`
  by prim root.
- In some runs (~20%, clustered in some shards) a camera's `_raw.json` colour → prim map is itself scrambled
  (e.g. the floor colour is labelled as a cone). The ground-plane calibration doubles as the check: such cameras
  get no fit or a large residual (> 5 cm) and are dropped; runs with < 2 valid cameras are rejected. A few
  rejected runs are kept in `data/physai_failed/` for inspection.

## Commands (wine4: `conda activate physgs-data` for data, the `physgs` env for LIBERO / training)
```
python -m physgs.data.physai_batch --scenes objects_falling,ball_mixer,rolling_ramp_objects,towers --shards_per_scene 5
python -m scripts.render_libero_multiview --suite libero_spatial --max_demos 1 --check
python -m scripts.render_libero_multiview --suite libero_spatial
python -m physgs.data.libero_process --src data/physgs_libero/libero_spatial --out data/libero_episodes/libero_spatial
python -m physgs.data.pack_episodes --src data/physai_episodes --out data/packed/physai --max_objects 24
python -m physgs.data.pack_episodes --src data/libero_episodes/libero_spatial --out data/packed/libero_spatial --max_objects 16 --native_order
python -m physgs.engine.train_encoder --data data/packed/physai --out runs/encoder_physai
python -m physgs.engine.train_encoder --data data/packed/libero_spatial --init runs/encoder_physai/last.pt --out runs/encoder_libero_spatial --steps 20000 --lr 1e-4
python -m physgs.engine.precompute_tokens --data data/packed/physai --ckpt runs/encoder_libero_spatial/last.pt
python -m physgs.engine.precompute_tokens --data data/packed/libero_spatial --ckpt runs/encoder_libero_spatial/last.pt
python -m physgs.engine.train_rssm --data data/packed/physai --out runs/rssm_physai
python -m physgs.engine.train_rssm --data data/packed/libero_spatial --init runs/rssm_physai/last.pt --out runs/rssm_libero_spatial --steps 20000 --lr 1e-4
python -m physgs.engine.precompute_forecast --data data/packed/libero_spatial --rssm runs/rssm_libero_spatial/last.pt --out data/forecast/libero_spatial
python -m scripts.split_libero_dataset --folder data/physgs_libero
python -m physgs.engine.train_policy train_dataset="[...bc_train_10 dirs...]" val_dataset="[...val dirs...]" forecast_dir=data/forecast/libero_spatial model_cfg.forecast_cfg.num_objects=16 model_cfg.forecast_cfg.feat_dim=146
python -m physgs.engine.eval_policy --suite libero_spatial --task <task> --policy runs/policy/<run>/model_best.ckpt --encoder runs/encoder_libero_spatial/last.pt --rssm runs/rssm_libero_spatial/last.pt --packed data/packed/libero_spatial
```

## Status
- PhysicalAI processing verified on objects_falling (conventions, label matching, visual check). Large-scale
  download waits on the depth-completeness index (HF rate limits anonymous header reads; set `HF_TOKEN`).
- Encoder, tokens and RSSM train end to end on a small packed set (GPU smoke test).
- LIBERO env on wine4 is being built; the LIBERO re-render, conversion, policy training and online eval are
  written but not yet run.
- Not yet done: physics-variation wrapper (Step 7), the ATM reference run.

