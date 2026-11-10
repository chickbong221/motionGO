#!/bin/bash
# install.sh (2026-11-10): Python environment for run_live.py on the robot PC (Linux, NVIDIA GPU, CUDA 12.8
# driver). The versions are those of the environment the pipeline was tested in on hessian-p2 (env_vla + the SAM 3.1
# overlay), plus LeRobot's hardware extras (Feetech motor bus, RealSense) that the cluster did not need.
#   bash install.sh [env dir, default ~/motiongo_env]
set -e
ENV=${1:-$HOME/motiongo_env}
python3.12 -m venv "$ENV"
source "$ENV/bin/activate"
pip install --upgrade pip
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install "lerobot[feetech,intelrealsense,smolvla]==0.7.0" "transformers==5.22.2" "numpy==2.2.6" \
    "huggingface_hub==2.4.0" "safetensors==0.8.0" "accelerate==1.15.0" "av==15.1.0" "h5py==3.16.0" \
    "omegaconf==2.4.0" "einops==0.8.2" "pandas==2.3.3" "pillow==13.0.0"
pip install --no-deps "robomimic==0.3.0"
# SAM 3.1 (the commit the lift used) and its dependencies; --no-deps keeps numpy 2 (sam3 pins numpy<2 but runs on 2.2)
pip install --no-deps "git+https://github.com/facebookresearch/sam3.git@f1647e8d0c1d280991f4d1b7d974c2704537fe66"
pip install --no-deps "timm==1.0.20" "iopath==0.1.10" portalocker "ftfy==6.1.1" wcwidth "scipy==1.15.3" \
    "pycocotools==2.0.11" "trimesh==5.2.0" tqdm regex
python - <<'PY'
import torch, lerobot, transformers, numpy, scipy, timm, trimesh
from sam3.model_builder import build_sam3_predictor
from lerobot.robots.so_follower import SO101Follower
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
print("OK torch", torch.__version__, "cuda", torch.cuda.is_available(), "lerobot", lerobot.__version__,
      "transformers", transformers.__version__, "numpy", numpy.__version__)
PY
echo "Next: hf auth login (an account with access to facebook/sam3.1), or pass --sam_ckpt <sam3.1_multiplex.pt>"
