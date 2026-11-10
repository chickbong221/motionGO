#!/bin/bash
# One-time setup on the H100 cluster. Everything heavy lives under $PHYSGS_ROOT (never $HOME).
#   bash physgs/setup/setup_env.sh            # run on the login node (CPU only)
set -e
export PHYSGS_ROOT=${PHYSGS_ROOT:-/mnt/data/duongnm2/physgs}
B=$PHYSGS_ROOT
mkdir -p $B/{code,envs,cache/{pip,conda_pkgs,hf,torch},data,runs,libero_cfg,logs}
export CONDA_PKGS_DIRS=$B/cache/conda_pkgs PIP_CACHE_DIR=$B/cache/pip HF_HOME=$B/cache/hf TORCH_HOME=$B/cache/torch

source ~/miniconda3/etc/profile.d/conda.sh
[ -d $B/envs/physgs ] || conda create -y -p $B/envs/physgs python=3.9
conda activate $B/envs/physgs

pip install -U "pip>=25.1"                     # can resume interrupted downloads (--resume-retries)
pip install setuptools==63.2.0 wheel==0.38.4   # gym 0.21 (used by LIBERO's vec envs) needs these
pip install --resume-retries 50 torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
# keep torch pinned: lightning / timm accept any torch and would otherwise pull the much larger latest build
printf "torch==2.0.1
torchvision==0.15.2
" > $B/constraints.txt
pip install --resume-retries 50 -c $B/constraints.txt --extra-index-url https://download.pytorch.org/whl/cu118 numpy==1.24.3 scipy==1.10.1 h5py==3.9.0 natsort==8.4.0 einops==0.6.1 easydict==1.10 click==8.1.7 \
    hydra-core==1.3.2 omegaconf==2.3.0 lightning==2.1.0 wandb==0.15.9 termcolor==2.3.0 tqdm==4.66.1 \
    opencv-python==4.8.0.76 pillow matplotlib==3.7.2 pandas==2.0.3 moviepy==1.0.3 imageio==2.31.2 \
    imageio-ffmpeg==0.4.9 flow-vis==0.1 cloudpickle==1.6.0 bddl==1.0.1 mujoco==2.3.7 \
    transformers==4.31.0 tokenizers==0.13.3 timm==0.9.6 huggingface-hub==0.16.4 requests future==0.18.2 protobuf==3.20.3
pip install "pip<24.1" && pip install --no-build-isolation gym==0.21.0  # gym 0.21 metadata is rejected by pip>=24.1

cd $B/code
[ -d ATM ] || git clone --recursive https://github.com/Large-Trajectory-Model/ATM.git
pip install -e ATM/third_party/robosuite
pip install -e ATM/third_party/robomimic

# LIBERO asks interactively for paths on first import unless a config exists.
cat > $B/libero_cfg/config.yaml <<EOF
benchmark_root: $B/code/ATM/libero
bddl_files: $B/code/ATM/libero/bddl_files
init_states: $B/code/ATM/libero/init_files
datasets: $B/data/libero_raw
assets: $B/code/ATM/libero/assets
task_embeddings: $B/code/ATM/libero/task_embedding_caches
EOF

cat > $B/activate.sh <<EOF
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $B/envs/physgs
export PHYSGS_ROOT=$B LIBERO_CONFIG_PATH=$B/libero_cfg
export PIP_CACHE_DIR=$B/cache/pip HF_HOME=$B/cache/hf TORCH_HOME=$B/cache/torch
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1  # LIBERO init-state files are plain pickles (torch>=2.6 default)
cd $B/code/ATM
EOF
echo "setup done; source $B/activate.sh"
