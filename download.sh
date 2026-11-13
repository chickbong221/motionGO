#!/bin/bash
# download.sh: world model, policies, calibration and task embeddings from Hugging Face (public dataset repo
# TuanDepZai/physgs-so101) into assets/, in the layout run_live.py reads.
#   bash download.sh
set -e
cd "$(dirname "$0")"
REPO=TuanDepZai/physgs-so101
T=assets/.hf
A=so101-left_arm_task_red_cube_out_of_pot_into_the_orange_hollow_cube
B=so101_left_arm_open_drawer_pick_red_cube_stack_on_blue_cube
C=so101-left_arm_task_drawer_placing_red_squre_block
hf download $REPO --repo-type dataset --local-dir $T \
  calib/kin_v5.json calib/$A.json calib/$B.json calib/$C.json \
  task_emb/$A.npy task_emb/$B.npy task_emb/$C.npy \
  so101_v3/norm_all.json \
  runs/so101_rel4_lab_80k/wm/best.pt \
  placeholder/physgo/config.yaml placeholder/physgo/model.ckpt \
  placeholder/plain_smolVLA/config.yaml placeholder/plain_smolVLA/model.ckpt
rm -rf assets/so101/calib assets/so101/task_emb assets/so101/policy_h5 assets/runs
mkdir -p assets/so101/policy_h5 assets/runs/so101/so101_rel4_lab_80k assets/runs/physgo/policy assets/runs/plain_smolVLA/policy
mv $T/calib $T/task_emb assets/so101/
mv $T/so101_v3/norm_all.json assets/so101/policy_h5/norm.json
mv $T/runs/so101_rel4_lab_80k/wm assets/runs/so101/so101_rel4_lab_80k/wm
mv $T/placeholder/physgo/config.yaml $T/placeholder/physgo/model.ckpt assets/runs/physgo/policy/
mv $T/placeholder/plain_smolVLA/config.yaml $T/placeholder/plain_smolVLA/model.ckpt assets/runs/plain_smolVLA/policy/
rm -rf $T
sha256sum -c assets.sha256
