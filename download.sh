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
  policy_h5/norm.json \
  runs/so101_rel4_lab/wm/best.pt \
  runs/so101_policy/so101_smolvla1_rel/policy/config.yaml \
  runs/so101_policy/so101_smolvla1_rel/policy/model_160.ckpt \
  runs/so101_policy/so101_smolvla1_rel_plain/policy/config.yaml \
  runs/so101_policy/so101_smolvla1_rel_plain/policy/model_50.ckpt
mkdir -p assets/so101 assets/runs/so101/so101_rel4_lab assets/runs/so101_policy
rm -rf assets/so101/calib assets/so101/task_emb assets/so101/policy_h5 assets/runs/so101/so101_rel4_lab/wm \
       assets/runs/so101_policy/so101_smolvla1_rel assets/runs/so101_policy/so101_smolvla1_rel_plain
mv $T/calib $T/task_emb $T/policy_h5 assets/so101/
mv $T/runs/so101_rel4_lab/wm assets/runs/so101/so101_rel4_lab/wm
mv $T/runs/so101_policy/so101_smolvla1_rel $T/runs/so101_policy/so101_smolvla1_rel_plain assets/runs/so101_policy/
rm -rf $T
sha256sum -c assets.sha256
