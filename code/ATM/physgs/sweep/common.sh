# Shared helpers for the H100 sweep (sourced by h100_all.sh and by queued work). Everything lives under /mnt/data.
B=/mnt/data/duongnm2/physgs
source $B/activate.sh > /dev/null 2>&1
D=$B/data; R=$B/runs/sweep; S=$B/code/ATM/physgs/sweep; Q=$R/queue
SUITE=libero_spatial
TASKS=$(ls $D/libero/$SUITE | sed 's/_demo.hdf5$//')
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p $R/logs $R/done $Q/pending $Q/running $Q/finished

log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

# GPU lanes: `lane 0` / `lane 1` pins the current (sub)shell to one of the job's GPUs by UUID
mapfile -t GPUS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null)
lane() { export CUDA_VISIBLE_DEVICES=${GPUS[$1]:-${GPUS[0]}} LANE=$1; }   # a 1-GPU job maps any lane to its GPU

# block until this lane's GPU has at least $1 GB free (several processes share each GPU)
gpu_wait() {
  while [ "$(nvidia-smi -i ${CUDA_VISIBLE_DEVICES} --query-gpu=memory.free --format=csv,noheader,nounits)" -lt $(( $1 * 1024 )) ]; do
    sleep 60
  done
}

# run NAME CMD...  full output in $R/logs/NAME.log; skipped when $R/done/NAME exists; up to 3 attempts
# (a later attempt usually succeeds when the first failed because another process on the GPU held memory)
run() {
  local name=$1; shift
  [ -f $R/done/$name ] && return 0
  for attempt in 1 2 3; do
    log "start $name (gpu $LANE, attempt $attempt)"
    if "$@" > $R/logs/$name.log 2>&1; then touch $R/done/$name; log "done  $name"; return 0; fi
    log "FAIL  $name: $(grep -v -i warn $R/logs/$name.log | tail -2 | tr '\n' ' ' | cut -c1-300)"
    cp $R/logs/$name.log $R/logs/$name.fail$attempt.log
    sleep 600
  done
  return 1
}

wait_done() { until [ -f $R/done/$1 ]; do sleep 120; done; }
wait_file() { until [ -f $1 ]; do sleep 120; done; }

# packed dataset view whose tokens.npy belongs to one encoder (all other files are symlinks to the packed data)
view() {
  local v=$D/views/$1/$2
  mkdir -p $v
  for f in $D/packed/$2/*; do [ "$(basename $f)" = tokens.npy ] || ln -sfn $f $v/; done
  echo $v
}

# encoder NAME DIM ENC_DEPTH DEC_DEPTH PRETRAIN_STEPS FINETUNE_STEPS LR GPU_GB [PHYSAI_DATA]
# PhysicalAI pre-training (batch 512) -> LIBERO fine-tuning (batch 256) -> tokens of both datasets
encoder() {
  local n=$1 a="--dim $2 --enc_depth $3 --dec_depth $4" lr=$7 data=${9:-$D/packed/physai}
  gpu_wait $8
  run enc_${n}_physai python -m physgs.engine.train_encoder --data $data --out $R/enc_$n/physai $a \
    --steps $5 --batch 512 --lr $lr --workers 4 --eval_every 5000 || return 1
  run enc_${n}_libero python -m physgs.engine.train_encoder --data $D/packed/$SUITE --init $R/enc_$n/physai/last.pt \
    --out $R/enc_$n/libero $a --steps $6 --batch 256 --lr $(python -c "print($lr / 3)") --workers 4 --eval_every 5000 || return 1
  tokens $n $R/enc_$n/libero/last.pt
}

tokens() {  # tokens NAME ENCODER_CKPT
  run tok_$1_physai python -m physgs.engine.precompute_tokens --data $(view $1 physai) --ckpt $2 --batch 2048 || return 1
  run tok_$1_libero python -m physgs.engine.precompute_tokens --data $(view $1 $SUITE) --ckpt $2 --batch 2048
}

# rssm VARIANT ENC_NAME DETER DIM LAYERS LR PRETRAIN_STEPS FINETUNE_STEPS GPU_GB
# PhysicalAI pre-training -> LIBERO fine-tuning (action-free, all 50 videos per task) -> checks -> forecasts
rssm() {
  local v=$1 e=$2 a="--deter $3 --dim $4 --layers $5" lr=$6
  local P=$D/views/$e/physai L=$D/views/$e/$SUITE
  wait_done tok_${e}_libero
  gpu_wait $9
  run rssm_${v}_physai python -m physgs.engine.train_rssm --data $P --out $R/$v/rssm_physai $a --steps $7 --batch 256 \
    --lr $lr --workers 3 --eval_every 5000 || return 1
  run rssm_${v}_libero python -m physgs.engine.train_rssm --data $L --init $R/$v/rssm_physai/last.pt --out $R/$v/rssm_libero $a \
    --steps $8 --batch 256 --lr $(python -c "print($lr / 3)") --workers 3 --eval_every 5000 || return 1
  forecasts $v $e
}

forecasts() {  # forecasts VARIANT ENC_NAME   (RSSM checks + policy inputs from $R/VARIANT/rssm_libero)
  local v=$1 P=$D/views/$2/physai L=$D/views/$2/$SUITE
  [ -f $R/$v/rssm_physai/last.pt ] && run evalrssm_${v}_physai python -m physgs.engine.eval_rssm --data $P --rssm $R/$v/rssm_physai/last.pt
  run evalrssm_${v}_libero python -m physgs.engine.eval_rssm --data $L --rssm $R/$v/rssm_libero/last.pt
  run fc_$v python -m physgs.engine.precompute_forecast --data $L --rssm $R/$v/rssm_libero/last.pt \
    --out $D/forecast/$v/$SUITE --batch 32
}

# policies VARIANT FORECAST_VARIANT ENCODER_CKPT [extra overrides...]
# GraphMimic / ATM protocol: one multi-task policy per suite, 10 labelled demos per task, 3 seeds.
# ATM's libero_vilt architecture; trained longer (150 epochs) at batch 128 per update.
policies() {
  local v=$1 fv=$2 enc=$3; shift 3
  local tr="" va=""
  for t in $TASKS; do tr="$tr,$D/physgs_libero/$SUITE/${t}_demo/bc_train_10"; va="$va,$D/physgs_libero/$SUITE/${t}_demo/val"; done
  for s in 0 1 2; do
    gpu_wait 24
    run pol_${v}_s$s python -m physgs.engine.train_policy out_dir=$R/$v/policy_s$s seed=$s num_workers=3 epochs=150 \
      save_freq=25 "train_dataset=[${tr#,}]" "val_dataset=[${va#,}]" forecast_dir=$D/forecast/$fv/$SUITE \
      model_cfg.forecast_cfg.num_objects=16 model_cfg.forecast_cfg.feat_dim=146 "$@" &
    sleep 300
  done
  wait
  for s in 0 1 2; do for ck in best final; do evaluate $v $s $fv $enc $ck; done; done
}

# evaluate VARIANT SEED FORECAST_VARIANT ENCODER_CKPT CKPT: 20 rollouts per task, horizon 600;
# each task runs as 10 parallel envs x 2 rounds, two tasks at a time
evaluate() {
  local v=$1 s=$2 fv=$3 enc=$4 ck=$5
  for t in $TASKS; do
    while [ $(jobs -rp | wc -l) -ge 2 ]; do sleep 30; done
    run ev_${v}_${ck}_s${s}_$t python -m physgs.engine.eval_policy_vec --suite $SUITE --task $t \
      --policy $R/$v/policy_s$s/model_$ck.ckpt --encoder $enc --rssm $R/$fv/rssm_libero/last.pt --packed $D/packed/$SUITE \
      --episodes 20 --num_envs 10 --horizon_steps 600 --libero_root $D/libero/ --out $R/$v/eval_$ck/s${s}_$t.json &
    sleep 20
  done
  wait
}
