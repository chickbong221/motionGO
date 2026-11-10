#!/bin/bash
#SBATCH -J physgs
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 7-00:00:00
#SBATCH -o /mnt/data/duongnm2/physgs/runs/sweep/logs/lane%a_%A.out
# The H100 sweep as two 1-GPU jobs (one per lane; one GPU frees up sooner than two):
#     sbatch --array=0 h100_lane.sh ; sbatch --array=1 h100_lane.sh
# Resumable: finished stages are skipped on resubmission. The lanes coordinate through done-markers in $R/done.
#
#   lane 0: data prep -> base encoder + RSSM-L (bL) -> policies;  BC baseline (forecast zeroed);
#           CPU: the rest of PhysicalAI (download + processing) into data/physai_episodes_full
#   lane 1: base encoder + duong's base RSSM (bB) -> policies;     encoder-L (45M) + RSSM-L (LL) -> policies
#   queue : a script dropped into $Q/pending whose first line is "# lane N" runs in lane N's job as soon as it
#           appears. A job ends when its lane and its queued items are finished, unless $Q/KEEP exists (then it
#           waits for more work, at most 4 h idle). $Q/STOP (or $Q/STOP<N>) ends it after running items finish.
LANE_ID=${SLURM_ARRAY_TASK_ID:-$1}
source /mnt/data/duongnm2/physgs/code/ATM/physgs/sweep/common.sh
lane $LANE_ID
log "lane $LANE_ID: job $SLURM_JOB_ID on $(hostname), $(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=name --format=csv,noheader), $(nproc) CPUs"

# ---------------------------------------------------------------- queue runner (this lane's items only)
mine() { ls $1/*.sh 2>/dev/null | while read f; do head -1 $f | grep -q "lane $LANE_ID" && echo $f; done; }
queue_runner() {
  while true; do
    for f in $(mine $Q/pending); do
      local n=$(basename $f .sh)
      mv $f $Q/running/
      ( log "queue: start $n"; bash -c "source $S/common.sh; lane $LANE_ID; source $Q/running/$n.sh" > $R/logs/queue_$n.log 2>&1
        log "queue: end $n (exit $?)"; mv $Q/running/$n.sh $Q/finished/ ) &
    done
    sleep 60
  done
}
queue_runner &
QPID=$!

ENC=$B/runs/base_duong/encoder_libero/last.pt
if [ "$LANE_ID" = 0 ]; then
  # ---------------------------------------------------------------- CPU: full PhysicalAI
  ( export HF_HUB_DISABLE_XET=1; wait_done pack_physai
    python -m physgs.data.physai_batch --scenes objects_falling,ball_mixer,rolling_ramp_objects,towers --shards_per_scene 1000 \
      --out $D/physai_episodes_full --raw $D/physai_raw --index_dir $D/physai_index --workers 4 --skip $D/physai_done_shards.txt \
      > $R/logs/physai_full.log 2>&1
    log "physai_full finished: $(find $D/physai_episodes_full -name '*.npz' | wc -l) new episodes" ) &
  DPID=$!
  # ---------------------------------------------------------------- data
  ( run pack_physai python -m physgs.data.pack_episodes --src $D/physai_episodes --out $D/packed/physai --max_objects 24 ) &
  for f in $D/libero/$SUITE/*.hdf5; do               # one render process per task
    t=$(basename $f .hdf5); mkdir -p $D/render_roots/$t/$SUITE; ln -sfn $f $D/render_roots/$t/$SUITE/
    ( run render_$t python -m scripts.render_libero_multiview --root $D/render_roots/$t/ --save $D/physgs_libero --suite $SUITE ) &
  done
  wait $(jobs -p | grep -vx -e $QPID -e $DPID)
  for i in $(seq 0 9); do
    ( run convert_$i python -m physgs.data.libero_process --src $D/physgs_libero/$SUITE --out $D/libero_episodes/$SUITE \
        --device cuda --shard $i --num_shards 10 ) &
  done
  wait $(jobs -p | grep -vx -e $QPID -e $DPID)
  run pack_libero python -m physgs.data.pack_episodes --src $D/libero_episodes/$SUITE --out $D/packed/$SUITE --max_objects 16 --native_order
  run split python -m scripts.split_libero_dataset --folder $D/physgs_libero
  for x in pack_physai pack_libero split; do [ -f $R/done/$x ] || { log "data preparation failed at $x"; touch $Q/STOP; exit 1; }; done
  touch $R/done/PREP
  log "data ready: $(ls $D/libero_episodes/$SUITE/*/*.npz | wc -l) LIBERO episodes; $(du -sh $D/packed/* | tr '\n' ' ')"
  # ---------------------------------------------------------------- models
  tokens base $ENC
  ( rssm bL base 1024 512 4 3e-4 30000 15000 30 && policies bL bL $ENC ) &                  # RSSM-L 22.6M
  ( wait_done fc_bB && policies bc bB $ENC model_cfg.forecast_cfg.use_zero_forecast=true ) &  # BC: same policy, no forecast
  L=$!
else
  wait_done PREP
  ( wait_file $R/bB/rssm_libero/last.pt; wait_done tok_base_libero                           # duong's RSSM (15k + 7.5k steps)
    forecasts bB base && policies bB bB $ENC ) &
  ( encoder L 512 8 4 80000 20000 4e-4 30 && rssm LL L 1024 512 4 3e-4 30000 15000 30 \
    && policies LL LL $R/enc_L/libero/last.pt ) &                                            # encoder-L 45.2M
  L=$!
fi

# ---------------------------------------------------------------- end of job
wait $(jobs -p | grep -vx -e $QPID -e ${DPID:-none})
log "lane $LANE_ID planned work finished"
python $S/summarize.py $R
idle=0
while true; do
  stop=0; { [ -f $Q/STOP ] || [ -f $Q/STOP$LANE_ID ]; } && stop=1
  busy=$( (mine $Q/pending; mine $Q/running) | wc -l)
  [ $stop = 1 ] && [ -z "$(mine $Q/running)" ] && break
  if [ "$busy" -eq 0 ]; then
    [ -f $Q/KEEP ] || break
    idle=$((idle + 1)); [ $idle -ge 240 ] && { log "idle for 4 h, ending"; break; }
  else idle=0; fi
  sleep 60
done
kill $QPID 2>/dev/null
[ -n "$DPID" ] && { pkill -P $DPID 2>/dev/null; kill $DPID 2>/dev/null; }   # PhysicalAI download is resumable
python $S/summarize.py $R
log "lane $LANE_ID JOB_DONE"
