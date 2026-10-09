#!/bin/bash
# Runs on Hessian: for each branch, wait for its VARb run to finish (done marker), then start a worker that runs its variance
# diagnostics and paired-noise evaluation in turn (GPU 0 for S0/S2, GPU 1 for S1/S3). Marker files only.
R=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/letuan_r2dreamergraph_2/morse
for k in S0 S1 S2 S3; do
  printf 'var-diag-%s\nvar-noise-%s\n' $k $k > $R/tools/queue/queue_h_varpost_$k.txt
done
declare -A GPU=([S0]=0 [S1]=1 [S2]=0 [S3]=1)
declare -A STARTED
for i in $(seq 1 240); do
  for k in S0 S1 S2 S3; do
    [ -n "${STARTED[$k]}" ] && continue
    if [ -e $R/hq/state/done/VARb-$k ]; then
      sleep 5
      setsid -f nohup bash $R/tools/queue/hrun2.sh queue_h_varpost_$k.txt ${GPU[$k]} data/offline_data/all_robots_suboptimal_70_forward_1m \
        > $R/hq/worker_varpost_$k.out 2>&1 < /dev/null
      STARTED[$k]=1; echo "$(date -u +%H:%M:%S) started post-processing for $k"
    elif [ -e $R/hq/state/failed/VARb-$k ]; then
      STARTED[$k]=failed; echo "$(date -u +%H:%M:%S) VARb-$k FAILED"
    fi
  done
  [ ${#STARTED[@]} = 4 ] && break
  sleep 60
done
echo "post launcher finished $(date -u +%H:%M:%S)"
