#!/bin/bash
# The four VAR continuation runs on Hessian: 4 workers (2 per GPU), full dataset, 90 s apart; list order S0, S1, S2, S3.
R=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/letuan_r2dreamergraph_2/morse
printf 'VARb-S0\nVARb-S1\nVARb-S2\nVARb-S3\n' > $R/tools/queue/queue_h_varb.txt
n=0
for g in 0 1 0 1; do
  n=$((n+1))
  setsid -f nohup bash $R/tools/queue/hrun2.sh queue_h_varb.txt $g data/offline_data/all_robots_suboptimal_70_forward_1m \
    > $R/hq/worker_varb_$n.out 2>&1 < /dev/null
  echo "[start_real $(date -u +%H:%M:%S)] worker $n on GPU $g"
  sleep 90
done
