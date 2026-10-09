#!/bin/bash
# On Hessian: when VARc-F0 is done, start the diagnostics/noise worker (GPU 1), the TB extraction, and the core-hash/parameter-distance script.
R=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/letuan_r2dreamergraph_2/morse
for i in $(seq 1 200); do
  if [ -e $R/hq/state/done/VARc-F0 ]; then
    sleep 5
    setsid -f nohup bash $R/tools/queue/hrun2.sh queue_h_frzpost.txt 1 data/offline_data/all_robots_suboptimal_70_forward_1m > $R/hq/worker_frzpost.out 2>&1 < /dev/null
    setsid -f nohup bash /tmp/extract_varc.sh > /tmp/extract_varc.log 2>&1 < /dev/null
    (export PATH=$R/env/bin:$PATH LD_LIBRARY_PATH=$R/env/lib HOME=$R PYTHONPATH=$R/repo; cd $R/repo; setsid -f nohup python /tmp/frz_post.py > /tmp/frz_post.log 2>&1 < /dev/null)
    echo "$(date -u +%H:%M:%S) post-processing started"; exit 0
  fi
  [ -e $R/hq/state/failed/VARc-F0 ] && { echo "$(date -u +%H:%M:%S) VARc-F0 FAILED"; exit 1; }
  sleep 30
done
