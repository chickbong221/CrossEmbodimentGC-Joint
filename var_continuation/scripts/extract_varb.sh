#!/bin/bash
# Extract the TensorBoard scalars of the four VARb runs (all tags) to /tmp/var_runs/tb on Hessian (read-only on the runs).
R=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/letuan_r2dreamergraph_2/morse
export PATH=$R/env/bin:$PATH LD_LIBRARY_PATH=$R/env/lib HOME=$R
mkdir -p /tmp/var_runs/tb
python $R/analysis/tb_extract.py /tmp/var_runs/tb 'morse/,eval/,steps/,ckpt/,sync/,policy_loss,time/,value_loss,critic_loss' \
  /tmp/var_runs/VARb/VARb-S0 /tmp/var_runs/VARb/VARb-S1 /tmp/var_runs/VARb/VARb-S2 /tmp/var_runs/VARb/VARb-S3 2>&1 | grep -v -i 'tensorflow\|cpu_feature\|rebuild'
