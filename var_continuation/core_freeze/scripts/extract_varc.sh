#!/bin/bash
R=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/letuan_r2dreamergraph_2/morse
export PATH=$R/env/bin:$PATH LD_LIBRARY_PATH=$R/env/lib HOME=$R
mkdir -p /tmp/var_runs/tbc
python $R/analysis/tb_extract.py /tmp/var_runs/tbc 'morse/,eval/,steps/,ckpt/,sync/,policy_loss,time/,value_loss,critic_loss' /tmp/var_runs/VARc/VARc-F0 2>&1 | grep -v -i 'tensorflow\|cpu_feature\|rebuild'
