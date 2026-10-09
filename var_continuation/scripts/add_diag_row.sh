#!/bin/bash
# Append one variance-diagnostics analysis row (exp analysis) to experiments_hessian.tsv: the flags of V7-CH-s0 without its
# checkpoint list, the URMA layer sizes of the V7 checkpoints, the variance mode, and the checkpoints to analyse.
# Usage: add_diag_row.sh ID MODE CKPT1,CKPT2,... [BATCHES]      Idempotent: an existing id is skipped.
R=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/letuan_r2dreamergraph_2/morse
T=$R/tools/queue/experiments_hessian.tsv
LEGACY="--algorithm.morse_joint_encoder_hidden=none --algorithm.morse_core_depth=3 --algorithm.morse_core_weight_norm=False"
ID=$1; MODE=$2; CKPTS=$3; NB=${4:-4}
grep -q "^$ID	" $T && { echo "exists $ID"; exit 0; }
base=$(awk -F'\t' '$1=="V7-CH-s0"' $T)
flags=$(echo "$base" | cut -f8 | sed 's/ --algorithm.morse_save_at=50000,75000//')
extra="--algorithm.morse_variance_mode=$MODE --algorithm.morse_ckpt_analysis=$CKPTS --algorithm.morse_ckpt_out=/tmp/var_runs/analysis/$ID.npz --algorithm.morse_ckpt_batches=0 --algorithm.morse_ckpt_var=$NB --algorithm.morse_ckpt_var_noise=True --runner.save_model=False"
printf '%s\tanalysis\tiql\tmorse\t0\t0.3\t0\t%s %s %s\n' "$ID" "$flags" "$LEGACY" "$extra" >> $T
echo "added $ID"
