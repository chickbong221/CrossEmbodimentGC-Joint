#!/bin/bash
# Append the variance-continuation rows (exp VAR, or analysis/dry rows) to experiments_hessian.tsv. Each takes the flags
# of V7-CH-s0 (a C-H run, seed 0) without its checkpoint list, the URMA layer sizes of the V7 checkpoints, a branch from
# its 75k checkpoint (at 75000 updates, the cosine schedule still spanning the original 100k), and the variance mode.
# Idempotent: existing ids are skipped.  Usage: add_var_rows.sh dry|real STOP SAVES
R=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/letuan_r2dreamergraph_2/morse
T=$R/tools/queue/experiments_hessian.tsv
CK=$R/repo/experiments/runs/morse/V7-CH/V7-CH-s0/models/at_75000
LEGACY="--algorithm.morse_joint_encoder_hidden=none --algorithm.morse_core_depth=3 --algorithm.morse_core_weight_norm=False"
KIND=$1; STOP=$2; SAVES=$3
[ "$KIND" = dry ] && P=dry-var || P=VAR
[ "$KIND" = dry ] && DIAG=100 || DIAG=250
[ -n "$4" ] && P=$4
cp -p $T $T.bak-var-$(date -u +%m%d%H%M)
base=$(awk -F'\t' '$1=="V7-CH-s0"' $T)
flags=$(echo "$base" | cut -f8 | sed 's/ --algorithm.morse_save_at=50000,75000//')
add() {  # add ID MODE
  grep -q "^$1	" $T && { echo "exists $1"; return; }
  extra="$LEGACY --algorithm.morse_branch_from=$CK --algorithm.morse_branch_updates=75000 --algorithm.morse_stop_at=$STOP --algorithm.morse_save_at=$SAVES --algorithm.morse_variance_mode=$2 --algorithm.morse_diag_frequency=$DIAG --algorithm.save_latest_frequency=100000000"
  printf '%s\t%s\tiql\tmorse\t0\t3.0\t9\t%s %s\n' "$1" "$P" "$flags" "$extra" >> $T
  echo "added $1"
}
add $P-S0 original
add $P-S1 detach
add $P-S2 private
add $P-S3 protected
awk -F'\t' -v p="$P-" 'index($1,p)==1 {print $1, NF}' $T
