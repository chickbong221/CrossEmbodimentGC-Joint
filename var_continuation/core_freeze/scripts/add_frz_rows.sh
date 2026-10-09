#!/bin/bash
# Rows of the core-freeze continuation (exp VARc, outputs node-local through a symlinked folder): the flags of the matched
# control VARb-S0 with morse_freeze_core=True and nothing else changed (the dry run only shortens it).   Usage: add_frz_rows.sh dry|real
R=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/letuan_r2dreamergraph_2/morse
T=$R/tools/queue/experiments_hessian.tsv
mkdir -p /tmp/var_runs/VARc; ln -sfn /tmp/var_runs/VARc $R/repo/experiments/runs/morse/VARc
base=$(awk -F'\t' '$1=="VARb-S0"' $T)
[ -n "$base" ] || { echo "no VARb-S0 row"; exit 1; }
flags=$(echo "$base" | cut -f8)
if [ "$1" = dry ]; then
  ID=dry-VARc-F0
  flags=$(echo "$flags" | sed 's/--algorithm.morse_stop_at=95000/--algorithm.morse_stop_at=75400/; s/--algorithm.morse_save_at=[0-9,]*/--algorithm.morse_save_at=75300/; s/--algorithm.morse_diag_frequency=250/--algorithm.morse_diag_frequency=100/')
else
  ID=VARc-F0
  flags=$(echo "$flags" | sed 's/--algorithm.morse_save_at=[0-9,]*/--algorithm.morse_save_at=80000,85000,95000/')
fi
grep -q "^$ID	" $T && { echo "exists $ID"; exit 0; }
cp -p $T $T.bak-frz-$(date -u +%m%d%H%M)
printf '%s\tVARc\tiql\tmorse\t0\t3.0\t9\t%s --algorithm.morse_freeze_core=True\n' "$ID" "$flags" >> $T
echo "added $ID"; awk -F'\t' -v i=$ID '$1==i {print substr($8, length($8)-330)}' $T
printf '%s\n' $ID > $R/tools/queue/queue_h_frz_$1.txt
