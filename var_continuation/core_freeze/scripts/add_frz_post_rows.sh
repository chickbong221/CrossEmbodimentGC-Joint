#!/bin/bash
# Post-run analysis rows of the core-freeze run: fixed-batch variance diagnostics of the source and the +5k/+10k/+20k checkpoints, and the
# paired action-noise evaluation (same protocol as the variance report) of the +20k checkpoint. Outputs in /tmp/var_runs/analysis.
R=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/letuan_r2dreamergraph_2/morse
SRC=$R/repo/experiments/runs/morse/V7-CH/V7-CH-s0/models/at_75000
B=$R/repo/experiments/runs/morse/VARc/VARc-F0/models
T=$R/tools/queue/experiments_hessian.tsv
LEGACY="--algorithm.morse_joint_encoder_hidden=none --algorithm.morse_core_depth=3 --algorithm.morse_core_weight_norm=False"
bash $R/analysis/var/add_diag_row.sh var-diag-F0 original "$SRC,$B/at_80000,$B/at_85000,$B/at_95000" 4
grep -q "^var-noise-F0	" $T || {
  base=$(awk -F'\t' '$1=="V7-CH-s0"' $T); flags=$(echo "$base" | cut -f8 | sed 's/ --algorithm.morse_save_at=50000,75000//')
  extra="--algorithm.morse_variance_mode=original --algorithm.morse_ckpt_analysis=$B/at_95000 --algorithm.morse_ckpt_out=/tmp/var_runs/analysis/var-noise-F0.npz --algorithm.morse_ckpt_batches=0 --algorithm.morse_ckpt_eval_noise=0.1@50 --algorithm.morse_ckpt_eval_episodes=30 --runner.save_model=False"
  printf '%s\tanalysis\tiql\tmorse\t0\t0.5\t0\t%s %s %s\n' var-noise-F0 "$flags" "$LEGACY" "$extra" >> $T; echo "added var-noise-F0"; }
printf 'var-diag-F0\nvar-noise-F0\n' > $R/tools/queue/queue_h_frzpost.txt
