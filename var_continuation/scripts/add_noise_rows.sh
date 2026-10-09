#!/bin/bash
# Paired action-noise evaluations (morse_ckpt_eval_noise) of the final +20k checkpoints (and the source checkpoint with S0): 30
# episodes with N(0, 0.1^2) noise on the first 50 steps, the same noise draws for every checkpoint. Outputs in /tmp/var_runs/analysis.
R=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/letuan_r2dreamergraph_2/morse
T=$R/tools/queue/experiments_hessian.tsv
LEGACY="--algorithm.morse_joint_encoder_hidden=none --algorithm.morse_core_depth=3 --algorithm.morse_core_weight_norm=False"
SRC=$R/repo/experiments/runs/morse/V7-CH/V7-CH-s0/models/at_75000
B=$R/repo/experiments/runs/morse/VARb
base=$(awk -F'\t' '$1=="V7-CH-s0"' $T)
flags=$(echo "$base" | cut -f8 | sed 's/ --algorithm.morse_save_at=50000,75000//')
add() {  # add ID MODE CKPTS
  grep -q "^$1	" $T && { echo "exists $1"; return; }
  extra="--algorithm.morse_variance_mode=$2 --algorithm.morse_ckpt_analysis=$3 --algorithm.morse_ckpt_out=/tmp/var_runs/analysis/$1.npz --algorithm.morse_ckpt_batches=0 --algorithm.morse_ckpt_eval_noise=0.1@50 --algorithm.morse_ckpt_eval_episodes=30 --runner.save_model=False"
  printf '%s\tanalysis\tiql\tmorse\t0\t0.5\t0\t%s %s %s\n' "$1" "$flags" "$LEGACY" "$extra" >> $T
  echo "added $1"
}
add var-noise-S0 original "$SRC,$B/VARb-S0/models/at_95000"
add var-noise-S1 detach "$B/VARb-S1/models/at_95000"
add var-noise-S2 private "$B/VARb-S2/models/at_95000"
add var-noise-S3 protected "$B/VARb-S3/models/at_95000"
printf 'var-noise-S0\nvar-noise-S1\nvar-noise-S2\nvar-noise-S3\n' > $R/tools/queue/queue_h_varnoise.txt
