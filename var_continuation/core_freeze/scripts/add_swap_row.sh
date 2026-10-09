#!/bin/bash
# Evaluation row for the four core-swap checkpoints: exact deterministic evaluation (twice each) and the paired action-noise
# protocol of the variance report (sigma 0.1 on the first 50 steps, 30 episodes, evaluation seeds 1000+e for every checkpoint).
R=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/letuan_r2dreamergraph_2/morse
T=$R/tools/queue/experiments_hessian.tsv
LEGACY="--algorithm.morse_joint_encoder_hidden=none --algorithm.morse_core_depth=3 --algorithm.morse_core_weight_norm=False"
S=/tmp/var_runs/swaps
grep -q "^swap-eval	" $T && { echo exists; exit 0; }
base=$(awk -F'\t' '$1=="V7-CH-s0"' $T)
flags=$(echo "$base" | cut -f8 | sed 's/ --algorithm.morse_save_at=50000,75000//')
extra="--algorithm.morse_ckpt_analysis=$S/ref75,$S/ref95,$S/core75_rest95,$S/core95_rest75 --algorithm.morse_ckpt_out=/tmp/var_runs/analysis/swap-eval.npz --algorithm.morse_ckpt_batches=0 --algorithm.morse_ckpt_eval=True --algorithm.morse_ckpt_eval_noise=0.1@50 --algorithm.morse_ckpt_eval_episodes=30 --runner.save_model=False"
printf '%s\tanalysis\tiql\tmorse\t0\t0.5\t0\t%s %s %s\n' swap-eval "$flags" "$LEGACY" "$extra" >> $T
printf 'swap-eval\n' > $R/tools/queue/queue_h_swap.txt
echo added
