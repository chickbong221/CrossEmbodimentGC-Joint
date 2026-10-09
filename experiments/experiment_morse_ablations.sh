#! /bin/bash
# Ablations and controls for MORSE (paper Sec. Experiments and appendix).
# Usage: bash experiment_morse_ablations.sh [iql|bc|td3bc] [seed] [variant ...]
# Without variants, every variant is launched one after another.
# MORSE_EXTRA_ARGS is appended to every run, e.g. MORSE_EXTRA_ARGS="--algorithm.nr_offline_updates=50000".
# Actor steps matched to Embodiment Grouping with M=4 (IQL): MORSE_EXTRA_ARGS="--algorithm.actor_steps_per_update=4".
# Each run logs the interference diagnostics used by analysis/analyze_morse.py.
set -e
ALGO=${1:-iql}
SEED=${2:-0}
shift $(( $# > 2 ? 2 : $# )) || true
SCRIPT="$(dirname "$0")/experiment_${ALGO}_morse.sh"

# The URMA architecture: one shared block each, original widths (the defaults since 2026-10-04 add decoder experts
# and set the expert width).
URMA_ARCH="--algorithm.morse_num_experts=1 --algorithm.morse_num_experts_decoder=0 --algorithm.morse_expert_hidden=0 --algorithm.morse_core_routing=body --algorithm.morse_topk_core=0 --algorithm.morse_topk_decoder=0 --algorithm.morse_mi_weight=0.0 --algorithm.morse_conflict_mode=none --algorithm.morse_eta=0.0"

declare -A VARIANTS=(
  # Main method (defaults since 2026-10-04: 8 experts in core and decoder with top-4 routing, the core mixed per joint,
  # joint-type MI with equal robot weights) and the two reference points of the paper.
  [morse]=""
  [urma]="$URMA_ARCH"
  # Ablations of the main method: dense routing, one core mixture per transition, no MI.
  [dense_routing]="--algorithm.morse_topk_core=0 --algorithm.morse_topk_decoder=0"
  [body_core]="--algorithm.morse_core_routing=body"
  [no_mi]="--algorithm.morse_mi_weight=0.0"
  # Capacity-matched dense policy (K=1, widened so that parameters match K=4).
  [dense_matched]="$URMA_ARCH --algorithm.morse_width_mult_core=2.57 --algorithm.morse_width_mult_decoder=4.0"
  # The variants below ablate the earlier interference-guided method; they need its defaults as explicit flags
  # (morse/config.py PREVIOUS_DEFAULTS and LEGACY_ARCHITECTURE).
  [moe_no_conflict]="--algorithm.morse_conflict_mode=none"
  # Is gradient evidence doing the work? Same conflict values, wrong pairs; STGC-style evidence (units only).
  [shuffled]="--algorithm.morse_conflict_mode=shuffled"
  [expert_mean]="--algorithm.morse_evidence=unit --algorithm.morse_conflict_mode=expert_mean"
  # Evidence: unit pairs of one update instead of accumulated types; same-batch type sums (with the noise
  # covariance of joints sharing a transition); relevance weights of unit evidence.
  [unit_evidence]="--algorithm.morse_evidence=unit"
  [same_batch_types]="--algorithm.morse_unbiased_types=False"
  [no_relevance]="--algorithm.morse_evidence=unit --algorithm.morse_relevance=False"
  # Threshold and normalization of the conflict objective.
  [calibrated_threshold]="--algorithm.morse_threshold=calibrated"
  [threshold_0]="--algorithm.morse_kappa=0.0"
  [mass_norm]="--algorithm.morse_conflict_norm=mass"
  # Router inputs.
  [router_morph]="--algorithm.morse_router_inputs=morph"
  [router_state]="--algorithm.morse_router_inputs=state"
  # Placement of experts.
  [blocks_decoder]="--algorithm.morse_blocks=decoder"
  [blocks_core]="--algorithm.morse_blocks=core"
  [blocks_all]="--algorithm.morse_blocks=joint_encoder,foot_encoder,core,decoder"
  # Number of experts and strength of the supervision.
  [k2]="--algorithm.morse_num_experts=2"
  [k8]="--algorithm.morse_num_experts=8"
  [lambda_0.1]="--algorithm.morse_lambda=0.1"
  [lambda_10]="--algorithm.morse_lambda=10.0"
  # The first version of the method (all blocks, calibrated threshold, mass-normalized loss, same-batch sums).
  [v1]="--algorithm.morse_blocks=joint_encoder,foot_encoder,core,decoder --algorithm.morse_threshold=calibrated --algorithm.morse_kappa=0.0 --algorithm.morse_conflict_norm=mass --algorithm.morse_unbiased_types=False"
)

SELECTED=("$@")
if [ ${#SELECTED[@]} -eq 0 ]; then SELECTED=("${!VARIANTS[@]}"); fi
for v in "${SELECTED[@]}"; do
  if [ -z "${VARIANTS[$v]+x}" ]; then echo "unknown variant $v"; exit 1; fi
  echo "=== $ALGO / $v / seed $SEED"
  bash "$SCRIPT" ${VARIANTS[$v]} --environment.seed="$SEED" --runner.exp_name="MORSE-${ALGO^^}-${v}" ${MORSE_EXTRA_ARGS}
done
