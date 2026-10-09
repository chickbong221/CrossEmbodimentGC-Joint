# MORSE on the cross-embodiment offline RL benchmark

MORSE (Morphology-aware Routing of Shared Experts) is implemented on top of the
URMA policy and the BC / IQL / TD3+BC learners of this repository. Embodiment
Grouping (EG) is replaced by expert routing: every robot updates together, and
selected policy blocks become dense mixtures of experts whose routers are
supervised by exact gradient-interference evidence.

## Files

| Path | Content |
|---|---|
| `cross_embodiment_offline_rl/algorithms/morse/moe.py` | Dense MoE block: K batched experts, router = morphology branch + state branch, capture of layer inputs and backward signals |
| `.../morse/policy.py` | URMA policy whose joint encoder, foot encoder, actor core, and action decoder are MoE blocks (with K=1 it is exactly URMA) |
| `.../morse/interference.py` | Exact Gram matrices, conflict evidence (unit and type level), control variants, routing losses, diagnostics |
| `.../morse/updater.py` | The MORSE actor update: per-robot backward passes, cross-robot evidence, router-only auxiliary gradients, logging, snapshots |
| `.../morse/config.py` | Hyperparameters and ablation switches (`--algorithm.morse_*`) |
| `.../uni_{iql,bc,td3bc}_morse/` | The three learners with MORSE (`--algorithm.name=uni_iql_morse.iql`, ...) |
| `experiments/experiment_{iql,bc,td3bc}_morse.sh` | Main runs; extra `--flag=value` arguments override the defaults |
| `experiments/experiment_morse_ablations.sh` | All ablations and controls |
| `experiments/analysis/analyze_morse.py` | Figures and tables for the paper and appendix |
| `tests/test_morse.py` | Correctness tests |

## Paper to code

| Paper | Code |
|---|---|
| Expert blocks and routed units (Eq. routed_units) | `MorsePolicy` blocks `joint_encoder`, `foot_encoder` (element level), `core` (whole body), `decoder` (element level); units are captured per transition and element |
| Router (Eq. router) | `moe.Router`: `morph_hidden/morph_out` + `state_hidden/state_out`, zero-initialized outputs |
| Dense mixture (Eq. expert_mixture) | `moe.MoEBlock` |
| Per-unit contributions, decomposition (Eqs. full_expert_gradient, interference_decomposition) | backward signals via `perturb` in `ExpertDense` / `ExpertLayerNorm` |
| Layer factorization, Gram (Eqs. linear_factorization, gram) | `interference.gram` (dense and LayerNorm parameters) |
| Alignment, calibrated relevance-weighted coefficient (Eqs. gradient_cosine, conflict_coefficient) | `interference.conflict_evidence` |
| Conflict loss (Eq. conflict_loss) | `interference.conflict_loss` |
| Utilization loss, objective (Eqs. utilization, training_objective) | `interference.utilization_loss`, `MorseActorUpdater._aux_loss` |
| Router-only auxiliary updates, warm-up | `MorseActorUpdater.update`, `lambda_at` |
| Router-only adaptation to new embodiments | `--algorithm.morse_router_only=True` |

## Implementation choices to reflect in the paper

1. **Normalization of the conflict loss.** `morse_conflict_norm=pairs` (default)
   averages over the unit pairs of the block:
   `L_conf = sum_k sum_{u<v} q p_uk p_vk / |P^(m)|`, so the pressure on the router
   is proportional to the amount of conflict and vanishes once conflicts are gone.
   `mass` divides by `sum q` instead (the first version): 1/K^2 under uniform routing
   however weak the conflicts, which kept the auxiliary router gradient 30-150x the
   task gradient in the first pilot run.
2. **Estimated utilization.** `p_bar` is averaged over the captured units
   `S^(m)`, which are a random subset of the minibatch.
3. **Captured units.** The first 8 transitions of every robot's batch (all of
   their joints or feet) for element-level blocks and the first 64 transitions for
   the core; batches are sampled uniformly, so these are random transitions.
4. Router state features pass through `symlog`; router output layers start at zero
   (uniform routing). Experts keep URMA's layer structure, including a per-expert
   LayerNorm whose parameters are part of the Gram matrix.
5. A block with one expert has no router, so `morse_num_experts=1` is exactly URMA
   (same parameters and outputs; tested). Critics stay shared.
6. **Evidence.** `morse_evidence=type` (default) aggregates contributions per unit
   type (a joint or foot of a robot for element-level blocks, a robot for the core)
   and over updates (EMA `morse_type_ema`); `unit` uses per-sample alignments of one
   update. With `morse_unbiased_types=True` (default) the off-diagonal type sums use
   only unit pairs from different transitions: joints of one transition share its
   sample, so the expectation of their product contains the covariance of their noise,
   which averaging over updates keeps (it can look like a conflict, or hide one). The
   diagonal keeps all pairs. Snapshots also store the same-batch sums
   (`type_gram_acc_samebatch`), and the motivation analysis reports both estimators.
   Threshold: fixed `morse_kappa=0.05` by default (`calibrated`: zeta x std of the
   alignments, which flags a roughly constant tail of pairs however small they are).
   Experts sit in the core and decoder by default: the joint and foot encoders have a
   few dozen parameters (attention logits), so experts there add routing but no capacity.
7. **Actor steps per update.** `actor_steps_per_update=n` (IQL) takes n actor steps
   per update, each on 1/n of every robot's minibatch, with the actor schedule over
   n x updates; critics still update once on the whole minibatch. n=4 matches EG with
   M=4 (one actor step per group): same actor steps and samples. Routing objectives,
   evidence, accumulators, and diagnostics stay once per update (first actor step).
8. **Router-only adaptation** masks the gradient of every non-router parameter and
   also keeps those parameters fixed after the optimizer step, so restored Adam
   momentum cannot move them.

## Logged diagnostics (TensorBoard / W&B, every `morse_diag_frequency` updates)

MORSE metrics are written in float32 (the base code stores other scalars as float16).
For every block `b` (also for K=1 blocks, so URMA is measured the same way):

| Metric `morse/b/...` | Definition | Supports |
|---|---|---|
| `cancelled_fraction` | C- / (S + C+), where \|\|sum_u g_u\|\|^2 = S + C+ - C- (self, constructive, destructive pair terms, summed over experts; Eq. interference_decomposition). In [0, 1] | share of the potential first-order progress that interference cancels |
| `destructive_share` | C- / (C+ + C-): share of the pair interactions that are destructive | same, independent of the self terms |
| `conflict_frac` | fraction of (pair, expert) entries with c_uv < -kappa | how common conflicts are |
| `conflict_mass` | routing-weighted fraction of conflicting co-routing, sum p_u p_v [c<-kappa] / sum p_u p_v | experts separate conflicting inputs |
| `conf_coroute` | the normalized L_conf value (1/K^2 = uniform, 0 = separated) | MoE without supervision does not separate (Jacobs et al.) |
| `conflict_frac_within_transition / within_robot / cross_robot` | conflict frequency by pair category (within a transition only for element-level blocks) | the unit of sharing should be finer than a robot |
| `robot_cos_mean`, `robot_neg_frac` | robot-level gradient cosine of the block (Abe et al. Sec. 5.1, per block) | link to the reference analysis |
| `router_entropy`, `eff_experts`, `usage_max/min`, `usage_kl` | routing sharpness and balance | dense-routing concern, no collapse |
| `cos_mean`, `cos_std`, `kappa`, `q_pos_frac` | alignment statistics and threshold | calibration of zeta |
| `type_conflict_frac`, `type_cos_mean`, `type_kappa` | the same at the type level (`morse_evidence=type` only) | persistent interference between types |
| `loss_conf`, `loss_use` | auxiliary losses | training health |
| `robot_cos_mean`, `robot_neg_frac`, `robot_cancelled_fraction` (+ `_acc`) | robot-level statistics from **full-minibatch per-robot gradients** of the block's expert parameters; `_acc` uses the Gram accumulated over updates (EMA `morse_robot_ema`) | inter-robot conflict and how routing removes it |

The same robot-level statistics are logged for `morse/actor/...`: every policy parameter
except routers, i.e. the per-embodiment actor-gradient cosine of Abe et al. (Eq. 2;
`robot_neg_frac` over training is their "fraction of negative pairwise cosines").
Parameters of different experts are disjoint, so robots routed to different experts do
not interact in these statistics.

**Unit-level magnitudes are noise-dominated.** `cancelled_fraction` and
`destructive_share` sum over thousands of per-sample units; independent noise alone gives
~0.98 and ~0.50 for 1,928 units (checked by simulation), which is what the runs show. They
are kept in the logs but are not evidence of interference; use the robot-level (`_acc`)
statistics and the joint-type analysis below instead. Conflict counts that depend on the
routing (`conflict_mass`, `conf_coroute`) remain useful for the routing claims.

Global: `morse/lambda` (current lambda), `morse/update_s` (policy-update time on a
diagnostic step, which always includes the Gram computation), `morse/router_grad_task`
and `morse/router_grad_aux` (router gradient norms of the task loss and of the
auxiliary objectives, for choosing lambda).

Snapshots (`<run>/morse_diagnostics/step_*.npz`, every `morse_snapshot_frequency`
updates) add, per block: robot x robot cosine (`robot_cos`, full-batch; `robot_cos_acc`,
accumulated), unit-level conflict frequencies per robot pair, robot x expert usage, per-unit
routing, a sample of pairs with their input overlap and demand agreement (for the "where
interference arises" figure), and the unit-type contributions accumulated over updates
(`type_gram_acc`, with `type_robot`, `type_elem`, `type_morph`; a type is joint/foot e of
robot r, or robot r for the core). `actor__robot_cos(_acc)` is the whole-actor matrix.

**Motivation: are conflicts organized by joints or by robots?** `analyze_morse.py`
(`joint_motivation`) uses the accumulated joint-type contributions of a URMA run (K=1) to
report (i) how destructive and constructive joint-pair interaction splits between joints of
the same robot, robots of the same EG group, and different EG groups (`joint_pairs.csv`),
and (ii) for the same number of groups, how much destructive and constructive interaction
stays inside groups when robots are grouped (EG, FGW morphology) versus when joint types
are grouped (by their morphology features, or by gradients as an oracle)
(`joint_grouping.csv`, `joint_grouping_<run>_<block>.pdf`), plus a joint-type alignment
heatmap for a quadruped, a biped, and the hexapod (`joint_cos_<run>_<block>.pdf`).
Accumulation happens on diagnostic steps, so run the URMA reference with a small
`morse_diag_frequency` (e.g. 10):

```bash
bash experiment_iql_morse.sh --algorithm.morse_num_experts=1 --algorithm.morse_conflict_mode=none \
    --algorithm.morse_eta=0.0 --algorithm.morse_diag_frequency=10 --runner.exp_name=URMA-IQL
```

## Experiments

Run from `experiments/`. Extra arguments (`--flag=value`) replace the script's
default for that flag (RL-X rejects a flag given twice, so the scripts drop the
default); `DATASET` and `COMMANDS` select the data and command file.

```bash
bash experiment_iql_morse.sh                     # MORSE + IQL (also: experiment_bc_morse.sh, experiment_td3bc_morse.sh)
bash experiment_morse_ablations.sh iql 0         # all ablations for seed 0
bash experiment_morse_ablations.sh iql 1 morse moe_no_conflict urma   # selected variants
MORSE_EXTRA_ARGS="--algorithm.nr_offline_updates=50000" bash experiment_morse_ablations.sh td3bc 2 shuffled
DATASET=data/offline_data/all_robots_expert_forward_1m bash experiment_iql_morse.sh --environment.seed=1
```

Ablation variants (`experiment_morse_ablations.sh`): `morse`, `moe_no_conflict`
(dense MoE, lambda=0), `urma` (K=1), `dense_matched` (K=1, parameter-matched width),
`shuffled` (same conflict values on random pairs), `expert_mean` (STGC-style
evidence), `unit_evidence`, `same_batch_types`, `no_relevance`,
`calibrated_threshold`, `threshold_0`, `mass_norm`, `router_morph`, `router_state`,
`blocks_decoder`, `blocks_core`, `blocks_all`, `k2`, `k8`, `lambda_0.1`, `lambda_10`,
and `v1` (the first version of the method). The first update logs
the parameter count of every block, so capacity matching can be checked.

Deployment and adaptation of a trained model. `--runner.load_model` takes the file
`runs/<project>/<exp_name>/<run>/models/tmp/checkpoint_<step>/checkpoint`; the
algorithm settings stored in the checkpoint are restored unless they are set on the
command line. Test mode needs TensorBoard, W&B, and model saving switched off.

```bash
CKPT=runs/cross-embodiment-offline/MORSE-IQL/<run>/models/tmp/checkpoint_<step>/checkpoint
# evaluation rollouts with top-k routing
bash experiment_iql_morse.sh --runner.mode=test --runner.load_model=$CKPT --runner.nr_test_episodes=5 \
    --runner.track_tb=False --runner.track_wandb=False --runner.save_model=False --algorithm.morse_eval_topk=2
# router-only adaptation (e.g. with --environment.train_robot_types set to new robots)
bash experiment_iql_morse.sh --runner.load_model=$CKPT --algorithm.morse_router_only=True --algorithm.reset_lr_scheduler=True
```

Headless servers: `export MUJOCO_GL=disable`; without `lstopo`, add
`--environment.cycle_cpu_affinity=False`; without W&B, add `--runner.track_wandb=False`
(or `export WANDB_MODE=disabled`).

Throughput (all six learners, MORSE and the EG baselines alike; none of this changes the
algorithms):

* `--algorithm.data_prefetch=True` (default): minibatches are drawn and copied to the GPU in
  background threads, a few updates ahead, from a NumPy generator seeded with
  `environment.seed` (uniform with replacement per robot, as before). `False` restores the
  original in-loop sampler on the training key (the minibatch stream then matches the
  original code; with prefetching it is a different random stream with the same
  distribution). `time/data_wait_s` logs how long an update waited for its batch.
* Each update is compiled as a few functions: the MORSE policy update (all robots, router
  objectives, optimizer step) is one; EG runs one per robot group (gradients, mean, Adam
  step); target networks and logged temperatures are compiled too. Scalars are copied to
  the host once per update, one update late, so the GPU never waits for logging.
* `--algorithm.dataset_mmap=True` memory-maps the dataset (both MORSE and EG); read the
  files once beforehand (e.g. `cat data/offline_data/<name>/*.npy > /dev/null`) so the
  pages are in the page cache, which runs on the same node share.

Analysis:

```bash
python analysis/analyze_morse.py \
  --run "MORSE=runs/cross-embodiment-offline/MORSE-IQL-morse/*" \
  --run "MoE, lambda=0=runs/cross-embodiment-offline/MORSE-IQL-moe_no_conflict/*" \
  --run "URMA=runs/cross-embodiment-offline/MORSE-IQL-urma/*" \
  --run "Shuffled=runs/cross-embodiment-offline/MORSE-IQL-shuffled/*" \
  --baseline URMA --out analysis_out
```

Outputs: `curves_<metric>.pdf`, `pair_categories.pdf`, `robot_cos_<run>_<block>.pdf`,
`morphology_vs_conflict.csv/.pdf`, `persistence.csv`, `usage_<run>_<block>.pdf`,
`factors_<run>_<block>.pdf`, `type_cos_<run>_<block>_e<k>.pdf`, `returns.csv/.tex`,
`conflict_vs_gain_<run>.pdf`, `diagnostics_summary.csv`, `cost.csv`
(see the docstring of the script). `--upto N` uses only what the runs logged up to N updates,
so that runs still in training can be compared with finished ones at the same number of updates
(`returns.csv` reports the update of the last evaluation used, `last_eval_update`).

Suggested use in the paper:

* **Motivation (URMA, K=1):** `curves_robot_neg_frac.pdf` (actor panel) reproduces the
  reference analysis of inter-robot conflict; `robot_cancelled_fraction_acc` shows how much
  of each block's update interference between robots cancels; `joint_grouping_*.pdf` and
  `joint_pairs.csv` show that conflicts are organized by joints rather than robots (joint-level
  grouping removes more destructive interaction than robot-level grouping at the same
  number of groups, and much of it lies inside a robot or an EG group);
  `morphology_vs_conflict` and `persistence.csv` (accumulated robot matrices) relate the
  structure to morphology and its stability; `factors_*.pdf` shows where conflicts arise.
* **Why supervision is needed (MoE, lambda=0):** `conf_coroute` stays near 1/K^2
  and `router_entropy` stays high: experts cooperate instead of separating.
* **MORSE works as intended:** lower `conf_coroute` and `conflict_mass`, and lower
  `robot_cancelled_fraction_acc` than both references; `usage_*.pdf` shows which robots share
  experts; `shuffled` and `expert_mean` isolate the value of pairwise gradient
  evidence; `dense_matched` rules out capacity.
* **Outcome:** `returns.tex` (all, quadrupeds, bipeds, hexapod, worst robot) and
  `conflict_vs_gain_*.pdf` (do robots whose conflict drops most gain most?).
* **Appendix tables:** `diagnostics_summary.csv`; **cost:** `cost.csv`.

## Tests

```bash
python tests/test_morse.py
```

1. The factorized Gram equals explicit per-unit gradient inner products (to machine
   precision, with LayerNorm parameters and an objective coupling joints).
2. Through the whole policy and across robots with different numbers of joints, the
   Gram entries sum to ||grad||^2 of every block and expert.
3. The auxiliary objectives only change router parameters.
4. The shuffled control preserves the conflict values of every expert.
5. A block with one expert is the plain shared MLP (no router).
6. With one expert per block, MORSE reproduces the original URMA policies (Gaussian
   and deterministic) exactly: same parameter count and outputs from mapped weights.
7. Robot-level statistics: the per-block and whole-actor Gram of full-batch per-robot
   gradients (routers excluded) and ||sum_r g_r||^2 = S + C+ - C-.
8. Type-level evidence: type sums, cosines, and expansion to unit pairs.
9. Independent type sums: noise shared by two joints of one transition (g_b = -g_a,
   zero mean) gives no conflict, while the same-batch sums give cosine -1; a genuine
   conflict hidden by positively shared noise is recovered.
10. The conflict loss averaged over pairs scales with the conflict coefficients; the
    mass-normalized loss is invariant to their scale.
11. The compiled update equals the step-by-step computation (per-robot passes, mean
    gradient, accumulators, router objectives, Adam step; unit and type evidence,
    router-only adaptation, both loss normalizations and type estimators), including
    the key sequence and the logged metrics.
12. Several actor steps per update: the MORSE step on the first chunk of every robot's
    minibatch and task-only steps on the others equal the step-by-step computation;
    accumulators advance once per update, one optimizer step per chunk.
13. The background sampler gathers exactly the dataset rows of `Dataset.sample`
    (in-RAM and memory-mapped), with the seeded draws and per-robot batch sizes.
