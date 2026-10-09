# Coordination tests: results (T3-MIcompat-s0, saved checkpoints "60k" and "90k")

Collected 2026-10-03 for the reviewer's three checkpoint tests. No policy retraining. The only new simulation is a
deterministic re-evaluation of the two saved checkpoints, one episode per robot, which gives the exact-step returns.
Full tables are in `experiments/analysis/results/coordination/tables.md`; all numbers are in `coordination_tests.json`
next to it.

## Bottom line

1. **The exact-step outcomes are the reverse of the assumption.** The saved "60k" policy (60,001 updates) **falls**
   with Cassie: return 1.0, 55 steps. The saved "90k" policy (90,002 updates) **walks** the full episode: return
   81.0, 1000 steps. The ~41 "at 60k" was a window average of 5 walks and 5 falls. Around 60k the logged evaluations
   alternate between walking and falling every 100 updates. Around 90k they all fall, yet the saved 90k policy walks.
   Between the windows, the share of evaluations that walk dropped from 50% (up to 60k) to 0% (up to 90k). So this pair
   cannot tie any action change to Cassie's falls, in either direction.
2. **Condition 1 holds.** Cross-joint relationships in successful reference episodes generalize across episodes. For
   every robot the full covariance predicts held-out reference actions better than the diagonal one. Cassie gains
   +3.28 nats per transition (95% CI [3.22, 3.34]) and G1 +8.02 [7.79, 8.24]. Real action combinations score far
   better than within-context shuffles. The structure is nearly identical across disjoint episode sets (similarity
   0.996 for Cassie, 0.985 for G1), and every context is supported by all fit episodes.
3. **Condition 2 fails.** Nothing in the checkpoint change is missed by action error or diagonal weighting:
   - **Imitation error:** every robot's action MSE improves from 60k to 90k.
   - **Cassie's coordination change is tiny and not coordination-specific.** Full exceedance rises by +0.11 pp
     [0.06, 0.16] and diagonal exceedance by +0.10 pp [0.04, 0.15], about the same.
   - **Stable robots change as much or more.** 13 of 16 robots show a similar small full-exceedance increase.
     The largest are anyc (+0.35 pp) and anyb (+0.26 pp), both stable. The stable comparison G1 is the one with
     the "full up, diagonal down" pattern: +0.10 pp vs −0.07 pp.
   - **The drift follows the reference correlations rather than breaking them,** in every robot. Coordination cost
     relative to an isotropic drift: Cassie median 0.77, random-direction null 0.96. No Cassie state exceeds 2
     (null 0.8%).
   - **The largest relationship change is small and isn't Cassie-specific.** Cassie, right hip roll / right knee:
     +0.048 [0.037, 0.060]. G1, left hip pitch / right ankle pitch: +0.079 [0.066, 0.093].
4. **By the reviewer's criterion, a coordination (covariance) loss is not justified by this evidence.** The temporal
   check (§5 of the request) was not run. It is meant for instantaneous tests that are informative but incomplete,
   and these are uninformative about the failure.

## How the exact-step returns were obtained

- **The labels are off by one.** A "latest" checkpoint labelled N updates holds the policy after N+1 updates: it is
  saved before the update counter increments. The stored step counters confirm it. "60k" has value-critic step 60001
  and policy step 240,004 (= 4 × 60,001). "90k" (resume) has value-critic step 90,002. The evaluation logged at
  60,000 is of the policy one update earlier, so the logs never evaluated the saved policies.
- **New option `morse_ckpt_eval`** evaluates saved checkpoints with the training-time evaluation code: mean actions,
  one episode per robot, done twice.
- **Validation:** on T1-URMA-A4-cal-s0 `at_2000` and `at_5000` (exactly labelled checkpoints), all 16 robots'
  returns and episode lengths match the training-time logs, within TensorBoard's float16 storage, at both
  checkpoints.
- **Repeats agree** to about 1e-14, except Cassie's 90k walk: 81.01 vs 80.99, the same outcome.

## Data and design

- **Episodes:** the dataset is stored in time order per robot (`obs[t+1] == next_obs[t]` at every sampled
  non-terminal step), so episodes are the runs between done flags.
- **Reference episodes:** reach the 1000-step horizon without termination, with return ≥ half the robot's best
  such return.
  - For the bipeds every reference episode lies in the last 300k samples, the expert segment. In the first 70% of
    the data the bipeds fall within tens of steps; op3's two full-length episodes there fail the return filter.
  - Cassie has 298 reference episodes, returns 73.1–81.5, which matches a walking evaluation (≈ 80). They are
    successful episodes, collected in the dataset's own (training-mode) conditions.
  - Quadrupeds' full-length episodes are bimodal in return, and the return filter keeps the high mode.
- **Splits:** 60/20/20 by episode (fit / calibration / test). Cassie: 179 / 60 / 59. G1: 179 / 60 / 60.
- **Contexts:** the command is [1, 0, 0] in every state, so contexts come from foot contact. For bipeds there are
  six: double support, flight, and left or right stance split into early and late swing by the swing foot's air
  time. Quadrupeds and the hexapod use contact patterns, with rare ones pooled. Badger's contact signal is constant,
  so it has one context.
- **Normalization and regularization:** actions are scaled per joint by their fit-episode std. α and ε were chosen
  on calibration NLL only. Cassie's selection is α = 0 (no shrinkage), ε = 3e-4.
- **Thresholds:** the 95th percentile of calibration reference actions' D² per model and context, fixed before
  scoring checkpoints. Real test reference actions exceed it 4–8% of the time (Cassie 4.7%).
- **Uncertainty:** paired bootstrap over test episodes (1000 resamples, 200 per context).
- **Logged-action noise:** the share of per-step noise (var of step differences / (2 var)) is low for Cassie (0.12)
  and moderate for G1 (0.34). Bittle's reference actions are essentially white noise (1.04), consistent with its
  weak Test 1 gain and 0% exceedance.

## Main table (test states of the reference episodes)

- **Exact-step:** the deterministic evaluation of the saved policy.
- **Window:** the mean of the last 10 logged evaluations up to the checkpoint's update, with the share that reached
  1000 steps in parentheses.
- **MSE:** action mean vs dataset action, mean over joints.
- **Exceedance:** the share of test states above the context threshold.
- **D² p95** grows with the number of joints, so it is not comparable across robots. The exceedance rates are.

| Robot | Ckpt | Exact-step return (length) | Window (walk share) | MSE | Full D² p95 | Full exceed. % | Diag. exceed. % |
|---|---|---:|---:|---:|---:|---:|---:|
| **cas** | 60k | **1.0 (55)** | 40.7 (50%) | 0.0088 | 17.38 | 3.08 | 3.34 |
|  | 90k | **81.0 (1000)** | 1.4 (0%) | 0.0084 | 17.65 | 3.19 | 3.44 |
| **g1** | 60k | 78.2 (1000) | 77.5 (100%) | 0.0075 | 32.11 | 2.75 | 3.93 |
|  | 90k | 78.4 (1000) | 78.1 (100%) | 0.0072 | 32.08 | 2.85 | 3.86 |
| a1 | 60k | 14.5 (1000) | 14.5 (100%) | 0.2135 | 11.09 | 0.55 | 1.76 |
|  | 90k | 14.4 (1000) | 14.4 (100%) | 0.2115 | 11.26 | 0.61 | 1.93 |
| go1 | 60k | 14.5 (1000) | 14.5 (100%) | 0.2214 | 11.79 | 0.98 | 1.57 |
|  | 90k | 14.5 (1000) | 14.5 (100%) | 0.2190 | 11.85 | 1.06 | 1.63 |
| go2 | 60k | 13.4 (1000) | 25.4 (70%) | 0.2443 | 12.52 | 0.96 | 1.44 |
|  | 90k | 13.6 (1000) | 13.6 (100%) | 0.2416 | 12.64 | 1.03 | 1.50 |
| anyb | 60k | 48.0 (1000) | 48.0 (100%) | 0.0103 | 19.10 | 3.55 | 3.92 |
|  | 90k | 48.3 (1000) | 48.2 (100%) | 0.0098 | 19.79 | 3.82 | 4.23 |
| anyc | 60k | 43.2 (1000) | 43.7 (100%) | 0.0111 | 19.46 | 2.47 | 2.92 |
|  | 90k | 44.1 (1000) | 43.8 (100%) | 0.0107 | 20.41 | 2.82 | 3.24 |
| bkv0 | 60k | 46.8 (1000) | 46.9 (100%) | 0.0440 | 19.15 | 2.30 | 2.89 |
|  | 90k | 47.1 (1000) | 47.3 (100%) | 0.0427 | 19.29 | 2.40 | 2.91 |
| bkvb | 60k | 55.0 (1000) | 55.0 (100%) | 0.0430 | 17.60 | 2.05 | 3.37 |
|  | 90k | 55.0 (1000) | 55.1 (100%) | 0.0420 | 17.82 | 2.16 | 3.48 |
| bg | 60k | 52.8 (1000) | 52.5 (100%) | 0.0522 | 20.37 | 3.01 | 2.94 |
|  | 90k | 52.5 (1000) | 52.6 (100%) | 0.0513 | 20.74 | 3.10 | 2.82 |
| bit | 60k | 45.9 (1000) | 45.8 (100%) | 0.5839 | 3.88 | 0.00 | 0.00 |
|  | 90k | 45.8 (1000) | 45.9 (100%) | 0.5809 | 3.90 | 0.00 | 0.00 |
| h1 | 60k | 54.4 (1000) | 54.2 (100%) | 0.0129 | 21.48 | 2.99 | 3.00 |
|  | 90k | 54.7 (1000) | 54.6 (100%) | 0.0124 | 21.93 | 3.04 | 3.11 |
| tl | 60k | 110.4 (1000) | 109.9 (100%) | 0.0129 | 23.78 | 1.49 | 1.78 |
|  | 90k | 111.5 (1000) | 110.8 (100%) | 0.0125 | 23.50 | 1.56 | 1.79 |
| op3 | 60k | 102.2 (1000) | 102.6 (100%) | 0.0740 | 18.19 | 1.07 | 1.42 |
|  | 90k | 102.4 (1000) | 102.6 (100%) | 0.0725 | 18.28 | 1.10 | 1.39 |
| nao5 | 60k | 90.7 (1000) | 81.8 (90%) | 0.0580 | 26.66 | 0.96 | 2.78 |
|  | 90k | 91.2 (1000) | 55.2 (60%) | 0.0567 | 26.99 | 1.06 | 2.85 |
| hx | 60k | 0.0 (1000) | 0.1 (100%) | 0.0336 | 22.97 | 1.76 | 4.00 |
|  | 90k | 0.0 (1000) | 0.0 (100%) | 0.0326 | 23.13 | 1.78 | 3.89 |

Changes from 60k to 90k with 95% intervals (paired bootstrap over test episodes):

| Robot | Test episodes | ΔMSE | Δ full p95 | Δ full exceed. (pp) | Δ diag. exceed. (pp) |
|---|---:|---:|---:|---:|---:|
| cas | 59 | −0.00040 [−0.00043, −0.00037] | +0.27 [0.05, 0.45] | +0.11 [0.06, 0.16] | +0.10 [0.04, 0.15] |
| g1 | 60 | −0.00033 [−0.00035, −0.00031] | −0.03 [−0.34, 0.43] | +0.10 [0.05, 0.14] | −0.07 [−0.11, −0.02] |
| anyb | 58 | −0.00047 [−0.00055, −0.00040] | +0.70 [0.34, 1.02] | +0.26 [0.18, 0.35] | +0.32 [0.19, 0.48] |
| anyc | 58 | −0.00040 [−0.00043, −0.00036] | +0.96 [0.80, 1.25] | +0.35 [0.25, 0.48] | +0.32 [0.23, 0.43] |
| h1 | 60 | −0.00048 [−0.00051, −0.00045] | +0.45 [0.13, 0.85] | +0.05 [0.01, 0.09] | +0.10 [0.07, 0.14] |
| nao5 | 59 | −0.00131 [−0.00136, −0.00126] | +0.33 [0.23, 0.42] | +0.11 [0.07, 0.15] | +0.07 [0.03, 0.11] |

All 16 robots are in `tables.md`. Every robot's MSE decreases, with the interval excluding 0. 13 of 16 robots show a
full-exceedance increase between +0.05 and +0.35 pp.

## Supporting results

### Test 1: held-out reference likelihood, shuffle control, stability, support

| Robot | NLL full / diag (nats per transition) | Gain [95% CI] | D²_full real / shuffled (AUC) | Similarity of disjoint halves (corr., partial) |
|---|---|---|---|---|
| cas | 6.12 / 9.40 | 3.28 [3.22, 3.34] | 9.7 / 31.6 (0.89) | 0.996, 0.985 |
| g1 | 18.81 / 26.83 | 8.02 [7.79, 8.24] | 22.7 / 70.3 (0.94) | 0.985, 0.973 |
| all 16 robots | — | 0.10 (bit) to 8.28 (hx); every interval > 0 | AUC 0.57 (bit) to 0.96 (bg) | ≥ 0.79 |

- **Better in every episode:** the full model is better in all 59 of Cassie's test episodes.
- **Diagonal check:** within-context shuffles leave the diagonal score sums unchanged, as they should.
- **Support:** each Cassie context draws on all 179 fit episodes, and 147 of them give ≥ 10 transitions to the rarest
  context (flight). The largest single-episode share of a context is 0.6% (1.2% for flight).
- **Main relationships:**
  - **Swing leg:** hip pitch is anti-correlated with knee, −0.83 for the right leg in left stance, late swing.
  - **Double support:** left hip roll correlates with right knee, 0.85.
  - These recur between fit and test episodes, for example 0.85 / 0.86.

### Test 3: beyond the size of the change

- **(A) Size vs coordination cost.** Rank correlation over test states is 0.75–0.99 across robots (Cassie 0.82,
  G1 0.81).
  - **Ratio medians:** relative to an isotropic drift, the ratio is 0.59–0.97 across robots. Random-direction drifts
    give 0.95–1.0, so the 60k→90k drift moves along the reference correlations in every robot.
  - **Matched drift size:** binned by drift size, Cassie's ratio stays at or below about 1.4 at p90 in every bin,
    with ≤ 0.3% of states above 2. So among changes of similar size, none departs much more from the reference
    relationships.
  - **States where MSE improved:** D²_full rises in 49% of them and D²_diag in 52%, a coin flip in both cases.
- **(B) Strongest relationship change, selected by a predefined ranking.** For each robot, take the context whose
  mean D²_full increases most, then the pair whose 2-D coordination excess increases most.
  - **Cassie:** right stance, late swing; right hip roll / right knee (reference corr. 0.72). Excess −0.223 → −0.175,
    Δ +0.048 [0.037, 0.060].
  - **G1 (stable comparison):** flight; left hip pitch / right ankle pitch (corr. 0.64). Excess −0.114 → −0.036,
    Δ +0.079 [0.066, 0.093].
  - **Figure** `pairs_cas_g1.png`: both checkpoints' means lie on the held-out reference cloud and are visually
    indistinguishable. These are changed statistical relationships, not causes of balance loss.

## Caveats

- **Expert states only.** Everything is measured on the expert dataset states, not on the states the policies visit
  when they fall. Without saved rollouts this cannot show what happens near a fall.
- **The checkpoint pair.** The analysis rests on one pair of checkpoints, whose exact outcomes run opposite to their
  windows. A competent→degraded comparison needs checkpoints whose exact outcome is known on both sides. Options:
  - `morse_save_at` checkpoints, which are labelled exactly, plus `morse_ckpt_eval` on each;
  - or a success rate per checkpoint, which needs randomized evaluation, since the test-mode evaluation is
    deterministic.
- **Model shape.** Within-context action distributions are not Gaussian (G1 flight is visibly multimodal). The
  Gaussian full model is a coarse summary, but it is enough to show the structure.

## What was added (code and runs)

- **`morse/coordination.py`:** reference-episode dump with episode reconstruction, both checkpoints' means, and
  features.
- **`morse_ckpt_eval`:** exact deterministic evaluation of saved checkpoints.
- **`morse_ckpt_batches=0`:** skips the older statistics.
- **`morse_coord_out`, `morse_coord_min_return`:** dump folder and reference-return filter.
- **Analysis scripts:** `experiments/analysis/coordination_tests.py` and `coordination_report.py`.
- **Tests:** `test_coordination_episodes_and_dump` in `tests/test_morse.py` (passes on Hessian).
- **Hessian runs (queue `analysis`):** `ckpt-T1cal-s0-2k-5k-eval` (validation) and `coord-MIcompat-s0-60k-90k` (dump
  plus exact evaluation).
- **Hessian outputs:** dumps (1.4 GB) in `$R/analysis/coord/MIcompat_s0_60k_90k/`; test outputs in
  `$R/analysis/coord/MIcompat_s0_60k_90k_tests/`.

## Follow-up (2026-10-04): start-up data and robustness checks

**Why these checks.** Cassie's evaluation falls happen in the first ~100 steps (99% of them). In those steps,
98–99% of the bipeds' training data comes from the dataset's falling episodes. The explanation under test was:
"next to the expert's start-up path, later training copies the falling policy more closely, so the margin shrinks."

**Check 1: the 60k and 90k policies on the first 50 steps of every episode.** Run `startup-MIcompat-s0-60k-90k`;
results in `experiments/analysis/results/coordination/startup_check.json`. It does **not** support that explanation:
- **The policy never copies the falling actions.** Its error against them is about 0.75–1.0 (Cassie 0.76 in steps
  0–10), against 0.03 for the expert's actions. The falling actions are very noisy, and the policy learns their
  average.
- **Falling episodes aren't next to the expert's path.** Only 2–4% of falling transitions are as close to an expert
  state as expert episodes are to each other, and almost all of them are in the first 10 steps. After that, falling
  states are far away: median squared distance 309 vs 4.7 between expert episodes (Cassie). The states just off the
  expert path have no data at all.
- **The critic separates expert from falling actions only right after reset.** Mean advantage weight is about 11 for
  expert vs 0.9 for falling actions in steps 0–10, and about 1 for both from step 10 on.
- **The 60k→90k shift toward falling actions near the start is small and doesn't separate victims from stable
  robots.** Cassie, NAO, G1 and OP3 move slightly toward the falling actions; H1, Talos and go2 move the other way.

**Check 2: robustness of the saved checkpoints.** Gaussian noise of σ (action units) on every joint's action during
the first 50 steps (`@50`) or the whole episode (`@1000`). There are 40 episodes per condition, and the noise is
identical for both checkpoints, so the comparison is paired (sign test on the episodes only one checkpoint survived).
Run `noise-MIcompat-s0-60k-90k`; results in `noise_tables.md` and `MIcompat_s0_60k_90k_noise.json`.

| Robot | No noise 60k / 90k | σ 0.03 @50 | σ 0.1 @50 | σ 0.1 @1000 |
|---|---|---|---|---|
| Cassie | fall / walk | 0% → 28% (p = 0.001) | 0% → 5% | 0% → 5% |
| NAO | walk / walk | **72% → 22%** (p = 1e-5) | **42% → 2%** (p = 3e-5) | **42% → 2%** (p = 3e-5) |
| Talos | walk / walk | 100% → 100% | 98% → 85% | **68% → 32%** (p = 0.007) |
| H1 | walk / walk | 100% → 100% | **60% → 100%** (p = 3e-5) | **60% → 100%** (p = 3e-5) |
| G1 | walk / walk | 100% → 100% | 100% → 98% | 100% → 98% |
| OP3 | walk / walk | 100% → 100% | 100% → 100% | 100% → 100% |

**Reading.** The margin is real and measurable, and training changes it without the deterministic evaluation
showing it:
- **NAO** walks deterministically at both checkpoints, yet under tiny noise its success falls from 72% to 22%. NAO is
  also a frequent late victim in other runs.
- **The change goes both ways:** NAO and Talos lose margin, while H1 gains it.
- **For Cassie this pair can't show a shrinking margin.** The saved 60k policy falls under every condition. The saved
  90k one walks only without noise, or 28% of the time with tiny noise. Both have almost no margin.
- **Revised explanation.** The expert data covers a narrow path, and the falling data covers a different region. The
  states just off the path have no data, so how the policy behaves there is set by the network's extrapolation. The
  loss doesn't constrain that extrapolation, so it drifts with training, sometimes toward more robust and sometimes
  less. Robots whose margin drifts below zero fall. This matches the cross-run pattern of late collapses and late
  gains, with victims varying by run and seed.
- **Practical.** A success rate under small noise (40 paired episodes) measures this. It is far more informative
  than one deterministic episode, both for checkpoint selection and for comparing methods.
