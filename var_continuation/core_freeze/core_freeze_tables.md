## 1. Checkpoint component swaps (evaluation only)

Core = every parameter under `core`: core experts, shared (global and group) experts, router and router noise head (4,112,812 parameters in 24 leaves: dense_0, dense_1, dense_2, ln_0, router, router_noise). Early = V7-CH-s0 at 75k, late = the S0-original continuation at 95k. Deterministic evaluation: one episode per robot, run twice (identical both times: True). Paired noise: N(0, 0.1^2) on the first 50 steps, 30 episodes, evaluation seeds 1000+e identical for every configuration.

| Configuration | Core | Rest of actor | Go2 exact | Go2 under noise: mean return (walks) | Mean over 16 robots, exact | Mean over 16 robots, under noise |
|---|---|---|---|---|---|---|
| Early reference | 75k | 75k | 50.8 | 7.9 (0.13) | 50.8 | 47.8 |
| Late reference | 95k | 95k | 2.8 | 5.8 (0.00) | 47.8 | 47.7 |
| Early-core swap | 75k | 95k | 51.0 | 49.2 (0.97) | 47.5 | 46.6 |
| Late-core swap | 95k | 75k | 13.9 | 10.9 (0.00) | 41.3 | 40.8 |

Deterministic return of every robot:

| Robot | Early reference | Late reference | Early-core swap | Late-core swap |
|---|---:|---:|---:|---:|
| a1 | 14.4 | 14.4 | 14.3 | 14.4 |
| go1 | 14.5 | 14.5 | 14.5 | 14.5 |
| go2 | 50.8 | 2.8 | 51.0 | 13.9 |
| anyb | 48.2 | 48.2 | 43.7 | 46.8 |
| anyc | 43.7 | 43.7 | 40.0 | 43.5 |
| bkv0 | 46.6 | 46.9 | 47.5 | 46.6 |
| bkvb | 55.1 | 55.1 | 55.0 | 55.1 |
| bg | 52.8 | 52.6 | 52.9 | 52.4 |
| bit | 45.8 | 45.9 | 45.8 | 45.8 |
| h1 | 54.6 | 54.7 | 54.4 | 54.2 |
| g1 | 78.8 | 78.8 | 34.6 | 76.0 |
| tl | 111.6 | 111.6 | 111.0 | 3.2 |
| op3 | 102.9 | 102.5 | 102.3 | 102.2 |
| nao5 | 91.1 | 91.8 | 91.7 | 90.8 |
| cas | 1.1 | 1.1 | 1.1 | 1.1 |
| hx | 0.1 | 0.0 | 0.0 | 0.1 |

Reference check. Both reference configurations are built by the same swap code and reproduce the saved policies bitwise (policy parameter hash equal: early True, late True); the swapped configurations carry the intended core and rest hashes (early core + late rest: True, late core + early rest: True). Their evaluations equal those of the saved checkpoints: exact Go2 50.8 and 2.8 as recorded in `checkpoints.jsonl`, and the paired-noise numbers 7.9 (walks 0.13) and 5.8 (0.00) equal those of the variance report.

## 2. Core freeze against the matched S0-original continuation

Common horizon 95000 updates (+20000). Returns: mean of the last 10 evaluations (one episode per robot every 100 updates) up to the horizon; "walks" is Go2's share of evaluations above the frozen threshold (26.2) in the last 2,000 updates.

| Run | Go2 return at +H | Go2 walks (last 2k) | Mean return, 16 robots | Go2 exact at +20k | Go2 under paired noise (walks) |
|---|---:|---:|---:|---:|---:|
| S0-original (control) | 4.3 | 0.00 | 47.9 | 2.8 | 5.8 (0.00) |
| core freeze | 1.6 | 0.00 | 47.7 | 1.7 | 1.8 (0.00) |
| source checkpoint (75k) | 29.9 (window) | | 49.4 | 50.8 | 7.9 (0.13) |

Go2, mean return per 1,000-update bin (share above the threshold in brackets):

| Run | +1k | +2k | +5k | +10k | +15k | +20k |
|---|---:|---:|---:|---:|---:|---:|
| S0-original | 37.5 (0.70) | 16.9 (0.30) | 20.5 (0.30) | 11.7 (0.20) | 21.5 (0.40) | 4.3 (0.00) |
| core freeze | 46.1 (0.90) | 31.1 (0.60) | 31.2 (0.60) | 6.5 (0.10) | 21.2 (0.40) | 1.6 (0.00) |
| original run (CH-s0) | 22.5 (0.40) | 11.5 (0.20) | 22.5 (0.40) | 4.6 (0.00) | 12.4 (0.20) | 2.4 (0.00) |

Exact deterministic evaluation of the saved checkpoints (Go2 / mean over the 16 robots): S0 +1000: 1.2 / 47.6, +5000: 50.7 / 50.7, +10000: 2.7 / 47.7, +15000: 50.8 / 50.7, +20000: 2.8 / 47.8; core freeze +5000: 50.5 / 50.8, +10000: 1.7 / 47.7, +20000: 1.7 / 47.7.

Per-robot return at the horizon (window mean of the last 10 evaluations); the source is the original run before 75k:

| Robot | Source | S0 | Core freeze | Freeze minus S0 |
|---|---:|---:|---:|---:|
| a1 | 14.4 | 14.4 | 14.4 | +0.0 |
| go1 | 14.5 | 14.5 | 14.5 | +0.0 |
| go2 | 29.9 | 4.3 | 1.6 | -2.7 |
| anyb | 48.0 | 48.1 | 48.1 | -0.0 |
| anyc | 43.7 | 43.8 | 43.8 | +0.1 |
| bkv0 | 47.2 | 47.0 | 47.0 | +0.1 |
| bkvb | 55.1 | 55.1 | 55.1 | +0.0 |
| bg | 52.6 | 52.6 | 52.6 | -0.0 |
| bit | 45.7 | 45.9 | 45.9 | +0.1 |
| h1 | 54.6 | 54.7 | 54.7 | -0.0 |
| g1 | 78.5 | 78.7 | 78.7 | +0.0 |
| tl | 111.2 | 111.5 | 111.5 | +0.0 |
| op3 | 102.8 | 102.8 | 102.8 | +0.0 |
| nao5 | 91.2 | 91.3 | 91.3 | +0.0 |
| cas | 1.2 | 1.2 | 1.2 | -0.0 |
| hx | 0.1 | 0.1 | 0.0 | -0.0 |

Status changes by the frozen thresholds: S0-original (control): lost ['go2'], gained none; core freeze: lost ['go2'], gained none.

## 3. Proof that the core stayed unchanged

- Core parameter hash (SHA-1 over all 24 core leaves) of the checkpoint d14c95f85844f519..., of the branched run's core at start d14c95f85844f519...: equal True.
- The run logs the core hash at every logged update (every 250 updates; 80 updates from 75000 to 94750): 1 distinct value(s), equal to the checkpoint's: True.
- Saved checkpoints: at_80000 d14c95f85844... (equal to the source's: True), at_85000 d14c95f85844... (equal to the source's: True), at_95000 d14c95f85844... (equal to the source's: True).
- Norm of the applied update to the core parameters at every logged step: max 0.0e+00 over 80 logged steps (the update is exactly zero: the gradient is zeroed and the restored Adam first moments of the core were zeroed). The trainable parameters moved: update norm of the whole actor 0.0024 on average against 0.0087 in S0.

**How the frozen core is excluded.** The core's gradient is computed (the core stays in the forward and backward pass, with its routing noise, and the gradient flows through it into the joint and foot encoders and the attention layers: nothing is stop-gradiented) and the optimizer's first step, `frozen_clip_by_global_norm`, sets it to zero before the global-norm clip, so it neither updates the core nor enters the clip's norm. The core router's MI and balance gradients (the core MI term, which reaches only the core router) are excluded the same way; the decoder MI and balance terms and the task loss are unchanged. The restored Adam state of every trainable parameter and the step counts (300,000 at start) are kept; only the core's first moments were set to zero. S0's global clip would have been computed over the core's gradient too; both factors are logged:

| Logged (every 250 updates) | Core freeze | S0 |
|---|---:|---:|
| Clip factor actually applied, minimum | 1.000 | 1.000 |
| Clip factor with the core's gradient included, minimum | 1.000 | (same as above) |
| Gradient norm of the trainable parameters, mean | 0.1206 | 0.1252 (all, core included) |
| Gradient norm of the core (computed, discarded), mean | 0.0324 | not logged |
| decoder MI cap factor, mean over steps | 1.000 | 1.000 |
| core MI cap factor, mean over steps | 1.000 | 1.000 |
| decoder router task-gradient norm, mean | 0.0135 | 0.0135 |

## 4. Data and critic synchronization with S0

80 logged updates in common (75000 to 94750): identical minibatch-stream hashes True, identical critic hashes True, identical value-critic hashes True. The routing noise has its own key stream (seed, update, actor step, robot), the main key stream splits the same way in every mode, the diagnostic steps are the same (every 250 updates) and the freeze changes no random draw, so the control is matched.

## 5. Go2 mean-action error and the remaining actor components

Fixed minibatches of the training data (4 per robot, the same for every checkpoint), advantage weights from each checkpoint's own critics; Go2 / mean over the 16 robots.

| Checkpoint | S0 weighted mean error | Freeze weighted mean error | S0 mean sigma | Freeze mean sigma |
|---|---|---|---|---|
| source (75k) | 0.769 / 0.599 | 0.769 / 0.599 | 0.879 / 0.781 | 0.879 / 0.781 |
| +5000 | 0.790 / 0.606 | 0.790 / 0.606 | 0.870 / 0.775 | 0.870 / 0.775 |
| +10000 | 0.786 / 0.604 | 0.787 / 0.604 | 0.878 / 0.781 | 0.878 / 0.781 |
| +20000 | 0.786 / 0.611 | 0.787 / 0.612 | 0.874 / 0.778 | 0.874 / 0.779 |

Parameter change by group (descriptive only; magnitudes do not identify the cause of a collapse):

```
L2 distance of the policy parameters per group at +20k (source = V7-CH-s0 at 75k; core experts and routers are frozen in the freeze run)
pair                             all      desc_embedding            std_head     decoder_experts        core_experts             routers joint_state_encoder  foot_state_encoder     joint_attention      foot_attention
S0 - source                  12.9920              0.3754              0.0191              2.5870             12.6666              1.2186              0.0144              0.0089              0.1380              0.1077
freeze - source               3.2479              0.3839              0.0187              3.1056              0.0000              0.8322              0.0189              0.0064              0.2112              0.1366
freeze - S0                  12.8711              0.1767              0.0011              1.9391             12.6666              1.1807              0.0291              0.0064              0.1648              0.0988
```
