# Gradient conflict on URMAv2 dense, 2k to 100k updates, two seeds

Written 2026-10-09. Model: URMAv2 dense (830,100 parameters), IQL actor loss, checkpoints 2k, 5k, 10k, 25k, 50k, 75k, 100k of
`U2-URMA-s0` and `U2-URMA-s1`. 32 minibatches x 1024 samples per robot, 64 episodes per class. Metric: cosine between gradients
of the actor loss (EG paper, PCGrad). Everything below is computed locally from the 14 verified `.npz` files in `data/`.

Jobs `U2-conf-r2-s0` / `-s1` (checkpoints 25k to 100k) finished cleanly on both H100 accounts (`queue/done`, nothing in `failed`).
The 8 new files were shrunk on the server (BC variant zeroed, IQL variant intact), downloaded, and md5-checked against the server.
No partial or corrupted file is left locally. Server copies (`analysis/conflict/slim/`, `slim*.py`, `bcw.py`) are still there.

## 0. Read this first: the 100k core numbers are not reliable

A cosine of mean gradients needs the mean gradient to be above the sampling noise. Reliability = agreement of the mean gradient
of two independent halves of the minibatches. A robot counts as reliable at >= 0.1.

| checkpoint | robots reliable in the core (seed 0 / 1) | joint units reliable in the core | mean reliability, core |
|---|---|---|---|
| 2k to 50k | 16 / 16 | 100% down to 93% | 0.98 down to 0.77 |
| 75k | 16 / 16 | 84% / 80% | 0.58 / 0.50 |
| **100k** | **5 / 3** | **34% / 37%** | **0.04 / 0.03** |

At 100k the core EG gap printed as -0.200 / -0.775 earlier came from **one** within-group pair. It is now printed `n/r`.
The decoder is still readable at 100k (13 / 10 robots, units 59% / 65%).
Why it drops (hypothesis, not tested): the core is near a stationary point at the end of training. Section 3 shows one possible reason.

## 1. Headline table (IQL variant, mean over 16 robots, seed 0 / seed 1)

| | 2k | 10k | 25k | 50k | 75k | 100k |
|---|---|---|---|---|---|---|
| **Episode S-F cosine, decoder** | -.50 / -.61 | -.47 / -.58 | -.50 / -.60 | -.57 / -.63 | -.60 / -.68 | -.62 / -.69 |
| Episode S-F cosine, log-std head | -.71 / -.76 | -.66 / -.71 | -.61 / -.69 | -.63 / -.66 | -.61 / -.66 | -.62 / -.67 |
| Episode S-F cosine, core | +.04 / +.02 | +.04 / +.04 | +.02 / +.01 | +.02 / +.02 | -.01 / -.01 | -.02 / -.02 |
| EG gap, core (same minus different group) | +.110 / +.128 | +.127 / +.301 | +.143 / +.151 | +.094 / +.155 | +.095 / +.170 | n/r |
| EG gap, decoder | -.001 / +.083 | +.065 / +.066 | +.197 / +.246 | -.029 / +.121 | +.041 / +.003 | -.004 / +.252 |
| Joint groups k=8 gap, core | +.032 / +.018 | +.046 / +.026 | +.034 / +.019 | +.019 / +.026 | +.019 / +.015 | +.007 / +.000 (n.s.) |
| Joint groups k=16 gap, core | +.046 / +.032 | +.057 / +.065 | +.053 / +.025 | +.028 / +.042 | +.032 / +.026 | +.027 / -.011 (n.s.) |
| Joint groups k=16 gap, decoder | +.032 / +.027 | +.048 / +.074 | +.124 / +.050 | +.027 / +.100 | +.039 / +.003 | -.034 / +.029 (n.s.) |

- Core EG gap: p <= 0.05 in all 12 cells from 2k to 75k (in 10 of them p <= 0.02). Seed 0 at 5k is p = 0.050.
- Core joint-group gap: k=8 has p <= 0.017 in 12/12 cells, k=16 has p < 0.02 in 11/12 (75k seed 1: p = 0.057). k=16 > k=8 in 12/12 cells.
- Decoder EG gap and decoder joint-group gap are real in about half the cells and **not stable**; both fade by 75k to 100k.
- p-values come from 300 permutations (floor 0.003). Full tables: `local_trend_full.txt` (all five parameter groups).

## 2. Two kinds of conflict, compared at the same level

Both rows use the noise-corrected cosine of **mean** gradients (class means for A, robot means for B).

| decoder | 2k | 10k | 25k | 50k | 75k | 100k |
|---|---|---|---|---|---|---|
| A. success-class vs fall-class, same robot (mean over robots) | -.71 / -.90 | -.76 / -.83 | -.86 / -.91 | -.87 / -.97 | -1.00 / -.99 | -1.00 / -1.00 |
| B. robot vs robot (mean over valid pairs) | +.08 / +.12 | +.22 / +.45 | +.27 / +.11 | +.39 / +.21 | +.08 / +.03 | +.06 / +.48 |

- **A is far more negative than B is positive**, and A gets stronger with training. Robots disagree with each other much less than good and
  bad data of the same robot disagree.
- Robot-pair conflicts: from 2k to 75k, 21% of the valid core pairs and 26% of the valid decoder pairs are negative. The negative ones are small
  (mean -0.07 core, -0.16 decoder). No pair is negative in more than 6 of 12 cells (core) or 8 of 12 (decoder).
  **Share of negative pairs inside an EG group: 19% core / 25% decoder; across groups: 21% / 26%.** EG grouping raises the mean cosine but does
  not change how often a pair is negative.
- Log-std head: A is -1.00 in every checkpoint. This is by construction of the Gaussian loss: the gradient in log sigma is 1 - z^2, positive for
  well-fit samples and negative for badly fit ones. A single std head cannot satisfy both, so this part is expected for any mixed-quality data.
  The **decoder** (mean path) result is the informative one.

Core, class-mean S-F (same estimator as A): +0.20 / +0.17 at 10k, about +0.1 up to 50k, **-0.13 / -0.14 at 75k and -0.58 / -0.58 at 100k**
(12 and 9 robots). Single-episode cosines in the core stay at 0, presumably because single-episode gradients are mostly noise. At 100k the success and
fall mean gradients in the core point against each other, while the robot's own batch gradient is mostly noise. This is consistent with
the two cancelling (hypothesis, not verified), and it means the quality conflict reaches the core late in training.

## 3. The critic weights are not a quality filter

The unweighted (w = 1, BC) and weighted (IQL) variants give the same success-fall cosine at every checkpoint, both seeds:
decoder -0.504 vs -0.510 (s0 2k) ... -0.621 vs -0.631 (s0 100k), -0.692 vs -0.699 (s1 100k). The largest difference in any group is 0.023 (core, seed 1, 50k).
A cosine ignores the overall scale of an episode, so I also measured the **effective weight** (episode gradient norm with weights divided by without):

| decoder, median over episodes, mean over robots | success | fall | low-return | fall / success |
|---|---|---|---|---|
| 2k to 100k, seed 0 | 1.01 to 1.17 | 0.88 to 1.01 | 0.87 to 1.01 | 0.80 to 0.95 |
| 2k to 100k, seed 1 | 0.98 to 1.14 | 0.93 to 0.96 | 0.95 to 1.00 | 0.82 to 0.97 |

Per robot at 2k to 10k, fall / success is 0.63 to 1.00 (bipeds lowest, e.g. TL 0.63 to 0.83; quadrupeds 0.84 to 0.99).
So the advantage weighting down-weights fall episodes by 3 to 20% at most; it does not remove their pull.

## 4. Per-robot view (decoder, success vs bad episodes, seed 0 / seed 1 at 100k)

- Strongest conflict (<= -0.79 in both seeds): Go1 -.86/-.83, AnyB -.88/-.95, AnyC -.82/-.90, H1 -.85/-.86, G1 -.87/-.86, TL -.80/-.83, OP3 -.88/-.91, NAO5 -.79/-.85.
- Weaker: A1 -.15/-.42, BG -.30/-.37, BKV0 -.45/-.64, Cassie -.54/-.58.
- **Bittle is the only robot with positive S-F (+.29/+.21).** Its success episodes also barely agree with each other (decoder S-S +0.04 to +0.24 at every checkpoint; every other robot is +0.91 or more), so its "success" class is not one coherent behaviour.
- Hexapod (no fall episodes, so success vs low-return): -.77/-.79 at 2k to -.93/-.96 at 100k.
- S-F strength follows how much the fall episodes agree with each other. Decoder F-F is high for Go1, AnyB, BKVB, Bittle, TL, Go2 (+0.4 to +0.9) and low for A1 and BG (about 0 to +0.2), BKV0 (+0.14 to +0.40) and Cassie (+0.19 to +0.36), which are the robots with the weakest S-F. Where the falls share no common direction, they do not oppose the successes either.
- In the core, F-F is below +0.1 for most robots after 5k (Bittle +0.3 to +0.5, BKVB about +0.2, Go1 +0.06 to +0.24 are the exceptions).
- Tables for every robot, group and checkpoint: `local_robots_full.txt` (S-F, S-L, S-S, F-F, alignment).

## 5. A1 and Go1 are not explained by gradient conflict

A1 stays at 13.9 to 14.5 return at every checkpoint in both seeds; Go1 is at 14.5 at 100k in both seeds (it reached 33.3 at 25k in seed 0 and 29.6 at 50k in seed 1, then fell back). Mean return over all 16 robots at 100k: 50.6 seed 0, 55.7 seed 1 (`returns_u2_by_ckpt.txt`).
- They are among the **best** aligned robots in the core: mean persistent cosine to the other robots over 2k to 75k is +0.19 (A1), +0.18 (Go1), +0.19 (Go2), +0.16 (BKVB) against +0.03 to +0.15 for the rest.
- Their share of the summed core signal is 4 to 10% (uniform = 6.25%), and their gradient has cosine +0.3 to +0.8 with the sum of all robots from 2k to 75k (one exception: Go1 seed 1 at 75k, +0.07). They are neither starved nor opposed.
- Their success-fall conflict is unrelated to failure: A1 is weak (-.15/-.42), Go1 strong (-.86/-.83), both fail; Go2 and AnyB are strong and learn.
- Bittle (+0.03) and the hexapod (+0.06) are the least aligned robots in the core (mean over 2k to 75k; Cassie, TL, G1, H1, NAO5 are at +0.07). The hexapod is the other robot that never learns. Seed 0 learned it at 10k (37.6) and lost it (0.1 at 75k); Cassie seed 0 did the same at the end (40.4 at 75k, 1.1 at 100k). I see no precursor of these collapses in the alignment numbers.

## 6. What this means for the intended paper story

| Component | What the measurement says | Strength |
|---|---|---|
| Shared expert (robots mostly aligned) | Robot-pair cosines are positive on average in the core (+0.06 to +0.22) and the decoder (+0.03 to +0.48); negative pairs are small and unstable | Supported |
| Morphology-routed experts | Same-EG-group pairs are more aligned in the core (+0.09 to +0.30, 12/12 cells). But sign of conflict is unchanged (19 to 25% negative inside groups, 21 to 26% outside). Joint-group labels add only +0.02 to +0.06 in the core, more with 16 groups | Supported as an alignment effect, **weak** as a conflict remover |
| Quality-aware routing in the decoder | Within-robot good vs bad data is the large conflict: -0.7 to -1.0 persistent in the decoder, stable from 2k to 100k, stronger than any robot-level effect | **Strongest evidence you have** |
| "Quality, not just robots": EG cannot fix this | EG groups robots; the conflict is inside each robot (and inside groups at the same rate). The critic weights do not fix it either (section 3) | Supported |

Consequences and cautions:
1. The story "conflict between embodiments" is the weak half; "conflict between data qualities inside an embodiment" is the strong half. Lead with the second.
2. Conflict is not harm. A negative cosine says the two classes pull in opposite directions; it does not say removing it improves return. A dose-response (filter or route the fall data, compare return) is the missing step.
3. Decoder-only routing may not be enough late: the core S-F turns negative at 75k to 100k. That is a measured reason to test a core variant, in either direction.
4. Do not credit "less conflict" for A1/Go1. The geometry does not show a conflict there (section 5); EG fixes them for some other reason that is not measured here.
5. The seeds agree on the conflict statistics (seed 0 vs 1 within about 0.1) but differ on which robots collapse, so the metric does not predict individual robot failures.

## 7. Limits

- Frozen checkpoints, one architecture, two seeds. Gradients are at fixed parameters, not along the training path.
- 64 episodes per class; fall episodes are short (29 to 54 steps for bipeds), so their gradients are noisier. S-F is measured on class-sampled episodes, not on the training mixture.
- Log-std head conflict is expected from the loss form (section 2); I did not test the claim that expert episodes lower the std and noisy ones raise it.
- Slimmed files (25k to 100k) have no BC variant; section 3 for those checkpoints was computed on the server from the full files.
- Joint-group results at 75k to 100k use fewer units and are not significant; this reflects missing signal, not proof that the structure is gone.

## Files

`local_trend_full.txt` (all groups, all checkpoints, with reliability columns), `local_robots_full.txt` (per robot),
`returns_u2_by_ckpt.txt` (per-robot returns of both runs at each checkpoint), scripts `conflict_trend.py`, `conflict_robots.py`
(new: reliability columns; per-robot tables), data in `data/U2-URMA-s{0,1}/`.
