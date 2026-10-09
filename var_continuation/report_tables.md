## 1. Starting checkpoint

- Run V7-CH-s0 (C-H: hierarchical shared core + MI), seed 0, checkpoint `models/at_75000` (75,000 updates), architecture of V7 (joint encoder 4 units, 3-layer core, no WeightNorm; passed as explicit legacy flags). Actor, value and both critics with target critics and optimizer state are in the checkpoint.
- Exact deterministic evaluation of this checkpoint (one episode per robot): Go2 50.8, mean over 16 robots 50.8; Cassie 1.1 and hexapod 0.1 were already lost.
- Original run, Go2 (evaluation every 100 updates): mean return per 5k bin to 65k: 33.4 (walks 0.64), 70k: 29.7 (walks 0.56), 75k: 30.4 (walks 0.54), 80k: 17.4 (walks 0.30), 85k: 14.3 (walks 0.22), 90k: 10.5 (walks 0.12), 95k: 3.9 (walks 0.02), 100k: 2.7 (walks 0.00).

## 2. Completed updates and starting checks

| Branch | Last update | Mean diff (start, vs checkpoint) | Log-std diff | Restored leaves (max diff) | Optimizer count | Evaluation noise floor |
|---|---:|---:|---:|---|---|---:|
| S0-original | 95000 | 5.5e-06 | 1.2e-07 | policy:0e+00, value_critic:0e+00, critic_1:0e+00, critic_2:0e+00, target_critic_1:0e+00, target_critic_2:0e+00 | [300000] (expected 300000) | 0.0e+00 |
| S1-detach | 95000 | 5.5e-06 | 1.2e-07 | policy:0e+00, value_critic:0e+00, critic_1:0e+00, critic_2:0e+00, target_critic_1:0e+00, target_critic_2:0e+00 | [300000] (expected 300000) | 0.0e+00 |
| S2-private-std | 95000 | 5.5e-06 | 1.2e-07 | policy:0e+00, value_critic:0e+00, critic_1:0e+00, critic_2:0e+00, target_critic_1:0e+00, target_critic_2:0e+00 | [300000] (expected 300000) | 0.0e+00 |
| S3-protected-mean | 95000 | 5.5e-06 | 1.2e-07 | policy:0e+00, value_critic:0e+00, critic_1:0e+00, critic_2:0e+00, target_critic_1:0e+00, target_critic_2:0e+00 | [300000] (expected 300000) | 0.0e+00 |

Spread of the checkpoint's log-std over states, per robot (max over joints; a state-independent per-joint table would be off by this much): a1: 0.152, go1: 0.196, go2: 0.170, anyb: 0.178, anyc: 0.208, bkv0: 0.148, bkvb: 0.138, bg: 0.192, bit: 0.215, h1: 0.226, g1: 0.089, tl: 0.180, op3: 0.141, nao5: 0.097, cas: 0.311, hx: 0.125. Learning-rate factor at the restored count: 4.39e-05.
New leaves of S2: opt_state/1/mu/params/logstd_offset, opt_state/1/nu/params/logstd_offset, params/params/logstd_offset; the head's Adam first moments were zeroed (frozen head).

Synchronization: 80 updates with content hashes in all branches (75000..94750); identical minibatch hashes: True, identical critic hashes: True, identical value-critic hashes: True.

## 3. Comparison at the common completed horizon

Common horizon: 95000 updates (+20000). Returns are means of the last 10 evaluations (one episode per robot every 100 updates) up to the horizon; "walks" is the share of Go2's evaluations above the frozen threshold (26.2) in the last 2,000 updates before it. Collapse = Go2 learned (window mean >= frozen threshold) at the start and not at the horizon.

| Branch | Go2 return at +H | Go2 walks (last 2k) | Mean return, 16 robots | Weighted mean error (Go2 / robot mean) | Std behavior (mean sigma, Go2 / robots) | Collapse observed |
|---|---:|---:|---:|---|---|---|
| S0-original | 4.3 | 0.00 | 47.9 | 0.7864 / 0.6111 | 0.874 / 0.778 (at bounds 0.000) | yes |
| S1-detach | 4.0 | 0.00 | 47.9 | 0.7864 / 0.6111 | 0.876 / 0.779 (at bounds 0.000) | yes |
| S2-private-std | 4.0 | 0.00 | 47.8 | 0.7865 / 0.6111 | 0.879 / 0.779 (at bounds 0.000) | yes |
| S3-protected-mean | 7.4 | 0.00 | 48.1 | 0.7862 / 0.6112 | 0.876 / 0.779 (at bounds 0.000) | yes |
| original run (CH-s0, same updates) | 2.4 | 0.00 | 47.8 | n/a | n/a | yes |

Go2's window mean at the start (original run, last 10 evaluations before 75000): 29.9.

Go2, mean return per bin after the start (walks in parentheses):

| Branch | +1k | +2k | +5k | +10k | +15k | +20k |
|---|---:|---:|---:|---:|---:|---:|
| S0-original | 37.5 (0.70) | 16.9 (0.30) | 20.5 (0.30) | 11.7 (0.20) | 21.5 (0.40) | 4.3 (0.00) |
| S1-detach | 37.5 (0.70) | 16.9 (0.30) | 20.3 (0.30) | 11.7 (0.20) | 16.7 (0.30) | 4.0 (0.00) |
| S2-private-std | 37.5 (0.70) | 16.9 (0.30) | 21.3 (0.30) | 11.7 (0.20) | 16.8 (0.30) | 4.0 (0.00) |
| S3-protected-mean | 42.4 (0.80) | 17.0 (0.30) | 15.5 (0.20) | 7.0 (0.10) | 7.2 (0.10) | 7.4 (0.00) |
| original run (CH-s0, same updates) | 22.5 (0.40) | 11.5 (0.20) | 22.5 (0.40) | 4.6 (0.00) | 12.4 (0.20) | 2.4 (0.00) |

Robots that change status by the horizon (learned = window mean >= frozen threshold; start = original run at 75k):

- S0-original: lost ['go2'], gained none; changes of 10 or more: go2 -26.
- S1-detach: lost ['go2'], gained none; changes of 10 or more: go2 -26.
- S2-private-std: lost ['go2'], gained none; changes of 10 or more: go2 -26.
- S3-protected-mean: lost ['go2'], gained none; changes of 10 or more: go2 -23.
- original run (CH-s0, same updates): lost ['go2'], gained none; changes of 10 or more: go2 -27.

## 4. Mean and variance diagnostics at the saved checkpoints

Fixed minibatches of the training data (the checkpoint's own training set: no held-out episodes exist for it), the same for every checkpoint; advantage weights from each checkpoint's own critics. Per cell: Go2 / mean over 16 robots.

**S0-original**

| Checkpoint | weighted mean err | mean err | sigma | at bounds | (a-mu)^2/sigma^2 | w/sigma^2 mean | w/sigma^2 p99 | w mean | w p99 | w clipped | ESS share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| at_75000 | 0.769 / 0.599 | 0.789 / 0.626 | 0.879 / 0.781 | 0 / 0 | 1.02 / 1.02 | 1.27 / 1.67 | 1.71 / 2.33 | 0.975 / 1 | 1.21 / 1.27 | 0 / 0.000214 | 0.685 / 0.535 |
| at_75000 (training noise) | 0.769 / 0.599 | 0.789 / 0.626 | 0.879 / 0.781 | 0 / 0 | 1.02 / 1.02 | 1.27 / 1.67 | 1.71 / 2.33 | 0.975 / 1 | 1.21 / 1.27 | 0 / 0.000214 | 0.685 / 0.535 |
| at_76000 | 0.782 / 0.605 | 0.79 / 0.626 | 0.871 / 0.777 | 0 / 0 | 1.04 / 1.03 | 1.35 / 1.71 | 1.83 / 2.38 | 1.01 / 1.01 | 1.31 / 1.29 | 0 / 0.000259 | 0.753 / 0.417 |
| at_76000 (training noise) | 0.782 / 0.605 | 0.79 / 0.626 | 0.871 / 0.777 | 0 / 0 | 1.04 / 1.03 | 1.35 / 1.71 | 1.83 / 2.38 | 1.01 / 1.01 | 1.31 / 1.29 | 0 / 0.000259 | 0.753 / 0.417 |
| at_80000 | 0.79 / 0.606 | 0.789 / 0.626 | 0.87 / 0.775 | 0 / 0 | 1.04 / 1.03 | 1.3 / 1.69 | 1.79 / 2.31 | 0.976 / 0.999 | 1.22 / 1.24 | 0 / 0.000229 | 0.929 / 0.548 |
| at_80000 (training noise) | 0.79 / 0.606 | 0.789 / 0.626 | 0.87 / 0.775 | 0 / 0 | 1.04 / 1.03 | 1.3 / 1.69 | 1.79 / 2.31 | 0.976 / 0.999 | 1.22 / 1.24 | 0 / 0.000229 | 0.929 / 0.548 |
| at_85000 | 0.786 / 0.604 | 0.789 / 0.626 | 0.878 / 0.781 | 0 / 0 | 1.02 / 1.02 | 1.2 / 1.65 | 1.76 / 2.51 | 0.916 / 0.984 | 1.27 / 1.38 | 0 / 0.000214 | 0.927 / 0.454 |
| at_85000 (training noise) | 0.786 / 0.604 | 0.789 / 0.626 | 0.878 / 0.781 | 0 / 0 | 1.02 / 1.02 | 1.2 / 1.65 | 1.76 / 2.51 | 0.916 / 0.984 | 1.27 / 1.38 | 0 / 0.000214 | 0.927 / 0.454 |
| at_90000 | 0.789 / 0.607 | 0.788 / 0.626 | 0.876 / 0.779 | 0 / 0 | 1.02 / 1.02 | 1.3 / 1.7 | 1.71 / 2.31 | 0.988 / 1.01 | 1.18 / 1.24 | 0 / 0.000214 | 0.981 / 0.581 |
| at_90000 (training noise) | 0.789 / 0.607 | 0.788 / 0.626 | 0.876 / 0.779 | 0 / 0 | 1.02 / 1.02 | 1.3 / 1.7 | 1.71 / 2.31 | 0.988 / 1.01 | 1.18 / 1.24 | 0 / 0.000214 | 0.981 / 0.581 |
| at_95000 | 0.786 / 0.611 | 0.788 / 0.625 | 0.874 / 0.778 | 0 / 0 | 1.03 / 1.03 | 1.25 / 1.68 | 1.68 / 2.3 | 0.947 / 0.999 | 1.16 / 1.24 | 0 / 0.000229 | 0.971 / 0.543 |
| at_95000 (training noise) | 0.786 / 0.611 | 0.788 / 0.625 | 0.874 / 0.778 | 0 / 0 | 1.03 / 1.03 | 1.25 / 1.68 | 1.68 / 2.3 | 0.947 / 0.999 | 1.16 / 1.24 | 0 / 0.000229 | 0.971 / 0.543 |

Gradient norms of the robot-averaged loss gradient through the action mean and through the log-std, by parameter group, and the cosine between the two pathways' gradients:

| Checkpoint | total (mean path) | total (std path) | embedding (mean path) | embedding (std path) | std head (std path) | cos total | cos embedding |
|---|---|---|---|---|---|---|---|
| at_75000 | 0.0342 | 0.0799 | 0.0071 | 0.0441 | 0.0665 | -0.001 | -0.018 |
| at_75000 (training noise) | 0.0341 | 0.0799 | 0.0071 | 0.0441 | 0.0665 | -0.001 | -0.015 |
| at_76000 | 0.0384 | 0.0538 | 0.0085 | 0.0213 | 0.0493 | -0.001 | -0.017 |
| at_76000 (training noise) | 0.0384 | 0.0538 | 0.0085 | 0.0212 | 0.0492 | -0.000 | -0.016 |
| at_80000 | 0.0340 | 0.0516 | 0.0072 | 0.0246 | 0.0453 | 0.002 | 0.019 |
| at_80000 (training noise) | 0.0339 | 0.0516 | 0.0071 | 0.0246 | 0.0453 | 0.002 | 0.019 |
| at_85000 | 0.0357 | 0.0593 | 0.0076 | 0.0302 | 0.0508 | 0.003 | 0.006 |
| at_85000 (training noise) | 0.0357 | 0.0593 | 0.0076 | 0.0302 | 0.0508 | 0.003 | 0.007 |
| at_90000 | 0.0328 | 0.0523 | 0.0067 | 0.0235 | 0.0466 | -0.000 | -0.002 |
| at_90000 (training noise) | 0.0328 | 0.0523 | 0.0067 | 0.0235 | 0.0466 | 0.000 | -0.001 |
| at_95000 | 0.0346 | 0.0508 | 0.0068 | 0.0230 | 0.0452 | 0.001 | 0.008 |
| at_95000 (training noise) | 0.0345 | 0.0508 | 0.0068 | 0.0230 | 0.0452 | 0.001 | 0.008 |

**S1-detach**

| Checkpoint | weighted mean err | mean err | sigma | at bounds | (a-mu)^2/sigma^2 | w/sigma^2 mean | w/sigma^2 p99 | w mean | w p99 | w clipped | ESS share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| at_75000 | 0.769 / 0.599 | 0.789 / 0.626 | 0.879 / 0.781 | 0 / 0 | 1.02 / 1.02 | 1.27 / 1.67 | 1.71 / 2.33 | 0.976 / 1 | 1.21 / 1.27 | 0 / 0.000214 | 0.686 / 0.535 |
| at_75000 (training noise) | 0.769 / 0.599 | 0.789 / 0.626 | 0.879 / 0.781 | 0 / 0 | 1.02 / 1.02 | 1.27 / 1.67 | 1.71 / 2.33 | 0.976 / 1 | 1.21 / 1.27 | 0 / 0.000214 | 0.686 / 0.535 |
| at_76000 | 0.782 / 0.605 | 0.79 / 0.626 | 0.873 / 0.778 | 0 / 0 | 1.03 / 1.03 | 1.34 / 1.7 | 1.82 / 2.37 | 1.01 / 1.01 | 1.31 / 1.29 | 0 / 0.000259 | 0.754 / 0.416 |
| at_76000 (training noise) | 0.782 / 0.605 | 0.79 / 0.626 | 0.873 / 0.778 | 0 / 0 | 1.03 / 1.03 | 1.34 / 1.7 | 1.82 / 2.37 | 1.01 / 1.01 | 1.31 / 1.29 | 0 / 0.000259 | 0.754 / 0.416 |
| at_80000 | 0.79 / 0.606 | 0.789 / 0.626 | 0.876 / 0.78 | 0 / 0 | 1.03 / 1.02 | 1.28 / 1.68 | 1.77 / 2.29 | 0.976 / 0.999 | 1.22 / 1.24 | 0 / 0.000229 | 0.929 / 0.548 |
| at_80000 (training noise) | 0.79 / 0.606 | 0.789 / 0.626 | 0.876 / 0.78 | 0 / 0 | 1.03 / 1.02 | 1.28 / 1.68 | 1.77 / 2.29 | 0.976 / 0.999 | 1.22 / 1.24 | 0 / 0.000229 | 0.929 / 0.548 |
| at_85000 | 0.786 / 0.604 | 0.789 / 0.626 | 0.875 / 0.778 | 0 / 0 | 1.03 / 1.03 | 1.21 / 1.66 | 1.78 / 2.53 | 0.916 / 0.984 | 1.27 / 1.38 | 0 / 0.000214 | 0.927 / 0.454 |
| at_85000 (training noise) | 0.786 / 0.604 | 0.789 / 0.626 | 0.875 / 0.778 | 0 / 0 | 1.03 / 1.03 | 1.21 / 1.66 | 1.78 / 2.53 | 0.916 / 0.984 | 1.27 / 1.38 | 0 / 0.000214 | 0.927 / 0.454 |
| at_90000 | 0.789 / 0.607 | 0.788 / 0.626 | 0.875 / 0.779 | 0 / 0 | 1.03 / 1.03 | 1.3 / 1.7 | 1.73 / 2.32 | 0.988 / 1.01 | 1.18 / 1.24 | 0 / 0.000214 | 0.981 / 0.581 |
| at_90000 (training noise) | 0.789 / 0.607 | 0.788 / 0.626 | 0.875 / 0.779 | 0 / 0 | 1.03 / 1.03 | 1.3 / 1.7 | 1.73 / 2.32 | 0.988 / 1.01 | 1.18 / 1.24 | 0 / 0.000214 | 0.981 / 0.581 |
| at_95000 | 0.786 / 0.611 | 0.788 / 0.625 | 0.876 / 0.779 | 0 / 0 | 1.03 / 1.02 | 1.24 / 1.68 | 1.68 / 2.29 | 0.947 / 0.999 | 1.16 / 1.24 | 0 / 0.000229 | 0.971 / 0.543 |
| at_95000 (training noise) | 0.786 / 0.611 | 0.788 / 0.625 | 0.876 / 0.779 | 0 / 0 | 1.03 / 1.02 | 1.24 / 1.68 | 1.68 / 2.29 | 0.947 / 0.999 | 1.16 / 1.24 | 0 / 0.000229 | 0.971 / 0.543 |

Gradient norms of the robot-averaged loss gradient through the action mean and through the log-std, by parameter group, and the cosine between the two pathways' gradients:

| Checkpoint | total (mean path) | total (std path) | embedding (mean path) | embedding (std path) | std head (std path) | cos total | cos embedding |
|---|---|---|---|---|---|---|---|
| at_75000 | 0.0342 | 0.0665 | 0.0071 | 0.0000 | 0.0665 | 0.000 | 0.000 |
| at_75000 (training noise) | 0.0341 | 0.0665 | 0.0071 | 0.0000 | 0.0665 | 0.000 | 0.000 |
| at_76000 | 0.0384 | 0.0498 | 0.0086 | 0.0000 | 0.0498 | 0.000 | 0.000 |
| at_76000 (training noise) | 0.0384 | 0.0498 | 0.0086 | 0.0000 | 0.0498 | 0.000 | 0.000 |
| at_80000 | 0.0336 | 0.0474 | 0.0071 | 0.0000 | 0.0474 | 0.000 | 0.000 |
| at_80000 (training noise) | 0.0336 | 0.0474 | 0.0071 | 0.0000 | 0.0474 | 0.000 | 0.000 |
| at_85000 | 0.0359 | 0.0479 | 0.0076 | 0.0000 | 0.0479 | 0.000 | 0.000 |
| at_85000 (training noise) | 0.0359 | 0.0478 | 0.0076 | 0.0000 | 0.0478 | 0.000 | 0.000 |
| at_90000 | 0.0328 | 0.0462 | 0.0066 | 0.0000 | 0.0462 | 0.000 | 0.000 |
| at_90000 (training noise) | 0.0328 | 0.0462 | 0.0066 | 0.0000 | 0.0462 | 0.000 | 0.000 |
| at_95000 | 0.0344 | 0.0457 | 0.0067 | 0.0000 | 0.0457 | 0.000 | 0.000 |
| at_95000 (training noise) | 0.0343 | 0.0457 | 0.0067 | 0.0000 | 0.0457 | 0.000 | 0.000 |

**S2-private-std**

| Checkpoint | weighted mean err | mean err | sigma | at bounds | (a-mu)^2/sigma^2 | w/sigma^2 mean | w/sigma^2 p99 | w mean | w p99 | w clipped | ESS share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| at_76000 | 0.782 / 0.605 | 0.79 / 0.626 | 0.881 / 0.78 | 0 / 0 | 1.02 / 1.02 | 1.32 / 1.7 | 1.78 / 2.36 | 1.01 / 1.01 | 1.31 / 1.29 | 0 / 0.000259 | 0.754 / 0.416 |
| at_76000 (training noise) | 0.782 / 0.605 | 0.79 / 0.626 | 0.881 / 0.78 | 0 / 0 | 1.02 / 1.02 | 1.32 / 1.7 | 1.78 / 2.36 | 1.01 / 1.01 | 1.31 / 1.29 | 0 / 0.000259 | 0.754 / 0.416 |
| at_80000 | 0.79 / 0.606 | 0.789 / 0.626 | 0.881 / 0.78 | 0 / 0 | 1.01 / 1.02 | 1.27 / 1.68 | 1.74 / 2.29 | 0.976 / 0.999 | 1.22 / 1.24 | 0 / 0.000229 | 0.929 / 0.548 |
| at_80000 (training noise) | 0.79 / 0.606 | 0.789 / 0.626 | 0.881 / 0.78 | 0 / 0 | 1.01 / 1.02 | 1.27 / 1.68 | 1.74 / 2.29 | 0.976 / 0.999 | 1.22 / 1.24 | 0 / 0.000229 | 0.929 / 0.548 |
| at_85000 | 0.786 / 0.604 | 0.789 / 0.626 | 0.879 / 0.778 | 0 / 0 | 1.02 / 1.03 | 1.19 / 1.66 | 1.75 / 2.53 | 0.916 / 0.984 | 1.27 / 1.38 | 0 / 0.000214 | 0.927 / 0.454 |
| at_85000 (training noise) | 0.786 / 0.604 | 0.789 / 0.626 | 0.879 / 0.778 | 0 / 0 | 1.02 / 1.03 | 1.19 / 1.66 | 1.75 / 2.53 | 0.916 / 0.984 | 1.27 / 1.38 | 0 / 0.000214 | 0.927 / 0.454 |
| at_90000 | 0.789 / 0.607 | 0.788 / 0.626 | 0.878 / 0.779 | 0 / 0 | 1.02 / 1.03 | 1.29 / 1.71 | 1.71 / 2.32 | 0.988 / 1.01 | 1.18 / 1.24 | 0 / 0.000214 | 0.981 / 0.581 |
| at_90000 (training noise) | 0.789 / 0.607 | 0.788 / 0.626 | 0.878 / 0.779 | 0 / 0 | 1.02 / 1.03 | 1.29 / 1.71 | 1.71 / 2.32 | 0.988 / 1.01 | 1.18 / 1.24 | 0 / 0.000214 | 0.981 / 0.581 |
| at_95000 | 0.786 / 0.611 | 0.788 / 0.625 | 0.879 / 0.779 | 0 / 0 | 1.02 / 1.02 | 1.23 / 1.68 | 1.66 / 2.29 | 0.947 / 0.999 | 1.16 / 1.24 | 0 / 0.000229 | 0.971 / 0.543 |
| at_95000 (training noise) | 0.786 / 0.611 | 0.788 / 0.625 | 0.879 / 0.779 | 0 / 0 | 1.02 / 1.02 | 1.23 / 1.68 | 1.66 / 2.29 | 0.947 / 0.999 | 1.16 / 1.24 | 0 / 0.000229 | 0.971 / 0.543 |

Gradient norms of the robot-averaged loss gradient through the action mean and through the log-std, by parameter group, and the cosine between the two pathways' gradients:

| Checkpoint | total (mean path) | total (std path) | embedding (mean path) | embedding (std path) | std head (std path) | cos total | cos embedding |
|---|---|---|---|---|---|---|---|
| at_76000 | 0.0384 | 0.0050 | 0.0086 | 0.0000 | 0.0050 | 0.000 | 0.000 |
| at_76000 (training noise) | 0.0384 | 0.0050 | 0.0086 | 0.0000 | 0.0050 | 0.000 | 0.000 |
| at_80000 | 0.0337 | 0.0050 | 0.0070 | 0.0000 | 0.0050 | 0.000 | 0.000 |
| at_80000 (training noise) | 0.0337 | 0.0050 | 0.0070 | 0.0000 | 0.0050 | 0.000 | 0.000 |
| at_85000 | 0.0361 | 0.0050 | 0.0076 | 0.0000 | 0.0050 | 0.000 | 0.000 |
| at_85000 (training noise) | 0.0361 | 0.0050 | 0.0076 | 0.0000 | 0.0050 | 0.000 | 0.000 |
| at_90000 | 0.0328 | 0.0049 | 0.0066 | 0.0000 | 0.0049 | 0.000 | 0.000 |
| at_90000 (training noise) | 0.0328 | 0.0049 | 0.0066 | 0.0000 | 0.0049 | 0.000 | 0.000 |
| at_95000 | 0.0342 | 0.0049 | 0.0067 | 0.0000 | 0.0049 | 0.000 | 0.000 |
| at_95000 (training noise) | 0.0342 | 0.0049 | 0.0067 | 0.0000 | 0.0049 | 0.000 | 0.000 |

**S3-protected-mean**

| Checkpoint | weighted mean err | mean err | sigma | at bounds | (a-mu)^2/sigma^2 | w/sigma^2 mean | w/sigma^2 p99 | w mean | w p99 | w clipped | ESS share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| at_75000 | 0.769 / 0.599 | 0.789 / 0.626 | 0.879 / 0.781 | 0 / 0 | 1.02 / 1.02 | 1.27 / 1.67 | 1.71 / 2.33 | 0.976 / 1 | 1.21 / 1.27 | 0 / 0.000214 | 0.686 / 0.535 |
| at_75000 (training noise) | 0.769 / 0.599 | 0.789 / 0.626 | 0.879 / 0.781 | 0 / 0 | 1.02 / 1.02 | 1.27 / 1.67 | 1.71 / 2.33 | 0.976 / 1 | 1.21 / 1.27 | 0 / 0.000214 | 0.686 / 0.535 |
| at_76000 | 0.782 / 0.605 | 0.789 / 0.626 | 0.873 / 0.778 | 0 / 0 | 1.03 / 1.03 | 1.34 / 1.71 | 1.82 / 2.37 | 1.01 / 1.01 | 1.31 / 1.29 | 0 / 0.000259 | 0.754 / 0.416 |
| at_76000 (training noise) | 0.782 / 0.605 | 0.789 / 0.626 | 0.873 / 0.778 | 0 / 0 | 1.03 / 1.03 | 1.34 / 1.71 | 1.82 / 2.37 | 1.01 / 1.01 | 1.31 / 1.29 | 0 / 0.000259 | 0.754 / 0.416 |
| at_80000 | 0.79 / 0.606 | 0.789 / 0.626 | 0.875 / 0.779 | 0 / 0 | 1.03 / 1.02 | 1.28 / 1.68 | 1.77 / 2.29 | 0.976 / 0.999 | 1.22 / 1.24 | 0 / 0.000229 | 0.929 / 0.548 |
| at_80000 (training noise) | 0.79 / 0.606 | 0.789 / 0.626 | 0.875 / 0.779 | 0 / 0 | 1.03 / 1.02 | 1.28 / 1.68 | 1.77 / 2.29 | 0.976 / 0.999 | 1.22 / 1.24 | 0 / 0.000229 | 0.929 / 0.548 |
| at_85000 | 0.786 / 0.604 | 0.788 / 0.626 | 0.874 / 0.778 | 0 / 0 | 1.03 / 1.03 | 1.21 / 1.66 | 1.77 / 2.53 | 0.916 / 0.984 | 1.27 / 1.38 | 0 / 0.000214 | 0.927 / 0.454 |
| at_85000 (training noise) | 0.786 / 0.604 | 0.788 / 0.626 | 0.874 / 0.778 | 0 / 0 | 1.03 / 1.03 | 1.21 / 1.66 | 1.77 / 2.53 | 0.916 / 0.984 | 1.27 / 1.38 | 0 / 0.000214 | 0.927 / 0.454 |
| at_90000 | 0.789 / 0.607 | 0.788 / 0.626 | 0.874 / 0.779 | 0 / 0 | 1.03 / 1.03 | 1.3 / 1.7 | 1.73 / 2.32 | 0.988 / 1.01 | 1.18 / 1.24 | 0 / 0.000214 | 0.981 / 0.581 |
| at_90000 (training noise) | 0.789 / 0.607 | 0.788 / 0.626 | 0.874 / 0.779 | 0 / 0 | 1.03 / 1.03 | 1.3 / 1.7 | 1.73 / 2.32 | 0.988 / 1.01 | 1.18 / 1.24 | 0 / 0.000214 | 0.981 / 0.581 |
| at_95000 | 0.786 / 0.611 | 0.788 / 0.626 | 0.876 / 0.779 | 0 / 0 | 1.03 / 1.02 | 1.24 / 1.68 | 1.68 / 2.29 | 0.947 / 0.999 | 1.16 / 1.24 | 0 / 0.000229 | 0.971 / 0.543 |
| at_95000 (training noise) | 0.786 / 0.611 | 0.788 / 0.626 | 0.876 / 0.779 | 0 / 0 | 1.03 / 1.02 | 1.24 / 1.68 | 1.68 / 2.29 | 0.947 / 0.999 | 1.16 / 1.24 | 0 / 0.000229 | 0.971 / 0.543 |

Gradient norms of the robot-averaged loss gradient through the action mean and through the log-std, by parameter group, and the cosine between the two pathways' gradients:

| Checkpoint | total (mean path) | total (std path) | embedding (mean path) | embedding (std path) | std head (std path) | cos total | cos embedding |
|---|---|---|---|---|---|---|---|
| at_75000 | 0.0210 | 0.0665 | 0.0045 | 0.0000 | 0.0665 | 0.000 | 0.000 |
| at_75000 (training noise) | 0.0210 | 0.0665 | 0.0045 | 0.0000 | 0.0665 | 0.000 | 0.000 |
| at_76000 | 0.0224 | 0.0496 | 0.0048 | 0.0000 | 0.0496 | 0.000 | 0.000 |
| at_76000 (training noise) | 0.0224 | 0.0496 | 0.0048 | 0.0000 | 0.0496 | 0.000 | 0.000 |
| at_80000 | 0.0201 | 0.0474 | 0.0044 | 0.0000 | 0.0474 | 0.000 | 0.000 |
| at_80000 (training noise) | 0.0201 | 0.0474 | 0.0044 | 0.0000 | 0.0474 | 0.000 | 0.000 |
| at_85000 | 0.0220 | 0.0485 | 0.0047 | 0.0000 | 0.0485 | 0.000 | 0.000 |
| at_85000 (training noise) | 0.0220 | 0.0485 | 0.0047 | 0.0000 | 0.0485 | 0.000 | 0.000 |
| at_90000 | 0.0199 | 0.0467 | 0.0040 | 0.0000 | 0.0467 | 0.000 | 0.000 |
| at_90000 (training noise) | 0.0199 | 0.0467 | 0.0040 | 0.0000 | 0.0467 | 0.000 | 0.000 |
| at_95000 | 0.0209 | 0.0463 | 0.0041 | 0.0000 | 0.0463 | 0.000 | 0.000 |
| at_95000 (training noise) | 0.0209 | 0.0463 | 0.0041 | 0.0000 | 0.0463 | 0.000 | 0.000 |

## 5. Training-time gradient diagnostics (logged every 250 updates; mean over the period)

| Branch | grad norm 0-5k | grad norm 5-10k | grad norm 10-20k | std-param grad 0-5k | std-param grad 5-10k | std-param grad 10-20k | embedding grad 0-5k | embedding grad 5-10k | embedding grad 10-20k | clip factor min | core router task grad | decoder router task grad | core MI cap | decoder MI cap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0-original | 0.1149 | 0.1255 | 0.1303 | 0.0829 | 0.0886 | 0.0935 | 0.0479 | 0.0555 | 0.0612 | 1.000 | 0.0041 | 0.0135 | 1.000 | 1.000 |
| S1-detach | 0.1047 | 0.1119 | 0.1152 | 0.0833 | 0.0877 | 0.0936 | 0.0125 | 0.0136 | 0.0140 | 1.000 | 0.0041 | 0.0135 | 1.000 | 1.000 |
| S2-private-std | 0.0631 | 0.0679 | 0.0654 | 0.0090 | 0.0096 | 0.0100 | 0.0125 | 0.0136 | 0.0141 | 1.000 | 0.0041 | 0.0135 | 1.000 | 1.000 |
| S3-protected-mean | 0.0922 | 0.0980 | 0.1034 | 0.0835 | 0.0886 | 0.0949 | 0.0077 | 0.0083 | 0.0086 | 1.000 | 0.0024 | 0.0083 | 1.000 | 1.000 |
