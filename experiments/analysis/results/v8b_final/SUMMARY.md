# V8b batch at 100k updates (read 2026-10-09)

All numbers are the mean return over all 16 robots (`analyze_morse.ROBOTS`, Cassie and hexapod included). Each value
is the mean of the last 10 evaluations up to the stated update. Full tables (per robot, per class, every 10k, gate
robots): `v8b_final_100k.md`. Script: `analysis/v8b_final.py`.

Runs: J = joint-group MI, R = robot MI, G = compatibility routing; 25/50/75 = expert width (¼ is 855k active
parameters, dense URMAv2 is 830k). Seed 0 for every V8b run. J50/R50/J75/R75 ran on Hessian and were restarted from
their 50k checkpoint after the shell died, so their curve is the original run to 50k and the resumed run after. J25,
R25, G25 and EG-s2 ran on the H100 slices, G50/G75/EG-s1/57cfg-s1 are the earlier reads.

| run | 50k | 75k | 100k | avg 80-100k |
|---|---:|---:|---:|---:|
| J25 | 51.2 | 51.0 | 51.7 | 49.7 |
| J50 | 55.1 | 55.3 | 55.8 | 55.8 |
| J75 | 49.9 | 44.9 | 46.2 | 45.1 |
| R25 | 53.9 | 51.2 | 49.6 | 49.5 |
| R50 | 54.6 | 47.7 | 47.7 | 48.4 |
| R75 | 55.6 | 53.1 | 48.6 | 49.7 |
| G25 | 53.4 | 55.5 | 55.8 | 55.7 |
| G50 | 54.5 | 54.7 | 53.4 | 53.8 |
| G75 | 49.8 | 51.8 | 52.7 | 51.6 |
| EG-s1 | 51.2 | 54.7 | 55.3 | 54.9 |
| EG-s2 | 48.3 | 46.5 | 47.3 | 47.2 |
| 57cfg-s1 | 45.0 | 42.5 | 38.4 | 40.5 |

## What the numbers say

1. **Nothing separates from seed noise.** The two EG seeds, identical configuration, end at 55.3 and 47.3 (8.0
   apart). Every MoE run lies between 46 and 56, inside that spread. Only 57cfg-s1 (38.4) is clearly below.
2. **Eight robots decide the ranking; the other eight are identical in every run.** A1, Go1, Go2, badger, Talos, NAO,
   Cassie and the hexapod are all-or-nothing (14 vs 53, 1 vs 81, 1 vs 91, ...). ANYmal B/C, Barkour v0/vB, Bittle, H1,
   G1 and OP3 end within 3 points of each other in all 12 runs. One gate robot is worth 2-7 points of the 16-robot
   mean (Cassie 5.1, NAO 5.6, Talos 6.8, Go2 2.4, A1 or Go1 2.4 each, badger 2.3, hexapod up to 1.2).
3. **A1 and Go1 are learned only by the EG baseline.** EG-s1 learns both, EG-s2 learns Go1. None of the nine MoE runs
   learns either (both stay at 14).
4. **Cassie is where the directions differ.** Learned (return at least 40) by J25, J50, G25, G50, G75 (5 of 9 MoE
   runs) and by none of R25, R50, R75, J75, nor by the EG and 57cfg runs. G holds it in all three widths (G25 from 60k,
   G50 from 70k, G75 only at 100k), R never keeps it in any width: R25 peaked at 71 (30k) and R50 at 64 (50k), both at 1 from 70k,
   R75 flickers (33 at 40k, 1 at 80k, 41 at 90k, 1 at 100k). J75 also had 41-50 at 30-50k and ended at 1. The
   5-point gap between G and R at each width (55.8 vs 49.6, 53.4 vs 47.7, 52.7 vs 48.6) is exactly Cassie. One
   seed per cell, so this is a pattern to test, not a result.
5. **Width has no consistent effect.** J: 51.7 / 55.8 / 46.2 for ¼ / ½ / ¾. R: 49.6 / 47.7 / 48.6. G: 55.8 / 53.4 /
   52.7. The ¼ width (855k active, +3% over dense) matches or beats the wider ones in G and R and trails J50.
6. **The final window is unstable for the gate robots.** J25 has Cassie 9 at 90k and 65 at 100k; R25 has the hexapod
   at 64 (50k), 38 (90k), 20 (100k); EG-s2 loses NAO (79 at 40k, 1 at 100k). Hence the `avg 80-100k` column. The
   ranking does not change with it (J50, G25, EG-s1 on top; J75, EG-s2, 57cfg-s1 at the bottom).
7. **The hexapod is learned by no run** except transiently by R25 (peak 64, 20 at the end).

## Changes since the interim read (81k, 2026-10-08)

J50 55.8 (same), J75 44.9 to 46.2, R50 48.3 to 47.7, R75 53.2 to 48.6 (Cassie 25 at 70k, 1 at 80k, 41 at 90k, 1 at 100k), G50 54.0 to
53.4, G75 51.5 to 52.7.

## Run facts

- Hessian runs finished 2026-10-08 19:07-19:30 UTC; outputs were on node-local `/tmp/v8b_runs` and are copied to
  `runs/morse/V8b-{J,R}-resume/` (final checkpoint in `models/tmp`, `at_75000`, TB cache, diagnostics, run log; tar md5
  verified against the source).
- H100: J25 about 9.0 h, R25 about 9 h, G25 about 10.9 h, EG-s2 about 3 h, no errors in any log.
- Local copies of the H100 runs hold only the TB cache, `sync_hashes.jsonl`, `morse_diagnostics`, `checkpoints.jsonl`
  and the run log. Their checkpoints (`at_25000/50000/75000`, `tmp`) stay on
  `/mnt/data/{tuannl,duongnm2}/morse/runs/morse/<exp>/<id>/models`.
