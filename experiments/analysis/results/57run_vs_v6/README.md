# The 57 run vs the v6 runs: what makes a good run

Starting point: the head of `MORSE_HANDOFF.md` ("Latest state: read §14 first"; §14 was last checked 2026-10-05
about 03:50 UTC). This file checks §14's numbers and its reading (§14.5) against the cached curves. All tables are in
[tables.md](tables.md), made by `experiments/analysis/compare_57run_v6.py`. The 57 run is `CAP-8x64-top4-s0`
(57.1 at 65k, 57.6 at 75k, 56.4 at 100k).

**Update, 2026-10-05 about 05:30 UTC.** The four Hessian runs (v6 ½ s0, ¾ s0, robot-type core s0 and s1) finished and
were pulled; the first version of this file had them at about 70k. Two conclusions changed: on the 57 run's own seed,
v6 ½ and robot-type now match it except for the hexapod, and v6 ¾'s advantage rests on seed 1 alone (¾ s0 collapsed late).

**Data.** Local cache (`tbcache/`). All eight v6 runs and the 57 run are finished (100k). The four Hessian runs were
cached on Hessian with `collect_v6.sh hessian` (CPU only, low priority; it writes one `tb_scalars.npz` of about 1.5 MB
next to each run folder) and copied back. The other four v6 runs are from the cache of about 02:30-03:45 UTC. Not
included: v6 ½ and ¾ seeds 3 and 4 (H100, still running at the last check).

## Final v6 results (mean return over 16 robots at 100k; Table K)

| | s0 | s1 | s2 | mean |
|---|---|---|---|---|
| v6 ¾ | 42.7 | **55.9** | 50.8 | 49.8 |
| v6 ½ | 50.6 | 38.4 | 47.6 | 45.5 |
| v6 ½ robot-type core | 50.7 | 45.1 | | 47.9 (2 seeds) |
| 57 run | **56.4** | | | |
| v5 / v4 / MoE / URMA | 50.7 / 50.7 / 43.2 / 42.4 | 42.8 / 31.4 / 50.7 / 35.5 | 50.9 / 53.3 / 48.1 / 50.3 | 48.1 / 45.1 / 47.3 / 42.7 |

## Answer

1. **The 57 is the plateau plus the hexapod.** Call the *plateau* the state where go2, g1, talos and nao5 are all kept.
   With the ten other robots and Cassie lost it scores 54.1 on the 15 robots without the hexapod. Seven seed-0 runs of
   seven different configurations end at 54.0-54.2, exactly there: v4, v5, v6 ½, v6 robot-type, the 57 run, and the dense 4×64
   and 8×64 siblings. The 57 run's 56.4 is that plateau plus the hexapod (91, +5.7). Against v6 ½ s0 (50.6) and
   robot-type s0 (50.7) it matches within 1 point on 15 of 16 robots; the hexapod is the whole gap. (Tables A, D, K)
2. **Six robots decide the mean** (go2, g1, talos, nao5, cassie, hexapod); each ends near 0 or at a full return. The
   other ten add about 30 (a1 and go1 sit at 14-15 in every run of Table A). All six at their best would give 61.6.
   The three best runs each keep five of the six: the 57 run (loses Cassie), dec top-2 (loses the hexapod) and v6 ¾ s1
   (loses the hexapod). The other 22 finished runs keep 4 (eleven runs), 3 (six), 2 (three), 1 or 0 (two). (Table B)
3. **The plateau is common; what differs is reliability on hard seeds and the rare bonus robot.** The plateau is
   reached by 8 of 11 runs on seed 0, 2 of 7 on seed 1 and 3 of 6 on seed 2. By configuration: the 57 family 4 of 4
   (seed 0 only), v6 ¾ 2 of 3, v5 2 of 3, v6 robot-type 1 of 2, v6 ½ 1 of 3, v4 1 of 3, MoE 1 of 3, URMA 1 of 3. A
   bonus robot (Cassie or hexapod) is kept in 4 of 25 runs: 57 run (hexapod), dec top-2, v6 ¾ s1 and v4 s2 (Cassie).
   (Table B)
4. **Both bonus robots are learned and then lost far more often than they are kept.** Cassie was learned at some time
   in 23 of 25 finished runs and kept in 3. The hexapod rose above 20 at some time in 12 of 52 cached runs (peaks
   23-43 in the other 11, mostly at 3-26k) and was kept in one, the 57 run. There it came late (11 at 55k, 58 at
   60k, about 90 from 85k) with no reversal. It never falls in any run (episode length 1000); elsewhere it stands
   still. The best hexapod value in any v6 run is 9.6. So the question for the 57 run is why the hexapod stayed, not
   why it was found. (Tables B, C, J)
5. **Compared on the same seed, v6 does not beat the older configurations.** Runs with the same seed see the same
   minibatches and the same critic (the critics never see the policy; the actor loss over the first 500 updates is
   1.489-1.502 for all twelve seed-0 runs, 1.345-1.360 for seed 1, 1.739-1.752 for seed 2; Table G). Deviation from the
   seed's mean of five reference configs (average over 50k-100k, Table D): v6 ¾ −1.4 / +13.6 / +1.0, v6 ½ +2.3 / +0.3 /
   −0.8, v5 +4.6 / +3.2 / +2.0, v4 +3.0 / −1.3 / +2.2, MoE 0.0 / +8.3 / −0.1 (seeds 0 / 1 / 2). v6 ¾ has the best
   mean of the three-seed configurations (49.8) but is 1.7 above v5 and 2.5 above MoE, inside the seed spread; at 100k it is −8.0 against v5 on
   seed 0 and +13.1 on seed 1.
6. **v6 ¾ s0 collapsed late, and not through the router.** It lost nao5 (75 at 70k, 1 at 95k) and badger (49 at 60k,
   16 at 100k) while go2 rose (8 at 60k, 51 at 100k). The ½ run and the 57 run on the same seed raised go2 without
   losing either robot. In ¾ s0 no robot's core expert selection moved more than 0.1 of 4 slots after 50k; nao5 moved
   0.00 and badger 0.01. Sharing a core expert set with another robot does not explain it either: 14 of 18
   (go2, nao5, tl) robot-run pairs have an identical set with some other robot (10 kept, 3 lost, 1 never learned), the
   4 without one were all kept, and nao5 shares a set with op3 in four runs and is lost in two. Too few to tell, but no
   sign of a link. (Tables M, N, O)
7. **Three readings in §14.5 are weaker than they look.**
   - *Core capacity, "the cleanest effect".* ¾ minus ½ at 100k is −7.9, +17.5, +3.2 on seeds 0, 1, 2 (50k-100k
     average −3.7, +13.4, +1.8). Seed 1 carries it and seed 0 reverses it. (Table E)
   - *Sparse decoder.* In the 57 family at 100k the 15-robot means are 54.2 (4×64 dense), 54.1 (8×64 dense), 59.4
     (top-2) and 54.1 (top-4). The top-4 gain over dense is the hexapod; the top-2 gain is Cassie. (Table H)
   - *Robot-type core* (one expert per robot) holds up so far: minus the free ½ core it is +0.1 and +6.7 at 100k, and
     +1.5 and +6.5 on the 50k-100k average (seeds 0 and 1). Two seeds, so still a lead. (Tables E, L)

**What makes the good run, as far as the data go.** It reaches the plateau and then keeps one of the two hardest
robots. The plateau is a common outcome that most configurations reach on an easy seed; the bonus robot is the rare
part, and no configuration gets it reliably. The 57 run's edge over the other good runs is one robot (the hexapod) in
one seed, and no v6 run came near it. Nothing in the logged diagnostics singles it out, and the router explanations
tested (changing selection, shared expert sets) do not account for the late losses. The best descriptive correlate is a
sparse per-joint decoder together with a large core (Table I: +6.3 over the seed's reference mean, against +0.2 for a
sparse decoder with a small core, +3.7 for a large core with a dense decoder, −2.8 for neither). That cell has 5 runs,
two on the same seed, and v6 ¾ s0 and s2 sit near zero in it, so it is a lead, not a result.

## What differs between the 57 run and v6

| | 57 run | v6 ½ | v6 ¾ | v6 ½ robot-type core |
|---|---|---|---|---|
| Core | 4 experts, full width 512-256-128, dense | 8 experts, 400-256-128-128, top-4 | 8 experts, 400-384-192-128, top-4 | as ½ |
| Active core capacity (× URMA core) | 4.0 | 1.65 | 2.74 | 1.65 |
| Core routing | body | body | body | body |
| Core router loss | utilization 0.01 | balance −H(E), 0.005 | same | full robot-level MI, 0.005 |
| Decoder | 8 × 64, top-4 per joint | same | same | same |
| Decoder router loss | utilization 0.01 | joint-type MI 0.001, joint-equal weights | same | same |
| Utilization loss (`morse_eta`) | 0.01 | 0 | 0 | 0 |
| Router init, top-k | zero; old tie rule, so the first step is dense | random std 0.01; exact top-k | same | same |
| Parameters | 1,686,527 | 1,425,415 | 2,231,815 | 1,425,415 |

Sources: handoff §12 for the 57 run (its old `keep_topk` and `Router` also read from the first session transcript:
top-k applied in training and evaluation, renormalized, router output layers zero); the v6 rows (`morse_eta=0.0`,
`morse_router_init=0.01`) for v6.

## What the data say about each difference

| Difference | Evidence | Strength |
|---|---|---|
| Core: 4.0× dense vs 1.65× or 2.74× top-4 | ¾ vs ½ at 100k: −7.9 / +17.5 / +3.2, mean +4.3 (Table E). Large core with a dense decoder +3.7 (Table I) | Seed 1 carries the v6 effect |
| Decoder router regime | 57 run: ~2.7 effective experts per joint, entropy 0.92-0.95, least-used expert 0.09-0.10, unchanged from 5k to 100k. All v6 runs at 95k: 1.3-1.5, 0.24-0.35, 0.03-0.09 (Table F) | A real difference, probably from the MI term. Its effect is untested: dec top-2 (1.5) and v6 ¾ s1 (1.4) are sharp and still good |
| Core router: free vs robot-type | Free: H(E) 2.08, H(E\|robot) 1.47-1.72. Robot-type: H(E\|robot) 0.00-0.01, one expert per robot, MI 1.95-2.01 (Table L). Results: see robot-type bullet above | 2 seeds |
| Zero init and dense first step vs random init | None. No v6 run shares this | Untested |

## Caveats

- One run per configuration at seed 0 (the whole 57 family), 2-3 seeds for the others. Seed-to-seed spread of the
  100k mean is 4-12 points (URMA 7.4, MoE 3.8, v3 8.5, v4 12.0, v5 4.6), and each contested robot moves the mean by
  3-7 points. Late collapses of single robots (v6 ¾ s0, v4 s1, MoE s0, URMA s0) are common, so single 100k numbers
  are noisy.
- "Same seed, same critic" rests on the critics' code and on the equal early loss, not on a direct comparison of
  critic values (the cache has no critic tags).
- Table I mixes in other differences (joint-routed core in v3-v5, MI weights), so it is descriptive. v3's very narrow
  core (0.41×) pulls the "sparse decoder, core < 2×" cell down; without v3 it is +1.7 (9 runs).
- The 57 run's core-router statistics are not in the cache (it was cached with `morse/decoder/` only), so the core
  routing of the 57 run and v6 cannot be compared yet.
- v6 ½ and ¾ seeds 3 and 4 would add two seeds to the v6 means; they are not in.

## Experiments that would settle it

1. **The exact 57 configuration at seeds 1 and 2.** It would share the critics of the v4, v5 and v6 runs there. It
   shows whether the hexapod, and the plateau reliability, belong to the configuration or to seed 0. Rerunning needs
   `PREVIOUS_DEFAULTS` and `LEGACY_ARCHITECTURE` as flags, plus the old tie rule or `morse_router_init=0.01` (§12).
2. **v6 ¾ with decoder MI 0 and utilization 0.01.** It gives the 57 run's decoder router regime inside v6, and it is the
   MI-0 control the paper needs (§14.6).
3. **The 57 run's core (4 full-width dense experts) with the v6 decoder.** It separates core from decoder.
4. **Cache `morse/core/*` for the 57 run** from the raw event logs (duongnm2, job 56652) to compare core routing.
5. **Dissect the ¾ s0 collapse.** The v6 runs saved checkpoints at 50k, 75k and 100k (`morse_save_at=50000,75000`
   plus the final one), so the checkpoint analysis of §8 can compare nao5 and badger across them.
