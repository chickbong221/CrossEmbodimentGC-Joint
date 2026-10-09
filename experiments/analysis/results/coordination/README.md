# Coordination tests: result files (T3-MIcompat-s0, saved checkpoints 60k vs 90k)

Summary and interpretation: `MORSE_COORDINATION_TESTS.md` in the repo root, including the 2026-10-04 follow-up.

| File | Content |
|---|---|
| `tables.md` | All tables (16 robots): main table, bootstrap changes, Test 1, Test 3A/3B, Cassie and G1 by context |
| `coordination_tests.json` | Every number of `analysis/coordination_tests.py` (seed 0, 1000 bootstrap resamples, 10 shuffles) |
| `MIcompat_s0_60k_90k_eval.json` | Exact deterministic evaluation of the two saved checkpoints (two repeats each) |
| `T1cal_s0_2k_5k_eval.json` | Validation of that evaluation on T1-URMA-A4-cal-s0 at_2000/at_5000 (matches the training logs) |
| `pairs_cas_g1.png`, `pairs/pair_<robot>.npz` | Test 3B scatter data and figure (top pair of the top context per robot) |
| `startup_check.json` | Follow-up check 1: policies vs expert and falling actions in the first 50 steps, near-path matching, critic weights (`analysis/startup_check.py`) |
| `MIcompat_s0_60k_90k_noise.json`, `noise_tables.md` | Follow-up check 2: success under paired action noise, both checkpoints, 16 robots (`analysis/noise_report.py`) |
| `envinfo.json` | Per robot: observation names (index order), joint and foot names, nominal positions, action scale, horizon |
| `scripts/` | One-off checks: time order of the dataset (`check_order.py`), observation layout (`envinfo.py`), episode/return/command exploration (`explore.py`), action-noise share (`noise.py`), optimizer step counters of checkpoints (`ckpt_steps.py`) |

The per-robot dumps stay on Hessian: `$R/analysis/coord/MIcompat_s0_60k_90k/<robot>.npz` (whole reference episodes,
1.4 GB) and `$R/analysis/coord/MIcompat_s0_60k_90k_startup/<robot>.npz` (first 50 steps of every episode, 1.6 GB).
Rebuild the tables: `python analysis/coordination_report.py <tests json> <eval json> <tb_scalars A> <tb_scalars B> --pairs <dir>`
(run with `PYTHONIOENCODING=utf-8` on Windows).
