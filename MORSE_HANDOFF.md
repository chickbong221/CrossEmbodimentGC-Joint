# MORSE handoff (state on 2026-10-03, ~09:30 UTC; §11 added ~12:00 UTC; §12 added 2026-10-04; §13–§14 added 2026-10-05; §15 added 2026-10-06)

This file is the full context for a new Claude session. Read it before touching any server.

**Latest state: read §15 first (V7), then §14.** §14 covers the v6 runs (body-routed core plus a sparse joint-level decoder), the
results so far, what drives the good runs, and the next steps. §13 has the v6 implementation and launch details.

**Earlier sections:**
- **§11 (coordination tests, second session) corrects §8.4:** the saved "60k" checkpoint falls with Cassie and the
  saved "90k" one walks.
- **§12 (third session):** the joint-routed method (v3–v5). The config defaults changed then, so read §12 before
  rerunning any older run definition.

The previous session's own notes are also on this machine, if a new session can read them:
`C:\Users\Lenovo\.claude\projects\C--Users-Lenovo-AppData-Roaming-Claude-scratch-workspaces-01d12a43-350f-44d6-be61-f854bdd47ef5-8404b0a0-4694-46bf-a0a5-722e1c8e0322-scratch-2026-09-29-af8bb3\memory\`
Everything important from them is already in this file.

---

## 1. Project in one paragraph

**MORSE** (Morphology-aware Routing of Shared Experts) is our method for cross-embodiment offline RL. One policy
controls 16 robots: 9 quadrupeds (A1, Go1, Go2, ANYmal B/C, Barkour v0/vB, Badger, Bittle), 6 bipeds (H1, G1, Talos,
OP3, NAO v5, Cassie) and a hexapod. The goal is a strong CVPR submission with a simple, plausible motivation.

- **Repo:** `E:\Code\cross_embodiment_offline_rl`. Locally it is not a git repository; the H100 copy is a git clone.
- **Base learner:** IQL. The actor is advantage-weighted regression on dataset actions (exp-advantage clipped at 100).
- **Policy:** URMA-style:
  - per-joint encoders;
  - a whole-body **core** MLP (512, 256, 128);
  - a per-joint **decoder** (260 → 128 LN-ELU → 1) that outputs each joint's action mean;
  - a per-joint log-std computed from the joint's descriptor embedding.
- **MoE versions:**
  - The core and/or decoder become K experts (K = 4 by default).
  - A learned router outputs `softmax(morphology branch + state branch)`; routing is per joint in the decoder and per transition in the core.
  - **Dense routing** takes the weighted sum over experts; **top-K routing** uses only the K largest weights, renormalized, in both training and evaluation.
- **Baselines:**
  - **URMA:** the same network with K = 1 (`morse_num_experts=1`).
  - **EG** (Embodiment Grouping, Abe et al. 2026, arXiv 2602.18025, same 16-robot suite): a shared URMA-like policy. The robots are clustered into M = 4 groups by embodiment distance, and each update takes one optimizer step per group.
  - All our runs use `actor_steps_per_update=4`, to match EG's 4 optimizer steps.
- **Dataset:** HuggingFace `haruki-abe/cross_embodiment_offline_rl`, folder `offline_data/all_robots_suboptimal_70_forward_1m`.
  - 8 files, 87.25 GB.
  - Shapes: obs (1M, 16, 668), actions (1M, 16, 24).
  - A local-only 20k subset, `subset_suboptimal_70_forward_20k` (1.7 GB), is used for dry runs.
- **Training and evaluation:**
  - 100k updates; minibatches of 1,024 transitions per robot.
  - Evaluation every 100 updates: one episode per robot, deterministic (see §8.3).
  - **Return** = mean of the last 10 evaluations before a horizon (50k, 75k, 90k, 100k).
  - **Learned** = return ≥ a frozen per-robot threshold (half the best return seen when the thresholds were set). The thresholds are in `learned_thresholds.json`: a1 26.5, go1 26.4, go2 26.2, anyb 24.0, anyc 21.8, bkv0 23.5, bkvb 27.5, bg 26.4, bit 22.9, h1 26.8, g1 38.6, tl 53.7, op3 51.2, nao5 45.2, cas 24.3, hx 3.0.
  - The hexapod threshold is fragile: no run learned it before the top-K runs.

---

## 2. Servers and access

### H100, account duongnm2 (ssh alias `H100-duong`)

- **Repo clone:** `/home/duongnm2/projects/cross_embodiment_offline_rl`. The conda env is `morse`.
- **`/mnt/data/duongnm2/morse`:** data, runs, logs, jobs and queue (`experiments/runs` links to `$M/runs`).
- **W&B key:** only in `~/.config/morse/wandb.env` (mode 600). **Never print it.**
- **Slurm:** the `main` partition is usually full, so use `mig` (3g.40gb slices) with `--mem=128G`. Per-user limit: 2 GPUs and 32 CPUs.
- **Submit only from login node `login-restricted-2`.** `login-restricted-1` cannot reach the Slurm controller.
- **Jobs:** `sbatch /mnt/data/duongnm2/morse/jobs/tests.sbatch LIST` runs the `run_fair` worker from `jobs/queue_lib.sh` over a list of run ids. Run definitions live in `jobs/experiments.tsv`; per-run logs are in `logs/runs/<id>.log`.
- **Queue state:** `queue/{claims,done,failed}`. Monitor through these marker files and logs, not repeated squeue/sacct, because the user wants minimal Slurm traffic.

### H100, account tuannl (ssh alias `H100`, same machine)

- Everything is in `/mnt/data/tuannl/morse` (97 GB):
  - `code/`: an exact copy of the duongnm2 tree, without `.git`. Set `RLX_DATA_ROOT`.
  - `env/`: a copied env.
  - `data/`, `runs/`, `logs/`, `jobs/tests.sbatch`, `queue/`, `cache/`, `tmp/`.
- W&B is disabled. **Never write to `/home/tuannl`**: that root filesystem is shared and 90% full.
- Allowed: the `mig` partition only. Never flood Slurm (no submission loops).

### Hessian (TU Darmstadt, Determined GPU shell)

- **Machine:** Determined shell `ea8a47a5-419f-45f6-816d-b450187be1e0`, with 2× A100 80 GB, 32 CPUs and 2 TB RAM. It has been up since Sep 30. **Don't kill it.**
- **Access:** `hx.sh "CMD"` from the old scratchpad (path below). It uses ssh through the `hessian-jump` ProxyCommand plus the Determined tunnel, and the private key `morse_shell_key` next to it. **Never print or copy that key.** The `det` CLI on the jump host asks for the user's password; never enter it.
- **Workspace:** `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/letuan_r2dreamergraph_2/morse` (`$R`).
  - `repo/`: the code; the dataset is in `repo/data/offline_data/`.
  - `env/`: Python 3.11.
  - `tools/queue/`:
    - `hrun2.sh LIST GPU`: a worker that runs a list's pending runs in order;
    - `slots.sh`;
    - `experiments_hessian.tsv`: run definitions, with `.bak-*` backups;
    - the run lists;
    - `start_resume.sh` and `start_after_resume.sh`.
  - `tools/`: `download_dataset.sh`, `download_prefix.sh` and `download_finish.sh`. These are the safe ones: they never delete. **Never use `tools/download_full.sh`**, which deletes its folder on failure.
  - `hq/`:
    - `state/{claims,done,failed,archive}`;
    - `runlogs/<id>.log`;
    - worker logs.
  - `ckpt/`: the URMA s1 checkpoint at 2k used for branching, and `resume/T3-MIrobot-s0`.
  - `tables/s1_persist/`: Test 2 tables.
  - `analysis/`: checkpoint-analysis outputs.
- **The parent folder holds the user's other project** (r2dreamer-graph: `logdir`, `miniconda3`, `data`, `checkpoints`, about 27 GB). **Never touch it.**
- **The group quota is shared and invisible.**
  - Measure free space only with brief `fallocate -l <bytes>` probes, deleting each right away.
  - Others' usage swings by 10–40 GB within minutes.
  - When the quota fills, runs crash with "Disk quota exceeded". This happened twice: Oct 2 about 11:40 and Oct 3 03:45 UTC.
  - At 15:10 Oct 2 about 36 GB was free, after the full dataset was in place.
- **Running Python by hand:** `export PATH=$R/env/bin:$PATH LD_LIBRARY_PATH=$R/env/lib HOME=$R`. To pip-install, also set `PIP_CONFIG_FILE=/dev/null`.

### Local helper files (old session scratchpad, in Temp, so back them up)

`C:\Users\Lenovo\AppData\Local\Temp\claude\C--Users-Lenovo-AppData-Roaming-Claude-scratch-workspaces-01d12a43-350f-44d6-be61-f854bdd47ef5-8404b0a0-4694-46bf-a0a5-722e1c8e0322-scratch-2026-09-29-af8bb3\4fd1fedc-8187-45f5-a299-25da41c82518\scratchpad`

| File | What it does |
|---|---|
| `hx.sh` + `morse_shell_key` | Hessian shell access (the key is private) |
| `hsync_paths.sh PATH...` | Copies local repo files into the Hessian repo. Never overwrite a script a running job executes: bash reads scripts lazily. Python files are fine |
| `collect.sh hessian\|h100 ID...` | Caches TensorBoard scalars into `tbcache/<exp>/<id>/tb_scalars.npz` (h100 = duongnm2) |
| `tbcache/` | Cached curves of all runs, including tuannl's and the resumes in `*-resume/` folders |
| `learned_thresholds.json` | The frozen thresholds |
| `watch_runs.sh` | Monitor: reports done/failed/stalled runs. Env vars `HQ_RUNS` (Hessian ids) and `SEEN` (already-reported ids) |
| `eta.sh` | Run progress from the logs |
| `tuannl/tests.sbatch`, `queue/*.sh` | Copies of the job scripts |

Newer analysis scripts are in the sibling `...\94ada265-5799-4a4e-a6a3-fe7744335911\scratchpad`:

| File | What it does |
|---|---|
| `results.py TBCACHE THRESHOLDS` | All result tables; stitches resumed runs |
| `ckpt_report.py` | Report of the checkpoint analysis |
| `ckpt_joints.py` | Per-joint detail |
| `ckpt_MIcompat_s0_60k_90k.npz`, `ckpt_dump.npz` | Checkpoint-analysis outputs |

---

## 3. Standing rules from the user (follow exactly)

**Placement and secrets:**
- Don't create files in shared folders. Keep our folders inconspicuous and named `morse`, with no personal names.
- On H100-duong, code may go in `/home/duongnm2/projects`, but data and checkpoints stay in `/mnt/data/duongnm2/morse`.
- On tuannl, data and checkpoints only go in `/mnt/data/tuannl`.
- Never echo the W&B key.

**Slurm:**
- Keep `--mem` at the maximum but expect the code to use about 110 GB.
- Never submit more than about 100 jobs per hour, and use no auto-resubmit loops.
- Submit only from login-restricted-2.
- After setup, don't run helper scripts that try to submit jobs. Let the servers finish; the user decides on new jobs.

**Testing and design:**
- Dry runs happen on Hessian before anything is submitted to H100.
- Ask before design-level changes.

**Data and deletions:**
- **Never delete datasets to free space.** Clean only the subset, dry/tune/failed run folders, caches or logs.
- **Before any `rm`, list the exact paths and sizes and get an explicit OK.** In this session, the permission classifier blocked a deletion list I had chosen myself.
- Download data from HuggingFace, not by copying from H100 (H100 is fragile).
- Hessian's quota is shared: probe with fallocate and delete the probe immediately.

**Other:**
- Don't kill the Hessian GPU shell.
- When the user asks to "keep it idle", start nothing until told.
- The user prefers no time estimates unless asked, and wants results analysis promptly.

---

## 4. Code (local repo; synced to the servers where noted)

All MORSE code lives in `cross_embodiment_offline_rl/algorithms/morse/`:

| File | Contents |
|---|---|
| `moe.py` | `MoEBlock`, `ExpertDense`, `Router`, `keep_topk` (training-time top-K), capture of layer inputs and routing |
| `policy.py` | `MorsePolicy` and `get_morse_policy`. Options: decoder expert count, width and top-K; fixed routing tables (`decoder_table`); `identical_experts`; per-joint `calibration` (one scale and bias per joint; our control, not part of URMA) |
| `updater.py` | `MorseActorUpdater`: router losses (pairwise/signed conflict, MI), crossfit compatibility estimator, joint-level PCGrad, diagnostics |
| `interference.py` | `type_grad_sums`, `crossfit_type_evidence`, `pcgrad`, `neg_mutual_information` |
| `probe.py` | `DecoderProbe`: Test 1 probes on frozen parameters with 5 independent data subsets |
| `branch.py` | `branch_state`: start a run from another checkpoint, tiling a shared block into K experts. Also used to **resume** runs |
| `ckpt_analysis.py` (new) | Checkpoint comparison, see §8 |
| `config.py` | All `morse_*` options |

**Other code:**
- `algorithms/uni_iql_morse/iql/iql.py`: the IQL trainer, with hooks for probes, `morse_save_at`, branching, and checkpoint analysis instead of training.
- `tests/test_morse.py`: 25 tests. The original 24 passed on Hessian on Oct 2. After the latest changes I reran only the new `test_checkpoint_analysis` and the two related tests (top-K, branching), and they pass. Rerun the full suite with `python tests/test_morse.py` before relying on the rest.
- `experiments/analysis/`:
  - `analyze_morse.py` (robot names, classes, `load_tb`);
  - `returns_table.py` (`--thresholds`);
  - `tb_cache.py`;
  - `probe_analysis.py`, `probe_turnover.py`, `probe_persistent.py`;
  - `make_tables.py` (Test 2 tables);
  - `review_checks.py`.
- **Not yet synced to H100 (Hessian only):** `ckpt_analysis.py` and the latest `config.py`, `iql.py` and `test_morse.py`.

**Key config flags:**

| Purpose | Flags |
|---|---|
| Which blocks get experts | `morse_blocks=core,decoder` |
| Expert count | `morse_num_experts`, `morse_num_experts_decoder` |
| Decoder width and top-K | `morse_width_mult_decoder`, `morse_topk_decoder` |
| Loss mode | `morse_conflict_mode=none\|pairwise\|signed`, `morse_conflict_blocks` |
| Compatibility estimator | `morse_compat_estimator=crossfit`, `morse_compat_ema=0.99`, `morse_compat_transitions=256`, `morse_compat_z=2.0`, `morse_lambda` |
| MI loss | `morse_mi_weight=0.001`, `morse_mi_target=type\|robot`, `morse_mi_weighting=units\|robot` |
| PCGrad control | `morse_pcgrad_blocks=decoder` |
| Probes | `morse_probe_at` |
| Extra checkpoints | `morse_save_at` |
| Branch or resume | `morse_branch_from`, `morse_branch_updates` |
| Fixed routing | `morse_decoder_table` |
| Calibration control | `morse_calibration` |
| Checkpoint analysis | `morse_ckpt_analysis=A,B`, `morse_ckpt_out`, `morse_ckpt_batches`, `morse_ckpt_dump=cas,g1` (short robot codes) |

**Checkpoints:**
- "Latest" checkpoints go to `<run>/models/tmp` every 10k updates with `keep=1`.
- **The 100k checkpoint is never written**, because of how the counter is checked. So finished runs hold only the **90k** checkpoint.
- `morse_save_at` writes extra checkpoints to `models/at_N`.
- The global step is updates × 16,384.

---

## 5. All runs (all finished)

**H100 duongnm2** (`/mnt/data/duongnm2/morse/runs/morse/<exp>/<id>`):
- URMA: URMA-IQL-A4-s0, URMA-IQL-s0.
- EG: EG-IQL-M4 (folder `1790757709_oghlb`).
- MORSE-IQL-v2-A4-s0 (MORSE-v2) and MORSE-IQL-v2-noconf-A4-s0 (plain MoE).
- Test 1: T1-URMA-A4-s1, s2 (probes at 1k/2k/5k/10k/20k/50k, checkpoints saved at 2k and 5k); T1t-early-s1, s2 (11 dense probes from 1k to 3k); T1t-mid-s1; T1t-cal-s2.
- Test 3: T3-MI-s0, T3-compat-s0, s1, T3-PCGrad-s0, T3-MIjointeq-s0.
- Capacity/top-K: CAP-8x64-top2-s0, CAP-8x64-top4-s0.
- Tuning: tune-cf-* (compatibility λ and EMA).
- Pilot: MORSE-IQL-type (`1790757709_vp3bq`).

**H100 tuannl** (`/mnt/data/tuannl/morse/runs/morse/...`): MORSE-IQL-v2-noconf-A4-s1, T3-compat-s2, CAP-4x64-s0, CAP-8x64-s0.

**Hessian** (`$R/repo/experiments/runs/morse/...`):
- T1-URMA-A4-cal-s0, s1, s2 and T1t-cal-s1.
- T2b-{shared,camps,random,robot}-s1.
- T2c-{pcamps,prandom}-s1, plus resumes `T2c-*-resume/<id>-from90k`.
- T3-MIcompat-s0 (crashed at 66.3k) plus the resume `T3-MIcompat-resume/T3-MIcompat-s0-from60k`.
- T3-MIrobot-s0 (crashed at 90k) plus the resume `T3-MIrobot-resume/T3-MIrobot-s0-from80k`, which branched from a clean copy in `ckpt/resume/T3-MIrobot-s0` because a half-written orbax tmp folder sat next to the 80k checkpoint.
- MORSE-IQL-v2-noconf-A4-s2.
- tune-MI-{0.001,0.01,0.1}-s0, tune-compat-{10,100}-s0, tune-none-s0.
- Analysis runs: ckpt-MIcompat-s0-60k-90k and -dump2.

**How resumes work and how to analyze them:**
- A resume is a new run id, so the queue doesn't move the original folder to `_failed`.
- It uses `morse_branch_from=<original>/models/tmp` and `morse_branch_updates=N`. A dry run confirmed every leaf is restored exactly ("tiled 0, kept 0").
- For analysis, use the original up to N and the resumed run after it (`results.py` `stitched`).
- The RNG stream restarts. For MIcompat, the compatibility estimates restart empty.

---

## 6. Results so far

Cells are **robots learned / mean return**. Thresholds are frozen.

### 6.1 Replication: URMA vs plain MoE vs MoE + compatibility (seeds 0, 1, 2)

| | 50k: robots learned [s0, s1, s2] → mean | 50k mean return | 100k: robots learned | 100k mean return |
|---|---|---|---|---|
| URMA | [10, 10, 11] → 10.3 | 39.3 | [10, 10, 12] → 10.7 | 42.7 |
| Plain MoE | [11, 13, 13] → 12.3 | 48.5 | [10, 12, 11] → 11.0 | 47.3 |
| MoE + compatibility | [13, 12, 12] → 12.3 | 49.4 | [11, 11, 11] → 11.0 | 47.5 |

- **The MoE beats URMA at 50k in every seed** (about +2 robots, +9 mean return). At 100k the gap shrinks (+4.6) and is no longer consistent.
- **The compatibility loss adds nothing over the plain MoE.** Its seed-0 lead was seed noise.

### 6.2 Calibration control (3 seeds)

| | 50k | 100k |
|---|---|---|
| URMA | 10.3 / 39.3 | 10.7 / 42.7 |
| URMA + per-joint calibration | 11.3 / 41.6 | 11.3 / 44.7 |

Calibration recovers only about a quarter of the MoE's gain, and inconsistently (seed 2 gets worse). It's useful as a **control**: the MoE gain needs real capacity, not just per-joint output scaling. It's not useful as a method.

### 6.3 Test 3 (seed 0): routing losses and controls

| | 50k | 75k | 90k | 100k |
|---|---|---|---|---|
| URMA | 10 / 38.0 | 11 / 42.6 | 11 / 44.6 | 10 / 42.4 |
| URMA + joint-level PCGrad | 11 / 38.0 | 10 / 39.4 | 9 / 38.6 | 9 / 37.0 |
| Plain MoE | 11 / 44.3 | 10 / 45.7 | 11 / 44.1 | 10 / 43.2 |
| MI, joint type, weighted by joint count | 12 / 47.7 | 12 / 49.3 | 11 / 46.2 | 10 / 43.2 |
| MI, joint type, robots weighted equally | 12 / 46.9 | 12 / 49.9 | 11 / 48.6 | 11 / 48.4 |
| MI at robot level (stitched at 80k) | 12 / 46.4 | 11 / 46.5 | 11 / 46.4 | 11 / 48.4 |
| Compatibility | 13 / 48.9 | 11 / 47.8 | 11 / 48.0 | 11 / 48.3 |
| MI + compatibility (stitched at 60k) | 13 / 49.2 | 12 / 48.0 | 11 / 46.1 | 11 / 48.4 |
| MORSE-v2 (original method, old noisy estimator) | 12 / 45.9 | 11 / 46.4 | 10 / 45.1 | 10 / 43.5 |
| EG | 13 / 48.3 | 14 / 52.4 | 14 / 53.0 | 13 / 50.8 |

- **No routing loss stands out** beyond seed noise (the MoE seed spread at 100k is 43–51).
- **PCGrad on a shared decoder hurts:** the bipeds h1, g1 and tl collapse. Resolving conflicts doesn't replace separate capacity.

### 6.4 Test 2: fixed routing tables (seed 1, decoder-only MoE branched from URMA s1 at 2k)

| Decoder routing | 50k | 100k |
|---|---|---|
| Shared (one expert) | 10 / 36.1 | 10 / 35.2 |
| Compatibility groups, one probe | 11 / 40.1 | 11 / 39.2 |
| Random groups, one probe | 11 / 41.1 | 11 / 40.1 |
| Split by robot | 10 / 38.4 | 11 / 38.7 |
| Compatibility groups, averaged probes (stitched at 90k) | 11 / 40.0 | 11 / 40.0 |
| Random groups, averaged probes (stitched at 90k) | 11 / 38.3 | 11 / 39.8 |

Any split beats shared (+1 robot, +4–5 mean return). **Which joints are grouped doesn't matter**: compatibility groups do the same as matched random groups.

### 6.5 Capacity / top-K (seed 0, plain MoE with only the decoder changed)

| Decoder | 50k | 75k | 100k |
|---|---|---|---|
| 4 × 128 dense (the plain MoE seed 0) | 11 / 44.3 | 10 / 45.7 | 10 / 43.2 |
| 4 × 64 dense | 12 / 51.1 | 12 / 50.7 | 12 / 50.8 |
| 8 × 64 dense | 13 / 52.0 | 11 / 49.5 | 12 / 50.7 |
| 8 × 64 top-2 | 13 / 50.6 | 14 / 54.0 | 13 / 55.7 |
| 8 × 64 top-4 | 13 / 51.6 | 14 / 57.6 | 13 / 56.4 |

- **Decoder size and expert count don't matter:** the dense variants stay within the MoE seed range.
- **Top-K gives the best mean returns of any run.** Top-2 is the only run that learns Cassie (81) and top-4 the only one that learns the hexapod (91).
- **Caveats:** one seed, each gain is a single all-or-nothing robot, and the advantage appears only after 50k.
- These runs have **no** compatibility or MI loss.

### 6.6 Which robots drive the differences

- Eight robots are learned by every run.
- **Bipeds h1 and g1:** every MoE variant learns them in all 3 seeds; URMA learns them only in seed 2. This is the robust joint-level win.
- **Go2:** URMA learns it in all 3 seeds; the plain MoE loses it in 2 of 3.
- **a1 and go1:** learned only by EG and URMA seed 1, never by a joint-level MoE. That's why EG is the best single run: robot-level grouping keeps the quadrupeds and the bipeds at the same time.
- **Late collapses** of single robots (nao5, tl, g1) make the 100k numbers noisy. Many runs peak at 50–75k.

### 6.7 Test 1 (mechanism)

- **Most of a single probe snapshot is transient:** about 70% of its reliable variance turns over within 10 updates.
- **The persistent part is reliable and shared across seeds:** averaged over the 11 dense probes, its reliability is 0.61. Comparing seeds 1 and 2: 0.60 raw and 0.74 normalized correlation.
- **It separates harmful from helpful joint pairs well.** A 2-way "camps" split keeps only 16–18% of harm inside camps but 75–79% of help. EG groups do much worse (about 55/60), as do morphology groups (61/70) and random splits (50/50). Joint descriptors can't predict it.
- **Calibration removes about 27% of early harmful interaction.** The remaining structure is still shared across seeds (0.52 raw / 0.82 normalized) and still separates harm from help.
- **Prediction of transfer:** strong at the same checkpoint (0.73–0.83 early, 0.44–0.60 later), weak from earlier probes (0.05–0.3), and about 0 from 3k or more updates earlier. The 0.98–0.99 P/T agreement is a check of the first-order approximation, not a validation of compatibility.
- **The online crossfit estimator** is valid (reliability 0.3–0.7) but doesn't act selectively: after 20k updates it covers only 1–2% of (expert, pair) combinations.
- **Probe interaction statistic:** routed models stay near N/S ≈ 7, while URMA grows to 17–21.

**Overall:** the compatibility structure is real but not actionable for routing (Tests 2 and 3). What the evidence supports is **better learning through modular, joint-level decoder capacity**. Selective, sparse activation is the most promising lead.

---

## 7. External review: current position (latest reviewer message)

- **Main finding:** Cassie's return deteriorated despite a slightly improved fit to its offline data and small average changes in routing and predicted actions. This weakens "Cassie's knowledge was broadly overwritten". It doesn't establish closed-loop fragility, or rule out damaging changes at a few important states.
- **Reviewer's conclusion table:**

| Explanation | Evidence |
|---|---|
| Worse average fit to Cassie's offline data | Not supported |
| Large average routing or expert drift on dataset states | Not supported |
| Localized degradation at important states or joints | Still possible |
| Small policy changes causing large rollout consequences | Plausible, unconfirmed |
| Other robots caused Cassie's decline | Not established |

- **Do not add GPM** (gradient projection memory) on the strength of this result.
- **The paper's justification becomes** "better learning with modular computation". The late performance loss remains unexplained.
- **No new training was requested.** The three clarifications the reviewer asked for were all collected (§8).

---

## 8. Checkpoint analysis (review item 7), with answers to the follow-up questions

**Setup:**
- **Tool:** `morse/ckpt_analysis.py`, run through iql.py with `morse_ckpt_analysis=A,B`.
- **Checkpoints:** the only competent → degraded pair available, T3-MIcompat-s0 at 60k (original run) → 90k (resumed run). Other runs keep only 90k.
- **Robots:** Cassie declines (return 41 → 1); G1 is stable (77 → 78).
- **Data:** 8,192 fixed transitions per robot (8 minibatches with a fixed seed).
- **Validity:** my recomputation of the decoder matches the policy to 7×10⁻⁷.
- **The resume isn't the cause:** Cassie's decline already started in the original run (60–66k).

### 8.1 Per robot, 60k → 90k, on fixed data

- **Every robot's fit improved slightly**, Cassie included. Overall NLL went 1.183 → 1.182; Cassie's NLL 1.062 → 1.060, and its AWR loss with advantage weights frozen at 60k 1.001 → 0.998.
- **Small changes for every robot:**
  - action-mean change 0.024–0.046 in units of the policy std (Cassie 0.036, G1 0.024);
  - decoder routing change (total variation) 0.017–0.028;
  - top decoder expert switching on 0.7–3% of states;
  - core routing change 0.003–0.007.
- **Cassie's routing:** its joints sit about 75% on decoder expert 1, which it shares with H1 and Talos. This didn't change.

### 8.2 Follow-up 1: per joint, Cassie vs G1

**Action-mean change, all joints pooled (median / p95 / max):**

| | Cassie | G1 |
|---|---|---|
| Policy-std units | 0.025 / 0.109 / 0.94 | 0.017 / 0.070 / 0.48 |
| Degrees of joint target | 0.6° / 2.5° / 25.8° | 0.4° / 1.6° / 10.4° |

- Physical mapping: joint target = nominal + action × 0.6 rad (Cassie) or 0.5 rad (G1).
- **Cassie's largest changes:** left foot max 25.8°, right foot 15.1°, right knee 14.0° (p95 3.2°), right hip pitch 13.4°, left knee 11.7°. **G1's:** right knee 10.4°.
- Normalized by the dataset action std: Cassie per-joint medians 0.018–0.028, p95 0.07–0.11, max 0.19–0.86.
- **Mean-squared error of the action mean:** Cassie 0.4980 → 0.4962 (lower on all 10 joints); G1 0.6619 → 0.6610.
- **Predicted std:** Cassie 0.690 → 0.687; G1 0.804 → 0.803.
- **NLL change split:** Cassie −0.0019 = +0.0019 (error/std² term) − 0.0039 (log-std term). Most of the likelihood gain is a narrower std; the mean accuracy improved only 0.4%.
- **Conclusion:** typical changes are small, but a small fraction of Cassie's states change a lot (up to 26°, about 2.5× G1's largest change), concentrated in the feet, knees and hip pitch. Whether those states matter for the falls is unknown.

### 8.3 Follow-up 2: how the decomposition was calculated

- **Model:** m(F, R, E) = Σₖ pₖ(R; morphⱼ, state_F) · hₖ(E; x_F), the decoder's output before its clip.
  - **F:** upstream features (encoders, the core MoE, the descriptor embedding).
  - **R:** decoder router parameters.
  - **E:** decoder expert parameters.
- Each of F, R and E is taken from A or B, and all 8 combinations are evaluated.
- **Exact Shapley value:** φᵢ = Σ over S ⊆ the others of |S|!(2−|S|)!/3! · [v(S ∪ {i}) − v(S)]. That is the **average over all 6 replacement orders**, and Σφᵢ = Δm exactly.
- **Share:** Σφᵢ·Δm / Σ Δm², pooled over transitions and joints. Shares sum to 1 and can be negative.
- **Results:** Cassie 71% features / 8% router / 21% experts; G1 86% / −4% / 18%.
- **The components cancel heavily:** Σ|φ| = 2.44× |Δm| for Cassie and 2.73× for G1. The router part opposes the total in about 47% of transition-joint cases.
- So the shares describe the geometry of a small action change on dataset states. **They are not causes of the return loss.**
- Code: `shapley3` and `per_state` in `ckpt_analysis.py`; test: `test_checkpoint_analysis`.

### 8.4 Follow-up 3: the evaluation protocol and episode outcomes

- **Each evaluation episode is deterministic for a given policy:**
  - mean actions (`eval_stochastic=False`), with no action processing beyond the policy's own clip;
  - the evaluation environment is seeded once at creation and never reseeded;
  - in test mode the start pose is fixed, with no domain randomization, perturbations, observation noise, dropout or action delay;
  - flat terrain and a fixed forward command;
  - the eval flag never toggles, so robot parameters are never resampled.
- So **differences between evaluations come only from the 100 training updates between them.**
- **Cassie's outcomes are binary:** a full 1,000-step walk (return ≈ 80) or a fall at about 54 steps (return ≈ 1).

| Window | Cassie episodes that walk |
|---|---|
| 55–60k | 23 of 50 (46%) |
| 60–66k, original run | 24% |
| 60–65k, resumed run | 28% |
| 85–95k | 2% (none after 86k) |

- G1 walks the full episode every time (75–79).
- The "41 at 60k" is 5 walks and 5 falls among the last 10 evaluations.
- **Interpretation:** an already unreliable, knife-edge policy whose chance of walking declined into a persistent loss.

**Possible next steps (no training needed for the first):**
1. Run both checkpoints on Cassie in simulation (for example 20 episodes each), then compare their actions on the states the 60k policy actually visits. This needs new rollout code.
2. Rerun one declining run with `morse_save_at=50000,75000,100000` to get the reviewer's preferred checkpoints. About 8 hours on Hessian.

---

## 9. Open decisions for the user

1. **Top-K:** seeds 1 and 2 for 8×64 top-2, top-4 and dense, to test the strongest lead.
2. **Combination:** robot-level grouping combined with joint-level top-K experts, aiming at a1/go1 plus the bipeds.
3. **Reporting:** at the peak (50–75k) or with a fixed early stop, given the late single-robot collapses.
4. **Rollout-state analysis for Cassie,** if the reviewer wants more (§8, next step 1).
5. **Paper narrative:** modular, joint-level decoder capacity helps learning (robust on the bipeds) and isn't explained by calibration or gradient surgery. Compatibility-based routing isn't supported. The late loss is unexplained, and GPM shouldn't be added.

---

## 10. Pitfalls already hit (avoid repeating)

- **pgrep/pkill over ssh match the remote `bash -c` command that contains the pattern,** so they kill or abort themselves. Anchor patterns to the real command line: `"^bash /pfss[^ ]*hrun2[.]sh"`, `"^python experiment"`, `"^curl "`.
- **A remote `nohup setsid` launch survives an interrupted or rejected tool call.** Check for running processes before relaunching; two downloads once wrote to the same file.
- **The queue's `launch` moves an existing run folder to `_failed/`** and restarts from zero. Resume under a new id.
- **After a quota crash,** a half-written `checkpoint_*.orbax-checkpoint-tmp-*` folder can sit next to the good checkpoint. Copy the good one to a clean folder before branching from it.
- **Worker markers may fail to write while the quota is full,** e.g. a missing failed marker. Check the run log.
- **Downloading on Hessian:**
  - A single curl stream manages only 12–20 MB/s; 8 parallel range requests reach about 110 MB/s; HuggingFace's own xet tool reaches about 290 MB/s.
  - GPFS has no keep-size preallocation.
  - Use `tools/download_dataset.sh`, which probes the quota, checks sha256 and never deletes.
- **Caching TensorBoard curves on Hessian is slow on GPFS:** about 15 minutes for 10 runs. Run it detached and fetch afterwards.
- **H100 login nodes kill long CPU-heavy processes,** such as pip builds. The tuannl env was copied instead of rebuilt.
- **tuannl's code copy has no `.git`,** so set `RLX_DATA_ROOT`.
- **Bash background tasks are limited to 2 hours** and Monitors to 30 minutes. Re-arm them, and use marker-file waits.

---

## 11. Coordination tests (second session, 2026-10-03 ~10:00–12:00 UTC)

The reviewer asked for three checkpoint tests of a proposed coordination (covariance) loss. All results are in
**`MORSE_COORDINATION_TESTS.md`** (repo root). Tables, JSON, figure and eval files are in
`experiments/analysis/results/coordination/`.

**Findings:**
- **Checkpoint labels are off by one.** A "latest" checkpoint labelled N holds the policy after N+1 updates (saved
  before the counter increments; the stored step counters confirm it). The evaluation logged at N is of the policy one
  update earlier, so the logs never evaluated the saved policies. `models/at_N` (`morse_save_at`) checkpoints are exact.
- **The exact-step outcomes contradict §8.4.** A deterministic re-evaluation, validated against the logs on
  T1-URMA-A4-cal-s0 `at_2000`/`at_5000` with all 16 robots matching, shows:
  - saved "60k" (60,001 updates): Cassie **falls** (1.0, 55 steps);
  - saved "90k" (90,002 updates): Cassie **walks** (81.0, 1000 steps).

  The window averages (50% → 0% walks) still describe the decline, but this pair cannot link any action change to
  falls.
- **Test 1:** reference coordination is real and reproducible for every robot. Cassie gains +3.28 nats per transition
  held out; stability ≥ 0.985.
- **Tests 2–3:** no coordination-specific deterioration. Cassie's full and diagonal exceedance both move by about
  +0.1 pp; stable robots move as much or more. The drift follows the reference correlations (ratio 0.77 vs null 0.96).
  By the reviewer's criterion, don't add a covariance/coordination loss.
- **Dataset structure:** the dataset is time-ordered per robot (episodes = runs between done flags).
  - **Bipeds:** samples 0–700k are a suboptimal policy that falls within tens of steps; the last 300k are expert
    full-length episodes (Cassie 298, returns 73–81.5).
  - **Commands:** always [1, 0, 0].
  - **Contacts:** binary, except Badger's, which is constant.

**Code added** (local, synced to Hessian only, **not to H100**):
- **`morse/coordination.py`:** reference-episode dump.
- **New flags:** `morse_ckpt_eval` (exact evaluation of both checkpoints, twice, written to
  `<morse_ckpt_out>_eval.json`), `morse_coord_out`, `morse_coord_min_return`. Also `morse_ckpt_batches=0` now skips
  the older statistics.
- **`evaluate_callback(..., force=True)`** in `iql.py`.
- **`experiments/analysis/coordination_tests.py`** (Tests 1–3) and **`coordination_report.py`** (tables and figure).
- **`tests/test_morse.py::test_coordination_episodes_and_dump`:** passes. The full suite was started on Hessian at
  ~11:50 UTC, with its log in `$R/analysis/coord/test_suite.log`. Check that the log ends with
  "all MORSE tests passed".

**Hessian state:**
- **Queue rows** `ckpt-T1cal-s0-2k-5k-eval` and `coord-MIcompat-s0-60k-90k` (exp `analysis`, lists
  `queue_h_coord_val.txt` and `queue_h_coord.txt`) are done; their workers have exited.
- **Outputs in `$R/analysis/coord/`:**
  - `MIcompat_s0_60k_90k/`: 16 robot dumps, 1.4 GB;
  - `MIcompat_s0_60k_90k_tests/`;
  - the two `*_eval.json`;
  - `envinfo.json`: every robot's observation names, joints, feet, nominal positions and action scale.
- **Backup:** the TSV backup from before these rows is `experiments_hessian.tsv.bak-10031128`.

**Follow-up runs (2026-10-04), see the follow-up section of `MORSE_COORDINATION_TESTS.md`:**
- `startup-MIcompat-s0-60k-90k`: `morse_coord_startup=50` dumps the first 50 steps of every complete episode with
  both checkpoints' means and advantage weights; output `$R/analysis/coord/MIcompat_s0_60k_90k_startup/`, 1.6 GB.
  Analysis: `experiments/analysis/startup_check.py`.
- `noise-MIcompat-s0-60k-90k`: `morse_ckpt_eval_noise="0@0,0.03@50,0.1@50,0.3@50,0.1@1000"` with
  `morse_ckpt_eval_episodes=40` (paired action noise). Output `$R/analysis/coord/MIcompat_s0_60k_90k_noise.json`.
  Summary: `experiments/analysis/noise_report.py`.
- **Findings:** the margin shrank for NAO (72% → 22% success at σ 0.03, although it walks deterministically at both
  checkpoints) and Talos, and grew for H1. The saved Cassie 60k falls under every condition; 90k walks 28% of the
  time at σ 0.03. The falling data is not next to the expert path, and the critic separates expert from falling
  actions only right after reset.
- **Tests:** the full suite passed on Hessian on 2026-10-03 (`$R/analysis/coord/test_suite.log`). The extended
  coordination test, with start-up mode and weights, passes.

**Pitfalls this session:**
- **`hx.sh` times out on detached launches.** It hangs until `timeout` when the remote command starts a detached
  process (`setsid nohup ... &`). The process survives; check it with `ps aux | grep`.
- **Console encoding.** Windows Python fails to print "Δ" to the console; run report scripts with
  `PYTHONIOENCODING=utf-8`.
- **Foot names.** `<foot>_cycles_since_last_ground_contact` also ends in `_ground_contact`; the dump excludes it when
  detecting feet.

---

## 12. Revised method (third session, 2026-10-04): implementation, tests, runs

**Method (the user's plan):**
- Core and decoder both have 8 experts with exact top-4 routing.
- **Joint-conditioned core routing.** The 8 core experts compute once per transition, h_e = F_e(x_body). Every joint
  has its own core router (inputs: its descriptor plus the robot's pooled descriptor, its joint state, and x_body), and
  z_j = Σ_{e∈top4} p̃_je h_e feeds that joint's decoder instead of one shared core output.
- **Expert width:** every hidden layer of the experts is 64, so the core is 64 → 64 → 128 and the decoder 64 → 1.
- **MI:** joint-type MI on both blocks with weight 0.005 and every robot weighted equally. It is router-only and is
  computed at every actor step from all transitions of that step's minibatch.
- No utilization, compatibility or PCGrad terms.

**Code changes** (local, synced to Hessian and, on 2026-10-04 ~02:35 UTC, to both H100 copies):
- **`moe.py`:**
  - `keep_topk` is now exact (index-based `lax.top_k`; ties go to the lower expert index).
  - `route(logits, topk, eval_topk)` returns (soft, mixing) weights and is used by training, evaluation and
    `ckpt_analysis.router_weights`.
  - `Router(out_init_std)` gives random output layers (std 0.01) so top-4 starts from varied expert sets.
  - `MoEBlock(joint_routing=True)` mixes whole-body experts per joint.
  - Routed blocks record their router inputs and weights for all transitions in the `morse_routing` collection.
    They never do so at init, so init still returns only `params`.
- **`policy.py`:** new options `topk_core`, `core_routing` (body | joint), `expert_hidden`, `router_init`.
- **`updater.py`:**
  - `_mi_loss` works on the routing records and runs at every actor step, including the further sub-steps.
  - Groups and prior are in `_mi_groups[block]` = (robot of each group, prior).
  - A joint-routed core takes only the MI objective; utilization or conflict on it raises an error.
  - New diagnostics:
    - `morse/<block>/mi`, `h_expert`, `h_expert_given_type`;
    - `router_grad_task`, `router_grad_mi`, `router_grad_mi_ratio`;
    - `select_e<k>` and `weight_e<k>` (robot-averaged).
  - Per-robot selection history goes to `morse_diagnostics/routing.npz` (arrays `<block>_select` and `<block>_weight`,
    each (diag steps, robots, 8)).
- **`interference.py`:** `group_neg_mutual_information(r, prior)` (shared by both MI functions).
- **`config.py`:**
  - The defaults are now the revised method.
  - `LEGACY_ARCHITECTURE` holds the values of the new keys that reproduce older runs. `IQL.load` applies them to
    checkpoints whose config lacks those keys.
  - `PREVIOUS_DEFAULTS` holds the old values of the defaults that changed.
  - **Rerunning an old TSV row needs both dicts as explicit flags**, without duplicating flags already in the row.
  - The same applies to checkpoint analyses of old runs.
- **`iql.py`:**
  - **"Latest" checkpoints are now saved after the counter advances.** `tmp/checkpoint_<step>` labelled N holds the
    policy after exactly N updates, and the 100k checkpoint is written.
  - Every saved checkpoint, latest or `at_N`, gets a line in `models/checkpoints.jsonl` with: checkpoint id, update
    count, seed, settings, and the deterministic evaluation of exactly those weights. The periodic evaluation of the
    same update is reused; otherwise a forced evaluation runs.
  - The same returns are logged as `ckpt/episode_return/<robot>`, separate from the `eval/` curves.
- **`experiments/experiment_morse_ablations.sh`:**
  - `urma` and `dense_matched` pin the URMA architecture. Under the new defaults, `morse_num_experts=1` alone would
    give 8 decoder experts and a 64-wide core.
  - New ablations: `dense_routing`, `body_core`, `no_mi`.
- **`tests/test_morse.py`:**
  - New test 21, `test_joint_core_topk_and_mi`.
  - `test_mi_robot_and_joint_groups` was adapted to the new MI groups.

**Verification:**
- **Full suite:** all 26 tests pass on Hessian (`$R/analysis/tests_revised.log`).
- **Dry run `dry-v3-s0`:** 200 updates on the 20k subset, 20 min, done.
  - All 292 logged tags are finite.
  - Checkpoints `tmp/checkpoint_1638400` (100 updates), `at_150` (forced evaluation) and `tmp/checkpoint_3276800`
    (200) are recorded in `checkpoints.jsonl` with exact update counts.
  - `routing.npz` is written.

**Dry-run finding: at weight 0.005 the MI objective takes over the routers almost at once.**
- **Its gradient dominates the routers.** On router parameters the MI gradient is 5–24× the task gradient from update
  50 on. The task gradient vanishes as the routing saturates.
- **Routing concentrates quickly.**
  - **H(E|type):** 2.08 → 0.38 (core) and 0.16 (decoder) within 150 updates.
  - **MI:** 0.95 (core) and 0.81 (decoder).
  - **Marginal H(E):** only 1.33 in the core.
- **The routing becomes hard and robot-level.** Every joint of a robot selects the same 4 experts, and most robots
  put 0.95–1.0 of their weight on one of them. The robots split into three groups:
  - a1, go1, go2, bkv0, bkvb, bg, bit, op3, nao5 and hx;
  - anyb, anyc, h1, tl and cas;
  - g1.
- **The earlier run was much milder.** In T3-MIjointeq (4 decoder experts, weight 0.001 once per update, 8 captured
  transitions per robot), H(E|type) went 1.39 → 0.99 at 500 updates and the marginal stayed near uniform.
- **Consequence.** Under MI 0.005 the dense and body-core ablations would also collapse to per-robot routing, which
  makes them uninformative. The informative tests are without MI and with a weaker MI.

**Runs (Hessian), started 2026-10-03 ~20:40 UTC by `tools/queue/start_v3.sh`:**
- **Workers:** four, two per GPU, each with its own list.
  - The lists are `queue_h_v3_a.txt` (s0, then s2), `_b` (noMI-s0), `_c` (mi0005-s0) and `_d` (s1).
  - Worker logs: `$R/hq/worker_queue_h_v3_<x>_g<gpu>.out`.
- **Backups:** the rows were appended after backups `experiments_hessian.tsv.bak-10032008` and `.bak-10032036`.

| Id | What |
|---|---|
| `dry-v3-s0` | 200 updates on the 20k subset (done) |
| `MORSE-v3-s0`, `-s1`, `-s2` | the revised method, seeds 0–2 |
| `MORSE-v3-noMI-s0` | same, `morse_mi_weight=0` |
| `MORSE-v3-mi0005-s0` | same, `morse_mi_weight=0.0005` (10× weaker) |
| `MORSE-v3-dense-s0`, `MORSE-v3-bodycore-s0` | defined but not queued (uninformative under MI 0.005) |

- **Flags:** every row spells out all the method flags, so it doesn't depend on the defaults.
- **Disk:**
  - **Exact checkpoints:** `morse_save_at=50000,75000`, plus the final 100k "latest" checkpoint.
  - **Snapshots:** off (`morse_snapshot_frequency=0`), because the group quota had only 1–2 GB free at launch
    (a 2 GB fallocate probe failed, 1 GB succeeded).
  - **Candidates to free, if the user agrees:** `$R/analysis/coord/MIcompat_s0_60k_90k` (1.4 GB) and
    `..._startup` (1.6 GB).
- **Outputs and logs:** outputs go to `$R/repo/experiments/runs/morse/<exp>/<id>` and logs to
  `$R/hq/runlogs/<id>.log`.

**Status 2026-10-04 ~02:30 UTC:**
- **Two Hessian runs crashed.** The shared Hessian quota filled again around 01:15 UTC, from other users' usage.
  - **`MORSE-v3-s0`** crashed at about 30k updates, in the `routing.npz` write. It has no failed marker, because the
    marker could not be written.
  - **`MORSE-v3-noMI-s0`** crashed at about 30k updates, in an orbax checkpoint write. It is marked failed.
  - **`MORSE-v3-s2`** never started. Its worker created the claim folder but could not write the owner file.
  - The partial run folders stay on Hessian.
- **Still running on Hessian:** `MORSE-v3-mi0005-s0` and `MORSE-v3-s1`, both past 50k.
- **H100 was reachable again (VPN on).** The 10 changed files were synced to both H100 copies, and the checksums match
  local. A short policy check passed in both envs (flax 0.8.1, jax 0.4.25).
- **The three affected runs were submitted to H100 as fresh runs (no resume):**
  - **duongnm2:** jobs 57125 and 57126, list `jobs/queue_v3_h100.txt` (`MORSE-v3-s0`, `MORSE-v3-s2`). The rows are
    in `jobs/experiments.tsv`.
  - **tuannl:** job 57127, list `jobs/queue_v3_tuannl.txt` (`MORSE-v3-noMI-s0`).
  - The rows are copied verbatim from the Hessian file. Both accounts have the backup
    `jobs/experiments.tsv.bak-10040232`.
- **H100 login node:** the ssh aliases land on `login-restricted-1` or `-2` at random. Run `sbatch` only after checking
  `hostname` = `login-restricted-2`, and retry the connection otherwise.

**Results (2026-10-04 ~09:30 UTC): all five v3 runs finished 100k updates.**
- Curves are cached in the old scratchpad `tbcache/MORSE-v3*/`. The tables come from `v3_results.py` (this session's
  scratchpad, `030f7cc8-...\scratchpad`), with the metric of `results.py`: the last 10 evaluations up to each horizon,
  robots learned by the frozen thresholds.

| Run | 25k | 50k | 75k | 100k |
|---|---|---|---|---|
| URMA, 3 seeds (mean) | — | 10.3 / 39.3 | 11.3 / 42.0 | 10.7 / 42.7 |
| Plain MoE, 3 seeds (mean) | — | 12.3 / 48.5 | 11.0 / 47.3 | 11.0 / 47.3 |
| EG s0 | 10 / 41.5 | 13 / 48.3 | 14 / 52.4 | 13 / 50.8 |
| Old MoE, decoder 8×64 top-4, s0 | 10 / 39.7 | 13 / 51.6 | 14 / 57.6 | 13 / 56.4 |
| **MORSE-v3 (MI 0.005), 3 seeds (mean)** | 6.7 / 28.1 | 9.7 / 39.0 | 10.3 / 40.8 | 9.7 / 39.4 |
| v3 s0 / s1 / s2 at 100k | | | | 41.3 / 30.1 / 46.7 |
| v3 MI 0.0005, s0 | 7 / 31.6 | 11 / 44.7 | 11 / 47.6 | 12 / 48.8 |
| v3 no MI, s0 | 6 / 29.2 | 11 / 42.6 | 13 / 47.9 | 11 / 48.0 |

- **The revised method at MI 0.005 is at the URMA level** and below the plain MoE, EG and the old decoder top-4 run.
  Seed 1 degrades late (H1, G1 and NAO lost after 50k).
- **MI 0.005 hurts.** On seed 0 it costs about 7 points at 100k against no MI or MI 0.0005. Its routing is nearly one
  expert per joint type in the core: H(E|type) is 0.06 at the end, while the marginal H(E) stays at 2.06.
  - At MI 0.0005, H(E|type) is 1.03 (core) and 0.89 (decoder).
  - The MI gradient on the routers ends at 0.2–0.3× the task gradient: it saturates, unlike the dry run's 5–24×.
- **Every v3 variant learns slower early** (25k: 6–8 robots), and the bipeds H1 and G1 are mostly not learned at 50k.
  - This points to the v3 architecture itself: a joint-routed core of narrow 64-wide experts, about 0.5M parameters in
    total.
  - The old MoE kept 512-256-128 core experts, about 1.7M parameters.
  - The current runs cannot separate joint-level core routing from the narrow core.
- **Not queued:** `MORSE-v3-dense-s0` and `MORSE-v3-bodycore-s0` would inherit MI 0.005 and the narrow core, so update
  them before any use. Hessian and our H100 jobs are idle. The user decides the next runs.

**MORSE-v4 (2026-10-04 ~10:50 UTC), the user's revised configuration:**
- **Changes from v3:**
  - Experts are half width, following Mod-Squad's width ÷ k rule: core 400 → 256 → 128 → 128 and decoder 260 → 64 → 1.
    The flags are `morse_expert_hidden=0`, `morse_width_mult_core=0.5`, `morse_width_mult_decoder=0.5`.
  - Still 8 experts with top-4 in both blocks.
  - MI with joint-equal weights in both terms (`morse_mi_weighting=units`) at 0.0005.
  - Everything else as in the v3 rows.
- **Policy parameters:** 1,427,079 (core 1,251,280, decoder 148,504), confirmed in the run logs. URMA is 431,648, v3
  512,647 and the old dense MoE 1,686,527.
- **Dry run `dry-v4-s0` (Hessian, 20k subset, 200 updates):** passed. All 292 tags are finite; routing.npz and the
  checkpoints (100, 150, 200) are recorded.
- **Running on H100:**
  - duongnm2: jobs 57192 and 57193, list `jobs/queue_v4_h100.txt` (`MORSE-v4-s0`, `MORSE-v4-s1`).
  - tuannl: job 57194, list `jobs/queue_v4_tuannl.txt` (`MORSE-v4-s2`).
  - The rows were copied verbatim from Hessian's `experiments_hessian.tsv`, which also holds unused Hessian lists
    `queue_h_v4_a/b.txt`.
  - Backups are `experiments.tsv.bak-10041020` on duongnm2 and the tuannl equivalent.
- **Hessian is idle.** A quota probe at ~10:25 UTC found 5–10 GB free (2 and 5 GB succeeded, 10 GB failed).

**MORSE-v5 (2026-10-04 ~10:50 UTC):** same as v4, but top-2 of 8 in both blocks and 3/4-width experts.
- **Experts:** core 400 → 384 → 192 → 128 and decoder 260 → 96 → 1 (`morse_width_mult_core/decoder=0.75`).
- **Policy parameters:** 2,301,063 (core 2,057,680, decoder 216,088), confirmed in the logs.
- **Dry run `dry-v5-s0`:** passed (292 finite tags, routing.npz, checkpoints at 100, 150 and 200).
- **Runs:**
  - Hessian: `MORSE-v5-s1` on GPU 1 from 10:41 UTC (list `queue_h_v5_b.txt`), and `MORSE-v5-s0` on GPU 0 from 10:51 UTC
    (`queue_h_v5_a.txt`).
  - H100 tuannl: `MORSE-v5-s2`, job 57197, list `jobs/queue_v5_tuannl.txt`.
  - Backups: Hessian TSV `.bak-1004xxxx` from before the v4 and v5 rows; tuannl `experiments.tsv.bak-*`.
- **Paper (`paper/morse_cvpr_revised.tex`):**
  - Abstract, introduction, related work and preliminaries were rewritten in the style of Abe et al.
  - The gradient decomposition is in the Preliminaries.
  - Section 4.4 now uses joint-equal MI weights, matching v4 and v5.

**v4 and v5 results (2026-10-04, all six runs finished):** curves cached in `tbcache/MORSE-v4|v5/`; script
`v45_results.py` (and `v45_nohx.py`) in this session's scratchpad.

| Run (mean return, last 10 evaluations) | 50k | 75k | 100k |
|---|---|---|---|
| v4, 3 seeds (s0 / s1 / s2 at 100k: 50.7 / 31.4 / 53.3) | 49.2 | 45.5 | 45.1 |
| v5, 3 seeds (s0 / s1 / s2 at 100k: 50.7 / 42.8 / 50.9) | 48.1 | 48.1 | 48.1 |
| URMA / old dense MoE / v3, 3 seeds | 39.3 / 48.5 / 39.0 | 42.0 / 47.3 / 40.8 | 42.7 / 47.3 / 39.4 |
| EG s0 / old decoder top-4 s0 | 48.3 / 51.6 | 52.4 / 57.6 | 50.8 / 56.4 |

- **Neither reaches ~57.**
  - The old decoder top-4 s0 run is the only run that ever learned the hexapod, which returns about 90 or about 0.
  - Without the hexapod (15 robots, 100k), dec top-4 s0 is 54.1, and v5 s0 and s2 are 54.1 and 54.2, with identical
    per-robot profiles.
  - 3-seed means without the hexapod at 100k: URMA 45.6, old MoE 50.5, v4 48.1, v5 51.3.
- **Seed 1 is hard for every method** (URMA 35.5, v3 30.1, v4 31.4, v5 42.8): it mostly loses NAO and Go2. v4 s1 also
  collapses late (G1, Talos).
- **v4 vs v5 behaviour:** v5 plateaus by about 50k and stays there in every seed. v4 peaks at about 45k.
- **Never learned by v4 or v5:** A1, Go1 (only EG), the hexapod (only dec top-4 s0), and Cassie except v4 s2 (81 at
  100k).
- **Routing at the end:**
  - v5: H(E|joint) core 0.47, decoder 0.53–0.68.
  - v4: H(E|joint) core 0.9, decoder 0.7–1.2.
  - Marginal H(E) is about 2.0 in both (all 8 experts used).
- **Open decision (user):** the fair reference would be seeds 1 and 2 of the old decoder top-4 configuration.

**Core routing structure (2026-10-05, seed 2, end of training; `core_overlap.py` on `routing.npz` fetched from tuannl):**
- Overlap = expected share of selected experts that two routing units have in common (1 = always the same experts).
  - v5 core: within a robot 0.59, between robots 0.23, chance 0.25. v5 decoder: 0.55, 0.42 and 0.25.
  - v4 core: 0.73, 0.52 and 0.50. v4 decoder: 0.68, 0.60 and 0.50.
- **The joint-routed core splits robots internally.** In v5, A1's 12 joints spread over 5 core experts, and the selection
  barely changes with the state. Only H1, Talos and the hexapod come close to one shared pair (0.81–0.88).
- **It groups robots by family:** the six small quadrupeds share one set, H1 and Talos another, OP3 and NAO a third.
  The decoder is where joints are shared across robots.
- **Suggested to the user (not run):** v5 with only `morse_core_routing=body` (the `body_core` ablation), seeds 0–2.

**The 57 run, `CAP-8x64-top4-s0` (2026-10-05 check; duongnm2 `jobs/experiments.tsv` line 87; run started 10-02 20:15, job 56652):**
- **Row flags:** `morse_blocks=core,decoder`, `morse_conflict_mode=none` (the evidence, threshold, kappa and lambda flags are
  inert), `actor_steps_per_update=4`, `morse_num_experts_decoder=8`, `morse_width_mult_decoder=0.5`, `morse_topk_decoder=4`.
- **Defaults then:**
  - 4 core experts, full width (512-256-128), dense, with body routing.
  - Router output layers zero-initialized; MI 0.
  - **Utilization loss `morse_eta=0.01` was active**: logged `loss_use` went 0.05 → 0.003, and its router gradient was
    about 10% of the task's.
- **Parameters:** 1,686,527 (core 1,510,728, decoder 148,504, shared 27,267, encoders 16 + 12).
- **Reproduction caveat.** The old `keep_topk` kept every weight ≥ the k-th largest, so the uniform start was dense for
  the first step. The current exact top-k with `morse_router_init=0.0` would start every joint on experts 0–3, and
  unselected zero-initialized logits get no task gradient.
  - So a rerun needs `PREVIOUS_DEFAULTS` + `LEGACY_ARCHITECTURE` flags, plus either the old tie rule or a small
    `morse_router_init` (for example 0.01).

**MI vs task on the routers in the full runs (2026-10-05, `mi_vs_task.py` on the cached curves):**
- The dry run's 5–24× does **not** hold in the full runs. Logged `router_grad_mi_ratio` (MI ÷ task gradient norm on router
  parameters):

  | Run (MI weight) | Core router | Decoder router |
  |---|---|---|
  | v3 s0 (0.005), 500 to 100k updates | 0.06–0.30 | 0.11–0.40 |
  | v4 s0, s2 (0.0005), 5k to 100k | 0.01–0.03 | 0.01–0.04 |
  | v5 s0, s2 (0.0005), 5k to 100k | 0.01–0.05 | 0.03–0.14 |

- **Why the dry run differed.** Its ratio was large because the router task gradient collapsed (core 0.0055 → about
  0.0001 by update 50, with H(E|type) 0.99 → 0.38). The MI gradient itself (about 0.002) matched the full run's.
- **Loss values:** the MI term is about 0.1% (v5) to 1.5% (v3) of the policy loss, which is about 1.15–1.35.

## 13. MORSE-v6 (approved by the user 2026-10-05; synced, tested and launched, see §13.1; current state in §14)

**Goal (user):** recreate the 57 run's success while using MI. The core router is left free (no robot-type pull),
like the 57 run, which was trained only by the task loss plus load balancing.

**Config = the MORSE-v4 row with these changes:**
- `morse_core_routing=body`, `morse_width_mult_core` = 0.5 or 0.75 (¼ dropped by the user). 8 + 8 experts, top-4 in
  both blocks; decoder 8 × 64 (= the 57 run's decoder).
- `morse_mi_weight=0.001` (decoder: full joint-type MI), `morse_mi_weight_core=0.005`, `morse_mi_balance_blocks=core`
  (the core takes only −H(E), which equals the 57 run's utilization loss up to log 8).
- **Robot-MI test:** the ½-width row without `morse_mi_balance_blocks`, so the core takes the full I(R;E) at 0.005.

**Code (local only):**
- `config.py`: new `morse_mi_weight_core` (−1 = use `morse_mi_weight`) and `morse_mi_balance_blocks` (""), both
  added to `RECORD_KEYS`.
- `updater.py`: per-block `mi_block_weight`; MI blocks need a positive weight. A balance block must be an MI block
  (ValueError otherwise). `_mi_loss` uses −h_marg for balance blocks.
- `tests/test_morse.py`: new test 22, `test_body_core_balance_mi`. It checks the body core's top-4 and shared z, the
  expected parameter counts, and that the core router moves by exactly −lr·w·∇(term) for the balance and robot-MI
  variants. **It has never been run** (no JAX locally).

**Expected parameters:**

| Width | Core | Decoder | Total |
|---|---|---|---|
| ½ | 1,249,616 | 148,504 | 1,425,415 |
| ¾ | 2,056,016 | 148,504 | 2,231,815 |

**Remaining steps:**
1. Sync `config.py`, `updater.py` and `test_morse.py` to Hessian (`hsync_paths.sh`) and both H100 copies; compare
   checksums.
2. Run the full suite on Hessian (27 tests).
3. Probe the Hessian quota (fallocate; delete the probe right away).
4. Add the rows, backing up each TSV first. Base them on `dry-v4-s0` / `MORSE-v4-s{n}` with sed, as in
   `add_v4_rows.sh` and `add_v5_rows.sh` (this session's scratchpad).
5. Dry runs on Hessian: `dry-v6-w50`, `dry-v6-w75`, `dry-v6-w50-robotMI`.
6. Runs: 12 in total, filling 8 slots.

**Runs and placement:**
- **Hessian** (4 workers, 2 per GPU, all on one list; start them about 150 s apart like `start_v3.sh` in session
  689bd97a's scratchpad): w50-s0, w75-s0, robotMI-s0, robotMI-s1, then w50-s3, w75-s3.
- **duongnm2** (2 jobs, one list): w50-s1, w75-s1, w50-s4.
- **tuannl** (2 jobs, one list): w50-s2, w75-s2, w75-s4.
- Submit only from login-restricted-2.
- **Ids:** `MORSE-v6-w50-s{0..4}`, `MORSE-v6-w75-s{0..4}`, `MORSE-v6-w50-robotMI-s{0,1}`.
- **Optional:** the 4 free second-wave slots could run seeds 1–2 of the exact 57 configuration. The user hasn't
  decided.

### 13.1 v6 launch (2026-10-05, Hessian clock 10-04 ~19:25–20:05)
- **Code synced** to Hessian and both H100 copies. The md5s of `config.py`, `updater.py` and `test_morse.py` match
  local. Before syncing, the server copies differed from local only by this change.
- **Tests:** the new test 22 passed (2.3 min on CPU), and the full suite passed (27 ok, `$R/analysis/tests_v6.log`).
- **Hessian quota:** fallocate probes of 5, 10 and 20 GB succeeded and were deleted. A finished v5 run is 423 MB.
- **Rows:** `scratchpad/add_v6_rows.sh` added and checked 15 rows on Hessian (backup `experiments_hessian.tsv.bak-10041928`).
  The 6 H100 rows were copied byte-identically (md5 checked) to duongnm2 and tuannl `jobs/experiments.tsv` (backups
  `.bak-*`).
  - Lists: Hessian `queue_h_v6.txt`; duongnm2 `jobs/queue_v6_h100.txt`; tuannl `jobs/queue_v6_tuannl.txt`.
- **Dry runs (`dry-v6-w50-s0`, `dry-v6-w75-s0`, `dry-v6-w50-robotMI-s0`, 20k subset, 200 updates) passed:**
  - 294 finite tags; checkpoints at 100, 150 and 200; `routing.npz` written.
  - Parameters 1,425,415 / 2,231,815.
  - Balance core: H(E) = 2.079 (uniform), all 8 experts in the top-4.
  - Robot-MI core: MI 0.86, H(E|robot) 0.57.
  - Decoder: H(E|joint) 0.45–0.66 by update 150, with an MI/task router-gradient ratio of 2–5 (the usual small-subset
    dry-run effect).
- **Running:**
  - **H100 duongnm2** jobs 57351 (`MORSE-v6-w50-s1`) and 57352 (`MORSE-v6-w75-s1`); `w50-s4` follows.
  - **H100 tuannl** jobs 57353 (`w50-s2`) and 57354 (`w75-s2`); `w75-s4` follows.
  - All four were submitted from login-restricted-2 only, by `scratchpad/submit_guarded.sh`.
  - **Hessian** has 4 workers (`start_v6.sh`, 2 per GPU, full dataset) on `queue_h_v6.txt`: w50-s0, w75-s0,
    robotMI-s0 and robotMI-s1 first, then w50-s3 and w75-s3. Worker logs: `$R/hq/worker_v6_{a,b,c,d}_g*.out`.
- **Monitoring:** use the marker files only: Hessian `$R/hq/state/{done,failed}`, H100 `queue/{done,failed}` (no
  squeue/sacct).
- **Hessian workers confirmed (20:05 Hessian time):** all 4 are running, with claims w50-s0, w75-s0, robotMI-s0 and
  robotMI-s1. No tracebacks. GPU memory is 4–5 GB per GPU. All 8 slots are busy; w50-s3, w75-s3 (Hessian), w50-s4
  (duongnm2) and w75-s4 (tuannl) are queued.

### 13.2 v6 partial results (2026-10-05 ~02:30 UTC; `v6_results.py`, `v6_latest.py`, `collect_v6.sh` in this session's scratchpad)
- **Status:**
  - H100 finished w50-s1, w75-s1 (duongnm2) and w50-s2, w75-s2 (tuannl); w50-s4 and w75-s4 run there.
  - The Hessian runs (w50-s0, w75-s0, robotMI-s0/s1) are at about 70k; w50-s3 and w75-s3 are queued.
- **Collection:** H100 became unreachable (ssh timeout, probably the VPN) during collection. The finished H100 runs are
  **not cached yet**; rerun `collect_v6.sh duong tuannl` once H100 answers.
- **At 65k** (mean return / without the hexapod):

  | Run | Seed 0 | Seed 1 |
  |---|---|---|
  | 57 run | 57.1 / 56.6 | — |
  | v5 | 51.1 / 54.5 | 42.6 |
  | v4 | 50.8 / 54.2 | 38.8 |
  | v6 robot-MI | 49.5 / 52.8 | **48.6 / 51.9** (best seed 1 so far) |
  | v6 w75 | 48.3 / 51.5 | — |
  | v6 w50 | 44.5 / 47.5 | — |
  | MoE | — | 46.3 |
  | URMA | — | 34.5 |

  No v6 run has the hexapod at 65k.
- **Routing at about 70k:**
  - Balance core: H(E) 2.08, H(E|robot) 1.6, MI 0.5 (some robot structure from the task alone), balance/task router
    gradient 0.01.
  - Robot-MI core: H(E|robot) ≈ 0, so one expert per robot; MI 2.0; MI/task router gradient 0.8–1.3.
  - Decoder (joint MI 0.001): H(E|joint) 0.6–0.85, MI 1.1–1.4, H(E) 2.0, ratio 0.03–0.06.

### 13.3 v6 results on seeds 1–2 (2026-10-05 ~03:50 UTC; all four finished H100 runs fetched)
- **100k** (mean return, robots learned / without the hexapod):

  | Run | Seed 1 | Seed 2 | Mean | Mean without hexapod |
  |---|---|---|---|---|
  | **v6 ¾ width** | **55.9 (13)** | 50.8 (12) | **53.3** | **56.9** |
  | v6 ½ width | 38.4 (10) | 47.6 (11) | 43.0 | 45.8 |
  | v5 | — | — | 46.8 | 49.9 |
  | v4 | — | — | 42.3 | 45.2 |
  | old MoE | — | — | 49.4 | 52.7 |
  | URMA | — | — | 42.9 | 45.8 |
  | 57 run (seed 0 only) | — | — | 56.4 | 54.1 |

  - v6 ¾ width over 50k → 75k → 100k: seed 1 51.1 → 53.3 → 55.9; seed 2 51.0 → 49.3 → 50.8.
  - v6 ¾ seed 1 at 100k learns everything except a1, go1 and the hexapod, including Cassie (81).
  - No v6 run learned the hexapod (≤ 1).
- **Housekeeping:**
  - Leftover cache jobs from the failed collection attempts were stopped on both login nodes of both H100 accounts
    (`scratchpad/clear_cache_jobs.sh`); Hessian has none.
  - **Lesson:** the login nodes are slow (load average about 37); each 270 MB log takes 15–20+ min at nice 19.
    Never set ssh timeouts shorter than the cache step. Prefer caching inside the job.
- **Moved to free H100 slots:** w50-s3 (duongnm2 job 57399) and w75-s3 (tuannl job 57401) were taken off the Hessian
  list (backup `queue_h_v6.txt.bak-*`) and added to `jobs/queue_v6b_{h100,tuannl}.txt`.

---

## 14. v6 status, results and analysis (2026-10-05, written for the next account/session)

### 14.1 What v6 is
v6 is the user's attempt to recreate the 57 run (`CAP-8x64-top4-s0`, see §12) while keeping an MI loss.

| Part | v6 setting | 57 run |
|---|---|---|
| Core | 8 experts, ½ (400→256→128→128) or ¾ width (400→384→192→128), **top-4 per state** | 4 full-width experts, dense |
| Core routing | **body**: one router per robot state (robot descriptor + x); every joint gets the same z | body, dense |
| Core router loss | **load balancing only**, −H(E), weight 0.005 (`morse_mi_weight_core=0.005`, `morse_mi_balance_blocks=core`) | utilization KL(p̄‖U) 0.01 (same term up to log 8) |
| Decoder | 8 experts × 64 hidden, **top-4 per joint** | same |
| Decoder router loss | joint-type MI, weight 0.001, joint-equal weights | utilization 0.01 |
| Router init / top-k | random std 0.01, exact top-k | zero init, old tie rule |
| Parameters | ½: 1,425,415; ¾: 2,231,815 | 1,686,527 |

**Robot-MI test:** ½ width, with the core taking the **full robot-level MI** I(R;E) at 0.005 (no
`morse_mi_balance_blocks`), so the core router follows the robot type.

### 14.2 Run status (last checked 2026-10-05 ~03:50 UTC)

| Run | Where | State |
|---|---|---|
| MORSE-v6-w50-s1, MORSE-v6-w75-s1 | H100 duongnm2 | **done** (100k), cached locally |
| MORSE-v6-w50-s2, MORSE-v6-w75-s2 | H100 tuannl | **done** (100k), cached locally |
| MORSE-v6-w50-s0, -w75-s0, -w50-robotMI-s0, -w50-robotMI-s1 | Hessian, 4 workers on `tools/queue/queue_h_v6.txt` | running, about 70k at 02:24 UTC; cached locally up to about 70k |
| MORSE-v6-w50-s4 | duongnm2 job 57352 (list `jobs/queue_v6_h100.txt`) | running, about 25k at 03:19 |
| MORSE-v6-w75-s4 | tuannl job 57353 (list `jobs/queue_v6_tuannl.txt`) | running, about 27k at 02:42 |
| MORSE-v6-w50-s3 | duongnm2 job 57399 (list `jobs/queue_v6b_h100.txt`) | running since 02:47 (moved from Hessian) |
| MORSE-v6-w75-s3 | tuannl job 57401 (list `jobs/queue_v6b_tuannl.txt`) | running since 02:49 (moved from Hessian) |

- No failures or tracebacks so far.
- Monitor **only through marker files**: Hessian `$R/hq/state/{done,failed}`, H100
  `/mnt/data/<account>/morse/queue/{done,failed}`. No squeue/sacct.
- The rows are in Hessian `tools/queue/experiments_hessian.tsv` and in each H100 account's `jobs/experiments.tsv`, all
  byte-identical (md5 checked). Backups `*.bak-*` sit next to them.

### 14.3 How to collect results (and avoid the slowness of 2026-10-05)
- **Local tbcache:** `<old scratchpad 4fd1fedc...>/tbcache/MORSE-v6-*/<id>/tb_scalars.npz` (+ `morse_diagnostics/routing.npz`).
- **Scripts** are in this session's scratchpad
  `C:\Users\Lenovo\AppData\Local\Temp\claude\E--Code-cross-embodiment-offline-rl\37c46568-6b61-4a71-b3f6-bcfd88cf97a7\scratchpad\`:

  | Script | What it does |
  |---|---|
  | `collect_v6.sh [hessian] [duong] [tuannl]` | caches on the server and copies here |
  | `fetch_h100_v6.sh` | copies H100 caches once they exist; edit its run list |
  | `clear_cache_jobs.sh HOST [kill or list]` | finds and stops leftover cache jobs on **both** login nodes |
  | `v6_results.py TBCACHE THRESHOLDS`, `v6_factors.py`, `v6_latest.py` | result tables |
  | `core_overlap.py`, `mi_vs_task.py` | routing analyses |

  The thresholds are `<old scratchpad>/learned_thresholds.json`.
- **Why it was slow:**
  - This PC has no tensorboard, so the 270 MB event logs are converted on the servers.
  - On H100 that happens on the **shared login nodes**, which are heavily loaded (load average about 37). At nice 19 a
    single finished run takes 15-20+ min.
- **Mistakes not to repeat:**
  - Never give the cache ssh a timeout shorter than the work (2 runs take 40+ min). The remote process survives a cut
    connection, its output is lost, and a retry starts a duplicate.
  - The H100 alias lands on login-restricted-1 or -2 at random, so duplicates can hide on either node. Use
    `clear_cache_jobs.sh HOST list` before starting new cache jobs.
  - Don't use `pkill -f` with a pattern that appears in your own command line (it kills its own ssh). Use a bracket
    pattern such as `[p]ython analysis/tb_cache`.
- **Better options for the future** (not done; the user decides):
  1. Cache inside the training job at its end (the job has its own CPUs).
  2. Install tensorboard locally and download the raw event logs.

### 14.4 Results so far

**100k, seeds 1-2** (mean return over 16 robots, robots learned):

| Run | Seed 1 | Seed 2 | Mean | Mean without hexapod |
|---|---|---|---|---|
| **v6 ¾ (free core)** | **55.9 (13)** | 50.8 (12) | **53.3** | **56.9** |
| old MoE | 50.7 (12) | 48.1 (11) | 49.4 | 52.7 |
| v5 | 42.8 (10) | 50.9 (12) | 46.8 | 49.9 |
| v6 ½ (free core) | 38.4 (10) | 47.6 (11) | 43.0 | 45.8 |
| URMA | 35.5 (10) | 50.3 (12) | 42.9 | 45.8 |
| v4 | 31.4 (8) | 53.3 (12) | 42.3 | 45.2 |
| 57 run (seed 0 only) | | | 56.4 (13) | 54.1 |

- v6 ¾ seed 1 is the best seed-1 run ever: 51.1 → 53.3 → 55.9 at 50/75/100k, with no late collapse. It learns
  everything except a1, go1 and the hexapod, including Cassie (81).
- No v6 run learned the hexapod (≤ 1). The hexapod is the 57 run's edge.

**65k, every v6 run** (the horizon all of them have reached):

| Run | s0 | s1 | s2 | Mean |
|---|---|---|---|---|
| v6 ¾, free core | 48.3 | 52.4 | 50.3 | **50.3** |
| v6 ½, free core | 44.5 | 41.5 | 50.2 | 45.4 |
| v6 ½, robot-type core | 49.5 | 48.6 | — | 49.1 (2 seeds) |
| v5 | 51.1 | 42.6 | 50.7 | 48.1 |
| v4 | 50.8 | 38.8 | 52.5 | 47.4 |
| 57 run | 57.1 | — | — | — |

**Routing at the end or latest point:**
- **Free core:** H(E) 2.08 (all 8 experts equally used); H(E|robot) 1.5-1.7, MI 0.4-0.6, so robots partly separate
  from the task alone. The balance/task router gradient is about 0.01.
- **Robot-type core:** H(E|robot) about 0.00-0.02, so effectively **one expert per robot** (an EG-like learned
  grouping); MI 1.95-2.0. The MI/task router gradient is 0.8-1.3.
- **Decoder:** H(E|joint) 0.57-0.85, MI 1.1-1.4, H(E) 1.9-2.0; MI/task router gradient 0.03-0.10.

### 14.5 Analysis: what brings the good results
**Isolated comparisons** (only one thing differs):

| Factor | Comparison | Result | Strength |
|---|---|---|---|
| **Core capacity** | v6 ½ vs ¾ width | 65k, 3 seeds: 45.4 vs **50.3**; 100k, seeds 1-2: 43.0 vs **53.3** | strong: the cleanest effect |
| **Sparse decoder** | 57-run family, seed 0: dense vs top-k decoder | dense 50.7-50.8 vs top-2 55.7 / top-4 **56.4** | fairly strong: one seed, every good run has it |
| **Robot-type core** | ½ width, seeds 0-1 at 65k | free 43.0 vs robot-type **49.1** | promising: 2 unfinished seeds |
| Joint vs body core routing | v4 (joint) vs v6 ½ (body); same active capacity | 42.3 vs 43.0 (100k, s1-2) | no effect |
| MI weight | v3 s0: 0.005 / 0.0005 / none | 41.3 / 48.8 / 48.0 | large MI hurts; small MI about equals none |
| Total model size | v5 (2.30M) vs v6 ¾ (2.23M) | 46.8 vs 53.3 | size alone isn't it |
| Load balancing | always on in the good runs | keeps all experts used; gradient about 1% of the task's | untested |

**Active core capacity per routing unit**, as a multiple of URMA's core (370,560 parameters), with the decoder for
context:

| Run | Active core | Decoder | Result |
|---|---|---|---|
| v3 | 0.41× | sparse | 39.4 (3 seeds) |
| v5 | 1.37× | sparse (top-2 of 96-wide) | 48.1 (3 seeds) |
| v4 | 1.65× | sparse | 45.1 (3 seeds) |
| v6 ½ | 1.65× | sparse | 43.0 (s1-2) |
| v6 ¾ | 2.74× | sparse | 53.3 (s1-2) |
| 57 run | 4.0× | sparse | 56.4 (s0) |
| old MoE | 4.0× | **dense** | 47.3 (3 seeds) |

**Which robots decide the mean:**
- **Talos:** about 110 when learned, about 5 points of the 16-robot mean.
  - ½ free core: lost in 2 of 3 seeds at 65k (36, 25, 108).
  - ¾ free core: kept in 3 of 3 (100, 110, 110).
  - ½ robot-type core: 61 and 110.
- **NAO and Go2** behave similarly. a1 and go1 are never learned by any MoE run; only the 57 run learned the hexapod.

**Reading** (1-3 seeds per config; single robots flip the mean by 3-7 points, so treat this as provisional):
1. **A sparse per-joint decoder** (8 small experts, top-k): present in every good run.
2. **Enough core capacity per robot**, so the large bipeds (Talos, NAO) are not crowded out. Two ways:
   - big shared experts (4 active ¾-width experts, about 2.7× URMA per state; the 57 run had 4×);
   - separating robots in the core (robot-type MI), which worked even at ½ width.
3. **Not shown to matter:** joint-level core routing, the MI loss at small weight (neutral; large weight hurts),
   total parameter count, and load balancing (never removed).

### 14.6 Next steps
1. Collect the remaining 8 runs when their markers appear (§14.3), then rerun `v6_results.py` and `v6_factors.py`.
   That gives the 5-seed means for ½ and ¾ and the finished robot-MI test.
2. **Proposed tests, not decided by the user:**
   - v6 ¾ with the robot-type core MI: do separation and capacity add up?
   - v6 ¾ with decoder MI 0: is MI needed? This matters for the paper's MI claim.
   - Optionally, seeds 1-2 of the exact 57 configuration (see the §12 rerun caveat).
3. **Paper** (`paper/morse_cvpr_revised.tex`): Sec. 4 still describes the joint-routed core (v3-v5). If v6 becomes
   the method, it needs rewriting:
   - Sec. 4.1/4.2: a body-routed core that every joint shares, with load balancing.
   - Sec. 4.3: joint-level decoder experts with joint-type MI.
   - The abstract and intro sentences on routing; the "Analysis of Joint Interactions" (localized interaction then
     applies to the decoder; the core localizes at the robot level).
   - The user prefers that MI not be over-claimed until the decoder-MI-0 test exists.
4. **Standing rules (§3) still apply:**
   - submit only from login-restricted-2;
   - no auto-resubmit loops; minimal Slurm traffic;
   - dry runs on Hessian first;
   - never echo the W&B key or `morse_shell_key`;
   - never write to /home/tuannl;
   - ask before design-level changes;
   - no `rm` without listing paths and sizes and getting an OK.
   - The Hessian coordination dumps (3 GB) still await the user's OK to delete.

---

## 15. V7 (2026-10-06): two ideas, shared core experts, 37-run plan

**Slides and design:** `MORSE_V7_SLIDES.md`.

**Arms:**

| Arm | Design |
|---|---|
| A | Balance only, original 16+16 expert architecture |
| B | Compatible-set routing |
| C | MI-only: EG-group core MI plus robot-joint decoder MI |
| A-H | Hierarchical shared core, no MI |
| C-H | Hierarchical shared core plus MI (the proposed method) |

**Shared V7 setup:**
- **Core:** 16 experts of 400→384→192→128, body routing, top-4 (top-2 routed in -H).
- **Decoder:** 16 experts of 260→64→1, top-4.
- **Routers:** zero-init router heads. Mod-Squad noise σ = softplus + 0.01 in training only. Auxiliary losses use the noisy pre-top-k probabilities with σ·ε detached.
- **Balance:** KL(p̄‖U) at 0.01 per block, on 64 core / 8 decoder transitions per robot, first actor substep.
- **Training:** 4 actor substeps.

**Hierarchical core (`morse_core_shared=hierarchical`):**
- Experts: 1 global + 4 EG-group + 11 routed.
- Output: h = 0.25·global + 0.25·own group + 0.5·renormalized top-2 routed mixture.
- The router, MI and balance cover the routed experts only. The shared experts get task gradients only.
- Groups (Ward, M = 4): {a1, go1, go2, bkv0, bkvb, bg, bit, hx}, {anyb, anyc, cas}, {h1, tl}, {g1, op3, nao5}.
- Parameters per expert: 253,376 → core 4,112,812 vs 4,114,112 for A.

**DeepSeekMoE source:**
- HF `deepseek-ai/deepseek-moe-16b-base` at rev 521d2bc4fb69a3f3ae565310fcc3b65f97af2580 (modeling_deepseek.py md5 cc30b472…).
- GitHub `deepseek-ai/DeepSeek-MoE` at 66edeee5… (README and finetune scripts only).
- Differences:
  - DeepSeek adds the shared MLP ungated, y = routed + shared.
  - Its top-k weights are not renormalized (norm_topk_prob=False).
  - It has no gate noise.
  - Its aux loss is α·Σ f_i P_i on routed experts.
  - It fuses its 2 shared experts into one wide MLP.
  - Ours uses fixed 0.25/0.25/0.5 coefficients, group experts, renormalized top-2 and noise, and computes every expert densely.

**Code (synced, md5-checked on Hessian and both H100 copies):**
- New flags: `morse_core_shared` (none|global|group|hierarchical), `morse_core_shared_coef`, `morse_eg_groups` (default 4).
- The EG table is `policy.eg_group_table`, shared by the policy and the updater; it fails on bad mappings.
- New diagnostics: branch norms, task-grad norms per expert category, routed eff/usage, noise scale, peak memory, wall time per update.
- Sync content hashes go to `<run>/sync_hashes.jsonl`.
- Test 27 covers the shared core. The full suite of 32 passes (`$R/analysis/tests_v7b.log`).
- **Bug fixed:** the old C rows had a core EG-MI that was identically 0, because MORSE's config.M=1 was used as the group count. Now `morse_eg_groups=4`, and C rows carry it.

**Dry runs (Hessian, 20k subset):** dry-v7-A2/B2/C2/AH2/CH2 all pass.
- A2 and B2 are bitwise equal to the original A and B dry runs (311 and 338 tags).
- C2's core MI is now nonzero.
- Minibatch and critic hashes are identical across all 5.

**Run placement (launched 2026-10-06 ~06:16-06:21 UTC; lists disjoint, 37 runs):**

| Where | List | Runs (in order) |
|---|---|---|
| Hessian, 4 workers (2 per GPU) | `queue_h_v7_main.txt` | AH-s0, CH-s0, A-s0, B-s0 (λ pilot), C-s0, B-s1, B-s2, urma-s0, dense4M-s0 |
| duongnm2, jobs 57726/57727 | `jobs/queue_v7_h100.txt` | AH-s1, CH-s1, A-s1, C-s1, EG-s1, 57cfg-s1, urma-s1, dense4M-s1, Talos exp/mix A-s0, A-s1, B-s0 |
| tuannl, jobs 57728/57729 | `jobs/queue_v7_tuannl.txt` | AH-s2, CH-s2, A-s2, C-s2, EG-s2, 57cfg-s2, urma-s2, dense4M-s2, Talos exp/mix A-s2, B-s1, B-s2 |

- **Hessian started with 2 workers** (`start_v7c.sh`; a on GPU 0, b on GPU 1), the user's choice because of the shared quota. `queue_h_v7_single.txt` is empty: the Talos runs moved to H100.
- **2026-10-06 06:51 UTC: 2 more Hessian workers** (the user asked whether Hessian could take 2 more jobs):
  - **Resources before:**
    - GPUs: 5 of 80 GB used each; utilization swings between ~90% (training) and ~20% (evaluation).
    - CPU and RAM: about 3 of 32 cores busy, 1.7 TB RAM free (each 16-robot run holds ~86 GB).
    - Quota probes: 12 and 8 GB failed; 5 and 4 GB passed.
  - **Workers:** `start_v7d.sh` started worker c on GPU 1 (takes A-s0, next to C-H) and worker d on GPU 0 (takes B-s0, next to A-H). Logs are in `hq/worker_v7d_*.out`.
  - **Moved from H100:** urma-s0 (duongnm2) and dense4M-s0 (tuannl), appended to the Hessian list.
    - They are commented out in the H100 lists, which the workers re-read before each run. Backups: `*.bak-10060700` on all three machines.
    - The two H100 lists are now about 72 estimated hours each.
  - **Talos runs stay on H100,** so the Talos A-vs-B comparison stays on one hardware type.
  - **Measured sharing cost (from per-minute log lines):**
    - CH-s0 ran at ~230 updates/min alone and ~144/min once A-s0 shared GPU 1; A-s0 runs at 133/min.
    - So each run is about 1.6× slower, and per-GPU throughput rises about 1.2×.
    - On H100, a 3g MIG slice runs ~193/min.
  - **Disk:** a V7 16-robot run writes ~0.45 GB (TensorBoard events ~2.7 MB per 1k updates plus 3 checkpoints), so the 9 Hessian runs need ~4 GB against 5-8 GB free. Probe the quota at every check.
- Deferred: C-global/C-group, Go2 quality runs, C-robot, robot-private, and B controls (only if B shows promise in either scenario).

**Hessian disk:**
- The user deleted the dry/tune runs, the coordination dumps, the branching copies, the old `models/` folders and the pre-V7 run logs (about 8 GB). Runs went 11.4 → 6.4 GB.
- A 6 GB quota probe then succeeded. The quota is shared, so probe before adding workers.

**Code freeze:** the code is identical on all three machines. Manifests of 590 files: `$R/tools/code_manifest_v7.txt` and `<M>/jobs/code_manifest_v7.txt` (manifest md5 9ef4cd9e…). Don't change training code while V7 runs are queued; use a separate copy if needed.

**Next checks:**
- **B pilot:** V7-B-s0 started on 2026-10-06 ~06:53 UTC with the extra Hessian workers.
  - Check its first ~12k updates (label shares, compat/task router-gradient ratio, update_s) before B-s1 starts. B-s1 starts when one of the first four Hessian runs finishes, roughly 13-15 h later.
  - If λ changes, keep B-s0 as the pilot and rerun it.
  - The Talos B runs sit at the end of the H100 lists.
- **B pilot result (2026-10-06 15:40 UTC, B-s0 at 68k):**
  - **How it was read:** `scratchpad/v7_check.py` on caches written by `tb_extract.py`. The extractor also sits at `$R/analysis/` and `<M>/analysis/`; caches are in `analysis/v7cache/`.
  - **Core, at full λ (10-20k updates):**
    - Labels: 56-58% compatible, 8-9% conflicting, 33-35% unknown. About 9 of 16 experts are compatible per unit, rising to 11 after 50k. No unit is left with 0 or 1 compatible experts.
    - Router gradient, compat/task: 0.37-0.39 (0.23 after 50k).
    - Signal: weak but present. Conflict labels are 1.4-1.9× as frequent as under independent references.
  - **Decoder:**
    - Labels: 33% compatible, 16-19% conflicting, ~50% unknown.
    - Agreement between the two references is 0.48-0.53, which equals the independence baseline. **The decoder labels carry no detectable signal.**
    - Yet the compat/task router-gradient ratio is 0.63-0.79.
  - **Cost:** wall time per update is +16% over A (0.44 vs 0.38 s on shared Hessian GPUs).
  - **Outcome at 50k:** B-s0 learns 12/16 (mean 47.2) vs A-s0 12/16 (50.4). B-s0 learns Go1 (40.5) but loses Go2.
  - **Decision pending (the user):** keep λ = 0.01 on both blocks, or drop or fix the decoder term (then rerun B-s0).
  - **Gating:** B-s1 and B-s2 now come after urma-s0 and dense4M-s0 on the Hessian list (backup `queue_h_v7_main.txt.bak-10061545`). The Talos B runs are still at the ends of the H100 lists.
- **First V7 results (2026-10-06 15:40 UTC):**
  - **Finished A-H/C-H runs at 100k:** AH-s1 12/16 (50.8), CH-s1 11/16 (47.7), AH-s2 12/16 (50.8), CH-s2 12/16 (50.8).
  - **Robots:** all four lose A1 and Go1 (14.4 each) and Cassie by 100k, and learn H1, G1, Talos, OP3 and NAO. That is the MoE-core pattern, so the shared experts did not recover the shared core's A1/Go1.
  - **Seed 0 at 50k:** A 12/16, AH 11/16, CH 12/16.
  - **C-H's EG-group MI saturates within 10k updates:** MI 1.23 nats = H(group), h_expert = ln 11. Every routed expert serves one EG group, so in C-H the global expert is the only path shared across groups.
  - **Global expert (50k+):**
    - Its weighted output norm is 7-8, against 13-14 for the group experts, 10-18 for the routed mixture and 20-27 for the output.
    - Its task gradient per parameter is similar to the group experts'.
    - The group-only ablation (`morse_core_shared=group`) is not queued.
- **Results analysis (2026-10-06 16:30 UTC; `scratchpad/v7_compare.py`, `per_robot_100k.py`):**
  - **At 50k, seed-paired over 3 seeds:**
    - A 50.3, A-H 51.7, C-H 51.4.
    - A-H − A = +1.5 and C-H − A = +1.2. Without Cassie these become −0.0 and +0.6, so the hierarchical "gain" is Cassie being up at 50k, and every finished run loses Cassie by 100k.
  - **Robots:**
    - 11 of 16 robots are within about 1 point in every V7 run.
    - The mean is decided by A1, Go1, Cassie and the hexapod. These sit at 14.4 for A1/Go1 (standing all 1000 steps; EG walks them at ~53), ~1 for Cassie (~80 when kept) and ~1 for the hexapod (~91 when learned).
    - Each flip is worth 2.4-5.6 mean points.
  - **The 57 run's 100k lead** (56.4 vs 50.8) is the hexapod alone: without it, both are 54.1. v6 ¾ s1's 55.9 is Cassie (81.5).
  - **V7 vs earlier runs:** V7 A at 50k (50.3) is the best MoE base so far (v6 ¾ 50.0, old MoE 48.5, URMA 39.3), with a faster start (25k: ~46). H1, G1, Talos and NAO are learned in all 11 V7 runs at 50k.
  - **C vs A:** C-s1 at 50k is 48.5 vs A-s1's 49.4.
- **Mechanism analysis (2026-10-06 ~16:45 UTC; `scratchpad/mech_analysis.py`, `run57_width.py`, `b_curves.py`). Seed-paired vs A, 10-50k window:**
  - **Core unit-pair conflict** (A 0.27): B −0.11, A-H −0.04, C-H −0.05, C 0.
  - **Core cancelled share** (A 0.87): A-H +0.03 and C-H +0.02, because the always-on experts collect the conflicting gradients. B −0.03, C −0.03.
  - **Robot-level opposing pairs** (core 0.46, whole actor 0.46) are unchanged in every arm.
  - **Decoder:** B pushes routing toward uniform (3.8 vs 2.7 effective experts) and raises cancellation slightly. Decoder MI (C, C-H) concentrates routing (1.6 effective) and lowers cancellation by 0.02-0.03.
  - **Outcomes:** unchanged. B-s0 loses NAO late (91 → 64 at 70k, 20 in the 70-80k window) and Go2, and its Go1 bump (26) fades.
  - **The 57-run family (seed 0, decoder variants):** every 64-wide decoder gives 57.8-57.9 on the 14 robots other than Cassie and the hexapod; AH-s1 is also 57.9. The 57 run's lead is the hexapod alone, learned steadily from 50k (9 → 35 → 70 → 87 → 90) and kept.
  - **Core width (v6 ½ vs ¾, 5 seeds, paired):** ¾ leads by +4.3 at 25k and +4.4 at 50k (all 5 seeds positive), +2.1 at 75k and +0.9 at 100k (sd 10.2). Without Cassie and the hexapod at 100k they are equal (54.2 vs 54.1).
  - **Decoder ½-width "gain":** one seed only. 4×128 dense scores 43.2/50.7/48.1 over its 3 seeds against 50.8 for 4×64 (seed 0); the gap is Talos collapsing in 4×128 seed 0.
- **URMA version (checked 2026-10-07):** our code is URMA v1, both the EG/URMA policy (`uni_iql_eg/iql/policy.py`) and the MORSE policy built on it.
  - **URMAv2** (Bohlinger & Peters, arXiv 2509.02815, Sec. III-A) is not used. Its changes:
    - no foot observations or foot encoder in the policy;
    - an extra per-joint "track nominal position" observation;
    - a 2×256 joint-observation MLP;
    - 5 core layers with WeightNorm;
    - an attention decoder, μ_j = z_action · α_j;
    - the action std from a linear layer on f_φ(d_j).
  - Our dataset has no nominal-tracking field.
- **Gradient split by parameter group (2026-10-06 ~18:15 UTC):**
  - **Method:** new flag `morse_ckpt_grads=N` (ckpt_analysis.gradient_split): per-robot policy-loss gradients on N fixed minibatches, a robot Gram per group, plus per-robot routing.
  - **Rows run:** `grads-{A-s0,CH-s0,C-s1}-50k-75k` (exp `analysis`) on the V7 checkpoints. The C-s1 checkpoints were copied to `$R/ckpt/V7-C-s1`. Outputs are in `$R/analysis/grads/`; reports are `scratchpad/grads_report.py` and `pair_conflict.py`.
  - **Where robot conflict sits (all 3 runs, 50k and 75k):**
    - Action log-std head (129 params): 58-65% of the robots' squared gradient and 37-65% of the cross-robot cancellation.
    - Decoder descriptor embedding (19.8k): 10-21% and 19-44%. Together with the std head: 78-89% of the cancellation.
    - State encoders (16 + 12): 10-20% of the cancellation.
    - Attention MLPs: ≤ 1%. Core experts: ≤ 0.4%. Decoder experts: ≤ 2.4%.
    - The input encoders are not the main source.
  - **Who drives it:** in C-s1, Cassie (49%) and the hexapod (14%) dominate the std-head gradient. OP3, Bittle and G1 pull the embedding against the rest; the most opposed pair, OP3-NAO (−0.8), is inside one EG group.
  - **Routing:**
    - C-H and C separate the EG groups completely: each routed expert serves one group. Within a group, experts are shared.
    - In C-H, H1 and Talos sit 100% on one expert, AnyB/AnyC/Cassie share 2 experts evenly, and G1/OP3/NAO partly specialize.
    - A (no MI) shares experts across groups (overlap up to 0.58).
- **Sync:** compare `sync_hashes.jsonl` across same-seed runs.
- **H100 time limit:** jobs have a 2-day limit, and each H100 list holds about 70-80 estimated hours for 2 jobs. Resubmit by hand if a job ends early; the claims of its unfinished run are released on the next start.
- **Variance continuation test (2026-10-06/07; report `var_continuation/REPORT.md`):** does variance learning drive the policy collapse?
  - **Setup:** one checkpoint, V7-CH-s0 at 75k, target Go2 (no A/C/C-H checkpoint precedes a Cassie collapse; Go2 walks at 75k, then falls to 2.6 by 100k in the original run, all other robots flat). Four +20k continuations on Hessian (ids `VARb-S0..S3`), seed 0, paired minibatches and critics (identical hashes):
    - S0 original; S1 `stopgrad(e)` into the log-std head; S2 frozen shared head plus a learned offset per robot-joint (the checkpoint's log-std varies with the state, spread up to 0.31, so a pure table is not equivalent); S3 detach plus split mean and variance losses and a separate clip for the head.
    - New code: `morse/variance.py`, `morse_variance_mode`, `morse_stop_at`, `morse_ckpt_var`, `check_branch` in iql.py (branch checks written to `branch_check.json`), test 28.
  - **Result:** all four lose Go2 like the original (return 4.3, 4.0, 4.0, 7.4 against 2.4 at +20k; walking share 0.00); no other robot changes. The log-std pathway holds most of the robot-averaged gradient (0.080 against 0.034; embedding 0.044 against 0.007) but is orthogonal to the mean pathway; sigma stays 0.87-0.88; advantage weights are near 1; clipping and the MI caps never acted. The interventions did move the embedding (S1 against S0 0.36, as far as S0 moved in 20k) and S3 the core experts (2.9), without changing Go2.
  - **Next (proposed, not run):** freeze the core block from the 75k checkpoint (it holds 12.7 of the 13.0 parameter movement), 3 data-order seeds against S0.
  - **Pitfalls:** the shared Hessian quota filled and crashed the first attempt (`VAR-S*`, left untouched). The rerun writes to node-local `/tmp/var_runs/VARb` (symlink `runs/morse/VARb`); checkpoints were copied to `$R/ckpt/VARb`. Two GPU evaluations of one function can differ by 1.9e-3 through near-tied top-k routing; compare on the CPU.
- **Core swap and core freeze (2026-10-07; report `var_continuation/core_freeze/REPORT.md`):** does updating the core contribute to Go2's deterioration?
  - **Swaps (evaluation only; core = all core experts, shared experts, router, router noise head):** early core + late rest walks (Go2 51.0, 49.2 under paired noise, 97% walking), late core + early rest does not (13.9); each mismatch hurts other robots (G1 34.6, AnyB 43.7, AnyC 40.0; Talos 3.2). References reproduce the saved policies bitwise.
  - **Freeze run `VARc-F0`** (`morse_freeze_core=True`: core grads zeroed before a clip that excludes them, restored Adam first moments of the core zeroed; core hash logged in `sync_hashes.jsonl`, unchanged at every logged update and checkpoint; data, critic and value hashes identical to S0): Go2 still collapses (1.6 against 4.3 for S0 at +20k, walking share 0.00), no other robot changes; the decline is slower for 15k updates (walking 0.76 against 0.34 in the first 5k). So core updates are not necessary for the deterioration; the swaps are co-adaptation-confounded.
  - **Next (proposed, not run):** freeze core + decoder together, train only embedding, encoders, attention and std head.
  - **Infra:** outputs on node-local `/tmp/var_runs/VARc` (symlink `runs/morse/VARc`), durable copy in `$R/ckpt/VARc/VARc-F0`.
