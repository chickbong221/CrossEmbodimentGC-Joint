# CrossEmbodimentGC-Joint

## Setup

### 1. Environment

Default installation for a Linux system with an NVIDIA GPU. For other configurations, see the RL-X
[documentation](https://nico-bohlinger.github.io/RL-X/#detailed-installation-guide).

```bash
conda create -n cross_offline_rl python=3.11.4 -y
conda activate cross_offline_rl
git clone https://github.com/haruki-abe/RL-X.git
cd RL-X
git checkout rlx-cross-v1.0.0
pip install -e .[all] --config-settings editable_mode=compat
pip uninstall $(pip freeze | grep -i '\-cu12' | cut -d '=' -f 1) -y
pip install "torch==2.2.1" --index-url https://download.pytorch.org/whl/cu118 --upgrade
pip install -U "jax[cuda12_pip]==0.4.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
cd ..
```

### 2. Project

```bash
git clone https://github.com/chickbong221/CrossEmbodimentGC-Joint.git
cd CrossEmbodimentGC-Joint
pip install -e .
```

### 3. Dataset

One dataset is used (about 80 GB of disk; training holds it in RAM, about 110 GB, or pass
`--algorithm.dataset_mmap=True` to memory-map it). If `hf` is missing, install `huggingface_hub` first.

```bash
cd data
hf download haruki-abe/cross_embodiment_offline_rl \
  --repo-type dataset \
  --include "offline_data/all_robots_suboptimal_70_forward_1m/**" \
  --local-dir .
cd ..
```

### 4. Weights & Biases (optional)

```bash
# put your entity on the first non-comment line, then apply it to the experiment scripts
vim experiments/wandb_entity.txt
bash experiments/apply_wandb_entity.sh
```

To train without W&B, add `--runner.track_wandb=False` to the commands below (TensorBoard logs are always written).

## Run

All runs use IQL as the base learner, the dataset above, and a budget of 50k updates
(`--algorithm.nr_offline_updates=50000`). Start them from `experiments/`; add `--environment.seed=N` to change the
seed (default 0) and `--runner.exp_name=NAME` to rename the run.

```bash
cd experiments
```

### Mutual-information loss (joint-type groups, expert width 1/2)

```bash
bash experiment_iql_morse.sh \
  --algorithm.nr_offline_updates=50000 \
  --runner.exp_name="MI-joint" \
  --algorithm.morse_arch=v8 \
  --algorithm.actor_steps_per_update=4 \
  --algorithm.morse_router_noise=True \
  --algorithm.morse_router_noise_floor=0.01 \
  --algorithm.morse_router_init=0.01 \
  --algorithm.morse_router_hidden=64 \
  --algorithm.morse_router_inputs=both \
  --algorithm.morse_v8_experts=16 \
  --algorithm.morse_v8_balance=0.01 \
  --algorithm.morse_v8_cap=0.0 \
  --algorithm.morse_conflict_mode=none \
  --algorithm.morse_lambda=0.0 \
  --algorithm.morse_eta=0.0 \
  --algorithm.morse_mi_weight=0.0 \
  --algorithm.morse_compat_route=off \
  --algorithm.morse_snapshot_frequency=0 \
  --algorithm.morse_v8_direction=mi_joint \
  --algorithm.morse_v8_mi_weight=0.0005 \
  --algorithm.morse_v8_joint_groups=v8_joint_groups.json \
  --algorithm.morse_v8_group_k=8 \
  --algorithm.morse_v8_width=0.5 \
  --algorithm.morse_warmup=10000 \
  --algorithm.morse_ramp=5000
```

### Gradient-conflict (compatibility) loss (expert width 1/4)

```bash
bash experiment_iql_morse.sh \
  --algorithm.nr_offline_updates=50000 \
  --runner.exp_name="Gradient-conflict" \
  --algorithm.morse_arch=v8 \
  --algorithm.actor_steps_per_update=4 \
  --algorithm.morse_router_noise=True \
  --algorithm.morse_router_noise_floor=0.01 \
  --algorithm.morse_router_init=0.01 \
  --algorithm.morse_router_hidden=64 \
  --algorithm.morse_router_inputs=both \
  --algorithm.morse_v8_experts=16 \
  --algorithm.morse_v8_balance=0.01 \
  --algorithm.morse_v8_cap=0.0 \
  --algorithm.morse_conflict_mode=none \
  --algorithm.morse_lambda=0.0 \
  --algorithm.morse_eta=0.0 \
  --algorithm.morse_mi_weight=0.0 \
  --algorithm.morse_compat_route=off \
  --algorithm.morse_snapshot_frequency=0 \
  --algorithm.morse_v8_direction=compat \
  --algorithm.morse_v8_compat_lambda=0.005 \
  --algorithm.morse_compat_tau=0.05 \
  --algorithm.morse_compat_core_transitions=64 \
  --algorithm.morse_compat_decoder_transitions=32 \
  --algorithm.morse_compat_min_support=4 \
  --algorithm.morse_v8_width=0.25 \
  --algorithm.morse_warmup=10000 \
  --algorithm.morse_ramp=5000
```

### Original EG (Embodiment Grouping, M=4)

The baseline scripts have their arguments written in, with a budget of 100k updates; the `sed` sets it to 50k. Change
the seed by editing `--environment.seed` in the script.

```bash
sed 's/--algorithm.nr_offline_updates=100000/--algorithm.nr_offline_updates=50000/' experiment_iql_eg.sh | bash
```

### Original URMA (no grouping, M=1)

```bash
sed 's/--algorithm.nr_offline_updates=100000/--algorithm.nr_offline_updates=50000/' experiment_iql.sh | bash
```

Logs and checkpoints are written under `experiments/runs/`.
