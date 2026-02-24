# Experiment Notes

These arguments are important when running experiments.

## 1) W&B entity (batch update)

- Save your entity in `experiments/wandb_entity.txt` (first non-comment line).
- Replace the placeholder (e.g., `your-wandb-entity`) with your own W&B account/team entity before running experiments.
- Apply it to all `experiment_*.sh` files:
  - `bash experiments/apply_wandb_entity.sh`

## 2) `algorithm.dataset_path`

- Set `--algorithm.dataset_path` to the dataset path you downloaded under `data/offline_data/`.
- Example:
  - `--algorithm.dataset_path='data/offline_data/all_robots_expert_forward_1m'`

## 3) `environment.test_command_path` mapping

- If the dataset is **forward**:
  - `--environment.test_command_path='command_forward.txt'`
- If the dataset is **backward**:
  - `--environment.test_command_path='command_backward.txt'`

## 4) Meaning of `algorithm.M` (EG algorithms)

- Applies to: `bc_eg`, `iql_eg`, `td3bc_eg`
- `--algorithm.M` is the number of **Embodiment Grouping** groups.
- When `M=1`, each method becomes equivalent to the non-EG (standard) version.

## 5) Embodiment distance matrix (reproduce)

- A precomputed matrix is already available at `cross_embodiment_offline_rl/data/meta/embodiment_distance.npy`.
- To regenerate it, run:

```bash
cd cross_embodiment_offline_rl/experiments
python create_distance_matrix.py
```
