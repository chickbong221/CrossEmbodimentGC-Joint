import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple
import json
from collections import defaultdict
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, cut_tree

from pathlib import Path
import os

def find_git_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / ".git").exists():
            return p
    return start  # fallback

def resolve_data_path(p: str) -> Path:
    path = Path(p).expanduser()
    if path.is_absolute():
        return path

    repo_root = find_git_root(Path(__file__).resolve())
    data_root = Path(os.environ.get("RLX_DATA_ROOT", repo_root))
    return (data_root / path).resolve()
class Transition(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray

class Dataset:
    def __init__(self, root: str, normalize: bool, mmap: bool = False):
        self.root = resolve_data_path(root)
        # `mmap=True` memory-maps .npy files; `mmap=False` fully loads them into RAM.
        mmap_mode = "r" if mmap else None

        self.obs = np.load(self.root / "observations.npy", mmap_mode=mmap_mode)
        self.act = np.load(self.root / "actions.npy", mmap_mode=mmap_mode)
        self.rew = np.load(self.root / "rewards.npy", mmap_mode=mmap_mode)
        self.done = np.load(self.root / "dones.npy", mmap_mode=mmap_mode)
        self.nobs = np.load(self.root / "next_observations.npy", mmap_mode=mmap_mode)
        with open(self.root / "metadata.json", 'r') as f:
            metadata = json.load(f)
        self.n_samples = int(metadata["n_samples"])
        self.n_envs = int(metadata["n_envs"])
        if normalize:
            self.mu = self.obs.mean(axis=0, keepdims=True)
            self.sig = self.obs.std(axis=0, keepdims=True) + 1e-5
        else:
            self.mu = self.sig = None

    def __len__(self):
        return self.n_samples

    def _empty_batch(self, env_id):
        """Create a length-0 Transition preserving env_id shape/dtype."""
        return Transition(
            observations=self.obs[:0, env_id, :],
            actions=self.act[:0, env_id, :],
            rewards=self.rew[:0, env_id],
            next_observations=self.nobs[:0, env_id, :],
            dones=self.done[:0, env_id],
        )

    def sample(self, key, batch_sizes):
        env_batches = []
        for env_type_id in range(len(batch_sizes)):
            batch_size = batch_sizes[env_type_id]
            key, subkey = jax.random.split(key)
            if batch_size > 0:
                indices = jax.random.choice(subkey, self.n_samples, (batch_size,), replace=True)
                indices = np.array(indices)
                env_batch = Transition(
                    observations=self.obs[indices, env_type_id, :],
                    actions=self.act[indices, env_type_id, :],
                    rewards=self.rew[indices, env_type_id],
                    next_observations=self.nobs[indices, env_type_id, :],
                    dones=self.done[indices, env_type_id],
                )
            else:
                env_batch = self._empty_batch(env_type_id)
            env_batches.append(env_batch)

        return env_batches, key

def create_robot_groups(embodiment_distance_path, robot_names, M, method='ward'):
    embodiment_distance_path = resolve_data_path(embodiment_distance_path)
    etas = np.load(embodiment_distance_path)
    X = etas.copy()
    np.fill_diagonal(X, 0)
    dists = squareform(X)
    Z = linkage(dists, method=method)
    labels = cut_tree(Z, n_clusters=M).flatten()
    groups = defaultdict(list)
    for name, lab in zip(robot_names, labels):
        groups[lab].append(name)
    return {f"group{i+1}": groups[i] for i in range(M)}
