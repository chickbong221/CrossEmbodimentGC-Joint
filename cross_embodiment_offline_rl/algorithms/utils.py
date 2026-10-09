import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple
import hashlib
import json
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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

# Robot order of the dataset's robot axis (the 16-robot suite, as in the environment's default train_robot_types).
DATASET_ROBOTS = ("unitree_a1", "unitree_go1", "unitree_go2", "anymal_b", "anymal_c", "barkour_v0", "barkour_vb", "badger",
                  "bittle", "unitree_h1", "unitree_g1", "talos", "robotis_op3", "nao_v5", "cassie", "hexapod")


def dataset_columns(train_robot_types, n_envs):
    """Dataset columns of the training robots, or None when they are the dataset's robots in its own order."""
    robots = tuple(str(r) for r in train_robot_types)
    if len(robots) == n_envs and (n_envs != len(DATASET_ROBOTS) or robots == DATASET_ROBOTS):
        return None
    return [DATASET_ROBOTS.index(r) for r in robots]


class Dataset:
    def __init__(self, root: str, normalize: bool, mmap: bool = False, rows=None, columns=None):
        """``rows`` = (lo, hi) keeps the dataset rows [lo, hi) of every robot; ``columns`` keeps these robots of the
        robot axis, in this order. A selection is copied into RAM from a memory map (only the selected data is read)."""
        self.root = resolve_data_path(root)
        with open(self.root / "metadata.json", 'r') as f:
            metadata = json.load(f)
        self.n_samples = int(metadata["n_samples"])
        self.n_envs = int(metadata["n_envs"])
        if rows is not None and not 0 <= rows[0] < rows[1] <= self.n_samples:
            raise ValueError(f"dataset rows {rows} outside [0, {self.n_samples}]")
        selected = rows is not None or columns is not None
        # `mmap=True` memory-maps .npy files; `mmap=False` fully loads them into RAM.
        mmap_mode = "r" if (mmap or selected) else None

        def load(name):
            a = np.load(self.root / f"{name}.npy", mmap_mode=mmap_mode)
            if rows is not None:
                a = a[rows[0]:rows[1]]
            if columns is not None:
                a = a[:, list(columns)]
            return np.ascontiguousarray(a) if selected else a

        self.obs = load("observations")
        self.act = load("actions")
        self.rew = load("rewards")
        self.done = load("dones")
        self.nobs = load("next_observations")
        if rows is not None:
            self.n_samples = rows[1] - rows[0]
        if columns is not None:
            self.n_envs = len(columns)
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

    def gather(self, indices, env_id):
        """Transitions of robot ``env_id`` at the given sample indices (host arrays, as in ``sample``)."""
        if len(indices) == 0:
            return self._empty_batch(env_id)
        # Row indices into the (sample * robot) axis: np.take copies whole contiguous rows and
        # releases the GIL, so several robots can be gathered in parallel threads.
        rows = np.asarray(indices, dtype=np.int64) * self.obs.shape[1] + env_id
        take = lambda a: np.take(a.reshape((-1,) + a.shape[2:]), rows, axis=0)
        return Transition(observations=take(self.obs), actions=take(self.act), rewards=take(self.rew),
                          next_observations=take(self.nobs), dones=take(self.done))


class BatchPrefetcher:
    """Minibatches sampled and copied to the device in background threads, ahead of the updates.

    Each robot's indices are drawn uniformly with replacement, as in ``Dataset.sample``, but from a
    NumPy generator seeded with ``seed`` instead of the training key, so sampling never waits for the
    device and the host-to-device copies overlap with the previous updates.
    """

    def __init__(self, dataset, batch_sizes, seed, depth=3, workers=4):
        self.dataset = dataset
        self.batch_sizes = [int(b) for b in batch_sizes]
        self.rng = np.random.default_rng(seed)
        self.wait_s = 0.0                      # time the training loop last waited for a batch
        self.checksum = 0                      # sum of the sample indices of every batch taken so far (runs with the
                                               # same seed and data selection must agree)
        self.sha1 = hashlib.sha1()             # content hash of every index array taken so far, in order
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._produce, name="batch-prefetch", daemon=True)
        self._thread.start()

    def _load(self, env_id, indices):
        return jax.device_put(self.dataset.gather(indices, env_id))

    def _produce(self):
        try:
            while not self._stop.is_set():
                indices = [self.rng.integers(0, len(self.dataset), size=b) for b in self.batch_sizes]
                batches = tuple(self._pool.map(self._load, range(len(indices)), indices))
                total = int(sum(int(i.sum()) for i in indices))
                blob = b"".join(np.ascontiguousarray(i, dtype=np.int64).tobytes() for i in indices)
                while not self._stop.is_set():
                    try:
                        self._queue.put((total, blob, batches), timeout=1.0)
                        break
                    except queue.Full:
                        pass
        except BaseException as error:        # surfaced in the training loop by next()
            self._queue.put(error)

    def next(self):
        start = time.perf_counter()
        item = self._queue.get()
        self.wait_s = time.perf_counter() - start
        if isinstance(item, BaseException):
            raise RuntimeError("batch prefetching failed") from item
        total, blob, batches = item
        self.checksum += total
        self.sha1.update(blob)
        return batches

    def close(self):
        self._stop.set()
        self._pool.shutdown(wait=False, cancel_futures=True)


def robot_batch_sizes(robot_loss_weights, batch_size, nr_env_types):
    """Per-robot batch sizes of the training loops (proportional to the robot loss weights).

    Same float32 arithmetic as the original per-update computation, done once on the host.
    """
    weights = np.asarray(robot_loss_weights, dtype=np.float32)
    total = np.float32(0.0)
    for w in weights:
        total = np.float32(total + w)
    return [int(np.float32(batch_size * nr_env_types) * np.float32(w / total)) for w in weights]

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
