"""Branch a run from a checkpoint of another (Test 2: interventions from a common starting policy).

Every leaf of the new training state (parameters and optimizer state) is taken from the checkpoint. A leaf
whose checkpointed version has a leading expert axis of 1 where the new one has K experts (a single shared
block turned into K identical experts) is tiled along that axis; a leaf that the checkpoint does not have
(a router, or calibration parameters) keeps its initial value. With one-hot routing to one expert per group,
every expert starts as an exact copy of the shared block, so the branched policy starts from the checkpoint's.
"""
import numpy as np
from flax import serialization


def _fill(new, old, path, tiled, kept):
    if isinstance(new, dict):
        return {k: _fill(v, old.get(k) if isinstance(old, dict) else None, f"{path}/{k}", tiled, kept)
                for k, v in new.items()}
    if old is None:
        kept.append(path)
        return new
    old = np.asarray(old)
    shape = np.shape(new)
    if old.shape == shape:
        return old.astype(np.asarray(new).dtype)
    if old.ndim == len(shape) and old.ndim > 0 and old.shape[0] == 1 and old.shape[1:] == shape[1:]:
        tiled.append(path)
        return np.broadcast_to(old, shape).astype(np.asarray(new).dtype).copy()
    raise ValueError(f"branch: {path} has shape {old.shape} in the checkpoint and {shape} in the new state")


def branch_state(new_state, checkpointed):
    """``new_state`` (a TrainState) with every leaf taken from ``checkpointed`` (its restored state dict)."""
    tiled, kept = [], []
    filled = _fill(serialization.to_state_dict(new_state), checkpointed, "", tiled, kept)
    return serialization.from_state_dict(new_state, filled), tiled, kept
