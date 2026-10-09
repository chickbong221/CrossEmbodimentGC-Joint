"""Variance-pathway interventions of the policy's action standard deviation (config.morse_variance_mode).

The policy's log-std is one linear unit on the decoder's descriptor embedding e (a function of the joint's static
description only), so every robot's variance gradient flows into the embedding that the mean decoder also reads.
Modes, all with the same forward pass at the start of a branched run:
  original   the Gaussian IQL (AWR) loss through the shared head.
  detach     the same loss, with e stop-gradiented into the log-std head: variance gradients reach the head only.
  private    the shared head's output is frozen (a fixed baseline, no gradient into the head or the embedding) and a
             learned offset per (robot, joint), initialized to 0, is added to it; the Gaussian loss is unchanged. The
             checkpoint's log-std depends on the state through the description (episode-level), so a pure table of
             per-joint values could not reproduce it; baseline plus offset does exactly.
  protected  detach, plus the mean and the variance fitted by separate losses (Stirn et al. 2023, Eq. 5; here with
             the advantage weights w): L_mu = E[w/2 mean_j (a - mu)^2] reaches the mean network only and
             L_sigma = E[w mean_j (rho + (a - sg(mu))^2 / (2 e^{2 rho}))] the log-std head only; the head's gradient is
             clipped on its own, so it cannot rescale the mean's through the global-norm clip.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax

MODES = ("original", "detach", "private", "protected")
EMBEDDING = ("Dense_4", "LayerNorm_2", "Dense_5")


def _keys(path):
    return tuple(getattr(k, "key", str(k)) for k in path)


def is_std_head(path):
    """The shared log-std head (a parameter of the policy's params tree, by its path)."""
    k = _keys(path)
    return len(k) > 1 and k[0] == "params" and k[1] == "Dense_6"


def is_std_param(path):
    """Any parameter that sets the log-std: the shared head or the private table."""
    k = _keys(path)
    return len(k) > 1 and k[0] == "params" and k[1] in ("Dense_6", "logstd_offset")


def is_core(path):
    """Any parameter of the core block: core experts, shared (global and group) experts, router and router noise head."""
    k = _keys(path)
    return len(k) > 1 and k[0] == "params" and k[1] == "core"


def is_embedding(path):
    k = _keys(path)
    return len(k) > 1 and k[0] == "params" and k[1] in EMBEDDING


def group_norm(tree, select):
    """Global norm of the leaves of ``tree`` whose path satisfies ``select``."""
    leaves = [g for p, g in jax.tree_util.tree_flatten_with_path(tree)[0] if select(p)]
    return optax.global_norm(leaves) if leaves else jnp.float32(0.0)


def protected_loss(w, mean, logstd, act):
    """Protected-mean loss of one robot's batch: w (B, 1) advantage weights, mean/logstd/act (B, joints)."""
    err2 = jnp.square(act - mean)
    mean_term = 0.5 * (w * err2).mean()
    var_term = (w * (logstd + 0.5 * jnp.square(act - jax.lax.stop_gradient(mean)) / jnp.exp(2.0 * logstd))).mean()
    return mean_term + var_term


def grouped_clip_by_global_norm(max_norm, in_group):
    """optax.clip_by_global_norm applied separately to the leaves selected by ``in_group(path)`` and to the others.
    Its state is empty, like the plain clip's, so a checkpoint's optimizer state restores into it unchanged."""
    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        norms = {sel: group_norm(updates, lambda p, sel=sel: bool(in_group(p)) == sel) for sel in (True, False)}

        def clip(path, g):
            n = norms[bool(in_group(path))]
            return jnp.where(n < max_norm, g, (g / n.astype(g.dtype)) * max_norm)
        return jax.tree_util.tree_map_with_path(clip, updates), state
    return optax.GradientTransformation(init_fn, update_fn)


def frozen_clip_by_global_norm(max_norm, is_frozen):
    """optax.clip_by_global_norm for a run with frozen parameters. The gradient of every leaf selected by ``is_frozen(path)``
    is set to zero before the norm is taken, so frozen leaves neither receive an update nor enter the clip's norm (they
    cannot rescale the trainable gradient); the others are clipped as usual. The state is empty, like the plain clip's."""
    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        zeroed = jax.tree_util.tree_map_with_path(lambda p, g: jnp.zeros_like(g) if is_frozen(p) else g, updates)
        norm = optax.global_norm(zeroed)
        scaled = jax.tree_util.tree_map(lambda g: jnp.where(norm < max_norm, g, (g / norm.astype(g.dtype)) * max_norm), zeroed)
        return scaled, state
    return optax.GradientTransformation(init_fn, update_fn)


def zero_first_moments(opt_state, select):
    """The optimizer state with the Adam first moments (``mu``) of the parameters whose path keys satisfy ``select(keys)``
    set to zero. With a zero gradient and a zero first moment the Adam update is exactly zero, so the parameter stays
    bitwise the same (a zero gradient alone would still move it by the restored momentum)."""
    def f(path, x):
        names = [getattr(k, "name", None) or getattr(k, "key", None) for k in path]
        return jnp.zeros_like(x) if "mu" in names and select(names) else x
    return jax.tree_util.tree_map_with_path(f, opt_state)


def logstd_state_spread(policy, params, inputs):
    """Largest difference over states of a policy's log-std per robot, (robots,): how far a state-independent
    per-joint value is from reproducing it. ``inputs[r]`` lists policy inputs (joint and foot descriptions and states,
    general state) of robot r from different minibatches."""
    out = []
    for r, batches in enumerate(inputs):
        vals = np.concatenate([np.asarray(policy.apply({"params": params}, *b, robot_id=r)[1]) for b in batches], axis=0)
        out.append(float((vals.max(axis=0) - vals.min(axis=0)).max()))
    return np.asarray(out)
