"""Gradient conflict between sets of samples, on frozen checkpoints (analysis, no training).

Conflict between two gradients is measured as in the cross-embodiment offline-RL paper whose Embodiment Grouping (EG)
is a baseline here (and as in PCGrad): the cosine similarity of the two actor gradients, taken of the IQL (AWR) policy
loss over all shared policy parameters, with conflict meaning a negative cosine. The headline statistic of that paper is
the fraction of negative pairwise cosines; this module also keeps the Gram matrices, so any grouping of the samples can
be scored afterwards (experiments/analysis/conflict_report.py) without recomputing a gradient.

For every checkpoint listed in ``morse_ckpt_analysis`` (any number, any run) it writes ``<morse_ckpt_out>_conflict_<name>.npz``
with two parts, both for two loss variants (``iql``: the AWR loss with the checkpoint's own advantage weights, the
actor loss that is trained; ``bc``: the same likelihood without weights) and for the parameter groups ``enc`` (joint and
foot state encoders and attention), ``core``, ``dec`` (action decoder and descriptor embedding), ``std`` (log-std
head) and ``all``:

1. Joint units. ``n_batches`` fixed minibatches (the training batch of every robot, drawn from the whole dataset with
   a fixed seed, identical for every checkpoint). For every unit a = (robot r, joint j) the gradient g_a of its term of
   the robot's policy loss (the robot's loss is the sum of its joint terms). ``unit_gram[k]`` is the Gram matrix of
   the unit gradients of minibatch k; ``half_cross``, ``half_same1``, ``half_same2`` are Gram matrices of the mean
   gradients of the first and second half of the minibatches (cross: half 1 against half 2), so that products of
   gradients from different data (whose expectation has no sampling noise) are available. Robot gradients, EG group
   gradients and joint-group gradients are sums of unit gradients, so every cosine between them follows from these
   matrices (g_A . g_B = m_A^T Gram m_B for the membership vectors m).
2. Trajectories. For each robot, up to ``morse_conflict_episodes`` episodes of each class: ``success`` (complete,
   full length without termination, return at least ``morse_coord_min_return`` of the best such return: the
   reference episodes of the coordination tests), ``fall`` (ended by the termination condition) and ``low`` (full
   length, below that return). The dataset is 30% expert and 70% suboptimal data, so the two failure classes are
   plentiful. For every episode the gradient of the mean loss over its transitions; ``ep_gram_<robot>`` is the Gram
   matrix of the episode gradients, ``ep_class_<robot>`` their class (0 success, 1 fall, 2 low), the episodes
   being in random order so that the first and the second half of each class are two independent samples.
"""
import os

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from cross_embodiment_offline_rl.algorithms.morse.ckpt_analysis import load_checkpoint, param_group
from cross_embodiment_offline_rl.algorithms.morse.coordination import episode_table, load_end_flags, reference_mask

HI = jax.lax.Precision.HIGHEST
CHUNK = 128          # samples per Jacobian chunk of the per-unit gradients
GROUPS = ("enc", "core", "dec", "std", "all")
VARIANTS = ("iql", "bc")
CLASSES = ("success", "fall", "low")
ENCODER_ROLES = ("joint_state_encoder", "foot_state_encoder", "joint_attention", "foot_attention")
# Parameter names of the expert policy (morse_arch=v8, morse/v8.py) and their group.
V8_PREFIXES = (("enc_", "enc"), ("foot_", "enc"), ("joint_", "enc"), ("core_", "core"), ("dec_moe", "dec"), ("std_", "std"))


def _whole(size):
    return [(0, size)]


def _enc_moe(keys, size):
    """The joint-encoder layer that the expert policy turns into an expert layer: the hidden 256 -> 256 layer of the
    dense joint-state encoder, or the experts of the expert policy's enc_moe (its router excluded)."""
    hit = len(keys) > 1 and ((keys[0] == "joint_encoder" and keys[1] == "dense_1")
                             or (keys[0] == "enc_moe" and keys[1] == "experts"))
    return _whole(size) if hit else []


def _dec_exp(keys, size):
    """The decoder that the expert policy replaces by a bank of experts: the dense decoder and its description
    embedding, or the experts of the expert policy's dec_moe (its router excluded)."""
    if len(keys) > 1 and keys[0] == "dec_moe":
        return _whole(size) if keys[1] == "experts" else []
    return _whole(size) if param_group(keys) in ("decoder_experts", "desc_embedding") else []


def _expert_part(layer, part):
    """The shared expert (index 0) or the routed experts (1..K-1) of an expert layer of the expert policy: every expert
    parameter is (K, ...), so each expert is a contiguous block of a leaf."""
    def member(keys, size, shape0=None):
        if len(keys) < 2 or keys[0] != layer or keys[1] != "experts" or not shape0:
            return []
        block = size // shape0
        return [(0, block)] if part == "shared" else [(block, size)]
    return member


# Extra parameter groups, reported after GROUPS (they overlap them). Each maps (keys, size[, K]) to ranges of a leaf.
SUBGROUPS = {"enc_moe": _enc_moe, "dec_exp": _dec_exp}
EXPERT_PARTS = {f"{tag}_{part}": _expert_part(layer, part)
                for tag, layer in (("enc", "enc_moe"), ("core1", "core_moe1"), ("core2", "core_moe2"), ("dec", "dec_moe"))
                for part in ("shared", "routed")}
SUBGROUPS.update(EXPERT_PARTS)


def analysis_group(path):
    """Group of a policy parameter from its path in params['params']."""
    role = param_group(path)
    if role == "other":
        for prefix, group in V8_PREFIXES:
            if path[0].startswith(prefix):
                return group
    if role in ENCODER_ROLES:
        return "enc"
    if role == "core_experts":
        return "core"
    if role == "std_head":
        return "std"
    return "dec"


class ConflictAnalysis:
    def __init__(self, policy, split_fn, awr_terms, robot_names, batch_sizes, loss_weights, a, seed, horizon,
                 route_policy=None, param_groups=None):
        self.policy, self.split_fn, self.awr_terms = policy, split_fn, awr_terms
        # Policies with auto-named layers (EG): the group (one of GROUPS) of every top-level parameter name; the
        # decoder group then also makes dec_exp.
        self.param_groups = dict(param_groups) if param_groups else None
        # Expert policy (morse/v8.py, not jitted): also record, per routing unit, the routing probabilities of every
        # expert layer and how often each routed expert is selected, averaged over the analysis minibatches.
        self.route_policy = route_policy
        self.robot_names = [str(n) for n in robot_names]
        self.batch_sizes = [int(b) for b in batch_sizes]
        self.loss_weights = [float(w) for w in loss_weights]
        self.n_batches = int(a.morse_conflict_batches)
        if self.n_batches < 4 or self.n_batches % 2:
            raise ValueError(f"morse_conflict_batches must be an even number >= 4, got {self.n_batches}")
        self.n_episodes = int(a.morse_conflict_episodes)
        self.horizon = int(horizon)
        self.min_return = float(a.morse_coord_min_return)
        self.rows = tuple(int(v) for v in str(a.morse_data_rows).split(",")) if getattr(a, "morse_data_rows", "") else None
        self.out_base = os.path.splitext(a.morse_ckpt_out)[0] if a.morse_ckpt_out else None
        self.seed = int(seed)
        self._fns = {}
        self._unravel = None
        self._slices = None
        self._gram_jit = jax.jit(self._gram)

    # ---- parameter bookkeeping -------------------------------------------------------------------------------

    def _bind(self, variables):
        """Flat parameter vector of a checkpoint; the first one fixes the tree structure and the parameter groups."""
        flat, unravel = ravel_pytree(variables)
        if self._unravel is None:
            self._unravel = unravel
            leaves = jax.tree_util.tree_flatten_with_path(variables)[0]
            slices, sub, off = [], [], 0
            for path, leaf in leaves:
                keys = [getattr(k, "key", str(k)) for k in path]
                keys = keys[1:] if keys and keys[0] == "params" else keys
                size = int(np.prod(leaf.shape))
                group = self.param_groups[keys[0]] if self.param_groups is not None else analysis_group(keys)
                slices.append((off, off + size, group))
                for name, member in SUBGROUPS.items():
                    if name == "dec_exp" and self.param_groups is not None:
                        ranges = _whole(size) if group == "dec" else []
                    elif name in EXPERT_PARTS:
                        ranges = member(keys, size, leaf.shape[0] if leaf.ndim else None)
                    else:
                        ranges = member(keys, size)
                    sub.extend((off + lo, off + hi, name) for lo, hi in ranges)
                off += size
            assert off == flat.shape[0]
            self._slices, self._sub = slices, sub
            self.group_sizes = {g: sum(hi - lo for lo, hi, gg in slices if gg == g) for g in GROUPS[:-1]}
            self.group_sizes["all"] = int(flat.shape[0])
            self.group_sizes.update({s: sum(hi - lo for lo, hi, ss in sub if ss == s) for s in SUBGROUPS})
        elif flat.shape[0] != self._slices[-1][1]:
            raise ValueError("the checkpoints have different parameter counts")
        return flat

    def _gram(self, A, B):
        """Per parameter group, the Gram matrices A_v B_v^T of (variants, N, P) and (variants, M, P) gradient arrays."""
        parts = {g: 0.0 for g in GROUPS[:-1]}
        for lo, hi, g in self._slices:
            parts[g] = parts[g] + jnp.einsum("vnp,vmp->vnm", A[:, :, lo:hi], B[:, :, lo:hi], precision=HI)
        subs = {s: 0.0 for s in SUBGROUPS}
        for lo, hi, s in self._sub:
            subs[s] = subs[s] + jnp.einsum("vnp,vmp->vnm", A[:, :, lo:hi], B[:, :, lo:hi], precision=HI)
        zero = jnp.zeros((A.shape[0], A.shape[1], B.shape[1]), A.dtype)
        parts = {g: zero if isinstance(v, float) else v for g, v in parts.items()}
        return jnp.stack([parts[g] for g in GROUPS[:-1]] + [sum(parts.values())]
                         + [subs[s] if not isinstance(subs[s], float) else zero for s in SUBGROUPS])

    # ---- joint units ----------------------------------------------------------------------------------------------

    def _unit_fn(self, r):
        if ("unit", r) in self._fns:
            return self._fns[("unit", r)]
        unravel, scale_w = self._unravel, self.loss_weights[r]

        def losses(params, extra, obs, act):
            out = self.policy.apply(params, *self.split_fn(obs, r), robot_id=r)
            exp_a, lp = self.awr_terms(out, obs, act, extra, r)
            return jnp.stack([-(exp_a * lp).mean(axis=0), -lp.mean(axis=0)]) * (scale_w / lp.shape[1])   # (2, n_joints)

        def jac(params, extra, obs, act):
            """(2, n_j, P) Jacobian in the order of the flat vector: taken with respect to the parameter tree and
            concatenated once (differentiating through unravel pads every leaf to P and runs out of memory)."""
            tree = jax.jacrev(losses)(params, extra, obs, act)
            return jnp.concatenate([x.reshape(x.shape[:2] + (-1,)) for x in jax.tree_util.tree_leaves(tree)], axis=-1)

        def fn(flat, extra, obs, act):
            # The losses are means over the samples with per-sample weights, so the Jacobian of the batch is the mean of
            # the Jacobians of equal chunks: chunks of at most CHUNK samples bound the memory of the expert policy.
            params = unravel(flat)
            n = obs.shape[0]
            c = max(d for d in range(1, min(n, CHUNK) + 1) if n % d == 0)
            oc, ac = obs.reshape((n // c, c) + obs.shape[1:]), act.reshape((n // c, c) + act.shape[1:])

            def step(carry, xs):
                o, a_ = xs
                return (carry[0] + losses(params, extra, o, a_), carry[1] + jac(params, extra, o, a_)), None

            l, j = losses(params, extra, oc[0], ac[0]), jac(params, extra, oc[0], ac[0])
            if n // c > 1:
                (l, j), _ = jax.lax.scan(step, (l, j), (oc[1:], ac[1:]))
            return l / (n // c), j / (n // c)                                                        # (2, n_j), (2, n_j, P)

        self._fns[("unit", r)] = jax.jit(fn)
        return self._fns[("unit", r)]

    def _route_fn(self, r):
        """Routing record of robot r on a batch: per expert layer, the mean routing probabilities and the selection
        frequency of every routed expert, per joint (joint-level layers) or per robot (body-level layers)."""
        if ("route", r) in self._fns:
            return self._fns[("route", r)]
        from cross_embodiment_offline_rl.algorithms.morse.v8 import V8_ROUTING
        unravel, pol = self._unravel, self.route_policy

        def fn(flat, obs):
            _, state = pol.apply(unravel(flat), *self.split_fn(obs, r), robot_id=r, mutable=[V8_ROUTING])
            rec = state[V8_ROUTING]
            return {layer: (rec[layer]["p"][0].mean(axis=0), (rec[layer]["w"][0] > 0).astype(jnp.float32).mean(axis=0))
                    for layer in rec}

        self._fns[("route", r)] = jax.jit(fn)
        return self._fns[("route", r)]

    def _unit_part(self, flat, extra, dataset):
        K = self.n_batches
        half = [None, None]
        grams, unit_loss = [], 0.0
        routes = {}
        for k in range(K):
            env_batches, _ = dataset.sample(jax.random.PRNGKey(self.seed * 1_000_003 + 271_828 + k), self.batch_sizes)
            losses, jacs = [], []
            for r, b in enumerate(env_batches):
                l, j = self._unit_fn(r)(flat, extra, jnp.asarray(b.observations), jnp.asarray(b.actions))
                losses.append(l)
                jacs.append(j)
                if self.route_policy is not None:
                    for layer, (p, sel) in self._route_fn(r)(flat, jnp.asarray(b.observations)).items():
                        acc = routes.setdefault(layer, [[None, None] for _ in self.robot_names])[r]
                        acc[0] = np.asarray(p) / K if acc[0] is None else acc[0] + np.asarray(p) / K
                        acc[1] = np.asarray(sel) / K if acc[1] is None else acc[1] + np.asarray(sel) / K
            if k == 0:
                n_joints = [int(j.shape[1]) for j in jacs]
            J = jnp.concatenate(jacs, axis=1)                      # (2, U, P)
            del jacs
            unit_loss = unit_loss + np.asarray(jnp.concatenate(losses, axis=1)) / K
            grams.append(np.asarray(self._gram_jit(J, J)))
            # The half sums stay on the host, so the device holds the gradients of one minibatch only.
            h = int(k >= K // 2)
            if half[h] is None:
                half[h] = np.array(J)
            else:
                half[h] += np.asarray(J)
            del J
        m1, m2 = jnp.asarray(half[0] / (K // 2)), jnp.asarray(half[1] / (K // 2))
        del half
        out = {"unit_gram": np.stack(grams).astype(np.float32),            # (K, groups, 2, U, U)
               "half_cross": np.asarray(self._gram_jit(m1, m2)), "half_same1": np.asarray(self._gram_jit(m1, m1)),
               "half_same2": np.asarray(self._gram_jit(m2, m2)), "unit_loss": unit_loss.astype(np.float32),
               "unit_robot": np.concatenate([np.full(n, r) for r, n in enumerate(n_joints)]),
               "unit_joint": np.concatenate([np.arange(n) for n in n_joints])}
        for layer, per_robot in routes.items():
            # joint-level layers: (n_joints, n_routed) per robot -> (U, n_routed); body-level: (n_routed,) -> (R, n_routed)
            stack = np.concatenate if per_robot[0][0].ndim == 2 else np.stack
            out[f"route_p_{layer}"] = stack([p for p, _ in per_robot]).astype(np.float32)
            out[f"route_sel_{layer}"] = stack([s for _, s in per_robot]).astype(np.float32)
        return out

    # ---- trajectories -------------------------------------------------------------------------------------------------

    def _episode_fn(self, r):
        if ("episode", r) in self._fns:
            return self._fns[("episode", r)]
        unravel, scale_w = self._unravel, self.loss_weights[r]

        def total(flat, extra, obs, act, mask):
            out = self.policy.apply(unravel(flat), *self.split_fn(obs, r), robot_id=r)
            exp_a, lp = self.awr_terms(out, obs, act, extra, r)
            per_t = jnp.stack([-(exp_a[:, 0] * lp.sum(axis=-1)), -lp.sum(axis=-1)])         # (2, L)
            return (per_t * (mask / jnp.maximum(mask.sum(), 1.0))).sum(axis=-1) * (scale_w / lp.shape[1])

        self._fns[("episode", r)] = jax.jit(jax.jacrev(total))                              # (2, P)
        return self._fns[("episode", r)]

    def _episode_part(self, flat, extra, dataset, terminated, truncated):
        out = {}
        for r, name in enumerate(self.robot_names):
            table = episode_table(np.asarray(dataset.done[:, r]).reshape(-1) > 0.5, terminated[:, r], truncated[:, r],
                                  np.asarray(dataset.rew[:, r]).reshape(-1))
            complete = table["ep_complete"]
            full = complete & table["ep_truncated"] & ~table["ep_terminated"] & (table["ep_len"] == self.horizon)
            ref = reference_mask(table, self.horizon, self.min_return)
            masks = (ref, complete & table["ep_terminated"], full & ~ref)
            rng = np.random.default_rng(self.seed * 7919 + r)
            chosen, label = [], []
            for c, m in enumerate(masks):
                idx = np.flatnonzero(m)
                n = min(len(idx), self.n_episodes)
                n -= n % 2
                if n < 8:
                    continue
                chosen.append(rng.choice(idx, n, replace=False))       # random order: first half vs second half
                label.append(np.full(n, c))
            if not chosen:
                continue
            chosen, label = np.concatenate(chosen), np.concatenate(label)
            fn, vecs = self._episode_fn(r), []
            for e in chosen:
                start, length = int(table["ep_start"][e]), int(table["ep_len"][e])
                tr = dataset.gather(np.arange(start, start + length), r)
                pad = self.horizon - length
                obs = np.concatenate([np.asarray(tr.observations), np.repeat(np.asarray(tr.observations)[-1:], pad, axis=0)])
                act = np.concatenate([np.asarray(tr.actions), np.repeat(np.asarray(tr.actions)[-1:], pad, axis=0)])
                mask = np.concatenate([np.ones(length), np.zeros(pad)]).astype(np.float32)
                vecs.append(fn(flat, extra, jnp.asarray(obs), jnp.asarray(act), jnp.asarray(mask)))
            E = jnp.stack(vecs, axis=1)                                # (2, n_episodes, P)
            out[f"ep_gram_{name}"] = np.asarray(self._gram_jit(E, E)).astype(np.float32)
            out[f"ep_class_{name}"] = label.astype(np.int8)
            out[f"ep_len_{name}"] = table["ep_len"][chosen].astype(np.int32)
            out[f"ep_return_{name}"] = table["ep_return"][chosen].astype(np.float32)
            out[f"ep_start_{name}"] = table["ep_start"][chosen].astype(np.int64)
        return out

    # ---- driver -----------------------------------------------------------------------------------------------------

    def run(self, dataset, paths):
        terminated, truncated = load_end_flags(dataset.root)
        if self.rows is not None:
            terminated, truncated = terminated[self.rows[0]:self.rows[1]], truncated[self.rows[0]:self.rows[1]]
        if terminated.shape[0] != dataset.n_samples or terminated.shape[1] != len(self.robot_names):
            raise ValueError(f"end flags {terminated.shape} do not match the dataset ({dataset.n_samples}, {len(self.robot_names)})")
        written = []
        for path in paths:
            variables, extra = load_checkpoint(path)
            flat = self._bind(variables)
            result = {"robot_names": np.array(self.robot_names), "checkpoint": np.array(os.path.abspath(path)),
                      "n_batches": np.array(self.n_batches), "batch_sizes": np.array(self.batch_sizes),
                      "groups": np.array(GROUPS + tuple(SUBGROUPS)), "variants": np.array(VARIANTS),
                      "classes": np.array(CLASSES), "horizon": np.array(self.horizon),
                      "min_return_frac": np.array(self.min_return),
                      "group_sizes": np.array([self.group_sizes[g] for g in GROUPS + tuple(SUBGROUPS)])}
            result.update(self._unit_part(flat, extra, dataset))
            result.update(self._episode_part(flat, extra, dataset, terminated, truncated))
            label = os.path.basename(os.path.normpath(path))
            base = self.out_base or os.path.join(os.path.abspath(path), "conflict")
            target = f"{base}_conflict_{label}.npz"
            os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
            np.savez_compressed(target, **result)
            written.append(target)
        return written
