"""Joint-type interactions in the action decoder, measured on frozen parameters (analysis, Test 1).

At chosen updates the probe draws five independent data subsets (Q1..Q5, ``n_batches`` full minibatches each)
and computes, for every unit type a (joint j of robot r), the decoder-parameter gradient g_a of its share of the
policy loss (the IQL loss of a robot is the sum of its per-joint terms), summed over the subset's minibatches.
The critic-derived advantage weights are fixed at their current values. From these it writes
(``<run>/morse_probe/probe_<updates>.npz``):

  G12_<grp>, G34_<grp>   symmetrized cross-subset Grams S_Q1 S_Q2^T and S_Q3 S_Q4^T: products of gradients from
                         different data, so the sampling noise of either factor does not enter their expectation
                         (two independent estimates of the same matrix give its reliability);
  Gfull_<grp>            (S_Q1 + S_Q2)(S_Q3 + S_Q4)^T, the estimate from all four subsets;
  N11_<grp>              diag(S_Q1 S_Q1^T): squared norms that include the noise (signal-to-noise per type);
                         per parameter group <grp>: dense0 (first layer), ln0 (LayerNorm), out_w (output weights),
                         out_b (output bias) and all;
  T_<i>, P_<i>           lookahead transfer for step norm eta_i: T[a, b] = L_b(theta) - L_b(theta - eta_i d_a), the
                         decrease of type b's loss on Q5 after a step along type a's unit gradient direction d_a
                         (from Q1..Q4); P[a, b] = eta_i g_b(Q5) . d_a is its first-order prediction;
  T_sign, P_sign         the same for one Adam-like step, -lr sign(g_a), the update of a sign-normalized optimizer;
  L5, type_robot, type_elem, type_desc   loss per type on Q5, robot and joint index, mean joint descriptor.
"""
import os

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


GROUPS = ("dense0", "ln0", "out_w", "out_b")


def _group_of(path):
    keys = [getattr(k, "key", str(k)) for k in path]
    if keys[0] == "dense_0":
        return "dense0"
    if keys[0] == "ln_0":
        return "ln0"
    if keys[0] == "dense_1":
        return "out_w" if keys[-1] == "kernel" else "out_b"
    raise ValueError(f"unexpected decoder parameter {keys}")


def _split_decoder(params):
    """Decoder expert parameters (without a router) and a function rebuilding the full parameters."""
    dec = params["params"]["decoder"]
    experts = {k: v for k, v in dec.items() if k != "router"}

    def rebuild(new_experts):
        return {**params, "params": {**params["params"], "decoder": {**dec, **new_experts}}}
    return experts, rebuild


class DecoderProbe:
    def __init__(self, policy, split_fn, loss_per_joint, robot_names, batch_sizes, n_batches, step_norms,
                 out_dir, seed, sign_lr=3e-4):
        self.policy = policy
        self.split_fn = split_fn
        self.loss_per_joint = loss_per_joint
        self.robot_names = list(robot_names)
        self.batch_sizes = list(batch_sizes)
        self.n_batches = int(n_batches)
        self.step_norms = tuple(float(s) for s in step_norms)
        self.sign_lr = float(sign_lr)
        self.out_dir = out_dir
        self.seed = int(seed)
        self._fns = {}

    def _robot_fns(self, r):
        if r in self._fns:
            return self._fns[r]

        def losses(params, extra, obs, act):
            out = self.policy.apply(params, *self.split_fn(obs, r), robot_id=r)
            return self.loss_per_joint(out, obs, act, extra, None, r)

        def grads(params, extra, obs, act):
            experts, rebuild = _split_decoder(params)
            flat, unravel = ravel_pytree(experts)
            f = lambda v: losses(rebuild(unravel(v)), extra, obs, act)
            loss, vjp = jax.vjp(f, flat)
            jac = jax.vmap(lambda e: vjp(e)[0])(jnp.eye(loss.shape[0], dtype=loss.dtype))
            return loss, jac                                     # (n_j,), (n_j, P)

        def shifted_losses(params, extra, obs, act, delta):
            experts, rebuild = _split_decoder(params)
            flat, unravel = ravel_pytree(experts)
            return losses(rebuild(unravel(flat + delta)), extra, obs, act)

        self._fns[r] = (jax.jit(losses), jax.jit(grads), jax.jit(shifted_losses))
        return self._fns[r]

    def _batches(self, dataset, subset, step):
        for i in range(self.n_batches):
            key = jax.random.PRNGKey(self.seed * 1_000_003 + step * 31 + subset * 1009 + i)
            env_batches, _ = dataset.sample(key, self.batch_sizes)
            yield [(jnp.asarray(b.observations), jnp.asarray(b.actions)) for b in env_batches]

    def run(self, params, extra, dataset, step):
        experts, _ = _split_decoder(params)
        flat0, _ = ravel_pytree(experts)
        leaves = jax.tree_util.tree_flatten_with_path(experts)[0]
        group_index = np.concatenate([np.full(int(np.prod(leaf.shape)), GROUPS.index(_group_of(path)))
                                      for path, leaf in leaves])
        n_robots = len(self.batch_sizes)
        sums = [[None] * n_robots for _ in range(5)]              # [subset][robot] -> (n_j, P)
        loss5 = [None] * n_robots
        desc = [None] * n_robots
        batches5 = []
        for q in range(5):
            for batch in self._batches(dataset, q, step):
                if q == 4:
                    batches5.append(batch)
                for r, (obs, act) in enumerate(batch):
                    _, fgrads, _ = self._robot_fns(r)
                    loss, jac = fgrads(params, extra, obs, act)
                    sums[q][r] = jac if sums[q][r] is None else sums[q][r] + jac
                    if q == 4:
                        loss5[r] = loss if loss5[r] is None else loss5[r] + loss
                    if desc[r] is None:
                        desc[r] = np.asarray(self.split_fn(obs, r)[0].mean(axis=0))
        S = [jnp.concatenate(sums[q], axis=0) for q in range(5)]   # (T, P)
        type_robot = np.concatenate([np.full(s.shape[0], r) for r, s in enumerate(sums[0])])
        type_elem = np.concatenate([np.arange(s.shape[0]) for s in sums[0]])
        out = {"step": np.array(step), "type_robot": type_robot, "type_elem": type_elem,
               "type_desc": np.concatenate(desc), "L5": np.asarray(jnp.concatenate(loss5)),
               "robot_names": np.array(self.robot_names), "groups": np.array(GROUPS + ("all",)),
               "step_norms": np.array(self.step_norms), "n_batches": np.array(self.n_batches)}
        hi = jax.lax.Precision.HIGHEST
        sym = lambda m: 0.5 * (m + m.T)
        for gi, grp in enumerate(GROUPS + ("all",)):
            cols = np.arange(len(group_index)) if grp == "all" else np.where(group_index == gi)[0]
            s = [x[:, cols] for x in S]
            out[f"G12_{grp}"] = np.asarray(sym(jnp.matmul(s[0], s[1].T, precision=hi)))
            out[f"G34_{grp}"] = np.asarray(sym(jnp.matmul(s[2], s[3].T, precision=hi)))
            out[f"Gfull_{grp}"] = np.asarray(sym(jnp.matmul(s[0] + s[1], (s[2] + s[3]).T, precision=hi)))
            out[f"N11_{grp}"] = np.asarray(jnp.sum(s[0] * s[0], axis=1))
        # Lookahead: one step per type from its gradient on Q1..Q4 (a descent step along the unit gradient direction
        # for each step norm, and one Adam-like sign step), effect on every type's loss on Q5. First-order
        # prediction: L_b(theta + delta) - L_b(theta) ~ g_b . delta, so T ~ -g_b(Q5) . delta.
        g_est = S[0] + S[1] + S[2] + S[3]
        unit = -g_est / jnp.maximum(jnp.linalg.norm(g_est, axis=1, keepdims=True), 1e-30)
        steps = {f"{i}": eta * unit for i, eta in enumerate(self.step_norms)}
        steps["sign"] = -self.sign_lr * jnp.sign(g_est)
        base = np.asarray(jnp.concatenate(loss5))
        for name, deltas in steps.items():
            T = np.zeros((len(type_robot), len(type_robot)), dtype=np.float32)
            for a in range(len(type_robot)):
                after = [None] * n_robots
                for batch in batches5:
                    for r, (obs, act) in enumerate(batch):
                        _, _, fshift = self._robot_fns(r)
                        lr = fshift(params, extra, obs, act, deltas[a])
                        after[r] = lr if after[r] is None else after[r] + lr
                T[a] = base - np.asarray(jnp.concatenate(after))
            out[f"T_{name}"] = T
            out[f"P_{name}"] = np.asarray(-jnp.matmul(deltas, S[4].T, precision=hi))
        out["sign_lr"] = np.array(self.sign_lr)
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"probe_{step:07d}.npz")
        np.savez_compressed(path, **out)
        return path
