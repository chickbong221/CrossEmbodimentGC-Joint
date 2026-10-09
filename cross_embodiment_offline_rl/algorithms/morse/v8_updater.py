"""The V8 actor update (morse/v8.py).

Every actor step runs one forward pass of every robot's minibatch chunk (jax.vjp) and one backward pass of the summed
losses: the task loss, the load-balancing loss, and the routing objective of the direction (MI, or compatibility
routing). All auxiliary losses are functions of the routing probabilities the MoE layers record, so their gradients
reach the routers and, through the router inputs, every upstream layer. The routing objective starts after
morse_warmup updates and ramps in over morse_ramp; the balance loss is on from the start.

Separate backward passes (task, balance, objective) run on diagnostic steps (to log the gradient ratios), on
compatibility scoring steps (the labels need the task's backward signals), and in every step with the optional
two-part cap (morse_v8_cap > 0) of the routing objective (not the balance loss), with c = morse_v8_cap:
  1. per MoE layer, its gradient on that layer's router is at most c times the task gradient on the same router;
  2. one factor for every non-router parameter it reaches: at most c times the task gradient's norm on those
     parameters.
Raw gradients, before the optimizer's clip.

Compatibility routing (direction compat; first actor step of an update, after morse_warmup with a morse_ramp ramp):
for every MoE layer and every scored unit u (the first morse_compat_core_transitions transitions of every robot at
the core, morse_compat_decoder_transitions at the joint-level layers, all their joints) and every expert k, the
candidate gradient ghat_k(u) is the gradient expert k's parameters would receive from u with routing weight 1 (the
backward signal of the layer output, sent unweighted into every expert). The reference is the expert's actual update
from all scored units of all robots, R_k = sum_v w_k(v) ghat_k(v), the unit itself included (STGC, ICLR 2025). Both
come exactly from layer inputs and backward signals (interference._layer_terms). Labels: compatible if
cos(ghat_k(u), R_k) >= -tau, conflicting if below, unknown if the expert has fewer than morse_compat_min_support
scored units or a zero reference. Loss: KL(uniform over compatible || routing probabilities renormalized over the
labelled experts) = -log p(compatible) + KL within the compatible set (interference.compat_loss).
"""
import logging
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax

from cross_embodiment_offline_rl.algorithms.morse import interference as itf
from cross_embodiment_offline_rl.algorithms.morse.v8 import (BODY_LAYERS, CAP_DELTA, CAP_IN, MOE_LAYERS, V8_DELTA,
                                                             V8_ROUTING, V8_SCORE, DecoderExperts, FFNExperts,
                                                             count_v8_params)

rlx_logger = logging.getLogger("rl_x")


def _keys(path):
    return [getattr(k, "key", None) for k in path]


def _is_router(path):
    return "router" in _keys(path)


def _layer_router(path, layer):
    k = _keys(path)
    return len(k) > 2 and k[1] == layer and k[2] == "router"


def _norm(leaves):
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves)) if leaves else jnp.zeros(())


def _select(tree, pred):
    return [leaf for path, leaf in jax.tree_util.tree_leaves_with_path(tree) if pred(path)]


def _chunk(x, sub, n_sub):
    size = x.shape[0] // n_sub
    return jax.lax.dynamic_slice_in_dim(x, sub * size, size)


def _chunk_struct(x, n_sub):
    return jax.ShapeDtypeStruct((x.shape[0] // n_sub,) + tuple(x.shape[1:]), x.dtype)


def balance_loss(ps):
    """KL(pbar || uniform) of a layer's routing, pbar the mean over robots (equal weight) of every robot's mean
    routing over its units. ps: per robot (n[, J], K)."""
    pbar = jnp.mean(jnp.stack([p.reshape(-1, p.shape[-1]).mean(axis=0) for p in ps]), axis=0)
    return jnp.sum(pbar * jnp.log(pbar.shape[-1] * pbar + 1e-12)), pbar


def robot_mi(ps):
    """-I(R;E) with equal robot priors: r_r is robot r's mean routing over its units."""
    r = jnp.stack([p.reshape(-1, p.shape[-1]).mean(axis=0) for p in ps])
    return itf.group_neg_mutual_information(r, jnp.full((len(ps),), 1.0 / len(ps)))


def joint_group_mi(ps, groups, n_groups):
    """-I(G;E) of joint groups at a joint-level layer. Every robot weighs 1/R, spread equally over its joints; a group's
    prior is its share of that weight and its routing the weighted mean routing of its joints. ps: per robot (n, J_r, K);
    groups: per robot (J_r,) group ids."""
    n_r = len(ps)
    k = ps[0].shape[-1]
    s = jnp.zeros((n_groups, k))
    prior = np.zeros(n_groups, np.float32)
    for p, g in zip(ps, groups):
        onehot = np.eye(n_groups, dtype=np.float32)[g] / (n_r * len(g))      # (J_r, G)
        s = s + jnp.asarray(onehot).T @ p.mean(axis=0)
        prior += onehot.sum(axis=0)
    r = s / jnp.maximum(jnp.asarray(prior)[:, None], 1e-30)
    return itf.group_neg_mutual_information(r, jnp.asarray(prior))


def inclusive_compat(robots, min_support, tau, rel_eps=1e-12):
    """Compatibility labels of every scored unit and expert from the exact per-unit candidate gradients.

    robots: per robot (layers, w, half) with layers a list of (kind, X, D) on (n, J, [K,] f) axes (J = 1 at a
    body-level layer), w (n, J, K) the units' mixing weights, half (n,) 0/1 (even/odd transitions, diagnostics).
    The reference of expert k is R_k = sum over all units of all robots of w_k(u) ghat_k(u), the unit included.
    Returns per robot (nonconf, conf, cos) of shape (n, J, K), the support (K,), and the label agreement between the
    two half references (diagnostic)."""
    parts, refs, half_refs = [], None, None
    for layers, w, half in robots:
        terms = [itf._layer_terms(kind, x, d) for kind, x, d in layers]
        norm = sum(t[0] for t in terms)
        r = [t[2](w) for t in terms]
        hw = [w * (half == h)[:, None, None].astype(w.dtype) for h in (0, 1)]
        hr = [[t[2](hw[h]) for t in terms] for h in (0, 1)]
        refs = r if refs is None else jax.tree_util.tree_map(jnp.add, refs, r)
        half_refs = hr if half_refs is None else jax.tree_util.tree_map(jnp.add, half_refs, hr)
        parts.append((terms, norm, w))
    sq = lambda ref: sum(jnp.sum(jnp.square(a), axis=tuple(range(1, a.ndim))) for layer in ref for a in layer)
    ref_norm = sq(refs)                                                         # (K,)
    half_norm = [sq(half_refs[h]) for h in (0, 1)]
    support = sum(jnp.sum((w > 0).astype(w.dtype), axis=(0, 1)) for _, _, w in parts)
    out, agree, both = [], 0.0, 0.0
    for terms, norm, w in parts:
        ip = sum(t[3](refs[i]) for i, t in enumerate(terms))                    # (n, J, K)
        ok = (norm > 0) & (ref_norm[None, None] > rel_eps * jnp.max(ref_norm)) & (support[None, None] >= min_support)
        cos = ip / jnp.sqrt(jnp.maximum(norm, 1e-30) * jnp.maximum(ref_norm, 1e-30)[None, None])
        nonconf = ok & (cos >= -tau)
        conf = ok & (cos < -tau)
        out.append((nonconf, conf, jnp.where(ok, cos, 0.0)))
        hc = []
        for h in (0, 1):
            hip = sum(t[3](half_refs[h][i]) for i, t in enumerate(terms))
            hc.append(hip / jnp.sqrt(jnp.maximum(norm, 1e-30) * jnp.maximum(half_norm[h], 1e-30)[None, None]) >= -tau)
        hok = ok & (half_norm[0] > 0)[None, None] & (half_norm[1] > 0)[None, None]
        agree = agree + jnp.sum((hc[0] == hc[1]) & hok)
        both = both + jnp.sum(hok)
    return out, support, agree / jnp.maximum(both, 1.0)


class V8ActorUpdater:
    def __init__(self, config, policy, split_fn, robot_names, run_path, joint_groups=None, n_groups=0):
        a = config.algorithm
        self.policy = policy
        self.split_fn = split_fn
        self.robot_names = list(robot_names)
        self.direction = str(a.morse_v8_direction)
        self.score_policy = policy.clone(record_score=True, record_delta=True)
        self.balance = float(a.morse_v8_balance)
        self.mi_weight = float(a.morse_v8_mi_weight) if self.direction in ("mi_joint", "mi_robot") else 0.0
        self.cap = float(a.morse_v8_cap)
        self.compat_lambda = float(a.morse_v8_compat_lambda) if self.direction == "compat" else 0.0
        self.compat_tau = float(a.morse_compat_tau)
        self.compat_min_support = int(a.morse_compat_min_support)
        self.compat_n = {layer: int(a.morse_compat_core_transitions if layer in BODY_LAYERS
                                    else a.morse_compat_decoder_transitions) for layer in MOE_LAYERS}
        self.warmup, self.ramp = int(a.morse_warmup), int(a.morse_ramp)
        if self.direction == "mi_joint" and joint_groups is None:
            raise ValueError("morse_v8_direction=mi_joint needs the joint-group table (morse_v8_joint_groups)")
        self.joint_groups, self.n_groups = joint_groups, n_groups
        # MI label of every layer: robot identity at the core; joint groups (mi_joint) or robots at the joint level.
        self.mi_label = {layer: ("joint_group" if self.direction == "mi_joint" and layer not in BODY_LAYERS else "robot")
                         for layer in MOE_LAYERS}
        self.router_noise = bool(a.morse_router_noise)
        seed = int(getattr(getattr(config, "environment", None), "seed", 0))
        self.routing_base = jax.random.fold_in(jax.random.PRNGKey(seed), 0x524F55)
        self.diag_frequency = int(a.morse_diag_frequency)
        self.snapshot_dir = os.path.join(run_path, "morse_diagnostics") if run_path else None
        self.loss_fn = None
        self._fns = {}
        self._templates = {}
        self._cap_acc = None
        self._routing_log = None
        self._logged = False
        self._last_diag = None

    def objective_factor_at(self, step):
        """Share of the routing objective's weight (MI or compatibility) in effect: 0 before morse_warmup, then a
        linear morse_ramp to 1. The balance loss is not delayed."""
        if (self.mi_weight <= 0.0 and self.compat_lambda <= 0.0) or step < self.warmup:
            return 0.0
        return 1.0 if self.ramp <= 0 else min(1.0, (step - self.warmup + 1) / self.ramp)

    def _routing_keys(self, step_sub, n_robots):
        if not self.router_noise:
            return [None] * n_robots
        base = jax.random.fold_in(jax.random.fold_in(self.routing_base, step_sub[0]), step_sub[1])
        return [jax.random.fold_in(base, r) for r in range(n_robots)]

    # ---------------------------------------------------------------------------------------------- forward pass
    def _delta_template(self, r, obs_struct):
        key = (r, obs_struct.shape)
        if key not in self._templates:
            init = lambda o: self.score_policy.init(jax.random.PRNGKey(0), *self.split_fn(o, r))
            self._templates[key] = jax.eval_shape(init, obs_struct)[V8_DELTA]
        return self._templates[key]

    def _forward(self, scoring, observations, actions, extra, rngs, routing_keys):
        """fwd(params, deltas) -> ((mean task loss, routing probabilities per robot and layer), (losses, records))."""
        policy = self.score_policy if scoring else self.policy
        mutable = [V8_ROUTING, V8_SCORE] if scoring else [V8_ROUTING]

        def fwd(params, deltas):
            losses, ps, recs = [], [], []
            for r, (obs, act) in enumerate(zip(observations, actions)):
                variables = {"params": params["params"]}
                if scoring:
                    variables[V8_DELTA] = deltas[r]
                out, state = policy.apply(variables, *self.split_fn(obs, r), robot_id=r, mutable=mutable,
                                          rngs=None if routing_keys[r] is None else {"routing": routing_keys[r]})
                losses.append(self.loss_fn(out, obs, act, extra, rngs[r], r))
                ps.append({layer: state[V8_ROUTING][layer]["p"][0] for layer in MOE_LAYERS})
                rec = {layer: {k: v[0] for k, v in state[V8_ROUTING][layer].items() if k != "p"} for layer in MOE_LAYERS}
                if scoring:
                    rec["score"] = {layer: {k: v[0] for k, v in state[V8_SCORE][layer].items()} for layer in MOE_LAYERS}
                recs.append(rec)
            losses = jnp.stack(losses)
            return (losses.mean(), tuple(ps)), (losses, tuple(recs))
        return fwd

    # ------------------------------------------------------------------------------------- compatibility labels
    def _expert_units(self, params, layer, rec, delta):
        """One robot's scored units of a layer: (kind, X, D) on (n, J, [K,] f) axes from the experts alone, run on the
        recorded expert inputs with the layer's backward signal sent unweighted into every expert, and the units'
        mixing weights (n, J, K)."""
        n = self.compat_n[layer]
        body = layer in BODY_LAYERS
        lift = (lambda t: t[:n, None]) if body else (lambda t: t[:n])
        inputs = tuple(lift(rec["score"][layer][f"u{i}"]) for i in range(2 if layer == "dec_moe" else 1))
        w = lift(rec["score"][layer]["w"])
        d_out = lift(delta[layer]["y"])                                         # (n, J, out)
        sub = params["params"][layer]["experts"]
        k = w.shape[-1]
        if layer == "dec_moe":
            mod = DecoderExperts(k, sub["desc_dense_0"]["kernel"].shape[-1], capture=True)
        else:
            mod = FFNExperts(k, sub["dense_0"]["kernel"].shape[-1], sub["dense_1"]["kernel"].shape[-1], capture=True)
        tmpl = jax.eval_shape(lambda *u: mod.init(jax.random.PRNGKey(0), *u), *inputs)[CAP_DELTA]
        zeros = jax.tree_util.tree_map(lambda s: jnp.zeros(s.shape, s.dtype), tmpl)

        def score(delta_vars):
            outs, state = mod.apply({"params": sub, CAP_DELTA: delta_vars}, *inputs, mutable=[CAP_IN])
            return jnp.sum(outs * jax.lax.stop_gradient(d_out)[..., None, :]), state

        (_, state), g = jax.value_and_grad(score, has_aux=True)(zeros)
        layers = []
        for kind, name in mod.LAYERS:
            if kind == "dense":
                layers.append((kind, state[CAP_IN][name]["x"][0], g[name]["z"]))
            else:
                layers.append((kind, state[CAP_IN][name]["xhat"][0], g[name]["y"]))
        return layers, w

    def _compat_labels(self, params, recs, g_deltas):
        labels, metrics = {}, {}
        for layer in MOE_LAYERS:
            robots = []
            for r, rec in enumerate(recs):
                layers, w = self._expert_units(params, layer, rec, g_deltas[r])
                half = (jnp.arange(w.shape[0]) % 2)
                robots.append((layers, w, half))
            out, support, agree = inclusive_compat(robots, self.compat_min_support, self.compat_tau)
            labels[layer] = [jax.lax.stop_gradient(jnp.stack(o[:2])) for o in out]  # per robot (2, n, J, K)
            nonconf = jnp.concatenate([o[0].reshape(-1, o[0].shape[-1]) for o in out])
            conf = jnp.concatenate([o[1].reshape(-1, o[1].shape[-1]) for o in out])
            n_a = nonconf.sum(axis=-1)
            metrics.update({f"morse/v8/{layer}/compat_nonconf_frac": nonconf.mean(),
                            f"morse/v8/{layer}/compat_conf_frac": conf.mean(),
                            f"morse/v8/{layer}/compat_mean_nonconf": n_a.mean(),
                            f"morse/v8/{layer}/compat_no_nonconf_frac": (n_a == 0).mean(),
                            f"morse/v8/{layer}/compat_support_min": support.min(),
                            f"morse/v8/{layer}/compat_half_agree": agree})
        return labels, metrics

    # ------------------------------------------------------------------------------------------------- losses
    def _balance(self, ps):
        total, metrics = jnp.zeros(()), {}
        for layer in MOE_LAYERS:
            loss, pbar = balance_loss([p[layer] for p in ps])
            total = total + self.balance * loss
            metrics[f"morse/v8/{layer}/balance_loss"] = loss
            metrics[f"morse/v8/{layer}/usage_min"] = pbar.min()
            metrics[f"morse/v8/{layer}/usage_max"] = pbar.max()
        return total, metrics

    def _routing_objective(self, ps, labels, factor):
        total, metrics = jnp.zeros(()), {}
        for layer in MOE_LAYERS:
            layer_ps = [p[layer] for p in ps]
            if self.mi_weight > 0.0:
                if self.mi_label[layer] == "joint_group":
                    neg_mi, h_cond, h_marg = joint_group_mi(layer_ps, self.joint_groups, self.n_groups)
                else:
                    neg_mi, h_cond, h_marg = robot_mi(layer_ps)
                total = total + factor * self.mi_weight * neg_mi
                metrics.update({f"morse/v8/{layer}/mi": -neg_mi, f"morse/v8/{layer}/h_expert_given_label": h_cond,
                                f"morse/v8/{layer}/h_expert": h_marg})
            if labels is not None:
                n = self.compat_n[layer]
                lift = (lambda t: t[:n, None]) if layer in BODY_LAYERS else (lambda t: t[:n])
                p = jnp.concatenate([lift(lp).reshape(-1, lp.shape[-1]) for lp in layer_ps])
                nonconf = jnp.concatenate([lab[0].reshape(-1, lab.shape[-1]) for lab in labels[layer]])
                conf = jnp.concatenate([lab[1].reshape(-1, lab.shape[-1]) for lab in labels[layer]])
                loss, parts = itf.compat_loss(p, nonconf, conf, "compat")
                total = total + factor * self.compat_lambda * loss
                metrics[f"morse/v8/{layer}/compat_loss"] = loss
                metrics.update({f"morse/v8/{layer}/compat_{k}": v for k, v in parts.items()})
        return total, metrics

    def _cap(self, g_task, g_obj):
        """Scale the routing objective's gradient by the two-part cap; returns (capped gradient, factors)."""
        factors = {}
        for layer in MOE_LAYERS:
            sel = lambda p, layer=layer: _layer_router(p, layer)
            task_n, obj_n = _norm(_select(g_task, sel)), _norm(_select(g_obj, sel))
            factors[layer] = jnp.where(obj_n > 0, jnp.minimum(1.0, self.cap * task_n / jnp.maximum(obj_n, 1e-30)), 1.0)
        reached = [(path, leaf) for path, leaf in jax.tree_util.tree_leaves_with_path(g_obj) if not _is_router(path)]
        mask = [jnp.sum(jnp.square(leaf)) > 0 for _, leaf in reached]
        task_leaves = {tuple(_keys(p)): l for p, l in jax.tree_util.tree_leaves_with_path(g_task)}
        obj_sq = sum(jnp.where(m, jnp.sum(jnp.square(l)), 0.0) for m, (_, l) in zip(mask, reached))
        task_sq = sum(jnp.where(m, jnp.sum(jnp.square(task_leaves[tuple(_keys(p))])), 0.0) for m, (p, _) in zip(mask, reached))
        obj_n, task_n = jnp.sqrt(obj_sq), jnp.sqrt(task_sq)
        factors["upstream"] = jnp.where(obj_n > 0, jnp.minimum(1.0, self.cap * task_n / jnp.maximum(obj_n, 1e-30)), 1.0)
        raw = {"upstream": (obj_n, task_n)}
        for layer in MOE_LAYERS:
            sel = lambda p, layer=layer: _layer_router(p, layer)
            raw[layer] = (_norm(_select(g_obj, sel)), _norm(_select(g_task, sel)))

        def scale(path, g):
            for layer in MOE_LAYERS:
                if _layer_router(path, layer):
                    return g * factors[layer]
            return g * factors["upstream"]
        if self.cap <= 0.0:
            return g_obj, {k: jnp.ones(()) for k in factors}, raw
        return jax.tree_util.tree_map_with_path(scale, g_obj), factors, raw

    # ------------------------------------------------------------------------------------------------- update
    def _step_fn(self, observations, n_sub, scoring, diag):
        key = (n_sub, scoring, diag, tuple((o.shape, o.dtype) for o in observations))
        if key in self._fns:
            return self._fns[key]
        structs = [_chunk_struct(o, n_sub) for o in observations]
        templates = [self._delta_template(r, s) for r, s in enumerate(structs)] if scoring else None
        has_objective = self.mi_weight > 0.0 or (scoring and self.compat_lambda > 0.0)

        # Separate backward passes (task, balance, objective) are needed for the cap, for the compatibility labels (the
        # task's backward signals) and for the diagnostics; otherwise one backward pass of the summed losses.
        separate = diag or scoring or self.cap > 0.0

        def fn(policy_state, extra, observations, actions, rng, chunk, step_sub, factor):
            if n_sub > 1:
                observations = tuple(_chunk(o, chunk, n_sub) for o in observations)
                actions = tuple(_chunk(a, chunk, n_sub) for a in actions)
            routing_keys = self._routing_keys(step_sub, len(observations))
            rngs = []
            for _ in observations:                     # the key sequence of the original updater
                rng, sub = jax.random.split(rng)
                rngs.append(sub)
            rng, _ = jax.random.split(rng)
            params = policy_state.params
            deltas = tuple(jax.tree_util.tree_map(lambda s: jnp.zeros(s.shape, s.dtype), t) for t in templates) \
                if scoring else tuple({} for _ in observations)
            fwd = self._forward(scoring, observations, actions, extra, rngs, routing_keys)
            (task, ps), vjp_fn, (losses, recs) = jax.vjp(fwd, params, deltas, has_aux=True)
            metrics = {}
            (_, bal_metrics), g_ps_bal = jax.value_and_grad(self._balance, has_aux=True)(ps)
            metrics.update(bal_metrics)
            factors, raw = {}, {}
            if not separate:
                cotangent = g_ps_bal
                if has_objective:
                    (_, obj_metrics), g_ps_obj = jax.value_and_grad(self._routing_objective, has_aux=True)(
                        ps, None, factor)
                    cotangent = jax.tree_util.tree_map(jnp.add, cotangent, g_ps_obj)
                    metrics.update(obj_metrics)
                total = vjp_fn((jnp.ones_like(task), cotangent))[0]
            else:
                zero_ps = jax.tree_util.tree_map(jnp.zeros_like, ps)
                g_task, g_deltas = vjp_fn((jnp.ones_like(task), zero_ps))
                labels = None
                if scoring:
                    labels, compat_metrics = self._compat_labels(params, recs, g_deltas)
                    metrics.update(compat_metrics)
                g_bal = vjp_fn((jnp.zeros_like(task), g_ps_bal))[0]
                total = jax.tree_util.tree_map(jnp.add, g_task, g_bal)
                if has_objective:
                    (_, obj_metrics), g_ps_obj = jax.value_and_grad(self._routing_objective, has_aux=True)(
                        ps, labels, factor)
                    g_obj = vjp_fn((jnp.zeros_like(task), g_ps_obj))[0]
                    g_obj, factors, raw = self._cap(g_task, g_obj)      # without a cap: unscaled, norms measured
                    total = jax.tree_util.tree_map(jnp.add, total, g_obj)
                    metrics.update(obj_metrics)
            new_state = policy_state.apply_gradients(grads=total)
            if diag:
                for layer in MOE_LAYERS:
                    sel = lambda p, layer=layer: _layer_router(p, layer)
                    t = _norm(_select(g_task, sel))
                    metrics[f"morse/v8/{layer}/router_grad_task"] = t
                    metrics[f"morse/v8/{layer}/router_grad_balance"] = _norm(_select(g_bal, sel))
                    if layer in raw:
                        metrics[f"morse/v8/{layer}/router_grad_objective"] = raw[layer][0]
                        metrics[f"morse/v8/{layer}/objective_task_ratio"] = raw[layer][0] / jnp.maximum(t, 1e-30)
                        if self.cap > 0.0:
                            metrics[f"morse/v8/{layer}/cap_factor"] = factors[layer]
                    ws = [rec[layer]["w"] for rec in recs]
                    flat = jnp.concatenate([w.reshape(-1, w.shape[-1]) for w in ws])
                    metrics[f"morse/v8/{layer}/selected_usage_min"] = (flat > 0).mean(axis=0).min()
                    metrics[f"morse/v8/{layer}/selected_usage_max"] = (flat > 0).mean(axis=0).max()
                    pn = flat / jnp.maximum(flat.sum(axis=-1, keepdims=True), 1e-30)
                    ent = -jnp.sum(jnp.where(pn > 0, pn * jnp.log(jnp.where(pn > 0, pn, 1.0)), 0.0), axis=-1)
                    metrics[f"morse/v8/{layer}/eff_experts"] = jnp.exp(ent).mean()
                    if "noise_scale" in recs[0][layer]:
                        metrics[f"morse/v8/{layer}/noise_scale"] = jnp.mean(jnp.concatenate(
                            [rec[layer]["noise_scale"].reshape(-1) for rec in recs]))
                up = lambda p: not _is_router(p)
                metrics["morse/v8/upstream_grad_task"] = _norm(_select(g_task, up))
                metrics["morse/v8/upstream_grad_balance"] = _norm(_select(g_bal, up))
                if "upstream" in raw:
                    metrics["morse/v8/upstream_grad_objective"] = raw["upstream"][0]
                    metrics["morse/v8/upstream_objective_task_ratio"] = raw["upstream"][0] / jnp.maximum(raw["upstream"][1], 1e-30)
                    if self.cap > 0.0:
                        metrics["morse/v8/upstream_cap_factor"] = factors["upstream"]
                metrics["morse/v8/grad_norm"] = optax.global_norm(total)
                metrics["morse/v8/objective_factor"] = factor
                for r, name in enumerate(self.robot_names):
                    metrics[f"morse/v8/loss_{name}"] = losses[r]
                stats = {layer: jnp.stack([(rec[layer]["w"].reshape(-1, rec[layer]["w"].shape[-1]) > 0).mean(axis=0)
                                           for rec in recs]) for layer in MOE_LAYERS}
            else:
                stats = {}
            return new_state, task, metrics, rng, (factors if self.cap > 0.0 else {}), stats

        self._fns[key] = jax.jit(fn)
        return self._fns[key]

    def update(self, policy_state, extra, env_batches, rng, step, sub=0, n_sub=1):
        """One actor step on chunk ``sub`` of n_sub chunks of the minibatches of update ``step``. Returns
        (policy_state, loss, metrics, rng); metrics only on diagnostic steps (first chunk)."""
        t0 = time.time()
        if not self._logged:
            counts = count_v8_params(policy_state.params, self.policy.n_shared + self.policy.topk)
            rlx_logger.info(f"V8 direction {self.direction}; experts per layer {self.policy.n_shared} shared + "
                            f"{self.policy.n_routed} routed, top-{self.policy.topk} ({self.policy.gate}); "
                            f"policy parameters {counts}")
            self._param_counts = {f"morse/v8/params_{k}": float(v) for k, v in counts.items()}
            self._logged = True
        observations = tuple(b.observations for b in env_batches)
        actions = tuple(b.actions for b in env_batches)
        diag = sub == 0 and self.diag_frequency > 0 and step % self.diag_frequency == 0
        factor = self.objective_factor_at(step)
        scoring = sub == 0 and self.compat_lambda > 0.0 and (factor > 0.0 or diag)
        step_sub = jnp.asarray([step, sub], jnp.uint32)
        policy_state, loss, metrics, rng, factors, stats = self._step_fn(observations, n_sub, scoring, diag)(
            policy_state, extra, observations, actions, rng, jnp.int32(sub), step_sub, jnp.float32(factor))
        self._add_cap(factors)
        if not diag:
            return policy_state, loss, {}, rng
        jax.block_until_ready(policy_state.params)
        out = {k: float(v) for k, v in jax.device_get(metrics).items()}
        out.update(self._cap_metrics())
        out.update(self._param_counts)
        self._log_routing(step, stats)
        now = time.time()
        out["morse/update_s"] = now - t0
        if self._last_diag is not None and step > self._last_diag[1]:
            out["morse/wall_per_update_s"] = (now - self._last_diag[0]) / (step - self._last_diag[1])
        self._last_diag = (now, step)
        try:
            peak = (jax.local_devices()[0].memory_stats() or {}).get("peak_bytes_in_use")
        except Exception:
            peak = None
        if peak is not None:
            out["morse/peak_mem_gb"] = peak / 2 ** 30
        return policy_state, loss, out, rng

    def _add_cap(self, factors):
        if not factors:
            return
        step = {k: jnp.stack([f, (f < 1.0).astype(f.dtype), jnp.ones_like(f)]) for k, f in factors.items()}
        self._cap_acc = step if self._cap_acc is None else {k: self._cap_acc[k] + v for k, v in step.items()}

    def _cap_metrics(self):
        """Mean cap factor and the share of actor steps in which the cap acted, since the last diagnostic step."""
        if self._cap_acc is None:
            return {}
        out = {}
        for k, (total, active, count) in jax.device_get(self._cap_acc).items():
            out[f"morse/v8/{k}/cap_factor_mean"] = float(total / count)
            out[f"morse/v8/{k}/cap_active_frac"] = float(active / count)
        self._cap_acc = None
        return out

    def _log_routing(self, step, stats):
        """Every robot's share of units selecting each routed expert, per layer, appended to morse_diagnostics/routing.npz
        (arrays <layer>_select of shape (steps, robots, experts))."""
        if not stats or self.snapshot_dir is None:
            return
        stats = jax.device_get(stats)
        if self._routing_log is None:
            self._routing_log = {"step": []}
        self._routing_log["step"].append(step)
        for layer, select in stats.items():
            self._routing_log.setdefault(f"{layer}_select", []).append(np.asarray(select))
        os.makedirs(self.snapshot_dir, exist_ok=True)
        path = os.path.join(self.snapshot_dir, "routing.npz")
        np.savez(path + ".tmp.npz", robot_names=np.array(self.robot_names),
                 **{k: np.asarray(v) for k, v in self._routing_log.items()})
        os.replace(path + ".tmp.npz", path)
