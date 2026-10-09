"""Correctness tests for the V8 architecture and update (morse/v8.py, morse/v8_updater.py).

Run with ``python tests/test_v8.py`` (or pytest). They check that
  1. parameter counts at the planned widths equal the closed-form counts (dense URMAv2 parts, pre-LN of every MoE
     layer, 16 experts per layer, 4 active);
  2. a MoE layer computes x + sum_k w_k E_k(LN(x)) with the DeepSeekMoE gate (shared expert weight 1, top-3 softmax
     scores not renormalized) or the renormalized top-4 gate;
  3. evaluation routes on clean logits deterministically; training noise changes the routing; the auxiliary
     probabilities hold the noise fixed;
  4. an auxiliary loss on a layer's routing reaches its router and every upstream layer feeding the router, but not
     its own experts, its noise head or anything downstream; the encoder router has nothing upstream;
  5. compatibility scoring: the candidate gradients from the layer's backward signal equal explicit per-unit
     gradients, the inclusive reference equals the task gradient of every expert, and the cosines match;
  6. the two-part cap bounds the objective's gradient per router and on the upstream parameters it reaches;
  7. MI and balance losses match their definitions (robot and joint-group priors);
  8. the compiled update runs for every direction with sub-steps, moves the parameters, and respects the cap;
  9. joint groups: mirror-invariant features, deterministic clustering, table validation;
 10. without the cap, the one-pass update equals the separate-pass update, and the objective warms up and ramps in.
"""
import json
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from cross_embodiment_offline_rl.algorithms.morse import interference as itf
from cross_embodiment_offline_rl.algorithms.morse.moe import Router
from cross_embodiment_offline_rl.algorithms.morse.v8 import (MOE_LAYERS, V8_DELTA, V8_ROUTING, V8_SCORE, DecoderExperts,
                                                             FFNExperts, MoELayer, V8Policy, build_joint_groups,
                                                             count_v8_params, direction_layer_form, joint_features,
                                                             load_joint_groups, write_joint_groups)
from cross_embodiment_offline_rl.algorithms.morse import v8_updater as vu

_ROBOTS = [(3, 2), (5, 4)]          # (joints, feet) of the toy robots


def _rel_err(a, b):
    return float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-30))


def _policy(direction="mi_robot", width=0.5, n_experts=6, noise=True, init=0.5, **kw):
    n_shared, n_routed, topk, gate = direction_layer_form(direction, n_experts)
    return V8Policy(std_dev=1.0, softmax_temperature=1.0, softmax_temperature_min=0.015, stability_epsilon=1e-8,
                    policy_mean_abs_clip=10.0, policy_std_min_clip=1e-8, policy_std_max_clip=2.0,
                    n_shared=n_shared, n_routed=n_routed, topk=topk, gate=gate, width=width, router_hidden=16,
                    router_init=init, router_noise=noise, **kw)


def _inputs(key, batch, n_joints, n_feet):
    ks = jax.random.split(key, 5)
    return (jax.random.normal(ks[0], (batch, n_joints, 23), jnp.float64),
            jax.random.normal(ks[1], (batch, n_joints, 3), jnp.float64),
            jax.random.normal(ks[2], (batch, n_feet, 10), jnp.float64),
            jax.random.normal(ks[3], (batch, n_feet, 2), jnp.float64),
            jax.random.normal(ks[4], (batch, 16), jnp.float64))


def _split(obs, r):
    nj, nf = _ROBOTS[r]
    a, b, c = nj * 23, nj * 26, nj * 26 + nf * 10
    return (obs[:, :a].reshape(-1, nj, 23), obs[:, a:b].reshape(-1, nj, 3),
            obs[:, b:c].reshape(-1, nf, 10), obs[:, c:c + nf * 2].reshape(-1, nf, 2), obs[:, -16:])


def _obs(key, batch, r):
    nj, nf = _ROBOTS[r]
    return jax.random.normal(key, (batch, nj * 26 + nf * 12 + 16), jnp.float64)


def _toy_loss(out, obs, act, extra, key, r):
    mean, logstd = out
    return extra * (jnp.mean((mean - act[:, :mean.shape[1]]) ** 2) + 0.1 * jnp.mean(logstd ** 2))


def _config(direction, **over):
    a = dict(morse_v8_direction=direction, morse_v8_balance=0.01, morse_v8_mi_weight=0.05, morse_v8_cap=0.75,
             morse_v8_compat_lambda=0.5, morse_compat_tau=0.05, morse_compat_min_support=1,
             morse_compat_core_transitions=8, morse_compat_decoder_transitions=4, morse_warmup=1, morse_ramp=1,
             morse_router_noise=True, morse_diag_frequency=2)
    a.update(over)
    return SimpleNamespace(algorithm=SimpleNamespace(**a), environment=SimpleNamespace(seed=0))


def _flat(tree):
    return {"/".join(str(getattr(k, "key", k)) for k in p): v for p, v in jax.tree_util.tree_leaves_with_path(tree)}


# ----------------------------------------------------------------------------------------------------------- 1
def test_parameter_counts():
    fixed = (3 * 256 + 256 + 512 + 256 * 4 + 4 + 2 * 4 + 4                          # encoder dense parts, foot
             + (23 * 64 + 64 + 128 + 64 * 64 + 64 + 1) + (10 * 32 + 32 + 64 + 32 * 32 + 32 + 1)   # aggregation
             + (400 * 512 + 512 + 512) + 1024 + (512 * 256 + 256 + 256) + (256 * 128 + 128 + 128)  # core (WN)
             + (23 * 128 + 128 + 256 + 128 * 128 + 128 + 128 + 1)                    # log-std branch
             + 512 + 1024 + 512)                                                     # pre-LN of the ffn MoE layers
    assert fixed == 403347
    for width, expect in ((0.5, 1302423), (0.75, 1749655)):
        h = lambda d: int(width * d / 2)
        w = int(width * 128)
        experts = (513 * h(256) + 256) + (1025 * h(512) + 512) + (513 * h(256) + 256) + (418 * w + 129)
        assert fixed + 4 * experts == expect
        for direction in ("mi_joint", "compat"):
            pol = _policy(direction, width, n_experts=16, noise=True, init=0.0)
            params = pol.init(jax.random.PRNGKey(0), *_inputs(jax.random.PRNGKey(1), 2, 12, 4))
            counts = count_v8_params(params, pol.n_shared + pol.topk)
            assert counts["active"] == expect, (direction, width, counts)
    print("parameter counts ok")


# ----------------------------------------------------------------------------------------------------------- 2
def test_moe_layer_matches_explicit_mixture():
    for gate, n_shared, n_routed, topk in (("deepseek", 1, 5, 3), ("renorm", 0, 6, 4)):
        layer = MoELayer(kind="ffn", n_shared=n_shared, n_routed=n_routed, topk=topk, gate=gate, hidden=8, features=12,
                         router_hidden=16, router_init=0.7)
        x = jax.random.normal(jax.random.PRNGKey(2), (5, 12), jnp.float64)
        morph = jax.random.normal(jax.random.PRNGKey(3), (5, 7), jnp.float64)
        params = layer.init(jax.random.PRNGKey(4), (x,), morph)
        y = layer.apply(params, (x,), morph)
        p = params["params"]
        mu, var = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
        xn = (x - mu) / jnp.sqrt(var + 1e-6) * p["ln"]["scale"] + p["ln"]["bias"]
        logits = Router(n_routed, 16, "both", 0.7).apply({"params": p["router"]}, morph, xn)
        prob = jax.nn.softmax(logits)
        top = jax.lax.top_k(prob, topk)[1]
        mask = jax.nn.one_hot(top, n_routed).sum(-2)
        w = prob * mask
        if gate == "renorm":
            w = w / w.sum(-1, keepdims=True)
            assert np.allclose(w.sum(-1), 1.0)
        w = jnp.concatenate([jnp.ones((5, n_shared)), w], -1)
        outs = FFNExperts(n_shared + n_routed, 8, 12).apply({"params": p["experts"]}, xn)
        ref = x + jnp.einsum("uk,uko->uo", w, outs)
        assert _rel_err(y, ref) < 1e-12, gate
        if gate == "deepseek":
            assert float(w[:, 1:].sum(-1).max()) < 1.0        # routed scores are not renormalized
    print("MoE layer mixture ok")


# ----------------------------------------------------------------------------------------------------------- 3
def test_noise_and_clean_evaluation():
    pol = _policy("compat")
    inp = _inputs(jax.random.PRNGKey(5), 6, 3, 2)
    params = pol.init(jax.random.PRNGKey(6), *inp)
    a = pol.apply(params, *inp)
    b = pol.apply(params, *inp)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
    (m1, _), s1 = pol.apply(params, *inp, mutable=[V8_ROUTING], rngs={"routing": jax.random.PRNGKey(7)})
    (m2, _), s2 = pol.apply(params, *inp, mutable=[V8_ROUTING], rngs={"routing": jax.random.PRNGKey(8)})
    assert not np.allclose(m1, m2)
    assert "noise_scale" in s1[V8_ROUTING]["core_moe1"]
    # A layer's auxiliary probabilities hold its noise fixed: no gradient reaches its own noise head (earlier layers'
    # noise heads shape its input, so theirs may receive one).
    for layer in MOE_LAYERS:
        def aux(prm):
            _, st = pol.apply(prm, *inp, mutable=[V8_ROUTING], rngs={"routing": jax.random.PRNGKey(7)})
            return jnp.sum(st[V8_ROUTING][layer]["p"][0] ** 2)
        g = _flat(jax.grad(aux)(params))
        assert all(float(jnp.abs(v).max()) == 0.0 for k, v in g.items() if k.startswith(f"params/{layer}/router_noise"))
        assert any(float(jnp.abs(v).max()) > 0.0 for k, v in g.items() if k.startswith(f"params/{layer}/router/"))
    print("noise and clean evaluation ok")


# ----------------------------------------------------------------------------------------------------------- 4
def test_aux_gradients_reach_upstream_layers():
    pol = _policy("mi_robot")
    inp = _inputs(jax.random.PRNGKey(9), 6, 3, 2)
    params = pol.init(jax.random.PRNGKey(10), *inp)

    def grads(layer):
        def f(prm):
            _, st = pol.apply(prm, *inp, mutable=[V8_ROUTING], rngs={"routing": jax.random.PRNGKey(11)})
            p = st[V8_ROUTING][layer]["p"][0]
            return jnp.sum(p * jnp.log(p))                    # any function of the routing probabilities
        return {k: float(jnp.abs(v).max()) for k, v in _flat(jax.grad(f)(params)).items()}

    g = grads("core_moe2")
    nz = lambda prefix: any(v > 0 for k, v in g.items() if k.startswith("params/" + prefix))
    for up in ("core_moe2/router/", "core_moe2/ln", "core_dense_2", "core_moe1/experts", "core_moe1/router/",
               "core_dense_0", "core_ln_0", "enc_dense_0", "enc_moe/experts", "joint_agg_0", "foot_dense_0"):
        assert nz(up), up
    for down in ("core_moe2/experts", "core_moe2/router_noise", "core_dense_4", "dec_moe", "std_"):
        assert not nz(down), down
    g = grads("enc_moe")
    assert all(v == 0 for k, v in g.items() if not k.startswith("params/enc_moe/router/"))
    g = grads("dec_moe")
    assert nz("core_dense_4") and nz("core_moe1/experts") and not nz("dec_moe/experts") and not nz("std_")
    print("auxiliary gradients reach upstream layers ok")


# ----------------------------------------------------------------------------------------------------------- 5
def _updater(direction, **over):
    pol = _policy(direction)
    groups = [np.array([0, 1, 0]), np.array([1, 2, 2, 0, 1])] if direction == "mi_joint" else None
    upd = vu.V8ActorUpdater(_config(direction, **over), pol, _split, ["a", "b"], None, joint_groups=groups, n_groups=3)
    upd.loss_fn = _toy_loss
    return upd, pol


def test_compat_scores_match_explicit_gradients():
    batch = 8
    upd, pol = _updater("compat", morse_compat_core_transitions=batch, morse_compat_decoder_transitions=batch)
    obs = tuple(_obs(jax.random.PRNGKey(20 + r), batch, r) for r in range(2))
    acts = tuple(jax.random.normal(jax.random.PRNGKey(30 + r), (batch, 6), jnp.float64) for r in range(2))
    params = pol.init(jax.random.PRNGKey(12), *_split(obs[0], 0))
    keys = [jax.random.PRNGKey(40), jax.random.PRNGKey(41)]
    templates = [upd._delta_template(r, jax.ShapeDtypeStruct(o.shape, o.dtype)) for r, o in enumerate(obs)]
    deltas = tuple(jax.tree_util.tree_map(lambda s: jnp.zeros(s.shape, s.dtype), t) for t in templates)
    fwd = upd._forward(True, obs, acts, 1.0, keys, keys)
    (task, ps), vjp_fn, (losses, recs) = jax.vjp(fwd, params, deltas, has_aux=True)
    g_task, g_deltas = vjp_fn((jnp.ones_like(task), jax.tree_util.tree_map(jnp.zeros_like, ps)))
    for layer in MOE_LAYERS:
        robots, explicit = [], []
        for r in range(2):
            layers, w = upd._expert_units(params, layer, recs[r], g_deltas[r])
            robots.append((layers, w, jnp.arange(batch) % 2))
            body = layer in vu.BODY_LAYERS
            lift = (lambda t: t[:, None]) if body else (lambda t: t)
            sc = recs[r]["score"][layer]
            ins = tuple(lift(sc[f"u{i}"]) for i in range(2 if layer == "dec_moe" else 1))
            d = lift(g_deltas[r][layer]["y"])
            sub = params["params"][layer]["experts"]
            k = w.shape[-1]
            mod = (DecoderExperts(k, sub["desc_dense_0"]["kernel"].shape[-1]) if layer == "dec_moe" else
                   FFNExperts(k, sub["dense_0"]["kernel"].shape[-1], sub["dense_1"]["kernel"].shape[-1]))

            def unit_grad(*args):
                *u, du = args
                f = lambda s: jnp.sum(mod.apply({"params": s}, *[x[None] for x in u])[0] * du[None, :])
                g = jax.grad(f)(sub)
                return jnp.concatenate([v.reshape(k, -1) for v in jax.tree_util.tree_leaves(g)], axis=1)  # (K, P)
            flat_in = [x.reshape((-1,) + x.shape[2:]) for x in ins]
            gu = jax.vmap(unit_grad)(*flat_in, d.reshape(-1, d.shape[-1]))          # (U, K, P)
            explicit.append((gu, w.reshape(-1, k)))
        ref = sum(jnp.einsum("uk,ukp->kp", w, gu) for gu, w in explicit)
        # The inclusive reference is every expert's task gradient.
        tg = jnp.concatenate([v.reshape(ref.shape[0], -1) for v in jax.tree_util.tree_leaves(g_task["params"][layer]["experts"])], 1)
        assert _rel_err(ref, tg) < 1e-5, layer          # float32 parameters
        out, support, _ = vu.inclusive_compat(robots, 1, 0.05)
        for (gu, _), (_, _, cos) in zip(explicit, out):
            c = jnp.einsum("ukp,kp->uk", gu, ref) / (jnp.linalg.norm(gu, axis=-1) * jnp.linalg.norm(ref, axis=-1)[None])
            valid = (jnp.linalg.norm(gu, axis=-1) > 0) & (jnp.linalg.norm(ref, axis=-1) > 0)[None]
            assert _rel_err(jnp.where(valid, c, 0.0), cos.reshape(c.shape)) < 1e-5, layer
    print("compatibility scores match explicit gradients ok")


# ----------------------------------------------------------------------------------------------------------- 6
def test_two_part_cap():
    upd, _ = _updater("mi_robot")
    r = lambda s: jnp.full((3,), s, jnp.float64)
    g_task = {"params": {"core_moe1": {"router": {"w": r(1.0)}, "experts": {"w": r(5.0)}},
                         "enc_moe": {"router": {"w": r(1.0)}}, "core_moe2": {"router": {"w": r(0.0)}},
                         "dec_moe": {"router": {"w": r(1.0)}}, "core_dense_0": {"w": r(2.0)}, "std_head": {"w": r(9.0)}}}
    g_obj = {"params": {"core_moe1": {"router": {"w": r(10.0)}, "experts": {"w": r(0.0)}},
                        "enc_moe": {"router": {"w": r(0.1)}}, "core_moe2": {"router": {"w": r(1.0)}},
                        "dec_moe": {"router": {"w": r(0.0)}}, "core_dense_0": {"w": r(4.0)}, "std_head": {"w": r(0.0)}}}
    capped, factors, _ = upd._cap(g_task, g_obj)
    n = lambda t: float(jnp.linalg.norm(t))
    assert abs(n(capped["params"]["core_moe1"]["router"]["w"]) - 0.75 * n(g_task["params"]["core_moe1"]["router"]["w"])) < 1e-12
    assert np.allclose(capped["params"]["enc_moe"]["router"]["w"], g_obj["params"]["enc_moe"]["router"]["w"])   # below cap
    assert float(factors["core_moe2"]) == 0.0                     # no task gradient on that router
    assert float(factors["dec_moe"]) == 1.0                       # no objective there
    # Upstream: only the leaves the objective reaches (core_dense_0) set the reference, not std_head or the experts.
    assert abs(n(capped["params"]["core_dense_0"]["w"]) - 0.75 * n(g_task["params"]["core_dense_0"]["w"])) < 1e-12
    print("two-part cap ok")


# ----------------------------------------------------------------------------------------------------------- 7
def test_mi_and_balance_definitions():
    key = jax.random.PRNGKey(13)
    ps = [jax.nn.softmax(jax.random.normal(jax.random.fold_in(key, r), (4, nj, 5)), -1)
          for r, (nj, _) in enumerate(_ROBOTS)]
    groups = [np.array([0, 1, 0]), np.array([1, 2, 2, 0, 1])]
    # Explicit: per unit weight 1/R * 1/J_r / n.
    w_units = [np.full((4, nj), 1.0 / (2 * nj * 4)) for nj, _ in _ROBOTS]
    prior = np.zeros(3)
    r_g = np.zeros((3, 5))
    for p, g, w in zip(ps, groups, w_units):
        for j, gj in enumerate(g):
            prior[gj] += w[:, j].sum()
            r_g[gj] += (np.asarray(p[:, j]) * w[:, j, None]).sum(0)
    r_g /= prior[:, None]
    h = lambda q: -np.sum(q * np.log(q + 1e-12), -1)
    expect = np.sum(prior * h(r_g)) - h(prior @ r_g)
    got = vu.joint_group_mi(ps, groups, 3)[0]
    assert abs(float(got) - expect) < 1e-6          # float32 group weights
    rr = np.stack([np.asarray(p).reshape(-1, 5).mean(0) for p in ps])
    expect_r = np.mean(h(rr)) - h(rr.mean(0))
    assert abs(float(vu.robot_mi(ps)[0]) - expect_r) < 1e-10
    pbar = rr.mean(0)
    assert abs(float(vu.balance_loss(ps)[0]) - np.sum(pbar * np.log(5 * pbar + 1e-12))) < 1e-10
    print("MI and balance definitions ok")


# ----------------------------------------------------------------------------------------------------------- 8
def test_update_runs_for_every_direction():
    from flax.training.train_state import TrainState
    import optax
    batch = 8
    for direction in ("mi_joint", "mi_robot", "compat", "none"):
        upd, pol = _updater(direction)
        obs = tuple(_obs(jax.random.PRNGKey(50 + r), batch, r) for r in range(2))
        acts = tuple(jax.random.normal(jax.random.PRNGKey(60 + r), (batch, 6), jnp.float64) for r in range(2))
        batches = [SimpleNamespace(observations=o, actions=a) for o, a in zip(obs, acts)]
        params = pol.init(jax.random.PRNGKey(14), *_split(obs[0], 0))
        state = TrainState.create(apply_fn=pol.apply, params=params, tx=optax.adam(1e-3))
        rng = jax.random.PRNGKey(15)
        for step in range(4):
            for sub in range(2):
                state, loss, metrics, rng = upd.update(state, 1.0, batches, rng, step, sub, 2)
                assert np.isfinite(float(loss))
                if step % 2 == 0 and sub == 0:
                    assert metrics and "morse/v8/core_moe1/balance_loss" in metrics, direction
                    if direction in ("mi_joint", "mi_robot"):
                        assert "morse/v8/enc_moe/mi" in metrics
                        assert metrics["morse/v8/core_moe1/router_grad_objective"] * metrics["morse/v8/core_moe1/cap_factor"] \
                            <= 0.75 * metrics["morse/v8/core_moe1/router_grad_task"] + 1e-9
                    if direction == "compat" and step >= 1:
                        assert "morse/v8/dec_moe/compat_nonconf_frac" in metrics
        moved = jax.tree_util.tree_map(lambda a, b: float(jnp.abs(a - b).max()), state.params, params)
        assert max(jax.tree_util.tree_leaves(moved)) > 0
    print("update runs for every direction ok")


def test_single_backward_pass_matches_separate_passes():
    """Without the cap, the one-pass update (summed losses) equals the update from separate task, balance and objective
    passes (the diagnostic path); the objective waits for the warm-up and ramps in."""
    from flax.training.train_state import TrainState
    import optax
    batch = 8
    for direction in ("mi_joint", "compat"):
        upd, pol = _updater(direction, morse_v8_cap=0.0, morse_warmup=10, morse_ramp=4)
        assert upd.objective_factor_at(9) == 0.0 and upd.objective_factor_at(10) == 0.25 and upd.objective_factor_at(13) == 1.0
        obs = tuple(_obs(jax.random.PRNGKey(70 + r), batch, r) for r in range(2))
        acts = tuple(jax.random.normal(jax.random.PRNGKey(80 + r), (batch, 6), jnp.float64) for r in range(2))
        params = pol.init(jax.random.PRNGKey(16), *_split(obs[0], 0))
        state = TrainState.create(apply_fn=pol.apply, params=params, tx=optax.sgd(1.0))
        args = (state, 1.0, obs, acts, jax.random.PRNGKey(17), jnp.int32(0), jnp.asarray([3, 0], jnp.uint32),
                jnp.float32(1.0))
        fast = upd._step_fn(obs, 1, False, False)(*args)
        sep = upd._step_fn(obs, 1, False, True)(*args)
        err = max(jax.tree_util.tree_leaves(jax.tree_util.tree_map(
            lambda a, b, c: float(jnp.abs(a - b).max() / (jnp.abs(c - a).max() + 1e-30)),
            fast[0].params, sep[0].params, params)))
        assert err < 1e-3, (direction, err)          # float32 parameters, different summation order
        if direction == "mi_joint":
            assert "morse/v8/core_moe1/objective_task_ratio" in sep[2] and "morse/v8/core_moe1/cap_factor" not in sep[2]
    print("single backward pass matches separate passes ok")


# ----------------------------------------------------------------------------------------------------------- 9
def test_joint_groups():
    rs = np.random.RandomState(0)
    desc = [rs.randn(nj, 23) for nj in (6, 9, 4)]
    mirrored = [d.copy() for d in desc]
    for d in mirrored:
        d[:, 3:6] *= -1
        d[:, 7] *= -1
        d[:, 14:16] = -d[:, 15:13:-1]
    assert np.allclose(joint_features(desc[0]), joint_features(mirrored[0]))
    # Torque and velocity limits are relative to the robot's largest: scaling all of a robot's limits changes nothing.
    pos = desc[0].copy()
    pos[:, 8:10] = rs.rand(6, 2) - 0.5                      # stored values of positive limits
    scaled = pos.copy()
    scaled[:, 8] = (pos[:, 8] + 1.0) * 1.7 - 1.0
    scaled[:, 9] = (pos[:, 9] + 1.0) * 0.4 - 1.0
    assert np.allclose(joint_features(pos), joint_features(scaled))
    assert np.isclose(joint_features(pos)[:, 5].max(), 1.0) and np.isclose(joint_features(pos)[:, 6].max(), 1.0)
    t1 = build_joint_groups(desc, ["a", "b", "c"], k=3)
    t2 = build_joint_groups(mirrored, ["a", "b", "c"], k=3)
    assert t1["groups"] == t2["groups"]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "groups.json")
        write_joint_groups(path, t1)
        groups, k = load_joint_groups(path, ["a", "b", "c"], [6, 9, 4])
        assert k == 3 and [len(g) for g in groups] == [6, 9, 4]
        try:
            load_joint_groups(path, ["a", "b", "c"], [6, 9, 5])
            raise AssertionError("joint-count mismatch not detected")
        except ValueError:
            pass
    print("joint groups ok")


if __name__ == "__main__":
    test_parameter_counts()
    test_moe_layer_matches_explicit_mixture()
    test_noise_and_clean_evaluation()
    test_aux_gradients_reach_upstream_layers()
    test_compat_scores_match_explicit_gradients()
    test_two_part_cap()
    test_mi_and_balance_definitions()
    test_update_runs_for_every_direction()
    test_single_backward_pass_matches_separate_passes()
    test_joint_groups()
    print("all V8 tests passed")
