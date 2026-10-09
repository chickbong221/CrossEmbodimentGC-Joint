"""Correctness tests for MORSE.

Run with ``python tests/test_morse.py`` (or pytest). They check that
  1. the factorized Gram matrix of a MoE block equals the Gram matrix of
     explicit per-unit gradients, including LayerNorm parameters and an
     objective that couples the outputs of different units;
  2. through the full policy and across robots with different numbers of
     joints, summing the Gram entries recovers ||grad||^2 of every block;
  3. the auxiliary routing objectives only change router parameters;
  4. the shuffled control keeps the conflict values of every expert;
  5. a block with one expert is the plain shared MLP (no router);
  6. with one expert per block, MORSE is exactly the original URMA policy;
  7. robot-level statistics use full-batch per-robot gradients (routers excluded) and satisfy
     ||sum_r g_r||^2 = S + C+ - C-;
  8. type-level evidence sums, normalizes, and expands correctly;
  9. the compiled update (all robots, router objectives, optimizer step in one function) equals the
     step-by-step computation, including the accumulators and router-only adaptation;
 10. the background minibatch sampler gathers exactly the rows of Dataset.sample and keeps the batch sizes;
 11. the type-expert mutual information loss equals I(A;E) of the routing distribution;
 12. signed compatibility evidence ignores cosines within z noise std (even/odd estimate);
 13. a fixed decoder routing table has no router and routes by the table; identical experts and the
     per-joint output calibration leave the initial policy unchanged;
 14. the probe's per-joint decoder gradients sum to the robot's gradient, its parameter groups add up,
     and small lookahead steps match the first-order prediction;
 15. cross-fitted compatibility: per-type gradient sums from the captures equal explicit per-unit gradients
     summed per element; without noise its cosines are exact, with noise its threshold follows the
     disagreement of two independent estimates; in the update, a stream holds every robot's decoder gradient.
 16. joint-level PCGrad: closed form for two gradients, no change without conflicts, and the PCGrad path with an
     identity projection reproduces the ordinary update.
 17. robot-level MI -I(R;E) and joint-type MI with unit-share or equal-robot priors match the definition.
 18. decoder top-k routing (renormalized mixture of the kept experts, no gradient to dropped ones) and a decoder
     expert count and width of its own next to the core's experts.
 19. checkpoint analysis: the decoder recomputation reproduces the policy and the Shapley parts add up.
 20. coordination dump: episodes from done flags, reference episodes, both policies' means on exactly their rows.
 21. revised method: exact top-k (ties included); joint-conditioned core routing with per-joint core representations;
     training, evaluation and restored policies agree; joint-type MI of core and decoder with equal robot weights
     reaches only the two routers, on every actor step, with eta = 0.
 22. v6: body-routed core with top-4 of 8 (one core representation per transition) and its parameter counts; the
     core's MI term with its own weight, as -H(E) only or the full robot-level -I(R;E), reaches only the core router.
 23. top-k tie rules; routing noise (per-transition draws shared by shorter batches, clean evaluation, recorded
     offset, noise head trained by the task); robot-fixed core and decoder use only the robot's expert.
 24. compatibility routing: per-unit cosines with each expert's actual update from the other transitions of each
     half equal explicit per-unit gradients; labels; the KL objective and its controls; stratified shuffle.
 25. EG groups in the core's MI; the per-block MI cap; routing-noise keys off the main key sequence; the
     compatibility step moves only routers.
 26. dataset row and robot-column selection; the sampler's index checksum and content hash.
 27. shared core experts (global, EG-group, hierarchical): the output equals the explicit weighted sum of the experts;
     each robot uses its own group's expert; two renormalized routed weights; task gradients reach the global expert
     from every robot, a group expert only from its group's robots, and routed experts only where selected; MI and
     balance act on the routed experts only and reach only the router (not the noise head or any expert); per-block MI
     caps; clean deterministic evaluation; parameter round trip; configurations without the new keys build the original
     architecture; inconsistent or missing group mappings fail.
 29. frozen core (core-freeze continuation): gradients reach the frozen core's inputs, the clip excludes it, zeroed first moments keep
     it bitwise fixed, the optimizer state keeps its structure.
 28. variance modes (continuation experiment): detach, private and protected-mean log-std pathways start from the
     original forward pass, separate the mean and variance gradients as specified, and keep the optimizer state.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
import jax.numpy as jnp
from flax.linen.initializers import lecun_normal

jax.config.update("jax_enable_x64", True)

from cross_embodiment_offline_rl.algorithms.morse import interference as itf
from cross_embodiment_offline_rl.algorithms.morse.moe import CAPTURE_DELTA, CAPTURE_IN, CAPTURE_ROUTER, MoEBlock
from cross_embodiment_offline_rl.algorithms.morse.policy import BLOCKS, MorsePolicy
from cross_embodiment_offline_rl.algorithms.morse.updater import BLOCK_LAYERS, extract_captures, _units


def _rel_err(a, b):
    return float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-30))


def _block(capture, n_capture=0, num_experts=3):
    return MoEBlock(layer_features=(6, 2), layer_post=("ln_elu", "none"),
                    kernel_inits=(lecun_normal(), lecun_normal()), num_experts=num_experts,
                    router_hidden=8, capture=capture, n_capture=n_capture)


def test_block_gram_matches_explicit_per_unit_gradients():
    k0, k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 5)
    n_trans, n_elem, d_in, n_experts = 3, 4, 5, 3
    x = jax.random.normal(k0, (n_trans, n_elem, d_in), jnp.float64)
    morph = jax.random.normal(k1, (n_trans, n_elem, 3), jnp.float64)
    state = jax.random.normal(k2, (n_trans, n_elem, 2), jnp.float64)
    w = jax.random.normal(k3, (n_trans, n_elem, 2), jnp.float64)

    def objective(out):
        # Couples all elements of a transition through a nonlinearity.
        return jnp.sum(jnp.tanh((out * w).sum(axis=(1, 2))) ** 2)

    block_c, block = _block(True, n_trans, n_experts), _block(False)
    variables = block_c.init(k4, x, morph, state)
    params = {"params": variables["params"]}
    # Break the uniform initial routing so that routing weights differ across units.
    params = jax.tree_util.tree_map(lambda a: a.astype(jnp.float64) + 0.3 * jax.random.normal(k3, a.shape, jnp.float64), params)
    delta0 = jax.tree_util.tree_map(jnp.zeros_like, variables[CAPTURE_DELTA])

    def f(prm, delta):
        out, st = block_c.apply({"params": prm["params"], CAPTURE_DELTA: delta}, x, morph, state,
                                mutable=[CAPTURE_IN, CAPTURE_ROUTER])
        return objective(out), st

    (_, st), (_, gd) = jax.value_and_grad(f, argnums=(0, 1), has_aux=True)(params, delta0)
    layers = [
        ("dense", _units(st[CAPTURE_IN]["dense_0"]["x"][0], 1), _units(gd["dense_0"]["z"], 2)),
        ("ln", _units(st[CAPTURE_IN]["ln_0"]["xhat"][0], 2), _units(gd["ln_0"]["y"], 2)),
        ("dense", _units(st[CAPTURE_IN]["dense_1"]["x"][0], 2), _units(gd["dense_1"]["z"], 2)),
    ]
    g_fact = itf.gram(layers)

    # Explicit per-unit contributions: give every unit its own copy of the expert parameters.
    n_units = n_trans * n_elem
    experts = {k: v for k, v in params["params"].items() if k != "router"}
    copies = jax.tree_util.tree_map(lambda a: jnp.broadcast_to(a, (n_units,) + a.shape), experts)

    def loss_from_copies(c):
        def one(cu, xu, mu, su):
            prm = {"params": {**cu, "router": params["params"]["router"]}}
            return block.apply(prm, xu[None], mu[None], su[None])[0]
        out = jax.vmap(one)(c, x.reshape(n_units, -1), morph.reshape(n_units, -1), state.reshape(n_units, -1))
        return objective(out.reshape(n_trans, n_elem, -1))

    per_unit = jax.grad(loss_from_copies)(copies)
    leaves = jax.tree_util.tree_leaves(per_unit)
    g_expl = jnp.stack([
        (lambda f_: f_ @ f_.T)(jnp.concatenate([leaf[:, k].reshape(n_units, -1) for leaf in leaves], axis=1))
        for k in range(n_experts)
    ])
    err = _rel_err(g_fact, g_expl)
    assert err < 1e-9, f"factorized Gram differs from explicit per-unit Gram (rel err {err:.2e})"
    print(f"[ok] block Gram == explicit per-unit Gram (rel err {err:.1e}, K={n_experts}, U={n_units})")


def _policy(capture, n_capture=0, num_experts=3, deterministic=False):
    return MorsePolicy(std_dev=1.0, softmax_temperature=1.0, softmax_temperature_min=0.015,
                       stability_epsilon=1e-8, policy_mean_abs_clip=10.0, policy_std_min_clip=1e-8,
                       policy_std_max_clip=2.0, deterministic=deterministic, num_experts=num_experts,
                       router_hidden=16, capture=capture, n_capture_element=n_capture, n_capture_body=n_capture)


def _robot_inputs(key, batch, n_joints, n_feet):
    ks = jax.random.split(key, 6)
    return (jax.random.normal(ks[0], (batch, n_joints, 23), jnp.float64),
            jax.random.normal(ks[1], (batch, n_joints, 3), jnp.float64),
            jax.random.normal(ks[2], (batch, n_feet, 10), jnp.float64),
            jax.random.normal(ks[3], (batch, n_feet, 2), jnp.float64),
            jax.random.normal(ks[4], (batch, 16), jnp.float64)), ks[5]


def test_policy_gram_sums_to_block_gradient_norm_across_robots():
    batch = 4
    robots = [(5, 2), (7, 3)]
    policy_c, policy = _policy(True, batch), _policy(False)
    inputs0, _ = _robot_inputs(jax.random.PRNGKey(1), batch, *robots[0])
    params = policy.init(jax.random.PRNGKey(2), *inputs0)
    params = jax.tree_util.tree_map(lambda a: a.astype(jnp.float64) + 0.2 * jax.random.normal(jax.random.PRNGKey(3), a.shape, jnp.float64), params)

    captures, grads = [], []
    for r, (n_joints, n_feet) in enumerate(robots):
        inputs, wkey = _robot_inputs(jax.random.PRNGKey(10 + r), batch, n_joints, n_feet)
        w = jax.random.normal(wkey, (batch, n_joints), jnp.float64)
        template = jax.eval_shape(lambda: policy_c.init(jax.random.PRNGKey(0), *inputs))[CAPTURE_DELTA]
        delta0 = jax.tree_util.tree_map(lambda s: jnp.zeros(s.shape, s.dtype), template)

        def f(prm, delta):
            (mean, logstd), st = policy_c.apply({"params": prm["params"], CAPTURE_DELTA: delta}, *inputs,
                                                mutable=[CAPTURE_IN, CAPTURE_ROUTER])
            loss = jnp.sum(jnp.tanh((mean * w).sum(axis=1)) ** 2) + jnp.mean(logstd ** 2)  # couples joints
            return loss, st

        (_, st), (gp, gd) = jax.value_and_grad(f, argnums=(0, 1), has_aux=True)(params, delta0)
        captures.append(extract_captures(st, gd))
        grads.append(gp)

    total_grad = jax.tree_util.tree_map(lambda a, b: a + b, *grads)
    for block in BLOCKS:
        layers = []
        for li, (kind, i) in enumerate(BLOCK_LAYERS[block]):
            x_tail = 1 if (kind == "dense" and i == 0) else 2
            layers.append((kind,
                           jnp.concatenate([_units(c[block]["x"][li], x_tail) for c in captures]),
                           jnp.concatenate([_units(c[block]["d"][li], 2) for c in captures])))
        g = itf.gram(layers)
        experts = {k: v for k, v in total_grad["params"][block].items() if k != "router"}
        leaves = jax.tree_util.tree_leaves(experts)
        for k in range(g.shape[0]):
            norm_sq = sum(jnp.sum(leaf[k] ** 2) for leaf in leaves)
            err = abs(float(g[k].sum() - norm_sq)) / (float(norm_sq) + 1e-30)
            assert err < 1e-9, f"{block}/expert {k}: sum of Gram != ||grad||^2 (rel err {err:.2e})"
    print(f"[ok] policy Gram sums to ||grad_phi_k L||^2 for every block and expert (robots {robots})")


def test_aux_objectives_only_touch_routers():
    batch = 4
    policy = _policy(False)
    inputs, _ = _robot_inputs(jax.random.PRNGKey(4), batch, 6, 2)
    params = policy.init(jax.random.PRNGKey(5), *inputs)
    params = jax.tree_util.tree_map(lambda a: a + 0.2 * jax.random.normal(jax.random.PRNGKey(6), a.shape, a.dtype), params)
    from cross_embodiment_offline_rl.algorithms.morse.moe import Router
    morph = jax.random.normal(jax.random.PRNGKey(7), (9, 58), jnp.float64)
    state = jax.random.normal(jax.random.PRNGKey(8), (9, 132), jnp.float64)
    q = jnp.triu(jax.random.uniform(jax.random.PRNGKey(9), (3, 9, 9), jnp.float64), k=1)

    def aux(prm):
        logits = Router(3, 16).apply({"params": prm["params"]["decoder"]["router"]}, morph, state)
        pt = jax.nn.softmax(logits, -1)
        return itf.conflict_loss(pt, q) + 0.1 * itf.utilization_loss(pt)

    g = jax.grad(aux)(params)
    for path, leaf in jax.tree_util.tree_leaves_with_path(g):
        is_router = any(getattr(p, "key", None) == "router" for p in path)
        if not is_router:
            assert float(jnp.max(jnp.abs(leaf))) == 0.0, f"aux gradient leaked into {path}"
    router_norm = sum(float(jnp.sum(leaf ** 2)) for leaf in jax.tree_util.tree_leaves(g["params"]["decoder"]["router"]))
    assert router_norm > 0.0
    print("[ok] auxiliary objectives update only router parameters")


def test_shuffle_preserves_conflict_values():
    q = jnp.triu(jax.random.uniform(jax.random.PRNGKey(11), (2, 12, 12)), k=1)
    pair = q > 0
    q_s = itf.shuffle_units(q, jnp.ones_like(pair), jax.random.PRNGKey(12))
    for k in range(2):
        a = np.sort(np.asarray(q[k][np.triu_indices(12, 1)]))
        b = np.sort(np.asarray(q_s[k][np.triu_indices(12, 1)]))
        assert np.allclose(a, b), "shuffled control changed the conflict values"
    print("[ok] shuffled control keeps each expert's conflict values")


def test_single_expert_is_shared_block():
    block = _block(False, num_experts=1)
    x = jax.random.normal(jax.random.PRNGKey(13), (4, 5), jnp.float64)
    variables = block.init(jax.random.PRNGKey(14), x, x[:, :3], x[:, :2])
    prm = variables["params"]
    assert set(prm) == {"dense_0", "ln_0", "dense_1"}, f"a single-expert block must have no router, got {sorted(prm)}"
    h = x @ prm["dense_0"]["kernel"][0] + prm["dense_0"]["bias"][0]
    h = (h - h.mean(-1, keepdims=True)) / jnp.sqrt(h.var(-1, keepdims=True) + 1e-6)
    h = jax.nn.elu(h * prm["ln_0"]["scale"][0] + prm["ln_0"]["bias"][0])
    want = h @ prm["dense_1"]["kernel"][0] + prm["dense_1"]["bias"][0]
    out = block.apply(variables, x, x[:, :3], x[:, :2])
    assert _rel_err(out, want) < 1e-9, "a single-expert block must equal the plain MLP"
    print("[ok] a single-expert block is the plain shared MLP (no router, routing weight 1)")


def _urma_params_as_morse(orig):
    """Map the parameters of the original URMA policy onto a MorsePolicy with one expert per block."""
    o = orig["params"]
    dense = lambda d: {"kernel": d["kernel"][None], "bias": d["bias"][None]}
    ln = lambda d: {"scale": d["scale"][None], "bias": d["bias"][None]}
    p = {
        "joint_log_softmax_temperature": o["joint_log_softmax_temperature"],
        "foot_log_softmax_temperature": o["foot_log_softmax_temperature"],
        "Dense_0": o["Dense_0"], "LayerNorm_0": o["LayerNorm_0"], "Dense_1": o["Dense_1"],     # joint aggregation
        "joint_encoder": {"dense_0": dense(o["Dense_2"])},
        "Dense_2": o["Dense_3"], "LayerNorm_1": o["LayerNorm_1"], "Dense_3": o["Dense_4"],     # foot aggregation
        "foot_encoder": {"dense_0": dense(o["Dense_5"])},
        "core": {"dense_0": dense(o["Dense_6"]), "ln_0": ln(o["LayerNorm_2"]),
                 "dense_1": dense(o["Dense_7"]), "dense_2": dense(o["Dense_8"])},
        "Dense_4": o["Dense_9"], "LayerNorm_2": o["LayerNorm_3"], "Dense_5": o["Dense_10"],    # description embedding
        "decoder": {"dense_0": dense(o["Dense_11"]), "ln_0": ln(o["LayerNorm_4"]), "dense_1": dense(o["Dense_12"])},
    }
    if "Dense_13" in o:
        p["Dense_6"] = o["Dense_13"]                                                           # log std (stochastic)
    return {"params": p}


def test_single_expert_policy_reproduces_urma():
    from cross_embodiment_offline_rl.algorithms.uni_iql_eg.iql.policy import Policy as UrmaGaussian
    from cross_embodiment_offline_rl.algorithms.uni_td3bc_eg.td3bc.policy import Policy as UrmaDeterministic
    hp = dict(std_dev=1.0, softmax_temperature=1.0, softmax_temperature_min=0.015, stability_epsilon=1e-8,
              policy_mean_abs_clip=10.0, policy_std_min_clip=1e-8, policy_std_max_clip=2.0)
    for deterministic, urma in ((False, UrmaGaussian(**hp)), (True, UrmaDeterministic(**hp, max_action=1.0))):
        inputs, _ = _robot_inputs(jax.random.PRNGKey(20), 5, 7, 3)
        orig = urma.init(jax.random.PRNGKey(21), *inputs)
        orig = jax.tree_util.tree_map(lambda a: a.astype(jnp.float64) + 0.3 * jax.random.normal(jax.random.PRNGKey(22), a.shape, jnp.float64), orig)
        morse = _policy(False, num_experts=1, deterministic=deterministic)
        mapped = _urma_params_as_morse(orig)
        own = morse.init(jax.random.PRNGKey(23), *inputs)
        shapes = lambda t: jax.tree_util.tree_map(lambda a: a.shape, t)
        assert shapes(own) == shapes(mapped), "K=1 MORSE parameters must be exactly the URMA parameters"
        n_orig = sum(a.size for a in jax.tree_util.tree_leaves(orig))
        n_morse = sum(a.size for a in jax.tree_util.tree_leaves(own))
        assert n_orig == n_morse, f"parameter counts differ: URMA {n_orig}, MORSE K=1 {n_morse}"
        for r, (n_joints, n_feet) in enumerate([(7, 3), (12, 4)]):
            x, _ = _robot_inputs(jax.random.PRNGKey(30 + r), 5, n_joints, n_feet)
            want, got = urma.apply(orig, *x), morse.apply(mapped, *x)
            for w, g in zip(jax.tree_util.tree_leaves(want), jax.tree_util.tree_leaves(got)):
                assert _rel_err(g, w) < 1e-9, f"K=1 MORSE output differs from URMA (rel err {_rel_err(g, w):.2e})"
        print(f"[ok] K=1 MORSE reproduces the {'deterministic' if deterministic else 'Gaussian'} URMA policy "
              f"({n_orig} parameters, same outputs)")


def test_robot_level_statistics():
    from cross_embodiment_offline_rl.algorithms.morse.updater import robot_block_grams
    policy = _policy(False)
    inputs, _ = _robot_inputs(jax.random.PRNGKey(40), 4, 6, 2)
    params = policy.init(jax.random.PRNGKey(41), *inputs)
    n_robots = 5
    grads = [jax.tree_util.tree_map(lambda a: jax.random.normal(jax.random.PRNGKey(50 + r), a.shape, jnp.float64), params)
             for r in range(n_robots)]
    grams = robot_block_grams(grads)
    for block in BLOCKS:
        flat = jnp.stack([jnp.concatenate([leaf.ravel() for path, leaf in jax.tree_util.tree_leaves_with_path(g["params"][block])
                                           if all(getattr(k, "key", None) != "router" for k in path)]) for g in grads])
        assert _rel_err(grams[block], flat @ flat.T) < 1e-12, f"{block}: robot Gram differs from explicit dot products"
        stats = itf.robot_level_stats(grams[block])
        m = grams[block]
        iu = np.triu_indices(n_robots, 1)
        self_term, pairs = float(jnp.trace(m)), np.asarray(m)[iu]
        total = float(jnp.sum(flat.sum(0) ** 2))                     # ||sum_r g_r||^2
        want = 1.0 - total / (self_term + 2 * pairs[pairs > 0].sum())
        assert abs(float(stats["cancelled_fraction"]) - want) < 1e-9, "cancelled fraction identity violated"
        assert 0.0 <= float(stats["cancelled_fraction"]) <= 1.0
        assert np.allclose(np.diag(stats["cos"]), 1.0)
    actor = jnp.stack([jnp.concatenate([leaf.ravel() for path, leaf in jax.tree_util.tree_leaves_with_path(g["params"])
                                        if all(getattr(k, "key", None) != "router" for k in path)]) for g in grads])
    assert _rel_err(grams["actor"], actor @ actor.T) < 1e-12, "actor-level robot Gram differs from explicit dot products"
    print("[ok] robot-level Gram of full-batch gradients (per block and whole actor, routers excluded) and cancelled fraction")


def test_type_level_evidence():
    rng = np.random.default_rng(0)
    n_units, n_types, n_experts = 9, 3, 2
    feats = rng.normal(size=(n_experts, n_units, 5))
    g = jnp.asarray(np.einsum("kud,kvd->kuv", feats, feats))
    type_ids = np.array([0, 0, 1, 1, 1, 2, 2, 0, 2])
    onehot = np.eye(n_types)[type_ids]
    s = itf.type_sums(g, onehot)
    for a in range(n_types):
        for b in range(n_types):
            expl = g[:, type_ids == a][:, :, type_ids == b].sum(axis=(1, 2))
            assert np.allclose(s[:, a, b], expl), "type sums differ from explicit block sums"
    ev = itf.type_evidence(s, zeta=0.0, kappa_fixed=0.0, calibrated=True)
    assert float(jnp.max(jnp.abs(ev["cos"]))) <= 1.0 + 1e-9
    q_unit = itf.expand_type_q(ev["q"], jnp.asarray(type_ids))
    full = ev["q"] + jnp.swapaxes(ev["q"], 1, 2)
    for u in range(n_units):
        for v in range(n_units):
            want = full[:, type_ids[u], type_ids[v]] if u < v else jnp.zeros(n_experts)
            assert np.allclose(q_unit[:, u, v], want), "unit pairs must inherit the coefficient of their types"
    print("[ok] type-level evidence: sums, cosines, and expansion to unit pairs")


def test_independent_type_sums_remove_shared_sample_covariance():
    # Two joints (types 0, 1) of one robot over n transitions. Each "expert" slot is an independent draw,
    # so the mean over slots is the expectation that accumulation over updates estimates.
    rng = np.random.default_rng(1)
    n_trans, dim, draws = 8, 6, 4000
    trans = np.repeat(np.arange(n_trans), 2)
    onehot = np.eye(2)[np.tile([0, 1], n_trans)]
    cross = (trans[:, None] != trans[None, :]).astype(np.float64)

    def cosines(joint0, joint1):
        vecs = np.stack([joint0, joint1], axis=2).reshape(draws, 2 * n_trans, dim)     # units (t, joint)
        g = jnp.asarray(np.einsum("kud,kvd->kuv", vecs, vecs))
        ind, same = itf.type_sums_split(g, onehot, cross)
        assert np.allclose(same, itf.type_sums(g, onehot)), "same-batch sums must equal plain type sums"
        assert np.allclose(jnp.diagonal(ind, axis1=1, axis2=2), jnp.diagonal(same, axis1=1, axis2=2))
        return [float(s[0, 1] / jnp.sqrt(s[0, 0] * s[1, 1])) for s in (ind.mean(0), same.mean(0))]

    # Shared zero-mean noise with opposite signs: no expected conflict, yet the same-batch estimate says -1.
    eps = rng.normal(size=(draws, n_trans, dim))
    c_ind, c_same = cosines(eps, -eps)
    assert c_same < -0.99 and abs(c_ind) < 0.05, (c_ind, c_same)
    # A genuine conflict (+m vs -m) hidden by positively shared noise: only independent pairs reveal it.
    m = np.full(dim, 0.3)
    c_ind, c_same = cosines(m + eps, -m + eps)
    assert c_same > 0.0 and c_ind < -0.3, (c_ind, c_same)
    print("[ok] independent type sums: no conflict from shared-sample noise, genuine conflicts kept")


def test_conflict_loss_scales_with_conflict():
    rng = np.random.default_rng(2)
    n_units, n_experts = 7, 3
    q = np.triu(rng.random((n_experts, n_units, n_units)), 1) * (rng.random((n_experts, n_units, n_units)) < 0.3)
    q = jnp.asarray(q)
    uniform = jnp.full((n_units, n_experts), 1.0 / n_experts)
    n_pairs = n_units * (n_units - 1) / 2
    assert abs(float(itf.conflict_loss(uniform, q, "pairs")) - float(q.sum()) / (n_experts ** 2 * n_pairs)) < 1e-12
    assert abs(float(itf.conflict_loss(uniform, q, "mass")) - 1.0 / n_experts ** 2) < 1e-9
    p = jax.nn.softmax(jnp.asarray(rng.normal(size=(n_units, n_experts))), axis=-1)
    for c in (0.1, 3.0):
        # "pairs": the pressure follows the strength of the evidence; "mass": invariant to it.
        assert _rel_err(itf.conflict_loss(p, c * q, "pairs"), c * itf.conflict_loss(p, q, "pairs")) < 1e-12
        assert _rel_err(itf.conflict_loss(p, c * q, "mass"), itf.conflict_loss(p, q, "mass")) < 1e-9
    print("[ok] conflict loss: mean over unit pairs scales with the conflict, mass-normalized does not")


_ROBOTS = [(5, 2), (7, 3), (4, 2)]      # (joints, feet) of three toy robots


def _split(obs, r):
    nj, nf = _ROBOTS[r]
    a, b, c = nj * 23, nj * 26, nj * 26 + nf * 10
    return (obs[:, :a].reshape(-1, nj, 23), obs[:, a:b].reshape(-1, nj, 3),
            obs[:, b:c].reshape(-1, nf, 10), obs[:, c:c + nf * 2].reshape(-1, nf, 2), obs[:, -16:])


def _toy_loss(out, obs, act, extra, key, r):
    mean, logstd = out
    return extra * (jnp.mean((mean - act[:, :mean.shape[1]]) ** 2) + 0.1 * jnp.mean(logstd ** 2))


def _updater(evidence, router_only, conflict_norm="pairs", unbiased_types=True):
    from types import SimpleNamespace
    from cross_embodiment_offline_rl.algorithms.morse.updater import MorseActorUpdater
    a = SimpleNamespace(morse_blocks="joint_encoder,foot_encoder,core,decoder", morse_num_experts=3, morse_router_hidden=16,
                        morse_router_inputs="both", morse_conflict_mode="pairwise", morse_lambda=1.0, morse_eta=0.01,
                        morse_conflict_norm=conflict_norm, morse_unbiased_types=unbiased_types,
                        morse_zeta=1.0, morse_kappa=0.0, morse_threshold="calibrated", morse_relevance=True,
                        morse_evidence=evidence, morse_type_ema=0.9, morse_conflict_every=1, morse_warmup=2, morse_ramp=2,
                        morse_router_only=router_only, morse_diag_frequency=2, morse_snapshot_frequency=0, morse_robot_ema=0.9)
    upd = MorseActorUpdater(SimpleNamespace(algorithm=a), _policy(True, 2), _split, ["a", "b", "c"], None)
    upd.loss_fn = _toy_loss
    return upd


def test_fused_update_matches_stepwise_reference():
    import optax
    from flax.training.train_state import TrainState
    from cross_embodiment_offline_rl.algorithms.morse.updater import _accumulate_robot_grams, _is_router, _router_norm
    from cross_embodiment_offline_rl.algorithms.utils import Transition
    batch = 6
    rng = np.random.default_rng(7)
    batches = []
    for nj, nf in _ROBOTS:
        obs = jnp.asarray(rng.normal(size=(batch, nj * 26 + nf * 12 + 16)))
        batches.append(Transition(obs, jnp.asarray(rng.normal(size=(batch, 8))), None, None, None))
    policy = _policy(False, num_experts=3)
    params = policy.init(jax.random.PRNGKey(60), *_split(batches[0].observations, 0))
    # float64 parameters: in float32, fused kernels (e.g. FMA contraction) change the last bit of Adam's moments.
    params = jax.tree_util.tree_map(lambda a: a.astype(jnp.float64) + 0.2 * jax.random.normal(jax.random.PRNGKey(61), a.shape, jnp.float64), params)
    tx = optax.chain(optax.clip_by_global_norm(5.0), optax.scale_by_adam(), optax.scale(-1e-2))
    extra = jnp.asarray(1.5)

    for evidence, router_only, norm, unbiased in (("type", False, "pairs", True), ("unit", False, "pairs", True),
                                                  ("type", True, "pairs", True), ("type", False, "mass", False)):
        fused, ref = _updater(evidence, router_only, norm, unbiased), _updater(evidence, router_only, norm, unbiased)
        state_f = state_r = TrainState.create(apply_fn=policy.apply, params=params, tx=tx)
        key_f = key_r = jax.random.PRNGKey(62)
        ref._init_state(state_r, extra, batches, key_r)
        robot_passes = [jax.jit(ref._robot_pass(r, b.observations)) for r, b in enumerate(batches)]
        for step in range(5):
            state_f, loss_f, metrics_f, key_f = fused.update(state_f, extra, batches, key_f, step)

            # Step-by-step reference (the computation before the update was compiled as one function).
            losses, grads, captures = [], [], []
            for r, b in enumerate(batches):
                key_r, sub = jax.random.split(key_r)
                loss, g, cap = robot_passes[r](state_r.params, extra, b.observations, b.actions, sub)
                losses.append(loss); grads.append(g); captures.append(cap)
            task = jax.tree_util.tree_map(lambda *gs: sum(gs) / len(gs), *grads)
            robot_grams, ref._robot_acc = _accumulate_robot_grams(ref._robot_acc, tuple(grads), ref.robot_decay)
            ref._robot_count += 1
            lam = ref.lambda_at(step)
            conflict, diag = lam > 0.0, step % 2 == 0
            key_r, sub = jax.random.split(key_r)
            metrics_r = {}
            if conflict or diag:
                ref._type_count += 1
                aux, metrics_r, _, ref._type_state = ref._interference(
                    state_r.params, tuple(captures), sub, jnp.float32(lam if conflict else 0.0), ref._type_state,
                    jnp.float32(ref._type_count), False)
            else:
                aux, metrics_r = ref._utilization(state_r.params, tuple(captures))
            total = jax.tree_util.tree_map(jnp.add, task, aux)
            if router_only:
                total = jax.tree_util.tree_map_with_path(lambda p, g: g if _is_router(p) else jnp.zeros_like(g), total)
                new = state_r.apply_gradients(grads=total)
                state_r = new.replace(params=jax.tree_util.tree_map_with_path(
                    lambda p, n, o: n if _is_router(p) else o, new.params, state_r.params))
            else:
                state_r = state_r.apply_gradients(grads=total)

            assert _rel_err(loss_f, jnp.mean(jnp.stack(losses))) < 1e-9, f"step {step}: loss differs"
            for (path, a), b in zip(jax.tree_util.tree_leaves_with_path(state_f.params), jax.tree_util.tree_leaves(state_r.params)):
                assert _rel_err(a, b) < 1e-9, f"{evidence}/router_only={router_only} step {step}: {path} differs"
            for blk in ref._type_state:
                assert _rel_err(fused._type_state[blk], ref._type_state[blk]) < 1e-9, f"step {step}: type accumulator differs"
            for lvl in ref._robot_acc:
                assert _rel_err(fused._robot_acc[lvl], ref._robot_acc[lvl]) < 1e-9, f"step {step}: robot accumulator differs"
            assert bool(jnp.all(key_f == key_r)), f"step {step}: key sequence differs"
            if diag:
                metrics_r["morse/router_grad_task"] = _router_norm(task)
                metrics_r["morse/router_grad_aux"] = _router_norm(aux)
                for k, v in metrics_r.items():
                    assert abs(metrics_f[k] - float(v)) <= 1e-9 * max(1.0, abs(float(v))), f"step {step}: metric {k} differs"
            else:
                assert metrics_f == {}, "metrics are returned on diagnostic steps only"
        if router_only:
            for (path, a), b in zip(jax.tree_util.tree_leaves_with_path(state_f.params), jax.tree_util.tree_leaves(params)):
                if not _is_router(path):
                    assert float(jnp.max(jnp.abs(a - b))) == 0.0, f"router-only update changed {path}"
    print("[ok] compiled update == step-by-step update (type/unit evidence, router-only; params, accumulators, keys, metrics)")


def _reference_step(ref, robot_passes, state, key, batches, extra, step, primary):
    """One actor step computed step by step: the MORSE step (primary) or a task-only further step."""
    from cross_embodiment_offline_rl.algorithms.morse.updater import _accumulate_robot_grams
    losses, grads, captures = [], [], []
    for r, b in enumerate(batches):
        key, sub = jax.random.split(key)
        loss, g, cap = robot_passes[r](state.params, extra, b.observations, b.actions, sub)
        losses.append(loss); grads.append(g); captures.append(cap)
    task = jax.tree_util.tree_map(lambda *gs: sum(gs) / len(gs), *grads)
    key, sub = jax.random.split(key)
    total = task
    if primary:
        _, ref._robot_acc = _accumulate_robot_grams(ref._robot_acc, tuple(grads), ref.robot_decay)
        ref._robot_count += 1
        lam = ref.lambda_at(step)
        conflict, diag = lam > 0.0, step % ref.diag_frequency == 0
        if conflict or diag:
            ref._type_count += 1
            aux, _, _, ref._type_state = ref._interference(
                state.params, tuple(captures), sub, jnp.float32(lam if conflict else 0.0), ref._type_state,
                jnp.float32(ref._type_count), False)
        else:
            aux, _ = ref._utilization(state.params, tuple(captures))
        total = jax.tree_util.tree_map(jnp.add, task, aux)
    return state.apply_gradients(grads=total), jnp.mean(jnp.stack(losses)), key


def test_actor_substeps_match_chunked_reference():
    """n_sub actor steps per update on equal chunks of the minibatch: the MORSE step (routing objectives,
    evidence, accumulators) on the first chunk, task-only steps on the others."""
    import optax
    from flax.training.train_state import TrainState
    from cross_embodiment_offline_rl.algorithms.utils import Transition
    batch, n_sub = 6, 2
    size = batch // n_sub
    rng = np.random.default_rng(8)
    batches = []
    for nj, nf in _ROBOTS:
        obs = jnp.asarray(rng.normal(size=(batch, nj * 26 + nf * 12 + 16)))
        batches.append(Transition(obs, jnp.asarray(rng.normal(size=(batch, 8))), None, None, None))
    chunks = [[Transition(b.observations[i * size:(i + 1) * size], b.actions[i * size:(i + 1) * size], None, None, None)
               for b in batches] for i in range(n_sub)]
    policy = _policy(False, num_experts=3)
    params = policy.init(jax.random.PRNGKey(70), *_split(batches[0].observations, 0))
    params = jax.tree_util.tree_map(lambda a: a.astype(jnp.float64) + 0.2 * jax.random.normal(jax.random.PRNGKey(71), a.shape, jnp.float64), params)
    tx = optax.chain(optax.clip_by_global_norm(5.0), optax.scale_by_adam(), optax.scale(-1e-2))
    extra = jnp.asarray(1.5)
    fused, ref = _updater("type", False), _updater("type", False)
    state_f = state_r = TrainState.create(apply_fn=policy.apply, params=params, tx=tx)
    key_f = key_r = jax.random.PRNGKey(72)
    ref._init_state(state_r, extra, chunks[0], key_r)
    robot_passes = [jax.jit(ref._robot_pass(r, b.observations)) for r, b in enumerate(chunks[0])]
    for step in range(4):
        for sub in range(n_sub):
            state_f, loss_f, metrics_f, key_f = fused.update(state_f, extra, batches, key_f, step, sub, n_sub)
            state_r, loss_r, key_r = _reference_step(ref, robot_passes, state_r, key_r, chunks[sub], extra, step, sub == 0)
            assert _rel_err(loss_f, loss_r) < 1e-9, f"step {step}.{sub}: loss differs"
            for (path, a), b in zip(jax.tree_util.tree_leaves_with_path(state_f.params), jax.tree_util.tree_leaves(state_r.params)):
                assert _rel_err(a, b) < 1e-9, f"step {step}.{sub}: {path} differs"
            assert bool(jnp.all(key_f == key_r)), f"step {step}.{sub}: key sequence differs"
            if sub > 0:
                assert metrics_f == {}, "further actor steps return no metrics"
        for blk in ref._type_state:
            assert _rel_err(fused._type_state[blk], ref._type_state[blk]) < 1e-9, f"step {step}: type accumulator differs"
        for lvl in ref._robot_acc:
            assert _rel_err(fused._robot_acc[lvl], ref._robot_acc[lvl]) < 1e-9, f"step {step}: robot accumulator differs"
        assert fused._robot_count == ref._robot_count == step + 1, "accumulators advance once per update"
    assert int(state_f.step) == 4 * n_sub, "one optimizer step per chunk"
    print("[ok] actor sub-steps: MORSE step on the first chunk, task-only steps on the others; accumulators once per update")


def test_prefetcher_gathers_dataset_rows():
    import json
    import tempfile
    from cross_embodiment_offline_rl.algorithms.utils import BatchPrefetcher, Dataset, robot_batch_sizes
    rng = np.random.default_rng(3)
    n, envs, dim = 50, 4, 6
    arrays = {"observations": rng.normal(size=(n, envs, dim)).astype(np.float32),
              "next_observations": rng.normal(size=(n, envs, dim)).astype(np.float32),
              "actions": rng.normal(size=(n, envs, 3)).astype(np.float32),
              "rewards": rng.normal(size=(n, envs)).astype(np.float32),
              "dones": rng.random(size=(n, envs)) < 0.2}
    with tempfile.TemporaryDirectory() as root:
        for name, a in arrays.items():
            np.save(os.path.join(root, f"{name}.npy"), a)
        json.dump({"n_samples": n, "n_envs": envs}, open(os.path.join(root, "metadata.json"), "w"))
        sizes = robot_batch_sizes([1.0, 1.0, 2.0, 0.0], 8, envs)
        assert sizes == [8, 8, 16, 0], sizes
        for mmap in (False, True):
            data = Dataset(root, normalize=False, mmap=mmap)
            idx = np.array([0, 7, 7, 49, 23])
            for e in range(envs):
                got = data.gather(idx, e)
                want = Transition_rows(data, idx, e)
                for g, w in zip(got, want):
                    assert np.array_equal(np.asarray(g), np.asarray(w)) and np.asarray(g).dtype == np.asarray(w).dtype
            pre = BatchPrefetcher(data, sizes, seed=0, depth=2, workers=2)
            draws = np.random.default_rng(0)
            for _ in range(3):
                batches = pre.next()
                for e, (b, size) in enumerate(zip(batches, sizes)):
                    idx = draws.integers(0, n, size=size)
                    assert b.observations.shape == (size, dim) and b.dones.shape == (size,)
                    for g, w in zip(b, Transition_rows(data, idx, e)):
                        assert np.array_equal(np.asarray(g), np.asarray(w)), "prefetched batch differs from the dataset rows"
            pre.close()
    print("[ok] background sampler: same rows as Dataset.sample indexing, seeded draws, batch sizes from robot weights")


def Transition_rows(data, idx, e):
    """Rows of robot e at sample indices idx, indexed as in Dataset.sample."""
    return (data.obs[idx, e, :], data.act[idx, e, :], data.rew[idx, e], data.nobs[idx, e, :], data.done[idx, e])


def test_mutual_information_loss_matches_definition():
    rng = np.random.default_rng(3)
    n_units, n_experts, n_types = 40, 4, 6
    logits = rng.normal(size=(n_units, n_experts))
    p = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    onehot = np.eye(n_types)[rng.integers(0, n_types, size=n_units)]
    neg_mi, h_cond, h_marg = itf.neg_mutual_information(jnp.asarray(p), onehot)
    joint = onehot.T @ p / n_units                       # P(A = a, E = e)
    pa, pe = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    mi = np.sum(np.where(joint > 0, joint * np.log(np.where(joint > 0, joint, 1.0) / (pa * pe)), 0.0))
    assert abs(-float(neg_mi) - mi) < 1e-9 and abs(float(h_marg - h_cond) - mi) < 1e-9
    # one expert per type, every expert used equally: I = log K
    hard = np.eye(n_experts)[np.arange(n_units) % n_experts]
    neg_mi_hard, _, _ = itf.neg_mutual_information(jnp.asarray(hard), np.eye(n_experts)[np.arange(n_units) % n_experts])
    assert abs(-float(neg_mi_hard) - np.log(n_experts)) < 1e-9
    print(f"[ok] -I(A;E) matches the definition (I = {mi:.4f}); hard balanced routing gives log K")


def test_signed_evidence_thresholds_by_noise():
    rng = np.random.default_rng(4)
    n_experts, n_types, d = 2, 8, 30
    vec = rng.normal(size=(n_experts, n_types, d))
    gram = lambda v: np.einsum("kad,kbd->kab", v, v)
    upper = np.triu(np.ones((n_types, n_types), bool), 1)

    def cosine(s):
        n = np.sqrt(np.einsum("kaa->ka", s))
        return s / (n[:, :, None] * n[:, None, :])
    s = gram(vec)
    ev = itf.signed_type_evidence(jnp.asarray(s), jnp.asarray(s), jnp.asarray(s), 2.0, 0.5)
    c = cosine(s)
    expect = (np.maximum(-c, 0) - 0.5 * np.maximum(c, 0)) * upper
    assert np.allclose(ev["q"], expect, atol=1e-9) and np.allclose(ev["sigma"], 0.0)
    s_e, s_o = gram(vec + rng.normal(scale=0.7, size=vec.shape)), gram(vec + rng.normal(scale=0.7, size=vec.shape))
    s_m = 0.5 * (s_e + s_o)
    ev = itf.signed_type_evidence(jnp.asarray(s_m), jnp.asarray(s_e), jnp.asarray(s_o), 2.0, 1.0)
    diff = (cosine(s_e) - cosine(s_o))[:, upper]
    sigma = np.sqrt((diff ** 2).mean(axis=1)) / 2
    assert np.allclose(ev["sigma"], sigma, atol=1e-9)
    cm = cosine(s_m)
    kappa = 2.0 * sigma[:, None, None]
    expect = (np.maximum(-cm - kappa, 0) - np.maximum(cm - kappa, 0)) * upper
    assert np.allclose(ev["q"], expect, atol=1e-9)
    assert np.all(np.asarray(ev["q"])[(np.abs(cm) <= kappa) & upper[None]] == 0.0)
    print(f"[ok] signed evidence: no noise -> q = [-cos]+ - rho[cos]+; with noise, |cos| <= {2.0:.0f} sigma gives none")


def test_fixed_routing_identical_experts_and_calibration():
    from flax.core import unfreeze
    robots, n_experts = [(5, 2), (7, 3)], 3
    table = np.zeros((2, 7, n_experts))
    table[:, :, 0] = 1.0
    table[1, 2] = [0.2, 0.3, 0.5]
    tup = tuple(tuple(tuple(float(v) for v in row) for row in robot) for robot in table)
    common = dict(std_dev=1.0, softmax_temperature=1.0, softmax_temperature_min=0.015, stability_epsilon=1e-8,
                  policy_mean_abs_clip=10.0, policy_std_min_clip=1e-8, policy_std_max_clip=2.0, num_experts=n_experts,
                  moe_blocks=("decoder",), router_hidden=16, n_robots=2)
    policy = MorsePolicy(decoder_table=tup, identical_experts=True, calibration=True, **common)
    inputs, _ = _robot_inputs(jax.random.PRNGKey(5), 4, *robots[1])
    params = policy.init(jax.random.PRNGKey(6), *inputs)
    dec = params["params"]["decoder"]
    assert "router" not in dec, "a fixed routing table must not create router parameters"
    k = np.asarray(dec["dense_0"]["kernel"])
    assert np.allclose(k, k[:1]), "identical_experts: every expert starts from the same parameters"
    out_none = policy.apply(params, *inputs)[0]
    out_r1 = policy.apply(params, *inputs, robot_id=1)[0]
    assert _rel_err(out_r1, out_none) < 1e-12, "at initialization routing and calibration leave the output unchanged"
    p2 = unfreeze(params)
    p2["params"]["out_bias"] = p2["params"]["out_bias"].at[1, 2].set(0.5)
    shift = np.asarray(policy.apply(p2, *inputs, robot_id=1)[0] - out_r1)
    assert np.allclose(shift[:, 2], 0.5) and np.allclose(np.delete(shift, 2, axis=1), 0.0)
    assert np.allclose(policy.apply(p2, *inputs, robot_id=0)[0], policy.apply(params, *inputs, robot_id=0)[0])
    # the captured routing weights are the robot's table rows
    cap = MorsePolicy(decoder_table=tup, capture=True, n_capture_element=4, n_capture_body=4, **common)
    pc = cap.init(jax.random.PRNGKey(6), *inputs)
    _, state = cap.apply({"params": pc["params"], CAPTURE_DELTA: pc[CAPTURE_DELTA]}, *inputs, robot_id=1,
                         mutable=[CAPTURE_IN, CAPTURE_ROUTER])
    p_cap = np.asarray(state[CAPTURE_ROUTER]["decoder"]["p"][0])
    assert np.allclose(p_cap, np.broadcast_to(table[1, :7], p_cap.shape))
    print("[ok] fixed routing table (no router, captured p = table), identical experts, per-joint calibration")


def test_probe_joint_gradients_and_lookahead():
    import tempfile
    from types import SimpleNamespace
    from jax.flatten_util import ravel_pytree
    from cross_embodiment_offline_rl.algorithms.morse.probe import DecoderProbe, _split_decoder
    policy = _policy(False, num_experts=1)
    batch = 6
    obs_dim = [nj * 26 + nf * 12 + 16 for nj, nf in _ROBOTS]

    class Data:
        def sample(self, key, sizes):
            out = []
            for r, n in enumerate(sizes):
                key, k1, k2 = jax.random.split(key, 3)
                out.append(SimpleNamespace(observations=jax.random.normal(k1, (n, obs_dim[r]), jnp.float64),
                                           actions=jax.random.normal(k2, (n, 8), jnp.float64)))
            return out, key

    def loss_per_joint(out, obs, act, extra, key, r):
        mean, logstd = out
        return extra * jnp.mean((mean - act[:, :mean.shape[1]]) ** 2 + 0.1 * logstd ** 2, axis=0) / mean.shape[1]

    data = Data()
    inputs = _split(data.sample(jax.random.PRNGKey(0), [batch] * 3)[0][0].observations, 0)
    params = policy.init(jax.random.PRNGKey(1), *inputs)
    params = jax.tree_util.tree_map(lambda a: a.astype(jnp.float64) + 0.2 * jax.random.normal(jax.random.PRNGKey(3), a.shape, jnp.float64), params)
    with tempfile.TemporaryDirectory() as tmp:
        probe = DecoderProbe(policy, _split, loss_per_joint, ["a", "b", "c"], [batch] * 3, 2, [1e-5, 1e-2], tmp, 0)
        b0 = data.sample(jax.random.PRNGKey(2), [batch] * 3)[0]
        for r in range(3):
            flosses, fgrads, _ = probe._robot_fns(r)
            obs, act = b0[r].observations, b0[r].actions
            loss, jac = fgrads(params, 1.5, obs, act)
            experts, rebuild = _split_decoder(params)
            flat, unravel = ravel_pytree(experts)
            g = jax.grad(lambda v: flosses(rebuild(unravel(v)), 1.5, obs, act).sum())(flat)
            assert _rel_err(jac.sum(axis=0), g) < 1e-9, "per-joint decoder gradients must sum to the robot's gradient"
        z = np.load(probe.run(params, 1.5, data, 3))
    n_types = sum(nj for nj, _ in _ROBOTS)
    assert z["G12_all"].shape == (n_types, n_types) and np.allclose(z["G12_all"], z["G12_all"].T)
    parts = sum(z[f"Gfull_{g}"] for g in ("dense0", "ln0", "out_w", "out_b"))
    assert _rel_err(parts, z["Gfull_all"]) < 1e-9, "parameter groups must add up to the whole decoder"
    t, pred = z["T_0"], z["P_0"]
    assert _rel_err(t, pred) < 1e-3, f"small lookahead steps must match the first-order prediction ({_rel_err(t, pred):.1e})"
    print(f"[ok] probe: per-joint gradients sum to the robot gradient; groups add up; lookahead ~ first order "
          f"(rel err {_rel_err(t, pred):.1e})")


def test_branch_from_shared_checkpoint():
    import tempfile
    import optax
    from flax.training import checkpoints
    from flax.training.train_state import TrainState
    from cross_embodiment_offline_rl.algorithms.morse.branch import branch_state
    robots = (7, 3)
    common = dict(std_dev=1.0, softmax_temperature=1.0, softmax_temperature_min=0.015, stability_epsilon=1e-8,
                  policy_mean_abs_clip=10.0, policy_std_min_clip=1e-8, policy_std_max_clip=2.0, router_hidden=16,
                  n_robots=2)
    inputs, _ = _robot_inputs(jax.random.PRNGKey(8), 4, *robots)
    inputs = tuple(x.astype(jnp.float32) for x in inputs)
    tx = lambda: optax.chain(optax.clip_by_global_norm(5.0), optax.scale_by_adam(),
                             optax.scale_by_schedule(optax.cosine_decay_schedule(-3e-4, 100)))
    shared = MorsePolicy(num_experts=1, **common)
    s1 = TrainState.create(apply_fn=shared.apply, params=shared.init(jax.random.PRNGKey(9), *inputs), tx=tx())
    for _ in range(2):
        g = jax.grad(lambda p: jnp.sum(shared.apply(p, *inputs)[0] ** 2) + jnp.sum(shared.apply(p, *inputs)[1]))(s1.params)
        s1 = s1.apply_gradients(grads=g)
    table = np.zeros((2, 7, 4), np.float32)
    table[:, :, 0] = 1.0
    table[1, ::2] = [0.0, 1.0, 0.0, 0.0]                     # two groups of joints on experts 0 and 1
    tup = tuple(tuple(tuple(float(v) for v in row) for row in robot) for robot in table)
    moe = MorsePolicy(num_experts=4, moe_blocks=("decoder",), decoder_table=tup, **common)
    s4 = TrainState.create(apply_fn=moe.apply, params=moe.init(jax.random.PRNGKey(10), *inputs), tx=tx())
    with tempfile.TemporaryDirectory() as tmp:
        checkpoints.save_checkpoint(ckpt_dir=tmp, target={"policy": s1}, step=2, keep=1)
        raw = checkpoints.restore_checkpoint(ckpt_dir=tmp, target=None)
    b, tiled, kept = branch_state(s4, raw["policy"])
    assert int(b.step) == 2 and not kept, f"every leaf must come from the checkpoint (kept {kept})"
    assert any("decoder" in p for p in tiled) and all("decoder" in p for p in tiled)
    out1 = shared.apply(s1.params, *inputs, robot_id=1)
    out4 = moe.apply(b.params, *inputs, robot_id=1)
    assert _rel_err(out4[0], out1[0]) < 1e-6 and _rel_err(out4[1], out1[1]) < 1e-6, "the branched policy starts from the checkpoint's"
    mu1 = s1.opt_state[1].mu["params"]["decoder"]["dense_0"]["kernel"]
    mu4 = b.opt_state[1].mu["params"]["decoder"]["dense_0"]["kernel"]
    assert np.allclose(np.asarray(mu4), np.broadcast_to(np.asarray(mu1), mu4.shape))
    assert int(b.opt_state[2].count) == int(s1.opt_state[2].count), "the learning-rate schedule continues"
    print(f"[ok] branch: {len(tiled)} decoder leaves tiled into 4 experts (params and Adam moments); same outputs, step, schedule")



def _flat_expert_grad(tree, k):
    """Expert k's parameters flattened in the order of itf.type_grad_sums (kernel, bias / scale, bias per layer)."""
    order = (("dense_0", "kernel"), ("dense_0", "bias"), ("ln_0", "scale"), ("ln_0", "bias"),
             ("dense_1", "kernel"), ("dense_1", "bias"))
    return jnp.concatenate([tree[m][n][k].reshape(-1) for m, n in order])


def test_crossfit_type_sums_and_evidence():
    from types import SimpleNamespace
    import optax
    from flax.training.train_state import TrainState
    from cross_embodiment_offline_rl.algorithms.morse.updater import MorseActorUpdater
    from cross_embodiment_offline_rl.algorithms.utils import Transition
    # (a) Per-type sums from the captures == explicit per-unit gradients summed over transitions, per element.
    k0, k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(5), 5)
    n_trans, n_elem, d_in, n_experts = 3, 4, 5, 3
    x = jax.random.normal(k0, (n_trans, n_elem, d_in), jnp.float64)
    morph = jax.random.normal(k1, (n_trans, n_elem, 3), jnp.float64)
    state = jax.random.normal(k2, (n_trans, n_elem, 2), jnp.float64)
    w = jax.random.normal(k3, (n_trans, n_elem, 2), jnp.float64)
    objective = lambda out: jnp.sum(jnp.tanh((out * w).sum(axis=(1, 2))) ** 2)
    block_c, block = _block(True, n_trans, n_experts), _block(False)
    variables = block_c.init(k4, x, morph, state)
    params = jax.tree_util.tree_map(lambda a: a.astype(jnp.float64) + 0.3 * jax.random.normal(k3, a.shape, jnp.float64),
                                    {"params": variables["params"]})
    delta0 = jax.tree_util.tree_map(jnp.zeros_like, variables[CAPTURE_DELTA])

    def f(prm, delta):
        out, st = block_c.apply({"params": prm["params"], CAPTURE_DELTA: delta}, x, morph, state,
                                mutable=[CAPTURE_IN, CAPTURE_ROUTER])
        return objective(out), st

    (_, st), (_, gd) = jax.value_and_grad(f, argnums=(0, 1), has_aux=True)(params, delta0)
    sums = itf.type_grad_sums([("dense", st[CAPTURE_IN]["dense_0"]["x"][0], gd["dense_0"]["z"]),
                               ("ln", st[CAPTURE_IN]["ln_0"]["xhat"][0], gd["ln_0"]["y"]),
                               ("dense", st[CAPTURE_IN]["dense_1"]["x"][0], gd["dense_1"]["z"])])
    n_units = n_trans * n_elem
    experts = {k: v for k, v in params["params"].items() if k != "router"}
    copies = jax.tree_util.tree_map(lambda a: jnp.broadcast_to(a, (n_units,) + a.shape), experts)

    def loss_from_copies(c):
        def one(cu, xu, mu, su):
            return block.apply({"params": {**cu, "router": params["params"]["router"]}}, xu[None], mu[None], su[None])[0]
        out = jax.vmap(one)(c, x.reshape(n_units, -1), morph.reshape(n_units, -1), state.reshape(n_units, -1))
        return objective(out.reshape(n_trans, n_elem, -1))

    per_unit = jax.grad(loss_from_copies)(copies)
    per_elem = jax.tree_util.tree_map(lambda a: a.reshape((n_trans, n_elem) + a.shape[1:]).sum(axis=0), per_unit)
    expl = jnp.stack([jnp.stack([_flat_expert_grad(jax.tree_util.tree_map(lambda a: a[e], per_elem), k)
                                 for e in range(n_elem)]) for k in range(n_experts)])
    err = _rel_err(sums, expl)
    assert sums.shape == expl.shape and err < 1e-9, f"per-type sums differ from explicit gradients (rel err {err:.2e})"

    # (b) Evidence: four identical streams give exact cosines and no noise; noisy streams set the threshold from the
    # disagreement of the two independent estimates.
    rng = np.random.default_rng(9)
    vec = rng.normal(size=(2, 8, 40))
    upper = np.triu(np.ones((8, 8), bool), 1)

    def cosine(a, b):
        g = 0.5 * (np.einsum("kap,kbp->kab", a, b) + np.einsum("kbp,kap->kab", a, b))
        n = np.sqrt(np.einsum("kaa->ka", g))
        return np.clip(g / (n[:, :, None] * n[:, None, :]), -1, 1)

    ev = itf.crossfit_type_evidence(jnp.asarray(np.stack([vec] * 4)), 2.0, 0.5)
    c = cosine(vec, vec)
    assert np.allclose(ev["q"], (np.maximum(-c, 0) - 0.5 * np.maximum(c, 0)) * upper, atol=1e-9)
    assert np.allclose(ev["sigma"], 0.0) and np.allclose(ev["reliability"], 1.0)
    st4 = np.stack([vec + rng.normal(scale=0.3, size=vec.shape) for _ in range(4)])
    ev = itf.crossfit_type_evidence(jnp.asarray(st4), 2.0, 1.0)
    diff = (cosine(st4[0], st4[1]) - cosine(st4[2], st4[3]))[:, upper]
    assert np.allclose(ev["sigma"], np.sqrt((diff ** 2).mean(axis=1)) / 2, atol=1e-9)
    cf = cosine(st4[0] + st4[1], st4[2] + st4[3])
    kappa = 2.0 * np.asarray(ev["sigma"])[:, None, None]
    assert np.allclose(ev["q"], (np.maximum(-cf - kappa, 0) - np.maximum(cf - kappa, 0)) * upper, atol=1e-9)

    # (c) In the update: the decoder captures the whole minibatch; after the first update stream 0 holds (1 - decay)
    # times every robot's decoder gradient split by joint, the other streams are empty.
    batch = 6
    batches = []
    for nj, nf in _ROBOTS:
        obs = jnp.asarray(rng.normal(size=(batch, nj * 26 + nf * 12 + 16)))
        batches.append(Transition(obs, jnp.asarray(rng.normal(size=(batch, 8))), None, None, None))
    a = SimpleNamespace(morse_blocks="core,decoder", morse_num_experts=3, morse_router_hidden=16, morse_router_inputs="both",
                        morse_conflict_mode="signed", morse_lambda=1.0, morse_eta=0.01, morse_conflict_norm="pairs",
                        morse_unbiased_types=True, morse_zeta=1.0, morse_kappa=0.0, morse_threshold="fixed",
                        morse_relevance=True, morse_evidence="type", morse_type_ema=0.9, morse_conflict_every=1,
                        morse_warmup=0, morse_ramp=0, morse_router_only=False, morse_diag_frequency=1,
                        morse_snapshot_frequency=0, morse_robot_ema=0.9, morse_conflict_blocks="decoder",
                        morse_compat_z=1.0, morse_compat_help=1.0, morse_compat_estimator="crossfit", morse_compat_ema=0.8)
    cap_policy = _policy(True, 2).clone(moe_blocks=("core", "decoder"), n_capture_decoder=batch)
    upd = MorseActorUpdater(SimpleNamespace(algorithm=a), cap_policy, _split, ["a", "b", "c"], None)
    upd.loss_fn = _toy_loss
    policy = _policy(False).clone(moe_blocks=("core", "decoder"))
    params = policy.init(jax.random.PRNGKey(70), *_split(batches[0].observations, 0))
    params = jax.tree_util.tree_map(lambda v: v.astype(jnp.float64) + 0.2 * jax.random.normal(jax.random.PRNGKey(71), v.shape, jnp.float64), params)
    tx = optax.chain(optax.clip_by_global_norm(5.0), optax.scale_by_adam(), optax.scale(-1e-2))
    train = TrainState.create(apply_fn=policy.apply, params=params, tx=tx)
    extra, key = jnp.asarray(1.5), jax.random.PRNGKey(72)
    upd._init_state(train, extra, batches, key)
    grads = [jax.jit(upd._robot_pass(r, b.observations))(params, extra, b.observations, b.actions, key)[1]
             for r, b in enumerate(batches)]
    dec = jax.tree_util.tree_map(lambda *gs: sum(gs), *[g["params"]["decoder"] for g in grads])
    _, _, metrics, _ = upd.update(train, extra, batches, key, 0)
    streams = upd._type_state["decoder_compat"]
    n_types = sum(nj for nj, _ in _ROBOTS)
    assert streams.shape[:3] == (4, 3, n_types) and float(jnp.abs(streams[1:]).max()) == 0.0
    total = streams[0].sum(axis=1) / (1.0 - 0.8)
    expect = jnp.stack([_flat_expert_grad(dec, k) for k in range(3)])
    err_u = _rel_err(total, expect)
    assert err_u < 1e-9, f"stream 0 differs from the robots' decoder gradient (rel err {err_u:.2e})"
    assert upd._meta["decoder"]["type_onehot"].shape[0] == 2 * n_types, "unit Grams keep n_capture_element transitions"
    for name in ("compat_reliability", "compat_sigma", "compat_valid_frac", "loss_conf"):
        assert f"morse/decoder/{name}" in metrics, name
    print(f"[ok] crossfit: per-type sums == explicit per-element gradients (rel err {err:.1e}); exact cosines without "
          f"noise, threshold from two independent estimates; a stream holds the robots' decoder gradients by joint "
          f"(rel err {err_u:.1e})")



def test_joint_pcgrad_control():
    from types import SimpleNamespace
    import optax
    from flax.training.train_state import TrainState
    from cross_embodiment_offline_rl.algorithms.morse import updater as upd_mod
    from cross_embodiment_offline_rl.algorithms.utils import Transition
    # PCGrad on two conflicting gradients matches the closed form; without conflicts nothing changes.
    g = jnp.asarray([[1.0, 0.0, 1.0], [-1.0, 2.0, 0.0]])
    out, frac = itf.pcgrad(g, jax.random.PRNGKey(0))
    d = float(g[0] @ g[1])
    expect = jnp.stack([g[0] - d / float(g[1] @ g[1]) * g[1], g[1] - d / float(g[0] @ g[0]) * g[0]])
    assert _rel_err(out, expect) < 1e-12 and abs(float(frac) - 1.0) < 1e-12
    g = jnp.abs(jax.random.normal(jax.random.PRNGKey(1), (6, 5), jnp.float64))
    out, frac = itf.pcgrad(g, jax.random.PRNGKey(2))
    assert float(jnp.max(jnp.abs(out - g))) == 0.0 and float(frac) == 0.0

    # In the update: with PCGrad replaced by the identity, the PCGrad path gives exactly the ordinary update
    # (the per-type sums are rebuilt into the decoder's gradient), for one expert (URMA) and for several.
    rng = np.random.default_rng(11)
    batch = 8
    batches = []
    for nj, nf in _ROBOTS:
        obs = jnp.asarray(rng.normal(size=(batch, nj * 26 + nf * 12 + 16)))
        batches.append(Transition(obs, jnp.asarray(rng.normal(size=(batch, 8))), None, None, None))

    def make(pcgrad, n_experts):
        a = SimpleNamespace(morse_blocks="decoder", morse_num_experts=n_experts, morse_router_hidden=16,
                            morse_router_inputs="both", morse_conflict_mode="none", morse_lambda=0.0, morse_eta=0.0,
                            morse_conflict_norm="pairs", morse_unbiased_types=True, morse_zeta=1.0, morse_kappa=0.0,
                            morse_threshold="fixed", morse_relevance=True, morse_evidence="type", morse_type_ema=0.9,
                            morse_conflict_every=1, morse_warmup=0, morse_ramp=0, morse_router_only=False,
                            morse_diag_frequency=1, morse_snapshot_frequency=0, morse_robot_ema=0.9,
                            morse_pcgrad_blocks="decoder" if pcgrad else "")
        cap = _policy(True, 2, num_experts=n_experts).clone(moe_blocks=("decoder",), n_capture_decoder=batch // 2)
        u = upd_mod.MorseActorUpdater(SimpleNamespace(algorithm=a), cap, _split, ["a", "b", "c"], None)
        u.loss_fn = _toy_loss
        return u

    identity = lambda g, key: (g, jnp.zeros(()))
    real = itf.pcgrad
    for n_experts in (1, 3):
        policy = _policy(False, num_experts=n_experts).clone(moe_blocks=("decoder",))
        params = policy.init(jax.random.PRNGKey(80), *_split(batches[0].observations, 0))
        params = jax.tree_util.tree_map(lambda v: v.astype(jnp.float64) + 0.2 * jax.random.normal(jax.random.PRNGKey(81), v.shape, jnp.float64), params)
        tx = optax.chain(optax.clip_by_global_norm(5.0), optax.scale_by_adam(), optax.scale(-1e-2))
        results = []
        for pc in (False, True):
            itf.pcgrad = identity
            try:
                u = make(pc, n_experts)
                st, key = TrainState.create(apply_fn=policy.apply, params=params, tx=tx), jax.random.PRNGKey(82)
                for step in range(2):
                    for sub in range(2):
                        st, _, metrics, key = u.update(st, jnp.asarray(1.5), batches, key, step, sub, 2)
                results.append(st.params)
            finally:
                itf.pcgrad = real
        for (path, x), y in zip(jax.tree_util.tree_leaves_with_path(results[0]), jax.tree_util.tree_leaves(results[1])):
            assert _rel_err(x, y) < 1e-9, f"K={n_experts}: PCGrad path with identity projection changed {path}"
    # With the real projection the decoder update changes and the metrics report it.
    u = make(True, 1)
    policy = _policy(False, num_experts=1).clone(moe_blocks=("decoder",))
    params = jax.tree_util.tree_map(lambda v: v.astype(jnp.float64),
                                    policy.init(jax.random.PRNGKey(83), *_split(batches[0].observations, 0)))
    st = TrainState.create(apply_fn=policy.apply, params=params, tx=tx)
    _, _, metrics, _ = u.update(st, jnp.asarray(1.5), batches, jax.random.PRNGKey(84), 0, 0, 2)
    assert 0.0 < metrics["morse/decoder/pcgrad_projected_frac"] < 1.0, metrics["morse/decoder/pcgrad_projected_frac"]
    print(f"[ok] joint PCGrad: closed form for two gradients; identity projection reproduces the ordinary update "
          f"(K=1 and K=3); real projection touches {100 * metrics['morse/decoder/pcgrad_projected_frac']:.0f}% of type pairs")



def test_mi_robot_and_joint_groups():
    from types import SimpleNamespace
    import optax
    from flax.training.train_state import TrainState
    from cross_embodiment_offline_rl.algorithms.morse.updater import MorseActorUpdater
    from cross_embodiment_offline_rl.algorithms.utils import Transition
    rng = np.random.default_rng(13)
    n_experts = 3
    joints = [5, 7, 4]                                       # joint types per robot; 2 units per joint type
    type_robot = np.repeat(np.arange(3), joints)
    types = np.repeat(np.arange(len(type_robot)), 2)
    p = rng.dirichlet(np.ones(n_experts), size=len(types))
    t_onehot = np.eye(len(type_robot))[types]
    r_onehot = np.eye(3)[type_robot[types]]

    def mi(group_of_unit, pi):
        groups = np.unique(group_of_unit)
        r = np.stack([p[group_of_unit == g].mean(0) for g in groups])
        h = lambda q: -(q * np.log(q + 1e-12)).sum(-1)
        return h(pi @ r) - pi @ h(r)

    neg, _, _ = itf.neg_mutual_information(jnp.asarray(p), jnp.asarray(t_onehot))
    assert abs(-float(neg) - mi(types, np.full(len(type_robot), 1 / len(type_robot)))) < 1e-9, "unit-share prior"
    neg, _, _ = itf.neg_mutual_information(jnp.asarray(p), jnp.asarray(r_onehot), weights=np.full(3, 1 / 3))
    assert abs(-float(neg) - mi(type_robot[types], np.full(3, 1 / 3))) < 1e-9, "robot-level MI, equal robots"
    w = 1.0 / (3 * np.bincount(type_robot)[type_robot])
    neg, _, _ = itf.neg_mutual_information(jnp.asarray(p), jnp.asarray(t_onehot), weights=w)
    assert abs(-float(neg) - mi(types, w)) < 1e-9, "joint-type MI, equal robots"

    # The updater builds the groups from its unit metadata: robots or joint types, with equal-robot priors.
    batch = 6
    batches = []
    for nj, nf in _ROBOTS:
        obs = jnp.asarray(rng.normal(size=(batch, nj * 26 + nf * 12 + 16)))
        batches.append(Transition(obs, jnp.asarray(rng.normal(size=(batch, 8))), None, None, None))
    policy = _policy(False, num_experts=3)
    params = jax.tree_util.tree_map(lambda v: v.astype(jnp.float64),
                                    policy.init(jax.random.PRNGKey(90), *_split(batches[0].observations, 0)))
    train = TrainState.create(apply_fn=policy.apply, params=params,
                              tx=optax.chain(optax.scale_by_adam(), optax.scale(-1e-3)))
    for target, weighting in (("robot", "robot"), ("type", "robot"), ("type", "units")):
        a = SimpleNamespace(morse_blocks="joint_encoder,foot_encoder,core,decoder", morse_num_experts=3,
                            morse_router_hidden=16, morse_router_inputs="both", morse_conflict_mode="none",
                            morse_lambda=0.0, morse_eta=0.01, morse_conflict_norm="pairs", morse_unbiased_types=True,
                            morse_zeta=1.0, morse_kappa=0.0, morse_threshold="fixed", morse_relevance=True,
                            morse_evidence="type", morse_type_ema=0.9, morse_conflict_every=1, morse_warmup=0,
                            morse_ramp=0, morse_router_only=False, morse_diag_frequency=1, morse_snapshot_frequency=0,
                            morse_robot_ema=0.9, morse_mi_weight=0.1, morse_mi_target=target, morse_mi_weighting=weighting)
        upd = MorseActorUpdater(SimpleNamespace(algorithm=a), _policy(True, 2), _split, ["a", "b", "c"], None)
        upd.loss_fn = _toy_loss
        _, _, metrics, _ = upd.update(train, jnp.asarray(1.5), batches, jax.random.PRNGKey(91), 0)
        robot_of, prior = upd._mi_groups["decoder"]
        n_joints = np.array([nj for nj, _ in _ROBOTS])
        if target == "robot":
            assert len(robot_of) == len(_ROBOTS) and np.allclose(prior, 1 / len(_ROBOTS))
        elif weighting == "robot":
            assert len(robot_of) == n_joints.sum() and np.allclose(np.bincount(robot_of, weights=prior), 1 / len(_ROBOTS))
        else:
            assert np.allclose(np.bincount(robot_of, weights=prior), n_joints / n_joints.sum())
        assert np.isfinite(metrics["morse/decoder/mi"])
    print("[ok] MI groups: robot-level -I(R;E) and joint-type MI with unit-share or equal-robot priors match the "
          "definition; the updater builds them from the routing records")



def test_decoder_experts_width_and_topk():
    from types import SimpleNamespace
    import optax
    from flax.training.train_state import TrainState
    from cross_embodiment_offline_rl.algorithms.morse.updater import MorseActorUpdater
    from cross_embodiment_offline_rl.algorithms.utils import Transition
    from cross_embodiment_offline_rl.algorithms.morse.moe import Router
    # Top-k routing: every unit keeps its k largest weights, renormalized; the output is the mixture of those experts,
    # and an expert outside a unit's top k receives no gradient from it.
    k0, k1, k2, k3 = jax.random.split(jax.random.PRNGKey(21), 4)
    n_trans, n_elem, n_experts, k = 2, 3, 4, 2
    x = jax.random.normal(k0, (n_trans, n_elem, 5), jnp.float64)
    morph = jax.random.normal(k1, (n_trans, n_elem, 3), jnp.float64)
    state = jax.random.normal(k2, (n_trans, n_elem, 2), jnp.float64)
    block_k = MoEBlock(layer_features=(6, 2), layer_post=("ln_elu", "none"), kernel_inits=(lecun_normal(), lecun_normal()),
                       num_experts=n_experts, router_hidden=8, train_topk=k)
    dense = block_k.clone(train_topk=0)
    params = block_k.init(k3, x, morph, state)
    params = jax.tree_util.tree_map(lambda a: a.astype(jnp.float64) + 0.3 * jax.random.normal(k3, a.shape, jnp.float64), params)
    logits = Router(n_experts, 8, "both").apply({"params": params["params"]["router"]}, morph, state)
    p_soft = jax.nn.softmax(logits, axis=-1)
    order = jnp.argsort(-p_soft, axis=-1)
    mask = jnp.sum(jax.nn.one_hot(order[..., :k], n_experts), axis=-2)
    p_k = p_soft * mask / jnp.sum(p_soft * mask, axis=-1, keepdims=True)
    out_k = block_k.apply(params, x, morph, state)
    # Expert outputs with one-hot routing, mixed by the expected top-k weights.
    def expert_out(e):
        prm = {"params": {**params["params"], "router": jax.tree_util.tree_map(jnp.zeros_like, params["params"]["router"])}}
        onehot = dense.clone(fixed_routing=True)
        return onehot.apply({"params": {kk: v for kk, v in prm["params"].items() if kk != "router"}}, x, morph, state,
                            jnp.broadcast_to(jax.nn.one_hot(e, n_experts), morph.shape[:-1] + (n_experts,)))
    expect = sum(p_k[..., e:e + 1] * expert_out(e) for e in range(n_experts))
    assert _rel_err(out_k, expect) < 1e-9, "top-k output is the renormalized mixture of the kept experts"
    assert bool(jnp.all(jnp.sum(mask, -1) == k))
    unit_mask = jnp.zeros((n_trans, n_elem, 1)).at[0, 0].set(1.0)
    g = jax.grad(lambda prm: jnp.sum(block_k.apply(prm, x, morph, state) * unit_mask))(params)
    dropped = [e for e in range(n_experts) if float(mask[0, 0, e]) == 0.0]
    for e in dropped:
        assert float(jnp.max(jnp.abs(g["params"]["dense_0"]["kernel"][e]))) == 0.0, "a dropped expert gets no gradient"

    # Decoder-only expert count and width: the core keeps its experts, the decoder has its own, and the updater uses
    # the policy's per-block counts; one update runs.
    batch = 6
    rng = np.random.default_rng(23)
    batches = []
    for nj, nf in _ROBOTS:
        obs = jnp.asarray(rng.normal(size=(batch, nj * 26 + nf * 12 + 16)))
        batches.append(Transition(obs, jnp.asarray(rng.normal(size=(batch, 8))), None, None, None))
    pol = _policy(False, num_experts=3).clone(moe_blocks=("core", "decoder"), num_experts_decoder=5, topk_decoder=2,
                                              width_mult_decoder=0.5)
    params = jax.tree_util.tree_map(lambda v: v.astype(jnp.float64),
                                    pol.init(jax.random.PRNGKey(24), *_split(batches[0].observations, 0)))
    dec, core = params["params"]["decoder"], params["params"]["core"]
    assert dec["dense_0"]["kernel"].shape[0] == 5 and core["dense_0"]["kernel"].shape[0] == 3
    hidden = dec["dense_0"]["kernel"].shape[-1]
    a = SimpleNamespace(morse_blocks="core,decoder", morse_num_experts=3, morse_router_hidden=16, morse_router_inputs="both",
                        morse_conflict_mode="none", morse_lambda=0.0, morse_eta=0.01, morse_conflict_norm="pairs",
                        morse_unbiased_types=True, morse_zeta=1.0, morse_kappa=0.0, morse_threshold="fixed",
                        morse_relevance=True, morse_evidence="type", morse_type_ema=0.9, morse_conflict_every=1,
                        morse_warmup=0, morse_ramp=0, morse_router_only=False, morse_diag_frequency=1,
                        morse_snapshot_frequency=0, morse_robot_ema=0.9)
    cap = _policy(True, 2, num_experts=3).clone(moe_blocks=("core", "decoder"), num_experts_decoder=5, topk_decoder=2,
                                                width_mult_decoder=0.5)
    upd = MorseActorUpdater(SimpleNamespace(algorithm=a), cap, _split, ["a", "b", "c"], None)
    upd.loss_fn = _toy_loss
    assert upd.num_experts["decoder"] == 5 and upd.num_experts["core"] == 3
    train = TrainState.create(apply_fn=pol.apply, params=params, tx=optax.chain(optax.scale_by_adam(), optax.scale(-1e-3)))
    new, loss, metrics, _ = upd.update(train, jnp.asarray(1.5), batches, jax.random.PRNGKey(25), 0)
    assert np.isfinite(float(loss)) and np.isfinite(metrics["morse/decoder/router_entropy"])
    mean, _ = pol.apply(new.params, *_split(batches[1].observations, 1))
    assert bool(jnp.all(jnp.isfinite(mean)))
    print(f"[ok] decoder top-{k}: renormalized mixture of the kept experts, no gradient to dropped ones; decoder with 5 "
          f"experts of width {hidden} next to a 3-expert core; the update uses the per-block counts")


def test_joint_core_topk_and_mi():
    import tempfile
    from types import SimpleNamespace
    import optax
    from flax.training import checkpoints
    from flax.training.train_state import TrainState
    from cross_embodiment_offline_rl.algorithms.morse.ckpt_analysis import load_checkpoint
    from cross_embodiment_offline_rl.algorithms.morse.moe import ROUTING, keep_topk
    from cross_embodiment_offline_rl.algorithms.morse.updater import MorseActorUpdater, _is_router
    from cross_embodiment_offline_rl.algorithms.utils import Transition
    # Exactly k experts per unit, also when every logit ties.
    for logits in (jnp.zeros((5, 8)), jnp.asarray(np.random.default_rng(40).normal(size=(5, 8)))):
        p = keep_topk(jax.nn.softmax(logits, axis=-1), 4)
        assert bool(jnp.all((p > 0).sum(-1) == 4)) and np.allclose(p.sum(-1), 1.0)

    # Joint-conditioned core: 8 whole-body experts mixed per joint (top-4), each joint's core representation feeds its
    # own decoder (top-4 of 8). Training (capture), evaluation and deterministic policies give the same actions, and
    # so do the weights after a checkpoint round trip.
    kw = dict(moe_blocks=("core", "decoder"), num_experts_decoder=8, topk_core=4, topk_decoder=4, core_routing="joint",
              expert_hidden=8, router_init=0.01)
    pol, cap = _policy(False, num_experts=8).clone(**kw), _policy(True, 2, num_experts=8).clone(**kw)
    det = _policy(False, num_experts=8, deterministic=True).clone(**kw)
    rng = np.random.default_rng(41)
    batch = 4
    batches = [Transition(jnp.asarray(rng.normal(size=(batch, nj * 26 + nf * 12 + 16))),
                          jnp.asarray(rng.normal(size=(batch, 8))), None, None, None) for nj, nf in _ROBOTS]
    variables = pol.init(jax.random.PRNGKey(42), *_split(batches[0].observations, 0))
    assert set(variables) == {"params"}, "init returns parameters only (no routing records in the train state)"
    params = jax.tree_util.tree_map(lambda v: v.astype(jnp.float64), variables)
    core = params["params"]["core"]
    assert [core[f"dense_{i}"]["kernel"].shape[-1] for i in range(3)] == [8, 8, 128]
    with tempfile.TemporaryDirectory() as tmp:
        tx = optax.adam(1e-3)
        target = {"config_algorithm": {"morse_core_routing": "joint"}, "policy": TrainState.create(apply_fn=pol.apply, params=params, tx=tx)}
        target.update({k: TrainState.create(apply_fn=None, params={"w": jnp.asarray(1.0)}, tx=tx)
                       for k in ("value_critic", "critic_1", "critic_2")})
        checkpoints.save_checkpoint(ckpt_dir=tmp, target=target, step=1, keep=1)
        restored, _ = load_checkpoint(tmp)
    for r, b in enumerate(batches):
        x, nj = _split(b.observations, r), _ROBOTS[r][0]
        (mean, _), st = pol.apply(params, *x, mutable=[ROUTING, "intermediates"], capture_intermediates=True)
        assert st["intermediates"]["core"]["__call__"][0].shape == (batch, nj, 128), "one core representation per joint"
        for block in ("core", "decoder"):
            p = st[ROUTING][block]["p"][0]
            assert p.shape == (batch, nj, 8) and bool(jnp.all((p > 0).sum(-1) == 4)), block
        template = jax.eval_shape(lambda: cap.init(jax.random.PRNGKey(0), *x))[CAPTURE_DELTA]
        zeros = jax.tree_util.tree_map(lambda s: jnp.zeros(s.shape, jnp.float64), template)
        (mean_c, _), _ = cap.apply({"params": params["params"], CAPTURE_DELTA: zeros}, *x,
                                   mutable=[CAPTURE_IN, CAPTURE_ROUTER, ROUTING])
        assert _rel_err(mean_c, mean) < 1e-12 and _rel_err(det.apply(params, *x), jnp.tanh(mean)) < 1e-12
        assert _rel_err(pol.apply(restored, *x)[0], mean) < 1e-12, "checkpoint round trip"

    # MI of both blocks, equal total weight per robot, with eta = 0: it reaches both routers and nothing else, on the
    # first and on a further actor step of an update (plain SGD, so other parameters move identically; the two
    # compiled updates may differ in the last bits).
    a = dict(morse_blocks="core,decoder", morse_num_experts=8, morse_router_hidden=16, morse_router_inputs="both",
             morse_conflict_mode="none", morse_lambda=0.0, morse_eta=0.0, morse_conflict_norm="pairs",
             morse_unbiased_types=True, morse_zeta=1.0, morse_kappa=0.0, morse_threshold="fixed", morse_relevance=True,
             morse_evidence="type", morse_type_ema=0.9, morse_conflict_every=1, morse_warmup=0, morse_ramp=0,
             morse_router_only=False, morse_diag_frequency=1, morse_snapshot_frequency=0, morse_robot_ema=0.9,
             morse_mi_blocks="core,decoder", morse_mi_target="type", morse_mi_weighting="robot")
    upds = {}
    for w in (0.1, 0.0):
        upds[w] = MorseActorUpdater(SimpleNamespace(algorithm=SimpleNamespace(**a, morse_mi_weight=w)), cap, _split,
                                    ["a", "b", "c"], None)
        upds[w].loss_fn = _toy_loss
    state = TrainState.create(apply_fn=pol.apply, params=params, tx=optax.sgd(1e-2))
    first = {w: u.update(state, jnp.asarray(1.5), batches, jax.random.PRNGKey(43), 0, 0, 2) for w, u in upds.items()}
    further = {w: u.update(first[0.0][0], jnp.asarray(1.5), batches, jax.random.PRNGKey(44), 0, 1, 2) for w, u in upds.items()}
    for out in (first, further):
        moved = set()
        for (path, x), y in zip(jax.tree_util.tree_leaves_with_path(out[0.1][0].params),
                                jax.tree_util.tree_leaves(out[0.0][0].params)):
            diff = float(jnp.max(jnp.abs(x - y)))
            if _is_router(path):
                moved |= {getattr(path[1], "key", None)} if diff > 1e-8 else set()
            else:
                assert diff < 1e-12, f"MI gradient leaked into {path} ({diff:.1e})"
        assert moved == {"core", "decoder"}, f"MI moved the routers of {moved}"
    n_joints = sum(nj for nj, _ in _ROBOTS)
    metrics = first[0.1][2]
    for block in ("core", "decoder"):
        robot_of, prior = upds[0.1]._mi_groups[block]
        assert len(robot_of) == n_joints and np.allclose(np.bincount(robot_of, weights=prior), 1 / len(_ROBOTS))
        assert np.isfinite(metrics[f"morse/{block}/mi"]) and metrics[f"morse/{block}/router_grad_mi"] > 0
        assert abs(sum(metrics[f"morse/{block}/select_e{e}"] for e in range(8)) - 4.0) < 1e-5   # float32 shares
    print("[ok] joint-conditioned core: exactly 4 of 8 experts per joint in both blocks (ties included), per-joint core "
          "representations; training, evaluation and restored policies agree; joint-type MI with equal robot weights "
          "moves only both routers, on every actor step, with eta = 0")


def test_body_core_balance_mi():
    from types import SimpleNamespace
    import optax
    from flax.training.train_state import TrainState
    from cross_embodiment_offline_rl.algorithms.morse.moe import ROUTING, Router
    from cross_embodiment_offline_rl.algorithms.morse.policy import count_params
    from cross_embodiment_offline_rl.algorithms.morse.updater import MorseActorUpdater, _is_router
    from cross_embodiment_offline_rl.algorithms.utils import Transition
    # v6: a body-routed core (one routing per transition, top-4 of 8, one representation shared by every joint) and the
    # per-joint decoder of the 57 run (8 experts of width 64, top-4). Parameter counts at the real router width.
    kw = dict(moe_blocks=("core", "decoder"), num_experts_decoder=8, topk_core=4, topk_decoder=4, core_routing="body",
              width_mult_core=0.5, width_mult_decoder=0.5, router_init=0.01)
    rng = np.random.default_rng(51)
    batch = 4
    batches = [Transition(jnp.asarray(rng.normal(size=(batch, nj * 26 + nf * 12 + 16))),
                          jnp.asarray(rng.normal(size=(batch, 8))), None, None, None) for nj, nf in _ROBOTS]
    for width, n_core, n_total in ((0.5, 1249616, 1425415), (0.75, 2056016, 2231815)):
        big = _policy(False, num_experts=8).clone(**{**kw, "width_mult_core": width}, router_hidden=64)
        counts = count_params(big.init(jax.random.PRNGKey(50), *_split(batches[0].observations, 0)))
        assert (counts["core"], counts["decoder"], counts["total"]) == (n_core, 148504, n_total), counts
    pol, cap = _policy(False, num_experts=8).clone(**kw), _policy(True, 2, num_experts=8).clone(**kw)
    params = jax.tree_util.tree_map(lambda v: v.astype(jnp.float64),
                                    pol.init(jax.random.PRNGKey(52), *_split(batches[0].observations, 0)))
    recs = []
    for r, b in enumerate(batches):
        _, st = pol.apply(params, *_split(b.observations, r), mutable=[ROUTING, "intermediates"],
                          capture_intermediates=True)
        assert st["intermediates"]["core"]["__call__"][0].shape == (batch, 128), "one core representation per transition"
        p = st[ROUTING]["core"]["p"][0]
        assert p.shape == (batch, 8) and bool(jnp.all((p > 0).sum(-1) == 4))
        recs.append((st[ROUTING]["core"]["morph"][0], st[ROUTING]["core"]["state"][0]))

    # Core MI term with its own weight: only -H(E) (balance block) or the full robot-level -I(R;E); the decoder keeps the
    # joint-type MI at morse_mi_weight. Against a run without the core term (plain SGD), the core router moves by
    # exactly -lr * w_core * grad of the explicit term, and nothing else moves.
    a = dict(morse_blocks="core,decoder", morse_num_experts=8, morse_router_hidden=16, morse_router_inputs="both",
             morse_conflict_mode="none", morse_lambda=0.0, morse_eta=0.0, morse_conflict_norm="pairs",
             morse_unbiased_types=True, morse_zeta=1.0, morse_kappa=0.0, morse_threshold="fixed", morse_relevance=True,
             morse_evidence="type", morse_type_ema=0.9, morse_conflict_every=1, morse_warmup=0, morse_ramp=0,
             morse_router_only=False, morse_diag_frequency=1, morse_snapshot_frequency=0, morse_robot_ema=0.9,
             morse_mi_blocks="core,decoder", morse_mi_target="type", morse_mi_weighting="units", morse_mi_weight=0.1)

    def updater(w_core, balance):
        u = MorseActorUpdater(SimpleNamespace(algorithm=SimpleNamespace(**a, morse_mi_weight_core=w_core,
                                                                         morse_mi_balance_blocks=balance)),
                              cap, _split, ["a", "b", "c"], None)
        u.loss_fn = _toy_loss
        return u

    try:
        updater(0.0, "core")
        raise AssertionError("a balance block without MI weight must be rejected")
    except ValueError:
        pass
    upds = {"balance": updater(0.5, "core"), "robot": updater(0.5, ""), "none": updater(0.0, "")}
    assert upds["none"].mi_blocks == ["decoder"] and upds["balance"].mi_blocks == ["core", "decoder"]
    state = TrainState.create(apply_fn=pol.apply, params=params, tx=optax.sgd(1e-2))
    outs = {k: u.update(state, jnp.asarray(1.5), batches, jax.random.PRNGKey(53), 0) for k, u in upds.items()}
    _, prior = upds["balance"]._mi_groups["core"]
    assert np.allclose(prior, 1 / len(_ROBOTS)), "body core: one MI group per robot, equal weights"
    router = Router(8, 16, "both")

    def core_term(rp, balance):
        r = jnp.stack([jax.nn.softmax(router.apply({"params": rp}, m, s), axis=-1).mean(0) for m, s in recs])
        neg_mi, _, h_marg = itf.group_neg_mutual_information(r, jnp.asarray(prior))
        return -h_marg if balance else neg_mi

    rp = params["params"]["core"]["router"]
    for key, balance in (("balance", True), ("robot", False)):
        expected = jax.tree_util.tree_map(lambda v: -1e-2 * 0.5 * v, jax.grad(core_term)(rp, balance))
        moved = jax.tree_util.tree_map(lambda x, y: x - y, outs[key][0].params["params"]["core"]["router"],
                                       outs["none"][0].params["params"]["core"]["router"])
        for x, y in zip(jax.tree_util.tree_leaves(moved), jax.tree_util.tree_leaves(expected)):
            assert _rel_err(x, y) < 1e-6, f"{key}: core router step {_rel_err(x, y):.1e} off the explicit term"
        for (path, x), y in zip(jax.tree_util.tree_leaves_with_path(outs[key][0].params),
                                jax.tree_util.tree_leaves(outs["none"][0].params)):
            if not (_is_router(path) and getattr(path[1], "key", None) == "core"):
                assert float(jnp.max(jnp.abs(x - y))) < 1e-12, f"core MI term leaked into {path}"
    assert abs(float(outs["balance"][2]["morse/core/h_expert"]) + float(core_term(rp, True))) < 1e-6
    print("[ok] body-routed core: top-4 of 8 per transition, one core representation for every joint, v6 parameter "
          "counts; the core's MI term takes its own weight, as -H(E) only (balance) or the full robot-level -I(R;E), "
          "and moves only the core router")


def test_checkpoint_analysis():
    import tempfile
    from types import SimpleNamespace
    import optax
    from flax.training import checkpoints
    from flax.training.train_state import TrainState
    from cross_embodiment_offline_rl.algorithms.utils import Transition
    from cross_embodiment_offline_rl.algorithms.morse.ckpt_analysis import CheckpointAnalysis, FACTORS
    # Two checkpoints of a core+decoder MoE policy that differ in chosen parts: the decoder recomputation reproduces
    # the policy, the three Shapley contributions add up to the decoder's output change, and a change confined to
    # one factor (decoder experts, decoder router, or the upstream core) is attributed to it entirely.
    batch = 5
    rng = np.random.default_rng(31)
    data = [Transition(jnp.asarray(rng.normal(size=(batch, nj * 26 + nf * 12 + 16))), jnp.asarray(rng.normal(size=(batch, 8))),
                       None, None, None) for nj, nf in _ROBOTS]
    pol = _policy(False, num_experts=3).clone(moe_blocks=("core", "decoder"))
    cap = _policy(True, num_experts=3).clone(moe_blocks=("core", "decoder"))
    noise = lambda t, k: jax.tree_util.tree_map(lambda v: v + 0.3 * jax.random.normal(jax.random.PRNGKey(k), v.shape, v.dtype), t)
    base = noise(jax.tree_util.tree_map(lambda v: v.astype(jnp.float64),
                                        pol.init(jax.random.PRNGKey(32), *_split(data[0].observations, 0))), 37)

    def changed(part):
        prm = dict(base["params"])
        dec = prm["decoder"]
        if part == "experts":
            prm["decoder"] = {**dec, **noise({k: v for k, v in dec.items() if k != "router"}, 33)}
        elif part == "router":
            prm["decoder"] = {**dec, "router": noise(dec["router"], 34)}
        elif part == "features":
            prm["core"] = noise(prm["core"], 35)
        else:
            prm = noise(prm, 36)
        return {"params": prm}

    def awr(out, obs, act, extra, r):
        mean, logstd = out
        lp = -0.5 * ((act[:, :mean.shape[1]] - mean) / jnp.exp(logstd)) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - logstd
        return jnp.ones((obs.shape[0], 1)) * extra[0]["w"], lp

    an = CheckpointAnalysis(cap, _split, awr, ["a", "b", "c"], [batch] * 3,
                            SimpleNamespace(morse_ckpt_analysis="A,B", morse_ckpt_out="", morse_ckpt_batches=1), 0)
    ex_a, ex_b = ({"w": jnp.asarray(1.0)},) * 3, ({"w": jnp.asarray(2.0)},) * 3
    for part in ("experts", "router", "features", "all"):
        var_b = changed(part)
        for r in range(len(_ROBOTS)):
            obs, act = data[r].observations, data[r].actions
            s = jax.device_get(an._robot_fn(r, obs)(base, var_b, ex_a, ex_b, obs, act))
            assert float(s["recon_err"]) < 1e-9, "the decoder recomputation reproduces the policy's mean"
            assert np.allclose(sum(s[f"proj_{f}"] for f in FACTORS), s["dm_sq"], rtol=1e-8, atol=1e-14)
            if part in FACTORS:
                assert np.allclose(s[f"proj_{part}"], s["dm_sq"], rtol=1e-8, atol=1e-14), part
                assert all(np.allclose(s[f"abs_{o}"], 0.0, atol=1e-12) for o in FACTORS if o != part), part
            drift_e, drift_f, tv = float(s["expert_drift"].max()), float(s["feature_drift"].max()), float(s["dec_tv"].max())
            if part == "experts":
                assert drift_e > 0 and drift_f == 0 and tv == 0
            if part == "router":
                assert drift_e == 0 and drift_f == 0 and tv > 0
            if part == "features":
                assert drift_e == 0 and drift_f > 0

    # End to end: two saved TrainState checkpoints and a toy dataset -> per-robot arrays padded over joints.
    class ToyData:
        def sample(self, key, sizes):
            return data, key
    with tempfile.TemporaryDirectory() as tmp:
        tx = optax.adam(1e-3)
        for name, var, w in (("A", base, 1.0), ("B", changed("all"), 2.0)):
            target = {"config_algorithm": {"name": "toy"}, "policy": TrainState.create(apply_fn=pol.apply, params=var, tx=tx)}
            target.update({k: TrainState.create(apply_fn=None, params={"w": jnp.asarray(w)}, tx=tx)
                           for k in ("value_critic", "critic_1", "critic_2", "target_critic_1", "target_critic_2")})
            checkpoints.save_checkpoint(ckpt_dir=os.path.join(tmp, name), target=target, step=1, keep=1)
        a = SimpleNamespace(morse_ckpt_analysis=f"{tmp}/A,{tmp}/B", morse_ckpt_out=f"{tmp}/out.npz", morse_ckpt_batches=2,
                            morse_ckpt_dump="b")
        z = np.load(CheckpointAnalysis(cap, _split, awr, ["a", "b", "c"], [batch] * 3, a, 0).run(ToyData()))
        nj_b = _ROBOTS[1][0]
        assert z["dump_b_mean_A"].shape == (2 * batch, nj_b) and z["dump_b_phi"].shape == (2 * batch, nj_b, 3)
        assert "dump_a_mean_A" not in z.files
        assert np.allclose(np.abs(z["dump_b_mean_B"] - z["dump_b_mean_A"]).mean(axis=0), z["out_change"][1, :nj_b])
        assert np.allclose(z["dump_b_phi"].sum(axis=-1), z["dump_b_dm"])
        n_max = max(nj for nj, _ in _ROBOTS)
        assert z["nll_A"].shape == (3, n_max) and list(z["n_joints"]) == [nj for nj, _ in _ROBOTS]
        assert np.isfinite(z["nll_A"][0, :5]).all() and np.isnan(z["nll_A"][0, 5:]).all()
        assert np.allclose(z["w_A"], 1.0) and np.allclose(z["w_B"], 2.0) and z["core_p_A"].shape == (3, 3)
        shares = sum(z[f"proj_{f}"] for f in FACTORS) / z["dm_sq"]
        assert np.allclose(shares[np.isfinite(shares)], 1.0)
    print("[ok] checkpoint analysis: decoder recomputation == policy; Shapley parts add up; a change of only the experts, "
          "the router, or the upstream core is attributed to it; saved checkpoints -> per-robot arrays")


def test_coordination_episodes_and_dump():
    import tempfile
    from types import SimpleNamespace
    from cross_embodiment_offline_rl.algorithms.morse.coordination import (CoordinationDump, GENERAL, episode_table,
                                                                           reference_mask)
    # Episodes are the runs up to a done flag (the last one may be cut by the end of the data); references reach the
    # horizon without termination and keep a fraction of the best such return.
    done = np.array([0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0], bool)
    term = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], bool)
    trunc = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0], bool)
    tab = episode_table(done, term, trunc, np.arange(11.0))
    assert list(tab["ep_start"]) == [0, 3, 5, 9] and list(tab["ep_len"]) == [3, 2, 4, 2]
    assert np.allclose(tab["ep_return"], [3, 7, 26, 19])
    assert list(tab["ep_complete"]) == [True, True, True, False] and list(tab["ep_terminated"]) == [True, False, False, False]
    assert list(reference_mask(tab, 4, 0.5)) == [False, False, True, False]

    # The dump: every transition of the reference episodes in order, with both policies' means on exactly those rows.
    n, horizon = 14, 4
    sizes = [nj * 26 + nf * 12 + 16 for nj, nf in _ROBOTS]
    rng = np.random.default_rng(41)
    d = np.zeros((n, 3), bool); d[[3, 5, 9, 13]] = True
    te = np.zeros((n, 3), bool); te[5] = True
    tr = d & ~te
    rew = np.ones((n, 3)); rew[10:14, 0] = 0.1           # robot 0: its last full episode keeps < half the best return
    data = SimpleNamespace(obs=rng.normal(size=(n, 3, max(sizes))), act=rng.normal(size=(n, 3, 8)), rew=rew, done=d)
    split = lambda o, r: _split(o[:, :sizes[r]], r)

    def names(r):
        nj, nf = _ROBOTS[r]
        a, c, ids = nj * 23, nj * 26 + nf * 10, {}
        for j in range(nj):
            ids.update({f"j{j}_position": a + 3 * j, f"j{j}_velocity": a + 3 * j + 1, f"j{j}_previous_action": a + 3 * j + 2})
        for f in range(nf):
            ids.update({f"f{f}_ground_contact": c + 2 * f, f"f{f}_cycles_since_last_ground_contact": c + 2 * f + 1})
        ids.update({g: sizes[r] - 16 + i for i, g in enumerate(GENERAL)})
        return ids

    pol = _policy(False, num_experts=3).clone(moe_blocks=("core", "decoder"))
    noise = lambda t, k: jax.tree_util.tree_map(lambda v: v + 0.3 * jax.random.normal(jax.random.PRNGKey(k), v.shape, v.dtype), t)
    var_a = noise(pol.init(jax.random.PRNGKey(42), *split(jnp.asarray(data.obs[:, 0]), 0)), 43)
    var_b = noise(var_a, 44)
    with tempfile.TemporaryDirectory() as tmp:
        dump = CoordinationDump(pol, split, ["a", "b", "c"], [names(r) for r in range(3)], horizon,
                                SimpleNamespace(morse_coord_out=tmp, morse_coord_min_return=0.5), chunk=3)
        dump.run(data, var_a, var_b, te, tr, checkpoints=("A", "B"))
        for r, name in enumerate(["a", "b", "c"]):
            z = np.load(os.path.join(tmp, f"{name}.npz"))
            nj, nf = _ROBOTS[r]
            ref = [0, 2] if r == 0 else [0, 2, 3]
            assert list(np.flatnonzero(z["ep_reference"])) == ref
            idx = np.concatenate([np.arange(z["ep_start"][e], z["ep_start"][e] + horizon) for e in ref])
            assert list(z["index"]) == list(idx) and list(z["t"]) == list(np.tile(np.arange(horizon), len(ref)))
            assert list(z["episode"]) == list(np.repeat(ref, horizon))
            assert np.allclose(z["act"], data.act[idx, r, :nj], atol=1e-6)
            for key, var in (("mean_A", var_a), ("mean_B", var_b)):
                mean, logstd = pol.apply(var, *split(jnp.asarray(data.obs[idx, r]), r), robot_id=r)
                assert np.allclose(z[key], mean, atol=1e-5) and np.allclose(z["logstd" + key[-2:]], logstd.mean(axis=0), atol=1e-5)
            pos = [names(r)[f"j{j}_position"] for j in range(nj)]
            air = [names(r)[f"f{f}_cycles_since_last_ground_contact"] for f in range(nf)]
            assert np.allclose(z["joint_pos"], data.obs[idx, r][:, pos], atol=1e-2)
            assert np.allclose(z["air_time"], data.obs[idx, r][:, air], atol=1e-2)
            assert list(z["joint_names"]) == [f"j{j}" for j in range(nj)] and z["general"].shape == (len(idx), len(GENERAL))

    # Start-up mode: the first steps of every complete episode, flagged, with each checkpoint's advantage weights.
    def awr(out, o, act, extra, r):
        mean, logstd = out
        lp = -0.5 * ((act[:, :mean.shape[1]] - mean) / jnp.exp(logstd)) ** 2
        return jnp.exp(extra["c"] * act[:, :1] + 0.1 * o[:, :1]), lp
    ex_a, ex_b = {"c": jnp.asarray(0.5)}, {"c": jnp.asarray(-0.5)}
    with tempfile.TemporaryDirectory() as tmp:
        dump = CoordinationDump(pol, split, ["a", "b", "c"], [names(r) for r in range(3)], horizon,
                                SimpleNamespace(morse_coord_out=tmp, morse_coord_min_return=0.5, morse_coord_startup=2),
                                chunk=3, awr_terms=awr)
        dump.run(data, var_a, var_b, te, tr, checkpoints=("A", "B"), extra_a=ex_a, extra_b=ex_b)
        z = np.load(os.path.join(tmp, "a.npz"))
        idx = np.array([0, 1, 4, 5, 6, 7, 10, 11])
        assert list(z["index"]) == list(idx) and list(z["t"]) == [0, 1] * 4
        assert list(z["reference"]) == [True, True, False, False, True, True, False, False]
        mean = pol.apply(var_a, *split(jnp.asarray(data.obs[idx, 0]), 0), robot_id=0)[0]
        assert np.allclose(z["mean_A"], mean, atol=1e-5)
        for key, ex in (("w_A", ex_a), ("w_B", ex_b)):
            assert np.allclose(z[key], np.exp(float(ex["c"]) * data.act[idx, 0, 0] + 0.1 * data.obs[idx, 0, 0]), rtol=1e-5)
    print("[ok] coordination dump: episodes from done flags, reference episodes, policy means on exactly their rows; "
          "start-up mode with advantage weights")


def test_routing_noise_ties_and_robot_fixed_routing():
    from cross_embodiment_offline_rl.algorithms.morse.moe import ROUTING, keep_topk, route
    # Tie rules: exact keeps exactly k (ties to the lower index), keep keeps every weight >= the k-th largest.
    p = jnp.asarray([[0.25, 0.25, 0.25, 0.25], [0.4, 0.2, 0.2, 0.2]])
    assert np.allclose(keep_topk(p, 2), [[0.5, 0.5, 0, 0], [2 / 3, 1 / 3, 0, 0]])
    assert np.allclose(keep_topk(p, 2, "keep"), p), "ties kept: a uniform start is dense"

    # Routing noise: training (a 'routing' stream) adds sigma * eps per transition, the first rows of a longer batch get
    # the same noise, evaluation (no stream) routes on the clean logits, and the recorded offset is the noise term.
    blk = MoEBlock(layer_features=(6, 2), layer_post=("ln_elu", "none"), kernel_inits=(lecun_normal(),) * 2,
                   num_experts=4, router_hidden=8, train_topk=2, router_noise=True, noise_floor=0.01)
    ks = jax.random.split(jax.random.PRNGKey(90), 4)
    x, m, s = (jax.random.normal(k, (5, 3, d)) for k, d in zip(ks[:3], (5, 3, 2)))
    params = blk.init(ks[3], x, m, s)
    assert "router_noise" in params["params"], "noise head parameters"
    rk = jax.random.PRNGKey(91)
    _, long = blk.apply(params, x, m, s, mutable=[ROUTING], rngs={"routing": rk})
    _, short = blk.apply(params, x[:2], m[:2], s[:2], mutable=[ROUTING], rngs={"routing": rk})
    _, clean = blk.apply(params, x, m, s, mutable=[ROUTING])
    off = long[ROUTING]["offset"][0]
    assert np.allclose(short[ROUTING]["offset"][0], off[:2]) and np.allclose(short[ROUTING]["p"][0], long[ROUTING]["p"][0][:2])
    assert np.allclose(clean[ROUTING]["offset"][0], 0.0), "no noise without the routing stream"
    sigma0 = float(jax.nn.softplus(0.0) + 0.01)
    assert abs(float(jnp.std(off)) / sigma0 - 1.0) < 0.35, "initial noise scale softplus(0) + floor"
    # The task loss trains the noise head through the noisy logits.
    g = jax.grad(lambda prm: jnp.sum(blk.apply(prm, x, m, s, rngs={"routing": rk}) ** 2))(params)
    assert float(sum(jnp.abs(v).sum() for v in jax.tree_util.tree_leaves(g["params"]["router_noise"]))) > 0.0
    assert np.allclose(route(jnp.zeros((1, 4)), 2)[1], [[0.5, 0.5, 0, 0]])

    # Robot-fixed routing: expert r serves robot r only (no router, no routing records); the other experts don't matter.
    pol = _policy(False, num_experts=3).clone(moe_blocks=("core", "decoder"), core_routing="body",
                                               fixed_robot_blocks=("core", "decoder"))
    rng = np.random.default_rng(92)
    obs = [jnp.asarray(rng.normal(size=(4, nj * 26 + nf * 12 + 16))) for nj, nf in _ROBOTS]
    params = pol.init(jax.random.PRNGKey(93), *_split(obs[0], 0))
    assert "router" not in params["params"]["core"] and "router" not in params["params"]["decoder"]
    for r in range(3):
        out, st = pol.apply(params, *_split(obs[r], r), robot_id=r, mutable=[ROUTING])
        assert "core" not in st.get(ROUTING, {}) and "decoder" not in st.get(ROUTING, {})
        for blk_name in ("core", "decoder"):
            for e in range(3):
                bumped = jax.tree_util.tree_map(lambda v: v, params)
                bumped["params"][blk_name] = jax.tree_util.tree_map(
                    lambda v: v.at[e].add(0.5) if v.ndim >= 1 and v.shape[0] == 3 else v, params["params"][blk_name])
                changed = not np.allclose(pol.apply(bumped, *_split(obs[r], r), robot_id=r)[0], out[0])
                assert changed == (e == r), f"robot {r}, {blk_name} expert {e}: changed={changed}"
    print("[ok] top-k tie rules; routing noise: per-transition draws shared by shorter batches, clean evaluation, "
          "recorded offset, noise head trained by the task; robot-fixed core and decoder use only the robot's expert")


def test_compat_scores_match_explicit_gradients():
    # A scored block with an element axis (like the decoder), top-2 of 3, two robots of 4 transitions with 3 elements.
    # Scoring pass: every expert receives the unweighted output gradient. Each unit's cosine with each expert's actual
    # update (actual routing weights) from the other transitions of each half must match explicit per-unit gradients.
    n, J, d_in, K = 4, 3, 5, 3

    def block(**kw):
        return MoEBlock(layer_features=(6, 2), layer_post=("ln_elu", "none"), kernel_inits=(lecun_normal(),) * 2,
                        num_experts=K, router_hidden=8, **kw)
    scorer, plain, single = block(capture=True, n_capture=n, score_mode=True, train_topk=2), block(train_topk=2), \
        block(fixed_routing=True)
    ks = jax.random.split(jax.random.PRNGKey(80), 12)
    inputs = [tuple(jax.random.normal(ks[4 * r + i], (n, J, d), jnp.float64) for i, d in enumerate((d_in, 3, 2, 2)))
              for r in range(2)]
    v = scorer.init(ks[10], *inputs[0][:3])
    params = {"params": jax.tree_util.tree_map(
        lambda a: a.astype(jnp.float64) + 0.3 * jax.random.normal(ks[11], a.shape, jnp.float64), v["params"])}
    experts = {k: val for k, val in params["params"].items() if k != "router"}
    robots, units = [], []
    for r, (x, m, s, w_obj) in enumerate(inputs):
        def objective(y):
            return jnp.sum(jnp.tanh((y * w_obj).sum(axis=(1, 2))) ** 2)

        zeros = jax.tree_util.tree_map(jnp.zeros_like, v[CAPTURE_DELTA])

        def f(delta):
            y, st = scorer.apply({"params": params["params"], CAPTURE_DELTA: delta}, x, m, s,
                                 mutable=[CAPTURE_IN, CAPTURE_ROUTER])
            return objective(y), st
        (_, st), gd = jax.value_and_grad(f, has_aux=True)(zeros)
        layers = [("dense", st[CAPTURE_IN]["dense_0"]["x"][0], gd["dense_0"]["z"]),
                  ("ln", st[CAPTURE_IN]["ln_0"]["xhat"][0], gd["ln_0"]["y"]),
                  ("dense", st[CAPTURE_IN]["dense_1"]["x"][0], gd["dense_1"]["z"])]
        w = st[CAPTURE_ROUTER]["p"][0]
        y = plain.apply(params, x, m, s)
        assert np.allclose(y, scorer.apply({"params": params["params"], CAPTURE_DELTA: zeros}, x, m, s,
                                           mutable=[CAPTURE_IN, CAPTURE_ROUTER])[0]), "scoring keeps the output"
        robots.append((layers, w))
        g_out = jax.grad(objective)(y)
        for t in range(n):
            for j in range(J):
                vecs = []
                for k in range(K):
                    def proj(ep):
                        yk = single.apply({"params": ep}, x[t, j][None], m[t, j][None], s[t, j][None],
                                          jax.nn.one_hot(k, K, dtype=jnp.float64))
                        return jnp.sum(yk[0] * g_out[t, j])
                    gk = jax.grad(proj)(experts)
                    vecs.append(jnp.concatenate([leaf[k].reshape(-1) for leaf in jax.tree_util.tree_leaves(gk)]))
                units.append(((r, t), t % 2, jnp.stack(vecs), w[t, j]))
    (scores, support) = itf.compat_scores(robots, min_support=1)
    for h in (0, 1):
        ref = sum(wu[:, None] * vec * (hu == h) for _, hu, vec, wu in units)                # (K, P)
        used = {key: np.asarray(sum(wu for kk, _, _, wu in units if kk == key) > 0) for key in {u[0] for u in units}}
        sup = sum(u for key, u in used.items() if key[1] % 2 == h)
        assert np.allclose(np.asarray(support[h]), sup), "support counts"
        n_checked = 0
        for i, (key, hu, vec, wu) in enumerate(units):
            own = sum(wv[:, None] * vv for kk, _, vv, wv in units if kk == key)
            r_ = ref - own if hu == h else ref
            want = jnp.sum(vec * r_, -1) / (jnp.linalg.norm(vec, axis=-1) * jnp.linalg.norm(r_, axis=-1))
            want_valid = (sup - (used[key] if hu == h else 0)) >= 1
            r, t = key
            got, got_valid = scores[r][0][h, t, i % J], scores[r][1][h, t, i % J]
            assert np.array_equal(np.asarray(got_valid), want_valid), f"half {h}, unit {key}: validity"
            assert np.allclose(np.asarray(got)[want_valid], np.asarray(want)[want_valid], atol=1e-8), \
                f"half {h}, unit {key}: {got} vs {want}"
            n_checked += int(want_valid.sum())
        assert n_checked > len(units), "most unit-expert pairs are checked"
    assert not bool(jnp.any(itf.compat_scores(robots, min_support=100)[0][0][1])), "support threshold"

    # Labels, the objective and its controls.
    rng = np.random.default_rng(81)
    cos = jnp.asarray(rng.uniform(-1, 1, size=(2, 30, K)))
    valid = jnp.asarray(rng.random(size=(2, 30, K)) < 0.8)
    nonconf, conf = itf.compat_labels(cos, valid, 0.05)
    both = np.asarray(valid[0] & valid[1])
    assert np.array_equal(nonconf, both & (np.asarray(cos) >= -0.05).all(0))
    assert np.array_equal(conf, both & (np.asarray(cos) < -0.05).all(0))
    p = jax.nn.softmax(jnp.asarray(rng.normal(size=(30, K))), -1)
    for mode in ("compat", "conflict_only", "uniform"):
        loss, parts = itf.compat_loss(p, nonconf, conf, mode)
        vals = []
        for u in range(30):
            a, c = np.asarray(nonconf[u]), np.asarray(conf[u])
            if not a.any():
                continue
            pa = np.asarray(p[u]) * (a | c)
            pt = pa / pa.sum()
            if mode == "compat":
                q = a / a.sum()
                vals.append(np.sum(q[a] * np.log(q[a] / pt[a])))
            elif mode == "conflict_only":
                vals.append(-np.log(pt[a].sum()))
            else:
                q = (a | c) / (a | c).sum()
                vals.append(np.sum(q[a | c] * np.log(q[a | c] / pt[a | c])))
        assert abs(float(loss) - np.mean(vals)) < 1e-6, f"{mode}: {float(loss)} vs {np.mean(vals)}"
        if mode == "compat":
            assert abs(float(parts["neg_log_mass"] + parts["kl_within"]) - float(loss)) < 1e-6, "KL decomposition"
    strata = jnp.asarray(np.repeat(np.arange(5), 6))
    sh_n, sh_c = itf.shuffle_within((nonconf, conf), strata, jax.random.PRNGKey(82))
    for s_ in range(5):
        rows = lambda a: sorted(map(tuple, np.asarray(a)[s_ * 6:(s_ + 1) * 6].astype(int)))
        assert rows(sh_n) == rows(nonconf) and rows(sh_c) == rows(conf), "labels permuted within strata"
    print("[ok] compatibility scores == explicit per-unit cosines with each expert's actual update from the other "
          "transitions of each half; labels; KL objective, conflict-only and uniform controls; stratified shuffle")


def test_mi_cap_eg_groups_noise_keys_and_compat_step():
    import tempfile
    from types import SimpleNamespace
    import optax
    from flax.training.train_state import TrainState
    from cross_embodiment_offline_rl.algorithms.morse.updater import MorseActorUpdater, _is_router
    from cross_embodiment_offline_rl.algorithms.utils import Transition
    rng = np.random.default_rng(95)
    batch = 4
    batches = [Transition(jnp.asarray(rng.normal(size=(batch, nj * 26 + nf * 12 + 16))),
                          jnp.asarray(rng.normal(size=(batch, 8))), None, None, None) for nj, nf in _ROBOTS]
    kw = dict(moe_blocks=("core", "decoder"), topk_core=2, topk_decoder=2, core_routing="body", router_noise=True)
    pol, cap = _policy(False, num_experts=4).clone(**kw), _policy(True, 2, num_experts=4).clone(**kw)
    params = jax.tree_util.tree_map(lambda v: v.astype(jnp.float64),
                                    pol.init(jax.random.PRNGKey(96), *_split(batches[0].observations, 0)))
    tmp = tempfile.mkdtemp()
    dist = os.path.join(tmp, "dist.npy")
    np.save(dist, np.array([[0, 0.1, 1.0], [0.1, 0, 1.0], [1.0, 1.0, 0]]))     # a, b close; c apart
    a = dict(morse_blocks="core,decoder", morse_num_experts=4, morse_router_hidden=16, morse_router_inputs="both",
             morse_conflict_mode="none", morse_lambda=0.0, morse_eta=0.0, morse_conflict_norm="pairs",
             morse_unbiased_types=True, morse_zeta=1.0, morse_kappa=0.0, morse_threshold="fixed", morse_relevance=True,
             morse_evidence="type", morse_type_ema=0.9, morse_conflict_every=1, morse_warmup=0, morse_ramp=0,
             morse_router_only=False, morse_diag_frequency=1, morse_snapshot_frequency=0, morse_robot_ema=0.9,
             morse_mi_blocks="core,decoder", morse_mi_target="type", morse_mi_weighting="units", morse_mi_weight=0.5,
             morse_mi_weight_core=-1.0, morse_mi_balance_blocks="", morse_router_noise=True,
             embodiment_distance_path=dist, morse_eg_groups=2)

    def updater(**over):
        u = MorseActorUpdater(SimpleNamespace(algorithm=SimpleNamespace(**{**a, **over})), cap, _split, ["a", "b", "c"], None)
        u.loss_fn = _toy_loss
        return u

    state = TrainState.create(apply_fn=pol.apply, params=params, tx=optax.sgd(1e-2))
    # EG groups of the core's MI need at least two groups (one group makes I(G;E) zero by construction).
    for bad in (1, 0):
        try:
            updater(morse_mi_core_groups="eg", morse_eg_groups=bad)
            raise AssertionError(f"morse_eg_groups={bad} must be rejected")
        except ValueError:
            pass
    # EG groups of the core's MI: {a, b} and {c}; group routing = mean of its robots', prior = summed prior.
    eg = updater(morse_mi_core_groups="eg")
    eg.update(state, jnp.asarray(1.5), batches, jax.random.PRNGKey(97), 0)
    assert sorted(np.bincount(eg.core_group_of).tolist()) == [1, 2] and eg.core_group_of[0] == eg.core_group_of[1]
    agg, (_, prior) = eg._mi_agg["core"], eg._mi_groups["core"]
    assert np.allclose(agg.sum(1), 1.0) and np.allclose(sorted(prior), [1 / 3, 2 / 3])
    assert eg._mi_agg["decoder"] is None

    # MI cap: the capped run's routers differ from the uncapped one by lr * (1 - c) * the MI gradient, per block;
    # nothing else differs. Noise keys come from their own stream: the main key sequence is unchanged.
    free, capped = updater(morse_mi_cap=0.0), updater(morse_mi_cap=0.5)
    key = jax.random.PRNGKey(98)
    out_f, out_c = free.update(state, jnp.asarray(1.5), batches, key, 0), capped.update(state, jnp.asarray(1.5), batches, key, 0)
    no_noise = updater(morse_router_noise=False)
    assert bool(jnp.all(out_f[3] == no_noise.update(state, jnp.asarray(1.5), batches, key, 0)[3])), "main key sequence"
    for blk_name in ("core", "decoder"):
        task, mi = out_c[2][f"morse/{blk_name}/router_grad_task"], out_c[2][f"morse/{blk_name}/router_grad_mi"]
        c = min(1.0, 0.5 * task / mi)
        assert abs(out_c[2][f"morse/{blk_name}/mi_cap_factor"] - c) < 1e-9 and c < 1.0, "the cap binds here"
        diff = jax.tree_util.tree_map(lambda x, y: x - y, out_c[0].params["params"][blk_name]["router"],
                                      out_f[0].params["params"][blk_name]["router"])
        norm = float(jnp.sqrt(sum(jnp.sum(d ** 2) for d in jax.tree_util.tree_leaves(diff))))
        assert abs(norm - 1e-2 * (1.0 - c) * mi) < 1e-6 * max(1.0, 1e-2 * mi), f"{blk_name}: {norm} vs {1e-2 * (1 - c) * mi}"
    assert abs(out_c[2]["morse/core/mi_cap_active_frac"] - 1.0) < 1e-12
    for (path, x), y in zip(jax.tree_util.tree_leaves_with_path(out_c[0].params), jax.tree_util.tree_leaves(out_f[0].params)):
        if not _is_router(path):
            assert float(jnp.max(jnp.abs(x - y))) == 0.0, f"the cap changed {path}"

    # Compatibility step: router-only, with label diagnostics.
    off = updater(morse_mi_weight=0.0, morse_compat_route="off")
    on = updater(morse_mi_weight=0.0, morse_compat_route="compat", morse_compat_lambda_core=0.1,
                 morse_compat_lambda_decoder=0.1, morse_compat_tau=0.05, morse_compat_core_transitions=4,
                 morse_compat_decoder_transitions=2, morse_compat_min_support=1)
    out_off, out_on = off.update(state, jnp.asarray(1.5), batches, key, 0), on.update(state, jnp.asarray(1.5), batches, key, 0)
    for blk_name in ("core", "decoder"):
        for k in ("nonconf_frac", "conf_frac", "valid_frac", "loss", "neg_log_mass", "kl_within"):
            assert np.isfinite(out_on[2][f"morse/{blk_name}/compat_{k}"]), k
        assert out_on[2][f"morse/{blk_name}/compat_valid_frac"] > 0.0
    moved = False
    for (path, x), y in zip(jax.tree_util.tree_leaves_with_path(out_on[0].params), jax.tree_util.tree_leaves(out_off[0].params)):
        delta = float(jnp.max(jnp.abs(x - y)))
        if _is_router(path):
            moved = moved or delta > 0.0
        else:
            assert delta == 0.0, f"compatibility routing changed {path}"
    assert moved, "the compatibility objective moves the routers"
    print("[ok] EG groups in the core's MI; per-block MI cap (raw router gradients) with its factor and activity; routing "
          "noise keys off the main key sequence; compatibility step on the routers only, with label diagnostics")


def test_dataset_selection_and_checksum():
    import hashlib
    import json
    import tempfile
    from cross_embodiment_offline_rl.algorithms.utils import BatchPrefetcher, DATASET_ROBOTS, Dataset, dataset_columns
    rng = np.random.default_rng(99)
    n, envs, dim = 40, 4, 5
    arrays = {"observations": rng.normal(size=(n, envs, dim)).astype(np.float32),
              "next_observations": rng.normal(size=(n, envs, dim)).astype(np.float32),
              "actions": rng.normal(size=(n, envs, 3)).astype(np.float32),
              "rewards": rng.normal(size=(n, envs)).astype(np.float32),
              "dones": rng.random(size=(n, envs)) < 0.2}
    with tempfile.TemporaryDirectory() as root:
        for name, a in arrays.items():
            np.save(os.path.join(root, f"{name}.npy"), a)
        json.dump({"n_samples": n, "n_envs": envs}, open(os.path.join(root, "metadata.json"), "w"))
        sel = Dataset(root, normalize=False, rows=(30, 40), columns=[2])
        assert sel.n_samples == 10 and sel.n_envs == 1 and sel.obs.shape == (10, 1, dim)
        assert np.array_equal(sel.obs[:, 0], arrays["observations"][30:40, 2])
        assert np.array_equal(sel.done[:, 0], arrays["dones"][30:40, 2])
        pre = BatchPrefetcher(sel, [6], seed=4, depth=2, workers=1)
        draws, total, digest = np.random.default_rng(4), 0, hashlib.sha1()
        for _ in range(3):
            b = pre.next()[0]
            idx = draws.integers(0, 10, size=6)
            total += int(idx.sum())
            digest.update(np.ascontiguousarray(idx, dtype=np.int64).tobytes())
            assert np.array_equal(np.asarray(b.observations), arrays["observations"][30 + idx, 2])
        assert pre.checksum == total, "checksum of the indices taken"
        assert pre.sha1.hexdigest() == digest.hexdigest(), "content hash of the indices taken, in order"
        pre.close()
    assert dataset_columns(DATASET_ROBOTS, 16) is None and dataset_columns(("talos",), 16) == [11]
    assert dataset_columns(("a", "b"), 2) is None
    print("[ok] dataset rows and robot columns selected exactly; the sampler's index checksum and content hash")


def test_shared_core_experts():
    import tempfile
    from types import SimpleNamespace
    import optax
    from flax import serialization
    from flax.training.train_state import TrainState
    from cross_embodiment_offline_rl.algorithms.morse.config import LEGACY_ARCHITECTURE
    from cross_embodiment_offline_rl.algorithms.morse.moe import ROUTING, Router, route
    from cross_embodiment_offline_rl.algorithms.morse.policy import eg_group_table, get_morse_policy
    from cross_embodiment_offline_rl.algorithms.morse.updater import MorseActorUpdater, _is_router
    from cross_embodiment_offline_rl.algorithms.utils import Transition

    def expert_sq(tree, n_experts):
        """Squared norm of every expert's slice of a block's expert leaves (routers excluded), (K,)."""
        out = np.zeros(n_experts)
        for name, sub in tree.items():
            if name in ("router", "router_noise"):
                continue
            for leaf in jax.tree_util.tree_leaves(sub):
                out += np.asarray(jnp.sum(jnp.square(leaf.reshape(leaf.shape[0], -1)), axis=1))
        return out

    # 1. One block: 1 global + 3 group + 5 routed experts, top-2 of the routed ones. Against the explicit outputs of every
    #    expert: y = 0.25 global + 0.25 own group + 0.5 renormalized top-2 mixture, for every group.
    n, d_in, G, R = 6, 5, 3, 5
    K = 1 + G + R
    kw = dict(layer_features=(6, 2), layer_post=("ln_elu", "none"), kernel_inits=(lecun_normal(),) * 2, num_experts=K,
              router_hidden=8)
    blk = MoEBlock(**kw, train_topk=2, n_global=1, n_group=G, shared_coef=0.25)
    blk_c = MoEBlock(**kw, train_topk=2, n_global=1, n_group=G, shared_coef=0.25, capture=True, n_capture=n)
    single = MoEBlock(**kw, fixed_routing=True)
    ks = jax.random.split(jax.random.PRNGKey(110), 6)
    x, m, s = (jax.random.normal(k, (n, d), jnp.float64) for k, d in zip(ks[:3], (d_in, 3, 2)))
    v = blk_c.init(ks[3], x, m, s)
    params = {"params": jax.tree_util.tree_map(
        lambda a: a.astype(jnp.float64) + 0.3 * jax.random.normal(ks[4], a.shape, jnp.float64), v["params"])}
    assert params["params"]["router"]["morph_out"]["kernel"].shape[-1] == R, "the router covers the routed experts only"
    zeros = jax.tree_util.tree_map(jnp.zeros_like, v[CAPTURE_DELTA])
    experts = {k: val for k, val in params["params"].items() if k != "router"}
    h = jnp.stack([single.apply({"params": experts}, x, m, s, jax.nn.one_hot(e, K, dtype=jnp.float64))
                   for e in range(K)], axis=1)                                              # (n, K, 2)
    p_r = route(Router(R, 8, "both").apply({"params": params["params"]["router"]}, m, s), 2)[1]
    assert bool(jnp.all((p_r > 0).sum(-1) == 2)) and np.allclose(p_r.sum(-1), 1.0), "two renormalized routed weights"
    w_obj = jax.random.normal(ks[5], (n, 2), jnp.float64)
    for g in range(G):
        gp = jax.nn.one_hot(g, G, dtype=jnp.float64)
        y = blk.apply(params, x, m, s, None, gp)
        parts = (0.25 * h[:, 0], 0.25 * h[:, 1 + g], 0.5 * jnp.einsum("nk,nko->no", p_r, h[:, 1 + G:]))
        assert _rel_err(y, sum(parts)) < 1e-12, f"group {g}: output != explicit weighted sum"
        yc, st = blk_c.apply({"params": params["params"], CAPTURE_DELTA: zeros}, x, m, s, None, gp,
                             mutable=[ROUTING, CAPTURE_IN, CAPTURE_ROUTER])
        assert _rel_err(yc, y) < 1e-12
        assert np.allclose(st[ROUTING]["p"][0], p_r), "routing records: the routed experts' weights"
        want_w = jnp.concatenate([jnp.full((n, 1), 0.25), jnp.broadcast_to(0.25 * gp, (n, G)), 0.5 * p_r], axis=-1)
        assert np.allclose(st[CAPTURE_ROUTER]["p"][0], want_w), "captured weights of every expert"
        norms = jnp.stack([jnp.linalg.norm(v_, axis=-1) for v_ in parts] + [jnp.linalg.norm(y, axis=-1)], axis=-1)
        assert np.allclose(st[ROUTING]["branch_norm"][0], norms), "branch norms"
        # Group selection: only this group's expert matters.
        for g2 in range(G):
            bumped = {"params": {**params["params"], **{
                k: jax.tree_util.tree_map(lambda a: a.at[1 + g2].add(0.5), val) for k, val in experts.items()}}}
            changed = not np.allclose(blk.apply(bumped, x, m, s, None, gp), y)
            assert changed == (g2 == g), f"group {g}: group expert {g2} changed the output: {changed}"
        # Task gradients (directly, not parameter movement): the global expert always, this group's expert only, the
        # routed experts exactly where some unit selects them.
        grad = jax.grad(lambda prm: jnp.sum(jnp.tanh(blk.apply(prm, x, m, s, None, gp) * w_obj) ** 2))(params)
        sq = expert_sq(grad["params"], K)
        assert sq[0] > 0 and sq[1 + g] > 0, f"group {g}: global and own group expert get gradients"
        assert all(sq[1 + g2] == 0.0 for g2 in range(G) if g2 != g), f"group {g}: other group experts get gradients"
        selected = np.asarray((p_r > 0).any(axis=0))
        assert np.array_equal(sq[1 + G:] > 0, selected), f"group {g}: routed gradients {sq[1 + G:]} vs selected {selected}"
    try:
        blk.apply(params, x, m, s)
        raise AssertionError("group experts without the robot's group must fail")
    except ValueError:
        pass

    # 2. Policy and update: three toy robots in two EG groups ({a, b}, {c}); core 1 global + 2 group + 5 routed (top-2),
    #    decoder 4 experts (top-2); routing noise; MI on both routers (EG groups at the core), balance, MI cap 0.5.
    tmp = tempfile.mkdtemp()
    dist = os.path.join(tmp, "dist.npy")
    np.save(dist, np.array([[0, 0.1, 1.0], [0.1, 0, 1.0], [1.0, 1.0, 0]]))
    table = eg_group_table(SimpleNamespace(morse_eg_groups=2, embodiment_distance_path=dist), ["a", "b", "c"])
    assert table[0] == table[1] != table[2] and sorted(set(table)) == [0, 1], table
    for bad in (dict(morse_eg_groups=1), dict(morse_eg_groups=3, names=["a", "b"])):
        try:
            eg_group_table(SimpleNamespace(morse_eg_groups=bad["morse_eg_groups"], embodiment_distance_path=dist),
                           bad.get("names", ["a", "b", "c"]))
            raise AssertionError(f"inconsistent EG groups {bad} must be rejected")
        except ValueError:
            pass
    shared_kw = dict(moe_blocks=("core", "decoder"), num_experts_decoder=4, topk_core=2, topk_decoder=2,
                     core_routing="body", router_noise=True, core_shared="hierarchical", core_group_of=table)
    pol, cap = _policy(False, num_experts=8).clone(**shared_kw), _policy(True, 2, num_experts=8).clone(**shared_kw)
    assert pol.core_shared_counts() == (1, 2) and pol.router_experts_of("core") == 5 and pol.router_experts_of("decoder") == 4
    rng = np.random.default_rng(111)
    batch = 4
    batches = [Transition(jnp.asarray(rng.normal(size=(batch, nj * 26 + nf * 12 + 16))),
                          jnp.asarray(rng.normal(size=(batch, 8))), None, None, None) for nj, nf in _ROBOTS]
    init = pol.init(jax.random.PRNGKey(112), *_split(batches[0].observations, 0))
    params = jax.tree_util.tree_map(lambda a: a.astype(jnp.float64) + 0.1 * jax.random.normal(
        jax.random.PRNGKey(113), a.shape, jnp.float64), init)
    # Per robot: the global expert from every robot, a group expert only from its group's robots, routed experts where
    # selected (robots of both groups take part).
    for r, b in enumerate(batches):
        def robot_loss(prm):
            out, st = pol.apply(prm, *_split(b.observations, r), robot_id=r, mutable=[ROUTING],
                                rngs={"routing": jax.random.PRNGKey(114 + r)})
            return _toy_loss(out, b.observations, b.actions, 1.5, None, r), st
        (_, st), grad = jax.value_and_grad(robot_loss, has_aux=True)(params)
        sq = expert_sq(grad["params"]["core"], 8)
        assert sq[0] > 0, f"robot {r}: global expert"
        assert sq[1 + table[r]] > 0 and sq[1 + (1 - table[r])] == 0.0, f"robot {r}: group experts {sq[1:3]}"
        sel = np.asarray((st[ROUTING]["core"]["p"][0] > 0).any(axis=0))
        assert np.array_equal(sq[3:] > 0, sel), f"robot {r}: routed {sq[3:]} vs selected {sel}"

    a = dict(morse_blocks="core,decoder", morse_num_experts=8, morse_router_hidden=16, morse_router_inputs="both",
             morse_conflict_mode="none", morse_lambda=0.0, morse_eta=0.01, morse_conflict_norm="pairs",
             morse_unbiased_types=True, morse_zeta=1.0, morse_kappa=0.0, morse_threshold="fixed", morse_relevance=True,
             morse_evidence="type", morse_type_ema=0.9, morse_conflict_every=1, morse_warmup=0, morse_ramp=0,
             morse_router_only=False, morse_diag_frequency=1, morse_snapshot_frequency=0, morse_robot_ema=0.9,
             morse_mi_blocks="core,decoder", morse_mi_target="type", morse_mi_weighting="units", morse_mi_weight=0.5,
             morse_mi_weight_core=-1.0, morse_mi_balance_blocks="", morse_router_noise=True, morse_mi_core_groups="eg",
             morse_mi_cap=0.5, embodiment_distance_path=dist, morse_eg_groups=2)

    def updater(policy=cap, **over):
        u = MorseActorUpdater(SimpleNamespace(algorithm=SimpleNamespace(**{**a, **over})), policy, _split,
                              ["a", "b", "c"], None)
        u.loss_fn = _toy_loss
        return u

    try:
        updater(policy=cap.clone(core_group_of=(table[2], table[2], table[0])))
        raise AssertionError("group experts and EG-group MI with different groupings must be rejected")
    except ValueError:
        pass
    try:
        updater(morse_compat_route="compat", morse_compat_lambda_core=0.1, morse_compat_lambda_decoder=0.1)
        raise AssertionError("compatibility routing with shared core experts must be rejected")
    except ValueError:
        pass
    upd = updater()
    assert upd.router_experts == {"joint_encoder": 1, "foot_encoder": 1, "core": 5, "decoder": 4}
    state = TrainState.create(apply_fn=pol.apply, params=params, tx=optax.sgd(1e-2))
    out = upd.update(state, jnp.asarray(1.5), batches, jax.random.PRNGKey(115), 0)[2]
    for k in ("branch_norm_global", "branch_norm_group", "branch_norm_routed", "branch_norm_out", "task_grad_global",
              "task_grad_group", "task_grad_routed", "task_grad_group_g0", "task_grad_group_g1", "routed_eff_experts",
              "routed_usage_min", "routed_usage_max", "noise_scale", "mi", "h_expert", "mi_cap_factor", "loss_use"):
        assert np.isfinite(out[f"morse/core/{k}"]), k
    assert np.isfinite(out["morse/decoder/noise_scale"]) and np.isfinite(out["morse/update_s"])
    assert out["morse/core/task_grad_group_g0"] > 0 and out["morse/core/task_grad_group_g1"] > 0
    assert 1.0 - 1e-9 <= out["morse/core/routed_eff_experts"] <= 2.0 + 1e-9, "top-2 of the routed experts"
    assert out["morse/core/h_expert"] <= np.log(5) + 1e-9, "MI over the 5 routed experts"
    per_expert = out["morse/core/params_global"]
    assert per_expert > 0 and out["morse/core/params_group"] == 2 * per_expert and \
        out["morse/core/params_routed"] == 5 * per_expert, "parameters by expert category"
    for blk_name in ("core", "decoder"):                     # independent per-block caps on raw router gradients
        task, mi = out[f"morse/{blk_name}/router_grad_task"], out[f"morse/{blk_name}/router_grad_mi"]
        assert abs(out[f"morse/{blk_name}/mi_cap_factor"] - min(1.0, 0.5 * task / mi)) < 1e-9, blk_name

    # MI and balance: on the routed experts only, and only the router moves (no expert, no noise head).
    recs = []
    for r, b in enumerate(batches):
        _, st = pol.apply(params, *_split(b.observations, r), robot_id=r, mutable=[ROUTING],
                          rngs={"routing": jax.random.PRNGKey(116 + r)})
        recs.append({name: {k: val[0] for k, val in rec.items()} for name, rec in st[ROUTING].items()})
    recs = tuple(recs)
    router5 = Router(5, 16, "both")
    rp = params["params"]["core"]["router"]
    r_rob = [np.asarray(jax.nn.softmax(router5.apply({"params": rp}, rec["core"]["morph"], rec["core"]["state"])
                                       + rec["core"]["offset"], axis=-1).mean(0)) for rec in recs]
    groups = sorted(set(table))
    pi = np.array([np.mean([t == g for t in table]) for g in groups])
    r_grp = [np.mean([r_rob[i] for i, t in enumerate(table) if t == g], axis=0) for g in groups]
    ent = lambda q: -float(np.sum(q * np.log(q)))
    mi_ref = ent(sum(p_ * q for p_, q in zip(pi, r_grp))) - sum(p_ * ent(q) for p_, q in zip(pi, r_grp))
    mi_loss, mi_metrics = upd._mi_loss(params, recs)
    # (relative 1e-5: the EG group priors and averaging matrix are float32)
    assert abs(float(mi_metrics["morse/core/mi"]) - mi_ref) < 1e-5 * abs(mi_ref), (float(mi_metrics["morse/core/mi"]), mi_ref)
    g_mi = jax.grad(lambda prm: upd._mi_loss(prm, recs)[0])(params)
    router_moved = 0.0
    for path, leaf in jax.tree_util.tree_leaves_with_path(g_mi):
        if _is_router(path):
            router_moved += float(jnp.sum(jnp.abs(leaf)))
        else:
            assert float(jnp.max(jnp.abs(leaf))) == 0.0, f"MI gradient on {path}"
    assert router_moved > 0.0
    ri = {"core": (jnp.concatenate([rec["core"]["morph"] for rec in recs]),
                   jnp.concatenate([rec["core"]["state"] for rec in recs])),
          "decoder": tuple(jnp.concatenate([_units(rec["decoder"][k], 1) for rec in recs]) for k in ("morph", "state"))}
    bal, bal_metrics = upd._aux_loss(params, ri, None, jnp.zeros(()))
    want_use = itf.utilization_loss(jax.nn.softmax(router5.apply({"params": rp}, *ri["core"]), axis=-1))
    assert abs(float(bal_metrics["morse/core/loss_use"]) - float(want_use)) < 1e-12, "balance over the routed experts"
    g_bal = jax.grad(lambda prm: upd._aux_loss(prm, ri, None, jnp.zeros(()))[0])(params)
    for path, leaf in jax.tree_util.tree_leaves_with_path(g_bal):
        if not _is_router(path):
            assert float(jnp.max(jnp.abs(leaf))) == 0.0, f"balance gradient on {path}"

    # Clean evaluation is deterministic, also through the jitted policy of the training loop (robot index traced);
    # training noise changes the routing.
    obs0 = _split(batches[2].observations, 2)
    e1, e2 = pol.apply(params, *obs0, robot_id=2), pol.apply(params, *obs0, robot_id=2)
    assert np.array_equal(e1[0], e2[0]) and np.array_equal(e1[1], e2[1])
    jitted = jax.jit(pol.apply)
    for r, b in enumerate(batches):
        want = pol.apply(params, *_split(b.observations, r), robot_id=r)[0]
        assert _rel_err(jitted(params, *_split(b.observations, r), robot_id=r)[0], want) < 1e-12, f"jitted robot {r}"
    assert not np.allclose(pol.apply(params, *obs0, robot_id=2, rngs={"routing": jax.random.PRNGKey(117)})[0], e1[0])
    for bad_id in (None, 3):
        try:
            pol.apply(params, *obs0, robot_id=bad_id)
            raise AssertionError(f"robot_id={bad_id} must fail with group experts")
        except ValueError:
            pass
    # Parameter round trip.
    restored = serialization.from_bytes(jax.tree_util.tree_map(jnp.zeros_like, params), serialization.to_bytes(params))
    for r, b in enumerate(batches):
        assert np.array_equal(pol.apply(restored, *_split(b.observations, r), robot_id=r)[0],
                              pol.apply(params, *_split(b.observations, r), robot_id=r)[0])

    # Configurations: without the new keys the original architecture (same tree, same outputs); every mode's allocation;
    # invalid settings fail.
    def config(robots=("a", "b", "c"), **over):
        alg = dict(std_dev=1.0, softmax_temperature=1.0, softmax_temperature_min=0.015, stability_epsilon=1e-8,
                   policy_mean_abs_clip=10.0, policy_std_min_clip=1e-8, policy_std_max_clip=2.0, max_action=1.0,
                   morse_blocks="core,decoder", morse_num_experts=8, morse_num_experts_decoder=4, morse_topk_core=2,
                   morse_topk_decoder=2, morse_core_routing="body", morse_expert_hidden=0, morse_router_init=0.0,
                   morse_router_hidden=16, morse_router_inputs="both", morse_detach_router_inputs=False,
                   morse_width_mult_core=1.0, morse_width_mult_decoder=1.0, morse_capture_transitions_element=2,
                   morse_capture_transitions_body=2, morse_eval_topk=0, morse_router_noise=True,
                   morse_router_noise_floor=0.01, embodiment_distance_path=dist, morse_eg_groups=2)
        return SimpleNamespace(algorithm=SimpleNamespace(**{**alg, **over}),
                               environment=SimpleNamespace(train_robot_types=list(robots)))
    assert LEGACY_ARCHITECTURE["morse_core_shared"] == "none"
    legacy = get_morse_policy(config())
    assert legacy.core_shared == "none" and legacy.core_shared_counts() == (0, 0) and legacy.router_experts_of("core") == 8
    old = _policy(False, num_experts=8).clone(moe_blocks=("core", "decoder"), num_experts_decoder=4, topk_core=2,
                                              topk_decoder=2, core_routing="body", router_noise=True)
    old_params = old.init(jax.random.PRNGKey(118), *obs0)
    assert jax.tree_util.tree_structure(old_params) == jax.tree_util.tree_structure(legacy.init(jax.random.PRNGKey(118), *obs0))
    assert np.array_equal(legacy.apply(old_params, *obs0, robot_id=2)[0], old.apply(old_params, *obs0, robot_id=2)[0])
    hier = get_morse_policy(config(morse_core_shared="hierarchical"))
    assert hier.core_group_of == table and hier.core_shared_counts() == (1, 2)
    for mode, topk, counts in (("global", 3, (1, 0)), ("group", 3, (0, 2)), ("hierarchical", 2, (1, 2))):
        pm = get_morse_policy(config(morse_core_shared=mode, morse_topk_core=topk))
        assert pm.core_shared_counts() == counts and pm.router_experts_of("core") == 8 - sum(counts), mode
        prm = pm.init(jax.random.PRNGKey(119), *obs0)
        assert prm["params"]["core"]["router"]["morph_out"]["kernel"].shape[-1] == 8 - sum(counts), mode
        _, st = pm.apply(prm, *obs0, robot_id=2, mutable=[ROUTING])
        bn = np.asarray(st[ROUTING]["core"]["branch_norm"][0])
        assert (bn[:, 0] > 0).all() == (counts[0] > 0) and (bn[:, 0] == 0).all() == (counts[0] == 0), mode
        assert (bn[:, 1] > 0).all() == (counts[1] > 0) and (bn[:, 1] == 0).all() == (counts[1] == 0), mode
        assert bool(jnp.all((st[ROUTING]["core"]["p"][0] > 0).sum(-1) == topk)), mode
    for bad in (dict(morse_core_shared="hierarchical", morse_core_routing="joint"),
                dict(morse_core_shared="hierarchical", morse_compat_route="compat"),
                dict(morse_core_shared="hierarchical", morse_num_experts=4),
                dict(morse_core_shared="hierarchical", morse_core_shared_coef=0.5),
                dict(morse_core_shared="group", morse_eg_groups=1),
                dict(morse_core_shared="group", robots=("a", "b")),
                dict(morse_core_shared="bogus")):
        try:
            get_morse_policy(config(**bad))
            raise AssertionError(f"{bad} must be rejected")
        except ValueError:
            pass
    print("[ok] shared core experts: output == explicit weighted sum (global, own group, top-2 routed); group selection; "
          "task gradients by category; MI and balance on the routed experts and the router only; per-block MI caps; "
          "clean deterministic evaluation; round trip; original architecture without the new keys; invalid settings fail")


def test_variance_modes():
    import optax
    from jax.lax import stop_gradient
    from flax import serialization
    from cross_embodiment_offline_rl.algorithms.morse import variance as var
    R, B, J = len(_ROBOTS), 6, 32
    base = _policy(False).clone(n_robots=R, policy_std_max_clip=1e8)
    modes = {m: base.clone(variance_mode=m) for m in var.MODES}
    key = jax.random.PRNGKey(51)

    def inputs(r, k, constant=True):
        (jd, js, fd, fs, g), _ = _robot_inputs(k, B, *_ROBOTS[r])
        if constant:              # a joint's description is the same in every transition of its robot
            jd, fd = jnp.broadcast_to(jd[:1], jd.shape), jnp.broadcast_to(fd[:1], fd.shape)
        return jd, js, fd, fs, g
    x = [inputs(r, jax.random.fold_in(key, r)) for r in range(R)]
    act = [jax.random.normal(jax.random.fold_in(key, 10 + r), (B, _ROBOTS[r][0]), jnp.float64) for r in range(R)]
    w = [jnp.exp(jax.random.normal(jax.random.fold_in(key, 20 + r), (B, 1), jnp.float64)) for r in range(R)]
    p0 = modes["original"].init(jax.random.PRNGKey(52), *x[0])
    p0 = jax.tree_util.tree_map(lambda a: a.astype(jnp.float64) + 0.2 * jax.random.normal(jax.random.PRNGKey(53), a.shape, jnp.float64), p0)

    # 1. Same forward pass at the start: detach and protected share the parameters of the original; the private table
    #    built from the original head reproduces its log-stds (they are state-independent), and the means are equal.
    ref = [modes["original"].apply(p0, *x[r], robot_id=r) for r in range(R)]
    for m in ("detach", "protected"):
        for r in range(R):
            out = modes[m].apply(p0, *x[r], robot_id=r)
            assert all(bool(jnp.all(a == b)) for a, b in zip(out, ref[r])), m
    pp = {"params": {**p0["params"], "logstd_offset": jnp.zeros((R, J), jnp.float64)}}
    for r in range(R):
        out = modes["private"].apply(pp, *x[r], robot_id=r)
        assert all(bool(jnp.all(a == b)) for a, b in zip(out, ref[r])), "baseline + zero offset is the original policy"
    assert var.logstd_state_spread(modes["original"], p0["params"], [[x[r]] for r in range(R)]).max() < 1e-9
    assert var.logstd_state_spread(modes["original"], p0["params"], [[inputs(0, key, constant=False)]])[0] > 1e-3,         "the log-std follows the description, so a state-independent table would not reproduce it"

    init_p = modes["private"].init(jax.random.PRNGKey(54), *x[0])           # initialization has no robot index
    assert init_p["params"]["logstd_offset"].shape == (R, J) and "Dense_6" in init_p["params"]
    try:
        modes["private"].apply(init_p, *x[0])
        raise AssertionError("private log-stds need the robot index outside initialization")
    except ValueError:
        pass

    def parts(mode, params, r):
        """Forward pass and the vjp of its outputs, to separate the pathways of a loss exactly."""
        (mean, logstd), vjp = jax.vjp(lambda prm: modes[mode].apply({"params": prm}, *x[r], robot_id=r), params["params"])
        return mean, logstd, vjp
    gauss = lambda m, s, r: -(w[r] * (-0.5 * ((act[r] - m) / jnp.exp(s)) ** 2 - s)).mean()
    emb = lambda g: [g[k] for k in var.EMBEDDING]
    norm = lambda t: float(optax.global_norm(t))

    # 2. Detach: the embedding gets the mean pathway's gradient only; the head's gradient is unchanged.
    r = 1
    mean, logstd, vjp_o = parts("original", p0, r)
    gm, gs = jax.grad(gauss, argnums=(0, 1))(mean, logstd, r)
    full = vjp_o((gm, gs))[0]
    via_mean_o = vjp_o((gm, jnp.zeros_like(gs)))[0]
    mean_d, logstd_d, vjp_d = parts("detach", p0, r)
    gm_d, gs_d = jax.grad(gauss, argnums=(0, 1))(mean_d, logstd_d, r)
    det = vjp_d((gm_d, gs_d))[0]
    via_std_d = vjp_d((jnp.zeros_like(gm_d), gs_d))[0]
    assert all(np.allclose(a, b, rtol=1e-10, atol=1e-14) for a, b in zip(jax.tree_util.tree_leaves(emb(det)), jax.tree_util.tree_leaves(emb(via_mean_o))))
    assert norm(emb(full)) > 0 and norm(jax.tree_util.tree_map(jnp.subtract, emb(full), emb(via_mean_o))) > 1e-8, \
        "without detaching, the variance pathway adds to the embedding's gradient"
    assert norm(emb(via_std_d)) == 0.0 and norm(via_std_d["Dense_6"]) > 0, "no variance gradient reaches the embedding"
    assert np.allclose(jax.tree_util.tree_leaves(det["Dense_6"])[0], jax.tree_util.tree_leaves(full["Dense_6"])[0], rtol=1e-10)

    # 3. Private: the offsets receive the gradient of the robot's own valid joints only; the variance pathway reaches no
    #    other parameter (the shared head and embedding are frozen); the offsets do not change the means.
    for r in range(R):
        mean, logstd, vjp_p = parts("private", pp, r)
        gm, gs = jax.grad(gauss, argnums=(0, 1))(mean, logstd, r)
        gt = vjp_p((jnp.zeros_like(gm), gs))[0]
        assert norm({k: v for k, v in gt.items() if k != "logstd_offset"}) == 0.0
        assert norm(vjp_p((gm, jnp.zeros_like(gs)))[0]["logstd_offset"]) == 0.0, "the mean pathway never reaches the offsets"
        mask = np.zeros((R, J), bool)
        mask[r, :_ROBOTS[r][0]] = True
        g_table = np.asarray(gt["logstd_offset"])
        assert np.all(g_table[~mask] == 0.0) and np.all(g_table[mask] != 0.0)
    shifted = {"params": {**pp["params"], "logstd_offset": pp["params"]["logstd_offset"] + 0.3}}
    assert all(bool(jnp.all(modes["private"].apply(shifted, *x[r], robot_id=r)[0] == modes["private"].apply(pp, *x[r], robot_id=r)[0])) for r in range(R))

    # 4. Protected: the loss against its formula; the variance loss reaches the head only, the mean loss never the
    #    head; the mean gradient is w (mu - a) / (B J) whatever the learned variance, and the head's gradient is the
    #    Gaussian loss's.
    r = 2
    mean, logstd, vjp_q = parts("protected", p0, r)
    err2 = (act[r] - mean) ** 2
    explicit = 0.5 * (w[r] * err2).mean() + (w[r] * (logstd + err2 / (2 * jnp.exp(2 * logstd)))).mean()
    assert np.allclose(var.protected_loss(w[r], mean, logstd, act[r]), explicit, rtol=1e-12)
    gm_p, gs_p = jax.grad(var.protected_loss, argnums=(1, 2))(w[r], mean, logstd, act[r])
    assert np.allclose(gm_p, w[r] * (mean - act[r]) / mean.size, rtol=1e-10, atol=1e-14)
    gm_g, gs_g = jax.grad(gauss, argnums=(0, 1))(mean, logstd, r)
    assert np.allclose(gs_p, gs_g, rtol=1e-10, atol=1e-14), "the variance term is the Gaussian loss's gradient in log-std"
    by_mean = vjp_q((gm_p, jnp.zeros_like(gs_p)))[0]
    by_std = vjp_q((jnp.zeros_like(gm_p), gs_p))[0]
    assert norm(by_mean["Dense_6"]) == 0.0, "the mean loss does not reach the log-std head"
    assert norm({k: v for k, v in by_std.items() if k != "Dense_6"}) == 0.0, "the variance loss reaches the head only"
    p_shift = {"params": {**p0["params"], "Dense_6": {**p0["params"]["Dense_6"], "bias": p0["params"]["Dense_6"]["bias"] + 1.0}}}
    mean2, logstd2, vjp_q2 = parts("protected", p_shift, r)
    gm_p2, _ = jax.grad(var.protected_loss, argnums=(1, 2))(w[r], mean2, logstd2, act[r])
    by_mean2 = vjp_q2((gm_p2, jnp.zeros_like(gs_p)))[0]
    assert all(np.allclose(a, b, rtol=1e-12, atol=1e-16) for a, b in zip(jax.tree_util.tree_leaves(by_mean), jax.tree_util.tree_leaves(by_mean2)))
    mo, so, vjp_o2 = parts("original", p_shift, r)
    go = jax.grad(gauss, argnums=(0, 1))(mo, so, r)
    by_mean_o2 = vjp_o2((go[0], jnp.zeros_like(so)))[0]
    by_mean_o = parts("original", p0, r)[2]((gm_g, jnp.zeros_like(gs_g)))[0]
    assert norm(jax.tree_util.tree_map(jnp.subtract, by_mean_o2, by_mean_o)) > 1e-6, "the Gaussian mean gradient depends on the variance"

    # 5. Grouped clipping: each group is clipped on its own, an unclipped group is untouched, the optimizer state has
    #    the structure of the plain clip's (a checkpoint's state restores into it).
    toy = {"params": {"Dense_6": {"kernel": jnp.ones((3,)), "bias": jnp.ones((1,))}, "Dense_4": {"kernel": jnp.ones((4,))}}}
    grads = {"params": {"Dense_6": {"kernel": 10.0 * jnp.ones((3,)), "bias": 10.0 * jnp.ones((1,))},
                        "Dense_4": {"kernel": 0.1 * jnp.ones((4,))}}}
    tx = var.grouped_clip_by_global_norm(1.0, var.is_std_head)
    upd, _ = tx.update(grads, tx.init(toy))
    assert np.isclose(norm(upd["params"]["Dense_6"]), 1.0, rtol=1e-6)
    assert np.allclose(upd["params"]["Dense_4"]["kernel"], grads["params"]["Dense_4"]["kernel"]), "the other group is not rescaled"
    plain, _ = optax.clip_by_global_norm(1.0).update(grads, optax.clip_by_global_norm(1.0).init(toy))
    assert not np.allclose(plain["params"]["Dense_4"]["kernel"], grads["params"]["Dense_4"]["kernel"]), \
        "the global clip rescales the other group (the interference removed)"
    small = jax.tree_util.tree_map(lambda g: 0.01 * g, grads)
    assert all(np.allclose(a, b) for a, b in zip(jax.tree_util.tree_leaves(tx.update(small, tx.init(toy))[0]), jax.tree_util.tree_leaves(small)))
    sched = optax.cosine_decay_schedule(-1e-3, 100)
    state_new = serialization.to_state_dict(optax.chain(tx, optax.scale_by_adam(), optax.scale_by_schedule(sched)).init(toy))
    state_old = serialization.to_state_dict(optax.chain(optax.clip_by_global_norm(1.0), optax.scale_by_adam(),
                                                        optax.scale_by_schedule(sched)).init(toy))
    assert jax.tree_util.tree_structure(state_new) == jax.tree_util.tree_structure(state_old)
    print("[ok] variance modes: equal starting forward passes (detach, protected, private baseline + zero offset); detach cuts the variance gradient into the embedding; private log-stds isolated per robot-joint; the "
          "protected losses touch disjoint parameters, the mean gradient is independent of the variance; grouped clipping "
          "keeps the optimizer state structure")


def test_frozen_core():
    import optax
    from flax import serialization
    from flax.training.train_state import TrainState
    from cross_embodiment_offline_rl.algorithms.morse import variance as var
    B, J = 6, 32
    pol = _policy(False).clone(n_robots=len(_ROBOTS), policy_std_max_clip=1e8)
    (jd, js, fd, fs, g), _ = _robot_inputs(jax.random.PRNGKey(61), B, *_ROBOTS[0])
    p0 = pol.init(jax.random.PRNGKey(62), jd, js, fd, fs, g)
    p0 = jax.tree_util.tree_map(lambda a: a.astype(jnp.float64) + 0.2 * jax.random.normal(jax.random.PRNGKey(63), a.shape, jnp.float64), p0)
    sched = optax.cosine_decay_schedule(-1e-2, 400000)        # the restored count sits at 3/4 of it, as in the runs
    norm = lambda t: float(optax.global_norm(t))

    def loss(prm):
        mean, logstd = pol.apply(prm, jd, js, fd, fs, g, robot_id=0)
        return jnp.sum(mean ** 2) + jnp.sum(logstd ** 2)

    def restored_state(tx, zero_core_moments):
        """A train state whose optimizer holds a checkpoint's momentum (noise in every first moment) and a step count."""
        state = TrainState.create(apply_fn=pol.apply, params=p0, tx=tx)
        adam = state.opt_state[1]
        mu = jax.tree_util.tree_map(lambda m: 0.5 * jax.random.normal(jax.random.PRNGKey(64), m.shape, m.dtype), adam.mu)
        st = (state.opt_state[0], adam._replace(mu=mu, count=jnp.asarray(300000)), state.opt_state[2]._replace(count=jnp.asarray(300000)))
        if zero_core_moments:
            st = var.zero_first_moments(st, lambda names: "core" in names)
        return state.replace(opt_state=st)

    # 1. The core receives a gradient, and the input encoders receive theirs through it (nothing is stopped): the joint
    #    encoder reaches the actor only through the core, so its gradient is zero only if the core path were cut.
    grads = jax.grad(loss)(p0)
    assert norm(grads["params"]["core"]) > 0 and norm(grads["params"]["joint_encoder"]) > 0

    # 2. Frozen clip: the core's gradient is excluded from the norm and from the update; the trainable gradient is clipped
    #    by its own norm. The optimizer state has the plain chain's structure.
    max_norm = 0.01
    clip = var.frozen_clip_by_global_norm(max_norm, var.is_core)
    clipped, _ = clip.update(grads, clip.init(p0))
    trainable = {k: v for k, v in grads["params"].items() if k != "core"}
    n_t = norm(trainable)
    assert n_t > max_norm and norm(grads["params"]["core"]) > 0
    assert norm(clipped["params"]["core"]) == 0.0
    assert np.isclose(norm({k: v for k, v in clipped["params"].items() if k != "core"}), max_norm, rtol=1e-9)
    ratio = max_norm / n_t
    assert all(np.allclose(a, ratio * b, rtol=1e-9) for a, b in zip(jax.tree_util.tree_leaves({k: v for k, v in clipped["params"].items() if k != "core"}),
                                                                     jax.tree_util.tree_leaves(trainable)))
    plain_ratio = max_norm / norm(grads)
    assert plain_ratio < ratio, "the plain clip would have scaled the trainable gradient down by the core's share too"
    chain_new = optax.chain(clip, optax.scale_by_adam(), optax.scale_by_schedule(sched))
    chain_old = optax.chain(optax.clip_by_global_norm(max_norm), optax.scale_by_adam(), optax.scale_by_schedule(sched))
    assert jax.tree_util.tree_structure(serialization.to_state_dict(chain_new.init(p0))) == \
        jax.tree_util.tree_structure(serialization.to_state_dict(chain_old.init(p0)))

    # 3. With a restored momentum, a zero gradient alone still moves a frozen parameter; zeroing its first moments too
    #    leaves it bitwise unchanged for many steps while the trainable parameters (encoders) keep moving.
    for zero_moments, core_moves in ((False, True), (True, False)):
        state = restored_state(chain_new, zero_moments)
        before = state.params
        for _ in range(5):
            state = state.apply_gradients(grads=jax.grad(loss)(state.params))
        same_core = all(bool(jnp.all(a == b)) for a, b in zip(jax.tree_util.tree_leaves(before["params"]["core"]),
                                                              jax.tree_util.tree_leaves(state.params["params"]["core"])))
        assert same_core == (not core_moves), (zero_moments, same_core)
        assert norm(jax.tree_util.tree_map(jnp.subtract, state.params["params"]["joint_encoder"], before["params"]["joint_encoder"])) > 0
    assert int(state.opt_state[1].count) == 300005 and int(state.opt_state[2].count) == 300005, "step counters continue"
    assert state.opt_state[1].mu["params"]["joint_encoder"]["dense_0"]["bias"].shape and \
        norm(state.opt_state[1].mu["params"]["core"]) == 0.0
    print("[ok] frozen core: gradients reach the core and, through it, the encoders; the clip sees the trainable gradient only; "
          "with zeroed first moments the core stays bitwise unchanged while the rest trains (and moves without that); "
          "optimizer state structure and counters unchanged")


if __name__ == "__main__":
    test_frozen_core()
    test_variance_modes()
    test_shared_core_experts()
    test_dataset_selection_and_checksum()
    test_routing_noise_ties_and_robot_fixed_routing()
    test_compat_scores_match_explicit_gradients()
    test_mi_cap_eg_groups_noise_keys_and_compat_step()
    test_body_core_balance_mi()
    test_joint_core_topk_and_mi()
    test_coordination_episodes_and_dump()
    test_checkpoint_analysis()
    test_decoder_experts_width_and_topk()
    test_mi_robot_and_joint_groups()
    test_joint_pcgrad_control()
    test_crossfit_type_sums_and_evidence()
    test_branch_from_shared_checkpoint()
    test_mutual_information_loss_matches_definition()
    test_signed_evidence_thresholds_by_noise()
    test_fixed_routing_identical_experts_and_calibration()
    test_probe_joint_gradients_and_lookahead()
    test_block_gram_matches_explicit_per_unit_gradients()
    test_policy_gram_sums_to_block_gradient_norm_across_robots()
    test_aux_objectives_only_touch_routers()
    test_shuffle_preserves_conflict_values()
    test_single_expert_is_shared_block()
    test_single_expert_policy_reproduces_urma()
    test_robot_level_statistics()
    test_type_level_evidence()
    test_independent_type_sums_remove_shared_sample_covariance()
    test_conflict_loss_scales_with_conflict()
    test_fused_update_matches_stepwise_reference()
    test_actor_substeps_match_chunked_reference()
    test_prefetcher_gathers_dataset_rows()
    print("all MORSE tests passed")
