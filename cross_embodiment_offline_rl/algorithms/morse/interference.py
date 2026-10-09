"""Interference evidence, routing losses, and diagnostics for MORSE.

Notation follows the paper. For block m and expert k, the contribution of routed
unit u to the expert's complete parameter vector is g_u = p_{u,k} * ghat_u, and
    ||grad_{phi_k} L||^2 = sum_u ||g_u||^2 + 2 sum_{u<v} <g_u, g_v>.
``gram`` computes G[k, u, v] = <g_u, g_v> exactly from layer inputs and backward
signals (Eq. gram), without materializing per-unit parameter gradients.
"""
import jax
import jax.numpy as jnp


HIGHEST = jax.lax.Precision.HIGHEST


def gram(layers):
    """Exact Gram matrices of full-expert gradient contributions, shape (K, U, U).

    ``layers`` is a list of ``(kind, X, D)`` for the trainable layers of the
    expert, where ``D`` (U, K, d_out) holds backward signals (they include the
    routing weight of each unit):
      * ``kind == "dense"``: ``X`` is the layer input, (U, d_in) for the first
        layer of a block (shared by all experts) or (U, K, d_in) otherwise;
        weight and bias contribute (D_u . D_v)(X_u . X_v + 1).
      * ``kind == "ln"``: ``X`` is the normalized input (U, K, d); scale and bias
        contribute (D_u * X_u) . (D_v * X_v) + D_u . D_v.
    """
    total = None
    for kind, x, d in layers:
        dd = jnp.einsum("uko,vko->kuv", d, d, precision=HIGHEST)
        if kind == "dense":
            if x.ndim == 2:
                xx = jnp.einsum("ui,vi->uv", x, x, precision=HIGHEST)[None]
            else:
                xx = jnp.einsum("uki,vki->kuv", x, x, precision=HIGHEST)
            term = dd * (xx + 1.0)
        else:
            dx = d * x
            term = jnp.einsum("uko,vko->kuv", dx, dx, precision=HIGHEST) + dd
        total = term if total is None else total + term
    return total


def _masked_std(values, mask):
    n = jnp.maximum(mask.sum(), 1)
    mean = (values * mask).sum() / n
    var = (jnp.square(values - mean) * mask).sum() / n
    return mean, jnp.sqrt(var)


def conflict_evidence(g, p, zeta, kappa_fixed, calibrated, relevance, rel_eps=1e-10):
    """Alignments, threshold, and conflict coefficients (Eqs. gradient_cosine, conflict_coefficient).

    g: (K, U, U) Gram matrices; p: (U, K) routing probabilities.
    Returns a dict with cos (K, U, U), q (K, U, U; zero outside u<v), the pair mask,
    the threshold kappa, and the relevance weights omega (K, U).
    """
    n_experts, n_units, _ = g.shape
    sq_norm = jnp.diagonal(g, axis1=1, axis2=2)                       # ||g_u||^2, (K, U)
    valid = sq_norm > rel_eps * jnp.max(sq_norm, axis=1, keepdims=True)
    norm = jnp.sqrt(jnp.maximum(sq_norm, 0.0))
    upper = jnp.triu(jnp.ones((n_units, n_units), dtype=bool), k=1)
    pair = upper[None] & valid[:, :, None] & valid[:, None, :]
    denom = norm[:, :, None] * norm[:, None, :]
    cos = jnp.where(pair, g / jnp.where(pair, denom, 1.0), 0.0)

    cos_mean, cos_std = _masked_std(cos, pair)
    kappa = zeta * cos_std if calibrated else jnp.asarray(kappa_fixed, cos.dtype)

    # Relevance: ||ghat_u|| = ||g_u|| / p_{u,k}, capped relative to the mean over units.
    ghat_norm = jnp.where(valid, norm / jnp.maximum(p.T, 1e-12), 0.0)
    mean_norm = (ghat_norm * valid).sum(axis=1, keepdims=True) / jnp.maximum(valid.sum(axis=1, keepdims=True), 1)
    if relevance:
        omega = jnp.where(valid, jnp.minimum(1.0, ghat_norm / jnp.maximum(mean_norm, 1e-20)), 0.0)
    else:
        omega = valid.astype(g.dtype)

    q = jax.nn.relu(-cos - kappa) * omega[:, :, None] * omega[:, None, :] * pair
    return {"cos": cos, "q": q, "pair": pair, "valid": valid, "kappa": kappa,
            "cos_mean": cos_mean, "cos_std": cos_std, "omega": omega}


def shuffle_units(q, pair, key):
    """Control: keep the conflict values of each expert but assign them to random unit pairs."""
    n_units = q.shape[-1]
    perm = jax.random.permutation(key, n_units)
    full = q + jnp.swapaxes(q, 1, 2)
    full = full[:, perm][:, :, perm]
    upper = jnp.triu(jnp.ones((n_units, n_units), dtype=bool), k=1)
    return full * upper[None] * pair


def expert_mean_evidence(g, omega, valid, zeta, kappa_fixed, calibrated):
    """STGC-style control: alignment of each unit with the expert's aggregate update.

    Returns q_u (K, U) = [-cos(ghat_u, sum_v g_v) - kappa]_+ * omega_u.
    """
    sq_norm = jnp.diagonal(g, axis1=1, axis2=2)
    total = jnp.maximum(g.sum(axis=(1, 2)), 1e-30)                   # ||sum_v g_v||^2, (K,)
    row = g.sum(axis=2)                                               # <g_u, sum_v g_v>, (K, U)
    align = jnp.where(valid, row / (jnp.sqrt(jnp.maximum(sq_norm, 1e-30)) * jnp.sqrt(total)[:, None]), 0.0)
    _, std = _masked_std(align, valid)
    kappa = zeta * std if calibrated else jnp.asarray(kappa_fixed, g.dtype)
    return jax.nn.relu(-align - kappa) * omega * valid, align


def conflict_loss(p_tilde, q, norm="pairs", eps=1e-12):
    """Co-routing of conflicting pairs (Eq. conflict_loss): sum_k sum_{u<v} q_kuv p_uk p_vk.

    norm="pairs" divides by the number of unit pairs, so the loss (and the pressure on the router) is
    proportional to the amount of conflict and vanishes with it; uniform routing gives mean(q_k)/K summed
    over experts. norm="mass" divides by sum q instead: uniform routing then gives 1/K^2 however weak
    the conflicts are.
    """
    pk = p_tilde.T
    num = jnp.einsum("ku,kuv,kv->", pk, q, pk, precision=HIGHEST)
    if norm == "mass":
        return num / (q.sum() + eps)
    n_units = q.shape[-1]
    return num / (n_units * (n_units - 1) / 2)


def expert_mean_loss(p_tilde, q_unit, norm="pairs", eps=1e-12):
    num = jnp.sum(q_unit * p_tilde.T)
    return num / (q_unit.sum() + eps) if norm == "mass" else num / q_unit.shape[-1]


def neg_mutual_information(p, type_onehot, eps=1e-12, weights=None):
    """-I(A;E) = sum_a pi_a H(r_a) - H(rbar) of unit group A and expert E (Mod-Squad's MI, per group).

    p: (U, K) routing probabilities of the units; type_onehot: (U, T) group of every unit (a joint type, or a
    robot). r_a is the mean routing of the units of group a, rbar = sum_a pi_a r_a. pi_a is the group's share of
    the units, or the given prior ``weights`` (T,) normalized over the groups present. Returns (-I, H(E|A), H(E)).
    """
    counts = type_onehot.sum(axis=0)
    present = counts > 0
    r = jnp.matmul(type_onehot.T, p, precision=HIGHEST) / jnp.maximum(counts, 1.0)[:, None]   # (T, K)
    if weights is None:
        pi = counts / jnp.maximum(counts.sum(), 1.0)
    else:
        w = jnp.where(present, jnp.asarray(weights, p.dtype), 0.0)
        pi = w / jnp.maximum(w.sum(), 1e-30)
    return group_neg_mutual_information(r, pi, eps)


def group_neg_mutual_information(r, pi, eps=1e-12):
    """-I(A;E) = sum_a pi_a H(r_a) - H(rbar) from the mean routing r (T, K) of every group and the group prior pi (T,)
    (summing to 1, 0 for absent groups), rbar = sum_a pi_a r_a. Returns (-I, H(E|A), H(E))."""
    h_r = -jnp.sum(r * jnp.log(r + eps), axis=-1)
    h_cond = jnp.sum(pi * h_r)
    rbar = jnp.matmul(pi, r, precision=HIGHEST)
    h_marg = -jnp.sum(rbar * jnp.log(rbar + eps))
    return h_cond - h_marg, h_cond, h_marg


def signed_type_evidence(s_main, s_even, s_odd, z, help_weight, rel_eps=1e-10):
    """Signed compatibility coefficients between unit types (conflict mode ``signed``).

    s_*: (K, T, T) accumulated type sums over independent unit pairs: all updates, even updates, odd updates.
    The noise std of the cosines is estimated per expert from the even/odd difference: the mean of two
    independent halves has noise std(cos_even - cos_odd) / 2. Cosines within z of that std carry no
    evidence. q = [-cos - kappa]_+ - help_weight [cos - kappa]_+ : positive for harmful pairs (routing overlap
    is penalized), negative for helpful pairs (overlap is rewarded).
    """
    main = type_evidence(s_main, 1.0, 0.0, False, rel_eps)
    even = type_evidence(s_even, 1.0, 0.0, False, rel_eps)
    odd = type_evidence(s_odd, 1.0, 0.0, False, rel_eps)
    pair = main["pair"] & even["pair"] & odd["pair"]
    n_pair = jnp.maximum(pair.sum(axis=(1, 2)), 1)
    diff = jnp.where(pair, even["cos"] - odd["cos"], 0.0)
    sigma = jnp.sqrt(jnp.sum(diff ** 2, axis=(1, 2)) / n_pair) / 2.0          # (K,)
    kappa = (z * sigma)[:, None, None]
    cos = jnp.where(pair, main["cos"], 0.0)
    harm = jax.nn.relu(-cos - kappa)
    help_ = jax.nn.relu(cos - kappa)
    q = (harm - help_weight * help_) * pair
    ce, co = jnp.where(pair, even["cos"], 0.0), jnp.where(pair, odd["cos"], 0.0)
    me, mo = ce.sum((1, 2)) / n_pair, co.sum((1, 2)) / n_pair
    cov = jnp.sum(jnp.where(pair, (ce - me[:, None, None]) * (co - mo[:, None, None]), 0.0), axis=(1, 2))
    var_e = jnp.sum(jnp.where(pair, (ce - me[:, None, None]) ** 2, 0.0), axis=(1, 2))
    var_o = jnp.sum(jnp.where(pair, (co - mo[:, None, None]) ** 2, 0.0), axis=(1, 2))
    reliability = cov / jnp.sqrt(jnp.maximum(var_e * var_o, 1e-30))
    return {"q": q, "pair": pair, "cos": cos, "sigma": sigma, "reliability": reliability,
            "harm_frac": (harm > 0).sum() / jnp.maximum(pair.sum(), 1),
            "help_frac": (help_ > 0).sum() / jnp.maximum(pair.sum(), 1)}


def type_grad_sums(layers):
    """Per-type gradient contributions of one robot's units to every expert's parameters, (K, E, P).

    ``layers`` is a list of ``(kind, X, D)`` as for ``gram`` but per robot, with the transition axis first and
    the element axis second: X (n, E, d_in) for the first layer of a block, otherwise (n, E, K, d_in); D (n, E, K,
    d_out). The contribution of element e (one unit type) is the sum over the n transitions. Parameters are
    flattened layer by layer as kernel, bias (dense) or scale, bias (LayerNorm).
    """
    parts = []
    for kind, x, d in layers:
        per_type = lambda a: jnp.swapaxes(a.sum(axis=0), 0, 1)                        # (E, K, f) -> (K, E, f)
        if kind == "dense":
            spec = "nei,neko->keio" if x.ndim == 3 else "neki,neko->keio"
            w = jnp.einsum(spec, x, d, precision=HIGHEST)
            parts += [w.reshape(w.shape[:2] + (-1,)), per_type(d)]
        else:
            parts += [per_type(d * x), per_type(d)]
    return jnp.concatenate(parts, axis=-1)


def crossfit_type_evidence(streams, z, help_weight, rel_eps=1e-10):
    """Signed compatibility from per-type gradient sums of four independent data streams (estimator ``crossfit``).

    streams: (4, K, T, P), each stream accumulated from different updates, hence different data. Products of
    different streams carry no sampling noise in expectation (as the decoder probe's cross-subset Grams):
    G12 = S1 S2^T and G34 = S3 S4^T are two independent estimates of the type Gram, whose agreement is the
    reliability and whose difference gives the noise std; the evidence is Gfull = (S1 + S2)(S3 + S4)^T. Cosines
    are normalized by the cross-fitted squared norms on the diagonal; a type whose cross-fitted squared norm is
    not positive has no signal. q = [-cos - kappa]_+ - help_weight [cos - kappa]_+ with kappa = z * noise std.
    """
    def cross(a, b):
        m = jnp.einsum("kap,kbp->kab", a, b, precision=HIGHEST)
        return 0.5 * (m + jnp.swapaxes(m, 1, 2))

    def cosine(g):
        diag = jnp.diagonal(g, axis1=1, axis2=2)
        valid = diag > rel_eps * jnp.maximum(jnp.max(diag, axis=1, keepdims=True), 0.0)
        norm = jnp.sqrt(jnp.where(valid, diag, 1.0))
        return jnp.clip(g / (norm[:, :, None] * norm[:, None, :]), -1.0, 1.0), valid

    c12, v12 = cosine(cross(streams[0], streams[1]))
    c34, v34 = cosine(cross(streams[2], streams[3]))
    cos, vf = cosine(cross(streams[0] + streams[1], streams[2] + streams[3]))
    n_types = cos.shape[-1]
    valid = v12 & v34 & vf
    upper = jnp.triu(jnp.ones((n_types, n_types), dtype=bool), k=1)
    pair = upper[None] & valid[:, :, None] & valid[:, None, :]
    n_pair = jnp.maximum(pair.sum(axis=(1, 2)), 1)
    diff = jnp.where(pair, c12 - c34, 0.0)
    sigma = jnp.sqrt(jnp.sum(diff ** 2, axis=(1, 2)) / n_pair) / 2.0          # (K,)
    kappa = (z * sigma)[:, None, None]
    cos = jnp.where(pair, cos, 0.0)
    harm = jax.nn.relu(-cos - kappa)
    help_ = jax.nn.relu(cos - kappa)
    q = (harm - help_weight * help_) * pair
    ca, cb = jnp.where(pair, c12, 0.0), jnp.where(pair, c34, 0.0)
    ma, mb = ca.sum((1, 2)) / n_pair, cb.sum((1, 2)) / n_pair
    da, db = jnp.where(pair, ca - ma[:, None, None], 0.0), jnp.where(pair, cb - mb[:, None, None], 0.0)
    reliability = (da * db).sum((1, 2)) / jnp.sqrt(jnp.maximum((da ** 2).sum((1, 2)) * (db ** 2).sum((1, 2)), 1e-30))
    return {"q": q, "pair": pair, "cos": cos, "sigma": sigma, "reliability": reliability,
            "harm_frac": (harm > 0).sum() / jnp.maximum(pair.sum(), 1),
            "help_frac": (help_ > 0).sum() / jnp.maximum(pair.sum(), 1),
            "valid_frac": valid.mean()}


def pcgrad(g, key, eps=1e-30):
    """PCGrad (Yu et al., 2020) over unit types. g: (T, P) per-type gradients. Every type's gradient is projected,
    in its own random order of the other types, onto the normal plane of each original gradient it conflicts with
    (negative inner product). Returns the projected gradients and the fraction of projections made."""
    n = g.shape[0]
    norms2 = jnp.sum(g * g, axis=1)
    order = jax.vmap(lambda k: jax.random.permutation(k, n))(jax.random.split(key, n))   # row a: the order of type a
    idx = jnp.arange(n)

    def step(carry, s):
        gt, count = carry
        b = order[:, s]
        gb = g[b]
        dot = jnp.sum(gt * gb, axis=1)
        hit = (dot < 0) & (b != idx) & (norms2[b] > eps)
        coef = jnp.where(hit, dot / jnp.maximum(norms2[b], eps), 0.0)
        return (gt - coef[:, None] * gb, count + hit.sum(dtype=count.dtype)), None

    (out, count), _ = jax.lax.scan(step, (g, jnp.zeros((), jnp.int32)), jnp.arange(n))
    return out, count / max(n * (n - 1), 1)


def utilization_loss(p_tilde, eps=1e-12):
    """KL(pbar || uniform) of aggregate routing probabilities (Eq. utilization)."""
    n_experts = p_tilde.shape[-1]
    pbar = p_tilde.mean(axis=0)
    return jnp.sum(pbar * jnp.log(n_experts * pbar + eps))


def block_diagnostics(g, p, ev, same_transition, same_robot, element_level, eps=1e-12):
    """Scalar diagnostics of interference and routing for one block, from the captured units.

    ``same_transition`` / ``same_robot`` are (U, U) boolean masks; for whole-body
    blocks every unit is its own transition, so ``element_level`` is False.
    Robot-level statistics use full-batch per-robot gradients instead (``robot_level_stats``).
    """
    pair, cos, kappa = ev["pair"], ev["cos"], ev["kappa"]
    n_pair = jnp.maximum(pair.sum(), 1)
    conflict = (cos < -kappa) & pair

    pk = p.T
    w = pk[:, :, None] * pk[:, None, :] * pair
    # ||sum_u g_u||^2 = S + C+ - C-  (Eq. interference_decomposition, summed over experts)
    self_term = jnp.sum(jnp.diagonal(g, axis1=1, axis2=2))
    constructive = 2.0 * jnp.sum(jax.nn.relu(g) * pair)
    destructive = 2.0 * jnp.sum(jax.nn.relu(-g) * pair)

    entropy = -jnp.sum(p * jnp.log(p + eps), axis=-1)
    pbar = p.mean(axis=0)

    out = {
        "conflict_frac": conflict.sum() / n_pair,
        "conflict_mass": jnp.sum(w * conflict) / jnp.maximum(w.sum(), eps),
        # Share of the potential first-order progress (self + constructive terms) cancelled by
        # destructive pairs; in [0, 1] because ||sum_u g_u||^2 >= 0.
        "cancelled_fraction": destructive / jnp.maximum(self_term + constructive, eps),
        "destructive_share": destructive / jnp.maximum(constructive + destructive, eps),
        "cos_mean": ev["cos_mean"],
        "cos_std": ev["cos_std"],
        "kappa": kappa,
        "q_pos_frac": (ev["q"] > 0).sum() / n_pair,
        "conf_coroute": conflict_loss(p, ev["q"]),
        "router_entropy": entropy.mean(),
        "eff_experts": jnp.exp(entropy).mean(),
        "usage_max": pbar.max(),
        "usage_min": pbar.min(),
        "usage_kl": utilization_loss(p),
    }
    categories = {"cross_robot": ~same_robot, "within_robot": same_robot & ~same_transition}
    if element_level:
        categories["within_transition"] = same_transition
    for name, mask in categories.items():
        m = pair & mask[None]
        out[f"conflict_frac_{name}"] = (conflict & m).sum() / jnp.maximum(m.sum(), 1)
    return out


def robot_level_stats(m, eps=1e-30):
    """Robot-level interference from a Gram matrix of per-robot gradients, M[r, s] = <g_r, g_s>.

    With full-batch per-robot gradients of a block this is the per-embodiment gradient analysis of
    Abe et al. restricted to the block. Parameters of different experts are disjoint, so robots
    routed to different experts do not interact. ``cancelled_fraction`` = C- / (S + C+), where
    ||sum_r g_r||^2 = S + C+ - C-, is the share of the robots' potential progress that
    interference between robots cancels.
    """
    n = m.shape[0]
    d = jnp.sqrt(jnp.maximum(jnp.diagonal(m), eps))
    cos = m / (d[:, None] * d[None, :])
    iu = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    n_pairs = jnp.maximum(iu.sum(), 1)
    self_term = jnp.trace(m)
    constructive = 2.0 * jnp.sum(jax.nn.relu(m) * iu)
    destructive = 2.0 * jnp.sum(jax.nn.relu(-m) * iu)
    return {"cos": cos,
            "cos_mean": jnp.sum(cos * iu) / n_pairs,
            "neg_frac": jnp.sum((cos < 0) & iu) / n_pairs,
            "cancelled_fraction": destructive / jnp.maximum(self_term + constructive, eps)}


def block_snapshot(g, p, ev, layers, robot_onehot, key, n_pair_samples=20000):
    """Arrays saved periodically for the appendix analyses (robot-level matrices are added by the updater)."""
    # Full precision: default float32 matmuls on A100 use TF32 (10-bit mantissa), which would make
    # counts above 2048 and routing averages inexact.
    conflict = (ev["cos"] < -ev["kappa"]) & ev["pair"]
    pair_counts = jnp.einsum("ur,kuv,vs->rs", robot_onehot, ev["pair"].astype(g.dtype), robot_onehot, precision=HIGHEST)
    conflict_counts = jnp.einsum("ur,kuv,vs->rs", robot_onehot, conflict.astype(g.dtype), robot_onehot, precision=HIGHEST)
    counts = jnp.maximum(robot_onehot.sum(axis=0), 1)
    snap = {
        "robot_conflict_frac": (conflict_counts + conflict_counts.T) / jnp.maximum(pair_counts + pair_counts.T, 1),
        "robot_usage": jnp.matmul(robot_onehot.T, p, precision=HIGHEST) / counts[:, None],
        "unit_p": p,
    }
    # Pair-level factors of the first expert layer (Eq. linear_factorization):
    # input overlap cos(x_u, x_v) and demand agreement cos(delta_u, delta_v).
    _, x, d = layers[0]
    n_units = x.shape[0]
    ku, kv = jax.random.split(key)
    u = jax.random.randint(ku, (n_pair_samples,), 0, n_units)
    v = jax.random.randint(kv, (n_pair_samples,), 0, n_units)
    keep = u != v
    xu, xv = (x[u], x[v]) if x.ndim == 2 else (x[u, 0], x[v, 0])
    in_cos = jnp.sum(xu * xv, -1) / (jnp.linalg.norm(xu, axis=-1) * jnp.linalg.norm(xv, axis=-1) + 1e-12)
    du, dv = d[u], d[v]
    out_cos = jnp.sum(du * dv, -1) / (jnp.linalg.norm(du, axis=-1) * jnp.linalg.norm(dv, axis=-1) + 1e-12)
    lo, hi = jnp.minimum(u, v), jnp.maximum(u, v)
    full_cos = ev["cos"][:, lo, hi].T
    snap.update({"pair_u": u, "pair_v": v, "pair_keep": keep, "pair_input_cos": in_cos,
                 "pair_delta_cos": out_cos, "pair_cos": full_cos})
    return snap


# ------------------------------------------------------------------ type-level evidence
# Per-sample alignments mix systematic interference with sample-specific demands
# (e.g. the sign of an action residual). Aggregating contributions over a unit
# type -- element e of robot tau for element-level blocks, robot tau for whole-body
# blocks -- and over updates keeps only persistent interference between types.

def type_sums(g, type_onehot):
    """S[k, a, b] = sum_{u in a, v in b} G[k, u, v]: inner products of per-type summed contributions."""
    return jnp.einsum("ua,kuv,vb->kab", type_onehot, g, type_onehot, precision=HIGHEST)


def type_sums_split(g, type_onehot, cross_transition):
    """Type sums over independent unit pairs, and over all unit pairs (same batch).

    Units of one transition (e.g. two joints of a robot) share its sample, so the expectation of their
    product includes the covariance of their noise, which averaging over updates keeps. Pairs of units
    from different transitions are independent: off the diagonal, the first result sums only those and
    estimates the product of the types' expected contributions. The diagonal keeps all pairs (norms).
    ``cross_transition`` is the (U, U) 0/1 mask of unit pairs from different transitions.
    """
    same = type_sums(g, type_onehot)
    cross = type_sums(g * cross_transition, type_onehot)
    return jnp.where(jnp.eye(same.shape[-1], dtype=bool), same, cross), same


def type_evidence(s, zeta, kappa_fixed, calibrated, rel_eps=1e-10):
    """Conflict coefficients between unit types from accumulated type sums (K, T, T)."""
    n_types = s.shape[-1]
    diag = jnp.diagonal(s, axis1=1, axis2=2)
    valid = diag > rel_eps * jnp.max(diag, axis=1, keepdims=True)
    norm = jnp.sqrt(jnp.maximum(diag, 0.0))
    upper = jnp.triu(jnp.ones((n_types, n_types), dtype=bool), k=1)
    pair = upper[None] & valid[:, :, None] & valid[:, None, :]
    denom = norm[:, :, None] * norm[:, None, :]
    cos = jnp.where(pair, s / jnp.where(pair, denom, 1.0), 0.0)
    cos_mean, cos_std = _masked_std(cos, pair)
    kappa = zeta * cos_std if calibrated else jnp.asarray(kappa_fixed, s.dtype)
    q = jax.nn.relu(-cos - kappa) * pair
    return {"cos": cos, "q": q, "pair": pair, "kappa": kappa, "cos_mean": cos_mean, "cos_std": cos_std}


def expand_type_q(q_type, type_ids):
    """Assign each unit pair the conflict coefficient of its two types (zero within a type)."""
    full = q_type + jnp.swapaxes(q_type, 1, 2)
    q = full[:, type_ids][:, :, type_ids]
    n_units = type_ids.shape[0]
    return q * jnp.triu(jnp.ones((n_units, n_units), dtype=bool), k=1)[None]


# ------------------------------------------------------------------ compatibility routing (2026-10-06)

def _layer_terms(kind, x, d):
    """Per-layer pieces of a scored block, one robot: x and d carry (transition, element[, expert], features) axes, as
    ``gram`` but with a transition and an element axis. Returns the units' squared norms (n, J, K), the
    within-transition Gram (n, K, J, J), and a function giving (the layer's reference sums from weights w (n, J, K),
    the units' inner products with given reference sums)."""
    dd = jnp.einsum("njko,nlko->nkjl", d, d, precision=HIGHEST)
    if kind == "dense":
        shared = x.ndim == 3                                                   # (n, J, d_in): first layer of a block
        x2 = jnp.sum(x * x, axis=-1)
        norm = jnp.sum(d * d, axis=-1) * ((x2[..., None] if shared else x2) + 1.0)
        xx = (jnp.einsum("nji,nli->njl", x, x, precision=HIGHEST)[:, None] if shared
              else jnp.einsum("njki,nlki->nkjl", x, x, precision=HIGHEST))
        gram_t = dd * (xx + 1.0)

        def reference(w):
            wd = w[..., None] * d
            kern = (jnp.einsum("njko,nji->koi", wd, x, precision=HIGHEST) if shared
                    else jnp.einsum("njko,njki->koi", wd, x, precision=HIGHEST))
            return kern, wd.sum(axis=(0, 1))

        def inner(ref):
            kern, bias = ref
            ip = (jnp.einsum("njko,koi,nji->njk", d, kern, x, precision=HIGHEST) if shared
                  else jnp.einsum("njko,koi,njki->njk", d, kern, x, precision=HIGHEST))
            return ip + jnp.einsum("njko,ko->njk", d, bias, precision=HIGHEST)
    else:
        dx = d * x
        norm = jnp.sum(dx * dx, axis=-1) + jnp.sum(d * d, axis=-1)
        gram_t = jnp.einsum("njko,nlko->nkjl", dx, dx, precision=HIGHEST) + dd

        def reference(w):
            return (w[..., None] * dx).sum(axis=(0, 1)), (w[..., None] * d).sum(axis=(0, 1))

        def inner(ref):
            scale, bias = ref
            return (jnp.einsum("njko,ko->njk", dx, scale, precision=HIGHEST)
                    + jnp.einsum("njko,ko->njk", d, bias, precision=HIGHEST))
    return norm, gram_t, reference, inner


def compat_scores(robots, min_support, rel_eps=1e-12):
    """Cosines between every routing unit's candidate gradient on every expert and that expert's actual update from the
    other transitions of the scored sample, in two halves (even and odd transitions of every robot).

    ``robots`` holds, per robot, ``(layers, w)``: ``layers`` a list of ``(kind, X, D)`` with a transition and an element
    axis (J = 1 at a body-routed block), where ``D`` (n, J, K, d_out) are the units' unweighted backward signals (the
    candidate update directions, also for experts a unit does not use), and ``w`` (n, J, K) the units' actual routing
    weights, so that a unit's actual contribution to expert k is w_k * ghat_k. A unit's own transition (all its elements)
    is left out of the reference half it belongs to. A cosine is valid when the unit's gradient and the reference are
    nonzero and the reference half has at least ``min_support`` transitions routed to the expert.

    Returns per robot (cos, valid), both (2, n, J, K), and the support of every half, (2, K)."""
    parts, refs = [], None                     # refs[h][l]: reference sums of layer l over half h, all robots
    for layers, w in robots:
        n = w.shape[0]
        in_half = [((jnp.arange(n) % 2) == h)[:, None, None].astype(w.dtype) for h in (0, 1)]
        norm, gram_t, fns, robot_refs = 0.0, 0.0, [], [[], []]
        for kind, x, d in layers:
            l_norm, l_gram, reference, inner = _layer_terms(kind, x, d)
            norm, gram_t = norm + l_norm, gram_t + l_gram
            fns.append(inner)
            for h in (0, 1):
                robot_refs[h].append(reference(w * in_half[h]))
        refs = robot_refs if refs is None else jax.tree_util.tree_map(jnp.add, refs, robot_refs)
        used = (w.sum(axis=1) > 0).astype(w.dtype)                             # (n, K): transitions routed to k
        parts.append((n, w, norm, gram_t, fns, used))
    ref_norm = jnp.stack([sum(jnp.sum(jnp.square(a), axis=tuple(range(1, a.ndim))) for ref in refs[h] for a in ref)
                          for h in (0, 1)])                                    # ||R_h||^2, (2, K)
    support = jnp.stack([sum(jnp.sum(u * ((jnp.arange(n) % 2) == h)[:, None], axis=0) for n, _, _, _, _, u in parts)
                         for h in (0, 1)])                                     # (2, K)
    out = []
    for n, w, norm, gram_t, fns, used in parts:
        half = jnp.arange(n) % 2
        own_ip = jnp.einsum("nkjl,nlk->njk", gram_t, w, precision=HIGHEST)      # <ghat_u, C_t>
        own_sq = jnp.einsum("njk,nkjl,nlk->nk", w, gram_t, w, precision=HIGHEST)  # ||C_t||^2
        cos, valid = [], []
        for h in (0, 1):
            ip = sum(f(refs[h][i]) for i, f in enumerate(fns))                 # <ghat_u, R_h>, (n, J, K)
            ref_own = jnp.einsum("njk,njk->nk", w, ip, precision=HIGHEST)      # <R_h, C_t>
            mine = (half == h).astype(w.dtype)
            ip_adj = ip - mine[:, None, None] * own_ip
            sq_adj = ref_norm[h][None] - mine[:, None] * (2.0 * ref_own - own_sq)  # ||R_h - C_t||^2, (n, K)
            sup = support[h][None] - mine[:, None] * used                      # (n, K)
            ok = (norm > 0.0) & (sq_adj[:, None, :] > rel_eps * jnp.max(ref_norm[h])) \
                & (sup[:, None, :] >= min_support)
            c = ip_adj / jnp.sqrt(jnp.maximum(norm, 1e-30) * jnp.maximum(sq_adj, 1e-30)[:, None, :])
            cos.append(jnp.where(ok, c, 0.0))
            valid.append(ok)
        out.append((jnp.stack(cos), jnp.stack(valid)))
    return out, support


def compat_labels(cos, valid, tau):
    """Non-conflicting (both halves >= -tau) and conflicting (both < -tau) unit-expert pairs; the rest is unknown."""
    both = valid[0] & valid[1]
    nonconf = both & (cos[0] >= -tau) & (cos[1] >= -tau)
    conf = both & (cos[0] < -tau) & (cos[1] < -tau)
    return nonconf, conf


def shuffle_within(labels, strata, key):
    """Label rows of the units permuted among the units of the same stratum (a robot, or a joint of a robot)."""
    order = jnp.lexsort((jax.random.uniform(key, strata.shape), strata))     # strata, random within
    base = jnp.lexsort((jnp.arange(strata.shape[0]), strata))                 # strata, original order within
    return tuple(jnp.zeros_like(a).at[base].set(a[order]) for a in labels)


def compat_loss(p, nonconf, conf, mode="compat", eps=1e-12):
    """Compatibility routing objective, the mean over units with at least one non-conflicting expert.

    p: (U, K) router probabilities; nonconf, conf: (U, K) labels (unknown experts are in neither and are left out:
    p is renormalized over the assessed experts, p~). With A the non-conflicting experts,
      compat (and shuffled): KL(U_A || p~) = -log p~(A) + KL(U_A || p~(.|A));
      conflict_only: -log p~(A);   uniform: KL(U_assessed || p~), the labels ignored.
    Returns (loss, parts) with parts the means of -log p~(A), KL(U_A || p~(.|A)), and the usable share of units."""
    assessed = (nonconf | conf).astype(p.dtype)
    a = nonconf.astype(p.dtype)
    n_a = a.sum(axis=-1)
    use = (n_a > 0).astype(p.dtype)
    p_ass = p * assessed
    p_t = p_ass / jnp.maximum(p_ass.sum(axis=-1, keepdims=True), eps)
    mass = jnp.sum(p_t * a, axis=-1)
    neg_log_mass = -jnp.log(jnp.maximum(mass, eps))
    q = a / jnp.maximum(n_a, 1.0)[..., None]
    p_in = p_t / jnp.maximum(mass, eps)[..., None]
    kl_within = jnp.sum(jnp.where(a > 0, q * (jnp.log(jnp.maximum(q, eps)) - jnp.log(jnp.maximum(p_in, eps))), 0.0), -1)
    if mode == "conflict_only":
        per_unit = neg_log_mass
    elif mode == "uniform":
        qs = assessed / jnp.maximum(assessed.sum(axis=-1), 1.0)[..., None]
        per_unit = jnp.sum(jnp.where(assessed > 0, qs * (jnp.log(jnp.maximum(qs, eps))
                                                          - jnp.log(jnp.maximum(p_t, eps))), 0.0), -1)
    else:
        per_unit = neg_log_mass + kl_within
    n_use = jnp.maximum(use.sum(), 1.0)
    mean = lambda v: jnp.sum(v * use) / n_use
    return mean(per_unit), {"neg_log_mass": mean(neg_log_mass), "kl_within": mean(kl_within),
                            "usable_frac": use.mean()}
