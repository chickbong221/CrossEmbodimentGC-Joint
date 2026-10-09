"""V8 policy: URMA with the URMAv2 layer widths and Mod-Squad-style mixture-of-experts layers.

The dense layers are those of URMA with the 2 x 256 joint encoder and the 5-layer core (512-512-256-256-128 with
WeightNorm). Every second layer is a mixture-of-experts layer, as in Mod-Squad's alternating placement:

    joint encoder:  dense 3->256 (LN, ELU) -> MoE(256) -> dense 256->4 (ELU)
    core:           dense 400->512 (LN, ELU) -> MoE(512) -> dense 512->256 (ELU) -> MoE(256) -> dense 256->128
    decoder:        one MoE bank; every expert owns its description MLP (23->w->128) and action MLP (260->w->1)
    log-std:        a separate shared branch with its own description embedding (23->128->128->1)

A feed-forward MoE layer is pre-LN with a residual connection, y = x + sum_k w_k E_k(LN(x)), each expert a
feed-forward block d -> h -> d with h = width * d / 2 (an expert of width 1 has the parameters of the dense d -> d
layer it replaces). The decoder bank has no residual (its output is the action mean of a joint).

Two layer forms (morse_v8_direction):
  * MI directions (DeepSeekMoE): one shared expert with weight 1 plus the top-3 of 15 routed experts, weighted by
    their softmax scores over the routed experts (not renormalized).
  * compatibility direction (Mod-Squad): the top-4 of 16 routed experts, weights renormalized over the four.

Every MoE layer records, in the collection ``V8_ROUTING``, the routing probabilities its auxiliary losses use (the
softmax of the noisy logits with the noise held fixed; gradients reach the router and, through the router inputs,
every upstream layer) and the mixing weights (no gradient). With ``record_score`` it also records its experts' inputs
and the mixing weights of every expert (``V8_SCORE``), and with ``record_delta`` it adds a zero perturbation to its
mixture output (``V8_DELTA``) whose gradient is the backward signal of the layer: the compatibility scoring pass.
"""
import json
import os
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, lecun_normal, orthogonal, zeros

from cross_embodiment_offline_rl.algorithms.morse.moe import Router, keep_topk, routing_noise


MOE_LAYERS = ("enc_moe", "core_moe1", "core_moe2", "dec_moe")
JOINT_LAYERS = ("enc_moe", "dec_moe")        # one routing unit per joint of a transition
BODY_LAYERS = ("core_moe1", "core_moe2")     # one routing unit per transition
DIRECTIONS = ("mi_joint", "mi_robot", "compat", "none")

V8_ROUTING = "v8_routing"
V8_SCORE = "v8_score"
V8_DELTA = "v8_delta"
CAP_IN = "v8_cap_in"
CAP_DELTA = "v8_cap_delta"


def expert_init(init_fn):
    """Initialize every expert slice of a (K, ...) parameter independently with ``init_fn``."""
    def init(key, shape, dtype=jnp.float32):
        keys = jax.random.split(key, shape[0])
        return jax.vmap(lambda k: init_fn(k, shape[1:], dtype))(keys)
    return init


class XDense(nn.Module):
    """K parallel dense layers. ``shared_input``: the input (..., d_in) feeds every expert; otherwise (..., K, d_in).
    Output (..., K, features). With ``capture`` the input is recorded (``CAP_IN``) and a zero perturbation is added to
    the output (``CAP_DELTA``), whose gradient is the layer's backward signal."""
    num_experts: int
    features: int
    kernel_init: object = lecun_normal()
    shared_input: bool = False
    capture: bool = False

    @nn.compact
    def __call__(self, x):
        kernel = self.param("kernel", expert_init(self.kernel_init), (self.num_experts, x.shape[-1], self.features))
        bias = self.param("bias", zeros, (self.num_experts, self.features))
        eq = "...i,kio->...ko" if self.shared_input else "...ki,kio->...ko"
        z = jnp.einsum(eq, x, kernel) + bias
        if self.capture:
            self.sow(CAP_IN, "x", x)
            z = self.perturb("z", z, collection=CAP_DELTA)
        return z


class XLayerNorm(nn.Module):
    """LayerNorm over the features with an affine transform per expert; with ``capture`` as XDense (normalized input)."""
    num_experts: int
    epsilon: float = 1e-6
    capture: bool = False

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", constant(1.0), (self.num_experts, x.shape[-1]))
        bias = self.param("bias", constant(0.0), (self.num_experts, x.shape[-1]))
        mean = x.mean(axis=-1, keepdims=True)
        var = jnp.square(x - mean).mean(axis=-1, keepdims=True)
        xhat = (x - mean) * jax.lax.rsqrt(var + self.epsilon)
        y = xhat * scale + bias
        if self.capture:
            self.sow(CAP_IN, "xhat", xhat)
            y = self.perturb("y", y, collection=CAP_DELTA)
        return y


class FFNExperts(nn.Module):
    """K feed-forward experts d -> hidden -> d on a shared input (..., d); output (..., K, d)."""
    num_experts: int
    hidden: int
    features: int
    capture: bool = False

    # Trainable layers in forward order (kind, module name): the exact per-unit gradient algebra of the scoring pass.
    LAYERS = (("dense", "dense_0"), ("dense", "dense_1"))

    @nn.compact
    def __call__(self, u):
        h = XDense(self.num_experts, self.hidden, orthogonal(np.sqrt(2)), shared_input=True, capture=self.capture,
                   name="dense_0")(u)
        h = nn.elu(h)
        return XDense(self.num_experts, self.features, orthogonal(1.0), capture=self.capture, name="dense_1")(h)


class DecoderExperts(nn.Module):
    """K decoder experts, each with its own description MLP (23 -> w -> 128, as URMA's shared embedding) and action
    MLP (core output, joint latent, own description embedding -> w -> 1). Inputs: the joint description (..., J, 23)
    and the context (..., J, 132); output (..., J, K, 1)."""
    num_experts: int
    hidden: int
    capture: bool = False

    LAYERS = (("dense", "desc_dense_0"), ("ln", "desc_ln_0"), ("dense", "desc_dense_1"),
              ("dense", "act_dense_0"), ("ln", "act_ln_0"), ("dense", "act_dense_1"))

    @nn.compact
    def __call__(self, jd, ctx):
        k, c = self.num_experts, self.capture
        e = XDense(k, self.hidden, orthogonal(np.sqrt(2)), shared_input=True, capture=c, name="desc_dense_0")(jd)
        e = nn.elu(XLayerNorm(k, capture=c, name="desc_ln_0")(e))
        e = XDense(k, 128, orthogonal(np.sqrt(2)), capture=c, name="desc_dense_1")(e)              # (..., J, K, 128)
        a = jnp.concatenate([jnp.broadcast_to(ctx[..., None, :], ctx.shape[:-1] + (k, ctx.shape[-1])), e], axis=-1)
        a = XDense(k, self.hidden, orthogonal(np.sqrt(2)), capture=c, name="act_dense_0")(a)
        a = nn.elu(XLayerNorm(k, capture=c, name="act_ln_0")(a))
        return XDense(k, 1, orthogonal(0.01), capture=c, name="act_dense_1")(a)


class MoELayer(nn.Module):
    """One routed layer: router (+ learned noise in training), mixing weights, experts, mixture.

    ``kind`` "ffn": pre-LN residual feed-forward layer, inputs (x,), router state = LN(x) unless ``state`` is given.
    ``kind`` "decoder": inputs (jd, ctx), no residual. ``gate`` "deepseek": shared experts with weight 1 plus the top-k
    routed softmax scores (not renormalized); "renorm": top-k routed weights renormalized."""
    kind: str
    n_shared: int
    n_routed: int
    topk: int
    gate: str
    hidden: int
    features: int = 0
    router_hidden: int = 64
    router_inputs: str = "both"
    router_init: float = 0.0
    router_noise: bool = False
    noise_floor: float = 0.01
    record_score: bool = False
    record_delta: bool = False

    @property
    def num_experts(self):
        return self.n_shared + self.n_routed

    def experts(self, capture=False):
        if self.kind == "ffn":
            return FFNExperts(self.num_experts, self.hidden, self.features, capture=capture, name="experts")
        return DecoderExperts(self.num_experts, self.hidden, capture=capture, name="experts")

    def mixing_weights(self, p):
        """Weights of the routed experts from the routing probabilities p (..., n_routed)."""
        if self.gate == "renorm":
            return keep_topk(p, self.topk)
        mask = jax.nn.one_hot(jax.lax.top_k(p, self.topk)[1], p.shape[-1], dtype=p.dtype).sum(axis=-2)
        return p * mask

    @nn.compact
    def __call__(self, inputs, morph, state=None):
        if self.kind == "ffn":
            (x,) = inputs
            xn = nn.LayerNorm(name="ln")(x)
            expert_inputs = (xn,)
            if state is None:
                state = xn
        else:
            expert_inputs = inputs
        logits = Router(self.n_routed, self.router_hidden, self.router_inputs, self.router_init, name="router")(morph, state)
        noise = jnp.zeros_like(logits)
        sigma = None
        if self.router_noise:
            raw = Router(self.n_routed, self.router_hidden, self.router_inputs, 0.0, name="router_noise")(morph, state)
            if self.has_rng("routing"):
                sigma = jax.nn.softplus(raw) + self.noise_floor
                flat = logits.reshape((logits.shape[0], -1))
                eps = routing_noise(self.make_rng("routing"), flat.shape, logits.dtype).reshape(logits.shape)
                noise = sigma * eps
        # Mixing: the noisy logits (the task trains the noise head). Auxiliary losses: the same probabilities with the
        # noise held fixed, so they reach the router and its inputs but not the noise head.
        p_mix = jax.nn.softmax(logits + noise, axis=-1)
        p_aux = jax.nn.softmax(logits + jax.lax.stop_gradient(noise), axis=-1)
        w_routed = self.mixing_weights(p_mix)
        w = jnp.concatenate([jnp.ones(w_routed.shape[:-1] + (self.n_shared,), w_routed.dtype), w_routed], axis=-1) \
            if self.n_shared else w_routed
        outs = self.experts()(*expert_inputs)                                   # (..., K, out)
        y = jnp.einsum("...k,...ko->...o", w, outs)
        if not self.is_initializing():
            self.sow(V8_ROUTING, "p", p_aux)
            self.sow(V8_ROUTING, "w", jax.lax.stop_gradient(w_routed))
            if sigma is not None:
                self.sow(V8_ROUTING, "noise_scale", jax.lax.stop_gradient(sigma.mean(axis=-1)))
            if self.record_score:
                for i, u in enumerate(expert_inputs):
                    self.sow(V8_SCORE, f"u{i}", jax.lax.stop_gradient(u))
                self.sow(V8_SCORE, "w", jax.lax.stop_gradient(w))
        if self.record_delta:
            y = self.perturb("y", y, collection=V8_DELTA)
        return x + y if self.kind == "ffn" else y


class WNDense(nn.Module):
    """Dense layer with WeightNorm as URMAv2's core (moe.ExpertDense with weight_norm, one expert): w = g v / ||v||,
    g starting at ||v||."""
    features: int
    kernel_init: object = orthogonal(np.sqrt(2))

    @nn.compact
    def __call__(self, x):
        kernel = self.param("kernel", self.kernel_init, (x.shape[-1], self.features))
        norm = jnp.sqrt(jnp.sum(jnp.square(kernel), axis=0) + 1e-12)
        scale = self.param("wn_scale", lambda key, shape: norm, (self.features,))
        bias = self.param("bias", zeros, (self.features,))
        return x @ (kernel * (scale / norm)[None, :]) + bias


class V8Policy(nn.Module):
    std_dev: float
    softmax_temperature: float
    softmax_temperature_min: float
    stability_epsilon: float
    policy_mean_abs_clip: float
    policy_std_min_clip: float
    policy_std_max_clip: float
    n_shared: int = 1
    n_routed: int = 15
    topk: int = 3
    gate: str = "deepseek"
    width: float = 0.5
    router_hidden: int = 64
    router_inputs: str = "both"
    router_init: float = 0.0
    router_noise: bool = False
    noise_floor: float = 0.01
    record_score: bool = False
    record_delta: bool = False

    def moe(self, name, kind, features=0):
        hidden = int(round(self.width * (features / 2 if kind == "ffn" else 128)))
        return MoELayer(kind=kind, n_shared=self.n_shared, n_routed=self.n_routed, topk=self.topk, gate=self.gate,
                        hidden=hidden, features=features, router_hidden=self.router_hidden,
                        router_inputs=self.router_inputs, router_init=self.router_init, router_noise=self.router_noise,
                        noise_floor=self.noise_floor, record_score=self.record_score, record_delta=self.record_delta,
                        name=name)

    def _aggregate(self, name, description, latent_state, slots):
        """Descriptor-conditioned aggregation of URMA (shared, unchanged)."""
        eps = self.stability_epsilon
        log_temperature = self.param(f"{name}_log_softmax_temperature",
                                     constant(jnp.log(self.softmax_temperature - self.softmax_temperature_min)), (1,))
        mask = nn.Dense(slots, name=f"{name}_agg_0")(description)
        mask = nn.LayerNorm(name=f"{name}_agg_ln")(mask)
        mask = nn.elu(mask)
        mask = nn.Dense(slots, name=f"{name}_agg_1")(mask)
        mask = jnp.clip(nn.tanh(mask), -1.0 + eps, 1.0 - eps)
        e_x = jnp.exp(mask / (jnp.exp(log_temperature) + self.softmax_temperature_min))
        mask = e_x / (e_x.sum(axis=-1, keepdims=True) + eps)
        mask = jnp.repeat(jnp.expand_dims(mask, axis=-1), latent_state.shape[-1], axis=-1)
        masked = mask * jnp.expand_dims(latent_state, axis=-2)
        masked = jnp.reshape(masked, masked.shape[:-2] + (masked.shape[-2] * masked.shape[-1],))
        return jnp.sum(masked, axis=-2)

    @nn.compact
    def __call__(self, dynamic_joint_description, dynamic_joint_state, dynamic_foot_description, dynamic_foot_state,
                 general_state, robot_id=None):
        jd, js = dynamic_joint_description, dynamic_joint_state
        fd, fs = dynamic_foot_description, dynamic_foot_state
        n_joints, n_feet = js.shape[-2], fs.shape[-2]
        batch_shape = jd.shape[:-2]
        pooled = jnp.concatenate([jd.mean(axis=-2), fd.mean(axis=-2),
                                  jnp.full(batch_shape + (1,), n_joints / 32.0, jd.dtype),
                                  jnp.full(batch_shape + (1,), n_feet / 8.0, jd.dtype)], axis=-1)
        joint_morph = jnp.concatenate([jd, jnp.broadcast_to(pooled[..., None, :], batch_shape + (n_joints, pooled.shape[-1]))],
                                      axis=-1)

        # Joint encoder: dense 3 -> 256 (LN, ELU), MoE(256), dense 256 -> 4 (ELU).
        h = nn.Dense(256, kernel_init=lecun_normal(), name="enc_dense_0")(js)
        h = nn.elu(nn.LayerNorm(name="enc_ln_0")(h))
        h = self.moe("enc_moe", "ffn", 256)((h,), joint_morph, js)
        z_joint = nn.elu(nn.Dense(4, kernel_init=lecun_normal(), name="enc_dense_2")(h))
        z_foot = nn.elu(nn.Dense(4, kernel_init=lecun_normal(), name="foot_dense_0")(fs))

        joint_latent = self._aggregate("joint", jd, z_joint, 64)
        foot_latent = self._aggregate("foot", fd, z_foot, 32)
        combined = jnp.concatenate([joint_latent, foot_latent, general_state], axis=-1)

        # Core: dense 400 -> 512 (LN, ELU), MoE(512), dense 512 -> 256 (ELU), MoE(256), dense 256 -> 128.
        c = nn.elu(nn.LayerNorm(name="core_ln_0")(WNDense(512, name="core_dense_0")(combined)))
        c = self.moe("core_moe1", "ffn", 512)((c,), pooled)
        c = nn.elu(WNDense(256, name="core_dense_2")(c))
        c = self.moe("core_moe2", "ffn", 256)((c,), pooled)
        z = WNDense(128, name="core_dense_4")(c)

        # Decoder bank: per joint, context = core output and the joint's latent (no gradient into the encoder, as URMA).
        z_b = jnp.broadcast_to(z[..., None, :], batch_shape + (n_joints, z.shape[-1]))
        ctx = jnp.concatenate([z_b, jax.lax.stop_gradient(z_joint)], axis=-1)
        mean = self.moe("dec_moe", "decoder")((jd, ctx), joint_morph, ctx).squeeze(-1)
        mean = jnp.clip(mean, -self.policy_mean_abs_clip, self.policy_mean_abs_clip)

        # Separate log-std branch with its own description embedding.
        s = nn.Dense(128, name="std_dense_0")(jd)
        s = nn.elu(nn.LayerNorm(name="std_ln_0")(s))
        s = nn.Dense(128, name="std_dense_1")(s)
        logstd = nn.Dense(1, kernel_init=orthogonal(0.1), bias_init=constant(np.log(self.std_dev)),
                          name="std_head")(s).squeeze(-1)
        logstd = jnp.clip(logstd, np.log(self.policy_std_min_clip), np.log(self.policy_std_max_clip))
        return mean, logstd


def direction_layer_form(direction, n_experts=16):
    """(n_shared, n_routed, topk, gate) of every MoE layer for a direction."""
    if direction in ("mi_joint", "mi_robot"):
        return 1, n_experts - 1, 3, "deepseek"
    if direction in ("compat", "none"):
        return 0, n_experts, 4, "renorm"
    raise ValueError(f"Unknown morse_v8_direction {direction}; choose from {DIRECTIONS}")


def get_v8_policy(config):
    a = config.algorithm
    direction = str(getattr(a, "morse_v8_direction", "none"))
    n_shared, n_routed, topk, gate = direction_layer_form(direction, int(getattr(a, "morse_v8_experts", 16)))
    width = float(getattr(a, "morse_v8_width", 0.5))
    if not 0.0 < width <= 1.0:
        raise ValueError(f"morse_v8_width {width} must be in (0, 1]")
    return V8Policy(
        std_dev=a.std_dev, softmax_temperature=a.softmax_temperature, softmax_temperature_min=a.softmax_temperature_min,
        stability_epsilon=a.stability_epsilon, policy_mean_abs_clip=a.policy_mean_abs_clip,
        policy_std_min_clip=a.policy_std_min_clip, policy_std_max_clip=a.policy_std_max_clip,
        n_shared=n_shared, n_routed=n_routed, topk=topk, gate=gate, width=width,
        router_hidden=int(a.morse_router_hidden), router_inputs=a.morse_router_inputs,
        router_init=float(a.morse_router_init), router_noise=bool(a.morse_router_noise),
        noise_floor=float(a.morse_router_noise_floor))


def count_v8_params(params, n_active):
    """Parameters of the policy: total, active (the shared parts plus n_active experts of every MoE layer; routers
    counted separately), and per MoE layer."""
    out = {"total": 0, "routers": 0, "active": 0}
    for name, sub in params["params"].items():
        for path, leaf in jax.tree_util.tree_leaves_with_path(sub):
            keys = [getattr(k, "key", None) for k in path]
            n = int(leaf.size)
            out["total"] += n
            if name in MOE_LAYERS and keys[0] in ("router", "router_noise"):
                out["routers"] += n
            elif name in MOE_LAYERS and keys[0] == "experts":
                out[f"{name}_experts"] = out.get(f"{name}_experts", 0) + n
                out["active"] += n // leaf.shape[0] * n_active
            else:
                out["active"] += n
    return out


# ------------------------------------------------------------------------------------------ joint groups (J direction)

# Joint-local descriptor features (indices into the 23-dimensional joint description, identical for every robot
# environment): rotation axis (3:6, torso frame), number of direct child joints (6), nominal position (7), torque
# limit (8), velocity limit (9), damping (10), armature (11), stiffness (12), friction loss (13), range (14:16).
# Excluded: torso-relative position (0:3), the robot-wide gains (16:19), mass (19) and dimensions (20:23).
# Mirror-invariant forms: |axis|, |nominal|, range width and |range centre|. Torque and velocity limits are relative to
# the robot's largest (strong vs weak joint of this robot, without the robot's size).
GROUP_FEATURES = ("abs_axis_x", "abs_axis_y", "abs_axis_z", "children", "abs_nominal", "torque_rel", "velocity_rel",
                  "damping", "armature", "stiffness", "friction", "range_width", "abs_range_centre")


def joint_features(desc):
    """(J, 23) joint descriptions of one robot -> (J, 13) mirror-invariant joint-local features. The descriptions store
    torque and velocity limits as limit / (1000 / 2) - 1 and limit / (35 / 2) - 1; they are converted back to limits
    and divided by the robot's largest."""
    d = np.asarray(desc, np.float64)
    lo, hi = d[..., 14], d[..., 15]
    torque, velocity = (d[..., 8] + 1.0) * 500.0, (d[..., 9] + 1.0) * 17.5
    rel = lambda v: v / max(float(np.max(np.abs(v))), 1e-12)
    return np.stack([np.abs(d[..., 3]), np.abs(d[..., 4]), np.abs(d[..., 5]), d[..., 6], np.abs(d[..., 7]),
                     rel(torque), rel(velocity), d[..., 10], d[..., 11], d[..., 12], d[..., 13],
                     np.abs(hi - lo), np.abs(0.5 * (hi + lo))], axis=-1)


def kmeans(x, k, seed=0, n_init=50, iters=300):
    """Lloyd's k-means with k-means++ starts; the best of n_init runs (deterministic for a seed)."""
    rs = np.random.RandomState(seed)
    best = None
    for _ in range(n_init):
        centres = [x[rs.randint(len(x))]]
        for _ in range(1, k):
            d2 = np.min(((x[:, None] - np.array(centres)[None]) ** 2).sum(-1), axis=1)
            centres.append(x[rs.choice(len(x), p=d2 / d2.sum())] if d2.sum() > 0 else x[rs.randint(len(x))])
        c = np.array(centres)
        for _ in range(iters):
            lab = np.argmin(((x[:, None] - c[None]) ** 2).sum(-1), axis=1)
            new = np.array([x[lab == j].mean(0) if np.any(lab == j) else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        lab = np.argmin(((x[:, None] - c[None]) ** 2).sum(-1), axis=1)
        inertia = float(((x - c[lab]) ** 2).sum())
        if best is None or inertia < best[0] - 1e-9:
            best = (inertia, lab)
    # Relabel groups by first appearance, so the table does not depend on the order of the centres.
    order = {}
    for g in best[1]:
        order.setdefault(int(g), len(order))
    return np.array([order[int(g)] for g in best[1]]), best[0]


def build_joint_groups(descriptions, robot_names, joint_names=None, k=8, seed=0):
    """Joint groups from the nominal joint descriptions of every robot: descriptions[r] is (J_r, 23) (the mean over
    the dataset of every joint's description). Features are standardized over all joints of all robots (constant
    features dropped), then clustered into k groups. Returns the table dict (written as JSON)."""
    feats = [joint_features(d) for d in descriptions]
    x = np.concatenate(feats)
    mu, sd = x.mean(0), x.std(0)
    keep = sd > 1e-6
    z = (x[:, keep] - mu[keep]) / sd[keep]
    labels, inertia = kmeans(z, k, seed)
    out, i = {}, 0
    for r, name in enumerate(robot_names):
        n = len(descriptions[r])
        out[str(name)] = [int(g) for g in labels[i:i + n]]
        i += n
    robots = np.concatenate([np.full(len(d), r) for r, d in enumerate(descriptions)])
    return {"k": k, "seed": seed, "features": [f for f, kp in zip(GROUP_FEATURES, keep) if kp],
            "inertia": inertia, "groups": out, "joint_names": joint_names or {},
            "nmi_group_robot": normalized_mutual_information(labels, robots)}


def normalized_mutual_information(a, b):
    """I(A;B) / sqrt(H(A) H(B)) of two label arrays (0: independent, 1: one determines the other)."""
    a, b = np.asarray(a), np.asarray(b)
    joint = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(joint, (a, b), 1.0)
    joint /= joint.sum()
    pa, pb = joint.sum(1), joint.sum(0)
    nz = joint > 0
    mi = float((joint[nz] * np.log(joint[nz] / (pa[:, None] * pb[None])[nz])).sum())
    ha = -float((pa[pa > 0] * np.log(pa[pa > 0])).sum())
    hb = -float((pb[pb > 0] * np.log(pb[pb > 0])).sum())
    return mi / max(np.sqrt(ha * hb), 1e-12)


def load_joint_groups(path, robot_names, n_joints):
    """The joint-group table of the training robots: a list (one int array per robot, in order) checked against the
    robots' joint counts."""
    with open(path) as f:
        table = json.load(f)
    groups = []
    for name, n in zip(robot_names, n_joints):
        if str(name) not in table["groups"]:
            raise ValueError(f"joint-group table {path} has no robot {name}")
        g = np.asarray(table["groups"][str(name)], np.int32)
        if len(g) != int(n):
            raise ValueError(f"joint-group table {path}: {name} has {len(g)} joints, the environment {n}")
        groups.append(g)
    return groups, int(table["k"])


def write_joint_groups(path, table):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(table, f, indent=1)
    os.replace(tmp, path)
