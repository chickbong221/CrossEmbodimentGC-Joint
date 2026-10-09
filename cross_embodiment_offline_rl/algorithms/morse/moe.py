"""Dense mixture-of-experts blocks used by MORSE.

A block holds ``num_experts`` experts with identical architecture. Their
parameters carry a leading expert axis, so all experts are evaluated in one
batched computation. The router adds the logits of a morphology branch and a
state branch.

With ``capture=True``, a block records everything needed to compute exact
inner products between the gradient contributions of different routed units
(Eq. linear_factorization in the paper). For the first ``n_capture`` batch
entries it stores:
  * the input of every expert linear layer and every normalized LayerNorm input
    (collection ``CAPTURE_IN``), and
  * a zero-valued perturbation on every linear-layer and LayerNorm output
    (collection ``CAPTURE_DELTA``), whose gradient is the backward signal,
  * the routing probabilities and the router inputs (collection ``CAPTURE_ROUTER``).
Capturing does not change the forward computation or the parameters.

Independently of capturing, a routed block records the router inputs and the routing weights of every batch entry
in the collection ``ROUTING`` when it is mutable (the MI objective and the routing diagnostics).

With ``router_noise``, a block adds learned noise to its routing logits in training (Mod-Squad's noisy gate): the
noise head's scale sigma = softplus(.) + floor multiplies N(0, 1) noise drawn from the ``routing`` RNG stream. Without
that stream (evaluation) the block routes on the clean logits. The noise term, held fixed (sigma * eps without
gradient), is recorded as ``offset`` so that router objectives can re-apply the router to the noisy logits.

Shared experts (DeepSeekMoE's shared-expert isolation, extended by group experts): the first ``n_global`` experts are
used by every unit and the next ``n_group`` are group experts, of which a unit uses its group's (``group_p``, one-hot).
Their weights are fixed (``shared_coef`` per branch); the router covers the remaining routed experts only, whose top-k
mixture takes the rest of the weight. Every expert is still computed in one batched pass. The ``ROUTING`` records
hold the routed experts' weights (the router's view); the captured ``p`` holds the weights of all experts (the
experts' view, for gradient Grams).
"""
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, lecun_normal, normal, zeros


CAPTURE_IN = "morse_in"
CAPTURE_DELTA = "morse_delta"
CAPTURE_ROUTER = "morse_router"
ROUTING = "morse_routing"


def expert_init(init_fn: Callable, identical: bool = False) -> Callable:
    """Initialize every expert slice independently with ``init_fn`` (or all slices with the same draw)."""
    def init(key, shape, dtype=jnp.float32):
        if identical:
            return jnp.broadcast_to(init_fn(key, shape[1:], dtype), shape)
        keys = jax.random.split(key, shape[0])
        return jax.vmap(lambda k: init_fn(k, shape[1:], dtype))(keys)
    return init


def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


class ExpertDense(nn.Module):
    """``num_experts`` parallel dense layers.

    If ``shared_input`` is True the input has no expert axis and is fed to every
    expert (first layer of a block); otherwise the input is ``(..., K, d_in)``.
    The output is ``(..., K, features)``.

    With ``weight_norm`` the weights of every output unit are w = g v / ||v||: ``kernel`` holds the directions v and
    ``wn_scale`` the lengths g, which start at ||v|| (the initial weights are those of the plain layer). Captured inputs
    and backward signals give gradients with respect to w.
    """
    num_experts: int
    features: int
    kernel_init: Callable = lecun_normal()
    bias_init: Callable = zeros
    shared_input: bool = False
    capture: bool = False
    n_capture: int = 0
    identical_init: bool = False
    weight_norm: bool = False

    @nn.compact
    def __call__(self, x):
        d_in = x.shape[-1]
        kernel = self.param("kernel", expert_init(self.kernel_init, self.identical_init),
                            (self.num_experts, d_in, self.features))
        if self.weight_norm:
            norm = jnp.sqrt(jnp.sum(jnp.square(kernel), axis=-2) + 1e-12)                  # (K, features)
            scale = self.param("wn_scale", lambda key, shape: norm, (self.num_experts, self.features))
            kernel = kernel * (scale / norm)[:, None, :]
        bias = self.param("bias", expert_init(self.bias_init, self.identical_init), (self.num_experts, self.features))
        if self.shared_input:
            z = jnp.einsum("...i,kio->...ko", x, kernel) + bias
        else:
            z = jnp.einsum("...ki,kio->...ko", x, kernel) + bias
        if self.capture:
            self.sow(CAPTURE_IN, "x", x[: self.n_capture])
            z = self.perturb("z", z, collection=CAPTURE_DELTA)
        return z


class ExpertLayerNorm(nn.Module):
    """LayerNorm over the feature axis with a separate affine transform per expert."""
    num_experts: int
    epsilon: float = 1e-6
    capture: bool = False
    n_capture: int = 0

    @nn.compact
    def __call__(self, x):
        d = x.shape[-1]
        scale = self.param("scale", constant(1.0), (self.num_experts, d))
        bias = self.param("bias", constant(0.0), (self.num_experts, d))
        mean = x.mean(axis=-1, keepdims=True)
        var = jnp.square(x - mean).mean(axis=-1, keepdims=True)
        xhat = (x - mean) * jax.lax.rsqrt(var + self.epsilon)
        y = xhat * scale + bias
        if self.capture:
            self.sow(CAPTURE_IN, "xhat", xhat[: self.n_capture])
            y = self.perturb("y", y, collection=CAPTURE_DELTA)
        return y


class Router(nn.Module):
    """Routing logits: morphology branch + state branch (Eq. router)."""
    num_experts: int
    hidden: int = 64
    inputs: str = "both"  # "both", "morph", or "state"
    out_init_std: float = 0.0  # > 0: random output layers N(0, std^2), so the experts' logits differ from the start
                               # (0: zero output layers, uniform initial routing)

    @nn.compact
    def __call__(self, morph, state):
        out_init = normal(self.out_init_std) if self.out_init_std > 0 else zeros
        logits = jnp.zeros(morph.shape[:-1] + (self.num_experts,), morph.dtype)
        if self.inputs in ("both", "morph"):
            h = nn.elu(nn.Dense(self.hidden, name="morph_hidden")(morph))
            logits = logits + nn.Dense(self.num_experts, kernel_init=out_init, bias_init=zeros, name="morph_out")(h)
        if self.inputs in ("both", "state"):
            h = nn.elu(nn.Dense(self.hidden, name="state_hidden")(symlog(state)))
            logits = logits + nn.Dense(self.num_experts, kernel_init=out_init, bias_init=zeros, name="state_out")(h)
        return logits


def keep_topk(p, k, ties="exact"):
    """Routing weights restricted to the k largest of every unit and renormalized over them; gradients reach the kept
    weights only. k <= 0 or k >= K keeps every expert. ties="exact" keeps exactly k (ties go to the lower expert
    index); ties="keep" keeps every weight >= the k-th largest (the rule before 2026-10-04, under which a uniform
    start is dense)."""
    if not 0 < k < p.shape[-1]:
        return p
    if ties == "keep":
        mask = (p >= jax.lax.top_k(p, k)[0][..., -1:]).astype(p.dtype)
    else:
        mask = jax.nn.one_hot(jax.lax.top_k(p, k)[1], p.shape[-1], dtype=p.dtype).sum(axis=-2)
    p = p * mask
    return p / p.sum(axis=-1, keepdims=True)


def route(logits, topk=0, eval_topk=0, ties="exact"):
    """The routing function of every MoE block, in training, evaluation and checkpoint analysis: the soft routing
    probabilities (the MI objective) and the weights that mix the expert outputs (top-k, then the evaluation top-k)."""
    p_soft = jax.nn.softmax(logits, axis=-1)
    return p_soft, keep_topk(keep_topk(p_soft, topk, ties), eval_topk, ties)


def routing_noise(key, shape, dtype):
    """N(0, 1) routing noise of the given shape, drawn per leading index i from fold_in(key, i): the first m rows are the
    same for any batch size, so a pass over the first m transitions of a minibatch sees the training pass's noise."""
    keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(shape[0]))
    return jax.vmap(lambda k: jax.random.normal(k, shape[1:], dtype))(keys)


class MoEBlock(nn.Module):
    """A dense mixture of ``num_experts`` identical MLP experts (Eq. expert_mixture).

    ``layer_features[i]`` is the width of linear layer ``i``; ``layer_post[i]`` is
    what follows it: ``"ln_elu"`` (LayerNorm then ELU), ``"elu"``, or ``"none"``.
    With ``num_experts == 1`` the block is an ordinary shared MLP (routing weight 1,
    no router), so its parameters are exactly those of the original block.
    """
    layer_features: Tuple[int, ...]
    layer_post: Tuple[str, ...]
    kernel_inits: Tuple[Callable, ...]
    num_experts: int = 4
    router_hidden: int = 64
    router_inputs: str = "both"
    detach_router_inputs: bool = False
    capture: bool = False
    n_capture: int = 0
    eval_topk: int = 0
    train_topk: int = 0           # > 0: every unit keeps its top-k routing weights (renormalized), in training and evaluation
    fixed_routing: bool = False   # routing weights come from a fixed table (no router parameters)
    identical_init: bool = False  # every expert starts from the same parameters
    router_init: float = 0.0      # std of the router's random output layers (0: zero layers, uniform initial routing)
    joint_routing: bool = False   # whole-body experts mixed per element: x has no element axis, morph and state have
                                  # one, and the output is (..., elements, features), one mixture per element
    router_noise: bool = False    # learned routing noise in training (applied when the 'routing' RNG stream is given)
    noise_floor: float = 0.01     # lower bound of the noise scale
    topk_ties: str = "exact"      # see keep_topk
    score_mode: bool = False      # compatibility scoring pass: the output keeps its value, but every expert receives
                                  # the unweighted output gradient, its candidate update direction (the gradient into
                                  # the block's input is not meaningful in this pass)
    n_global: int = 0             # always-on experts used by every unit (the first n_global experts)
    n_group: int = 0              # always-on group experts (the next n_group): a unit uses its group's, given by group_p
    shared_coef: float = 0.25     # fixed weight of each always-on branch (global, group); the routed mixture takes the rest
    weight_norm: bool = False     # WeightNorm on every dense layer of the experts (ExpertDense)

    def _all_weights(self, p, group_p):
        """Weights of every expert: the fixed shared branches, then the routed experts' renormalized top-k mixture."""
        n_branches = int(self.n_global > 0) + int(self.n_group > 0)
        parts = []
        if self.n_global:
            parts.append(jnp.full(p.shape[:-1] + (self.n_global,), self.shared_coef / self.n_global, p.dtype))
        if self.n_group:
            if group_p is None:
                if not self.is_initializing():
                    raise ValueError("group experts need the robot's group (pass robot_id to the policy)")
                group_p = jnp.zeros(p.shape[:-1] + (self.n_group,), p.dtype)       # init: shapes only
            parts.append(self.shared_coef * jnp.broadcast_to(group_p.astype(p.dtype), p.shape[:-1] + (self.n_group,)))
        parts.append((1.0 - n_branches * self.shared_coef) * p)
        return jnp.concatenate(parts, axis=-1)

    @nn.compact
    def __call__(self, x, morph, state, fixed_p=None, group_p=None):
        offset = None
        n_shared = self.n_global + self.n_group
        if n_shared and (self.num_experts - n_shared < 1 or self.fixed_routing or self.joint_routing
                         or self.score_mode):
            raise ValueError("shared experts need routed experts, a learned router, one mixture per routing unit, and "
                             "no compatibility scoring")
        w = None                      # weights of every expert when the block has shared experts
        if self.num_experts == 1:
            # A single expert is the shared block: routing weight 1 and no router parameters.
            p = jnp.ones(morph.shape[:-1] + (1,), x.dtype)
        elif self.fixed_routing:
            # (n_elements, K) table row of this robot (or one-hot by robot), or uniform weights when no row is given
            # (init).
            p = (jnp.full(morph.shape[:-1] + (self.num_experts,), 1.0 / self.num_experts, x.dtype) if fixed_p is None
                 else jnp.broadcast_to(fixed_p.astype(x.dtype), morph.shape[:-1] + (self.num_experts,)))
        else:
            r_morph = jax.lax.stop_gradient(morph) if self.detach_router_inputs else morph
            r_state = jax.lax.stop_gradient(state) if self.detach_router_inputs else state
            n_routed = self.num_experts - n_shared
            logits = Router(n_routed, self.router_hidden, self.router_inputs, self.router_init,
                            name="router")(r_morph, r_state)
            offset = jnp.zeros_like(logits)
            sigma = None
            if self.router_noise:
                # Noise head as in Mod-Squad (zero output layers: sigma starts at softplus(0) + floor). The task loss
                # trains it through the noisy logits; the recorded offset carries no gradient to it.
                raw = Router(n_routed, self.router_hidden, self.router_inputs, 0.0,
                             name="router_noise")(r_morph, r_state)
                if self.has_rng("routing"):
                    sigma = jax.nn.softplus(raw) + self.noise_floor
                    noise = sigma * routing_noise(self.make_rng("routing"), logits.shape, logits.dtype)
                    logits = logits + noise
                    offset = jax.lax.stop_gradient(noise)
            _, p = route(logits, self.train_topk, self.eval_topk, self.topk_ties)
            if n_shared:
                w = self._all_weights(p, group_p)
            if not self.is_initializing():     # init returns parameters only, never routing records
                self.sow(ROUTING, "morph", jax.lax.stop_gradient(r_morph))
                self.sow(ROUTING, "state", jax.lax.stop_gradient(r_state))
                self.sow(ROUTING, "p", jax.lax.stop_gradient(p))
                self.sow(ROUTING, "offset", offset)
                if sigma is not None:          # mean noise scale of every unit (diagnostics)
                    self.sow(ROUTING, "noise_scale", jax.lax.stop_gradient(sigma.mean(axis=-1)))

        h = x
        for i, (features, post, init) in enumerate(zip(self.layer_features, self.layer_post, self.kernel_inits)):
            h = ExpertDense(self.num_experts, features, kernel_init=init, bias_init=constant(0.0),
                            shared_input=(i == 0), capture=self.capture, n_capture=self.n_capture,
                            identical_init=self.identical_init, weight_norm=self.weight_norm, name=f"dense_{i}")(h)
            if post == "ln_elu":
                h = ExpertLayerNorm(self.num_experts, capture=self.capture, n_capture=self.n_capture, name=f"ln_{i}")(h)
                h = nn.elu(h)
            elif post == "elu":
                h = nn.elu(h)
        if w is not None:
            y = jnp.einsum("...k,...ko->...o", w, h)
            if not self.is_initializing():
                # Norm of every branch's weighted output (global, group, routed mixture) and of the output, per unit
                # (diagnostics; not a measure of each branch's causal contribution).
                hs = jax.lax.stop_gradient(h)
                wv = jax.lax.stop_gradient(w)
                bounds = (0, self.n_global, n_shared, self.num_experts)
                parts = [jnp.einsum("...k,...ko->...o", wv[..., lo:hi], hs[..., lo:hi, :])
                         for lo, hi in zip(bounds[:-1], bounds[1:])]
                self.sow(ROUTING, "branch_norm", jnp.stack(
                    [jnp.linalg.norm(v, axis=-1) for v in parts] + [jnp.linalg.norm(sum(parts), axis=-1)], axis=-1))
            p = w                              # the captured weights are those of every expert
        else:
            y = jnp.einsum("...ek,...ko->...eo" if self.joint_routing else "...k,...ko->...o", p, h)
        if self.score_mode:
            if self.joint_routing:
                raise ValueError("compatibility scoring is defined for blocks with one mixture per routing unit")
            y = jax.lax.stop_gradient(y) + (h - jax.lax.stop_gradient(h)).sum(axis=-2)

        if self.capture:
            n = self.n_capture
            if offset is None:
                offset = jnp.zeros_like(p)
            if self.joint_routing:
                # The experts compute once per transition, so a transition is the unit of their gradient
                # contributions; its routing weights are the mean over elements (diagnostics only).
                p, morph, state, offset = p.mean(axis=-2), morph.mean(axis=-2), state.mean(axis=-2), offset.mean(axis=-2)
            self.sow(CAPTURE_ROUTER, "p", jax.lax.stop_gradient(p[:n]))
            self.sow(CAPTURE_ROUTER, "morph", jax.lax.stop_gradient(morph[:n]))
            self.sow(CAPTURE_ROUTER, "state", jax.lax.stop_gradient(state[:n]))
            self.sow(CAPTURE_ROUTER, "offset", offset[:n])
        return y
