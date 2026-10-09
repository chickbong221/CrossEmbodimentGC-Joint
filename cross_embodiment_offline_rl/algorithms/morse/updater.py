"""The MORSE actor update.

For every robot, one ordinary backward pass of the base learner's policy loss
yields the parameter gradient and, through the capture collections, the layer
inputs and backward signals of the first few transitions. These are combined
across robots into exact Gram matrices per expert, from which the conflict
coefficients, the router-only auxiliary gradients, and the diagnostics follow.
The same pass records every routed block's router inputs and routing weights for
all transitions, from which the router-only MI objective is computed at every
actor step. A whole update (all robots, router objectives, optimizer step) is
one compiled function, so the host only dispatches it.
"""
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax

from cross_embodiment_offline_rl.algorithms.morse import interference as itf
from cross_embodiment_offline_rl.algorithms.morse.moe import CAPTURE_DELTA, CAPTURE_IN, CAPTURE_ROUTER, ROUTING, Router
import logging

from cross_embodiment_offline_rl.algorithms.morse.policy import (BLOCKS, WHOLE_BODY_BLOCKS, count_core_categories,
                                                                count_params, eg_group_table, parse_blocks)
from cross_embodiment_offline_rl.algorithms.morse.variance import (group_norm, is_core, is_embedding, is_std_head,
                                                                   is_std_param)


# Trainable layers of each block in forward order for URMA's layer sizes; the updater takes the layers of its policy
# (MorsePolicy.block_layers).
BLOCK_LAYERS = {
    "joint_encoder": (("dense", 0),),
    "foot_encoder": (("dense", 0),),
    "core": (("dense", 0), ("ln", 0), ("dense", 1), ("dense", 2)),
    "decoder": (("dense", 0), ("ln", 0), ("dense", 1)),
}


def extract_captures(state, g_delta, layout=BLOCK_LAYERS):
    """Layer inputs and backward signals of the captured transitions, per block, and (key ROUTING) the router inputs
    and routing weights of every transition of every routed block."""
    captured = {ROUTING: {block: {k: v[0] for k, v in rec.items()} for block, rec in state.get(ROUTING, {}).items()}}
    for block in BLOCKS:
        s_in, g_in, router = state[CAPTURE_IN][block], g_delta[block], state[CAPTURE_ROUTER][block]
        xs, ds = [], []
        for kind, i in layout[block]:
            if kind == "dense":
                x = s_in[f"dense_{i}"]["x"][0]
                d = g_in[f"dense_{i}"]["z"]
            else:
                x = s_in[f"ln_{i}"]["xhat"][0]
                d = g_in[f"ln_{i}"]["y"]
            xs.append(x)
            ds.append(d[: x.shape[0]])
        captured[block] = {"x": xs, "d": ds, "p": router["p"][0],
                           "morph": router["morph"][0], "state": router["state"][0], "offset": router["offset"][0]}
    return captured


def _block_units(state, g_delta, block, layout=BLOCK_LAYERS):
    """One robot's scored units of a block from a scoring pass: the layers (kind, X, D) with a transition and an element
    axis (J = 1 at the body-routed core), and the units' routing weights, router inputs, and noise offsets."""
    s_in, g_in, router = state[CAPTURE_IN][block], g_delta[block], state[CAPTURE_ROUTER][block]
    element = block not in WHOLE_BODY_BLOCKS
    lift = (lambda a: a) if element else (lambda a: a[:, None])
    layers = []
    for kind, i in layout[block]:
        x = s_in[f"dense_{i}"]["x"][0] if kind == "dense" else s_in[f"ln_{i}"]["xhat"][0]
        d = g_in[f"dense_{i}"]["z"] if kind == "dense" else g_in[f"ln_{i}"]["y"]
        layers.append((kind, lift(x), lift(d[: x.shape[0]])))
    return layers, {k: lift(router[k][0]) for k in ("p", "morph", "state", "offset")}


def _is_router(path):
    return any(getattr(k, "key", None) == "router" for k in path)


def _router_norm(grads, block=None):
    """Global norm of the router parameters' gradient (to compare task and auxiliary signals), of one block if given."""
    leaves = [leaf for path, leaf in jax.tree_util.tree_leaves_with_path(grads)
              if _is_router(path) and (block is None or any(getattr(k, "key", None) == block for k in path[:2]))]
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)) if leaves else jnp.zeros(())


# Robot-level diagnostics are computed per expert block and for the whole actor ("actor": every
# policy parameter except routers, i.e. the per-embodiment actor-gradient cosine of Abe et al., Eq. 2).
ROBOT_LEVELS = BLOCKS + ("actor",)


def robot_block_grams(grads):
    """Gram matrix of the per-robot gradients, (R, R), for every block's expert parameters and the actor.

    ``grads`` holds one full-minibatch gradient per robot; router parameters are excluded.
    The Gram is summed leaf by leaf (concatenating all leaves of all robots makes one fused
    kernel with too many arguments for the GPU compiler).
    """
    per_robot = [jax.tree_util.tree_leaves_with_path(g["params"]) for g in grads]
    grams = {}
    for items in zip(*per_robot):                       # the same parameter leaf of every robot
        path = items[0][0]
        if _is_router(path):
            continue
        v = jnp.stack([leaf.reshape(-1) for _, leaf in items])
        m = jnp.dot(v, v.T, precision=itf.HIGHEST)
        top = getattr(path[0], "key", None)                # module name: an expert block or a shared module
        for level in ("actor", top):
            if level in ROBOT_LEVELS:
                grams[level] = grams[level] + m if level in grams else m
    return grams


def _accumulate_robot_grams(acc, grads, decay):
    grams = robot_block_grams(grads)
    return grams, {b: decay * acc[b] + (1.0 - decay) * grams[b] for b in ROBOT_LEVELS}


def _units(a, n_tail):
    """Flatten the leading (transition[, element]) axes into one unit axis."""
    return a.reshape((-1,) + a.shape[a.ndim - n_tail:])


def _chunk_struct(x, n_sub):
    """Shape of one of n_sub equal chunks of a minibatch array."""
    return jax.ShapeDtypeStruct((x.shape[0] // n_sub,) + tuple(x.shape[1:]), x.dtype)


def _chunk(x, sub, n_sub):
    size = x.shape[0] // n_sub
    return jax.lax.dynamic_slice_in_dim(x, sub * size, size)


class MorseActorUpdater:
    def __init__(self, config, capture_policy, split_fn, robot_names, run_path):
        a = config.algorithm
        self.capture_policy = capture_policy
        self.split_fn = split_fn
        self.robot_names = list(robot_names)
        self.moe_blocks = parse_blocks(a.morse_blocks)
        self.num_experts = {b: capture_policy.num_experts_of(b) for b in BLOCKS}
        # Experts each router chooses among (the core's always-on shared experts excluded): the router objectives, MI
        # and balance act on these only. Shared core experts are the first n_global + n_group of its expert axis.
        self.router_experts = {b: capture_policy.router_experts_of(b) for b in BLOCKS}
        self.core_shared = capture_policy.core_shared_counts()
        self.block_layers = {b: capture_policy.block_layers(b) for b in BLOCKS}
        self.variance_mode = str(getattr(a, "morse_variance_mode", "original"))
        self.freeze_core = bool(getattr(a, "morse_freeze_core", False))
        self.max_grad_norm = float(getattr(a, "max_grad_norm", 0.0))
        self._param_counts = None
        # Blocks with fixed routing (a decoder table, or one-hot by robot) have no router: no routing objectives.
        fixed_robot = parse_blocks(getattr(a, "morse_fixed_robot_blocks", "") or "")
        self.fixed_blocks = tuple(b for b in BLOCKS
                                  if b in fixed_robot or (b == "decoder" and getattr(a, "morse_decoder_table", "")))
        # Routing noise: its keys come from a stream of their own (seed, update, actor step, robot), so the main key
        # sequence, the minibatches and the critics are those of a run without noise.
        self.router_noise = bool(getattr(a, "morse_router_noise", False))
        seed = int(getattr(getattr(config, "environment", None), "seed", 0))
        self.routing_base = jax.random.fold_in(jax.random.PRNGKey(seed), 0x524F55)
        self.routed_blocks = [b for b in BLOCKS if self.num_experts[b] > 1 and b not in self.fixed_blocks]
        self.router_hidden = a.morse_router_hidden
        self.router_inputs = a.morse_router_inputs
        self.mode = a.morse_conflict_mode
        if self.mode not in ("pairwise", "shuffled", "expert_mean", "signed", "none"):
            raise ValueError(f"Unknown morse_conflict_mode {self.mode}")
        if self.mode == "signed" and a.morse_evidence != "type":
            raise ValueError("signed compatibility uses type evidence")
        self.compat_z = float(getattr(a, "morse_compat_z", 2.0))
        self.compat_help = float(getattr(a, "morse_compat_help", 1.0))
        conf_spec = getattr(a, "morse_conflict_blocks", "")
        self.conf_blocks = [b for b in parse_blocks(conf_spec) if b in self.routed_blocks] if conf_spec else list(self.routed_blocks)
        # Cross-fitted compatibility: the decoder captures every transition of the chunk (the capture policy's
        # n_capture_decoder); unit Grams, diagnostics and routing objectives keep the first n_gram transitions.
        self.compat_estimator = getattr(a, "morse_compat_estimator", "products")
        if self.compat_estimator not in ("products", "crossfit"):
            raise ValueError(f"Unknown morse_compat_estimator {self.compat_estimator}")
        self.crossfit_blocks = []
        if self.compat_estimator == "crossfit":
            if self.mode != "signed" or self.conf_blocks != ["decoder"]:
                raise ValueError("the crossfit estimator is for signed compatibility at the decoder "
                                 "(morse_conflict_mode=signed, morse_conflict_blocks=decoder)")
            self.crossfit_blocks = ["decoder"]
        self.compat_decay = float(getattr(a, "morse_compat_ema", 0.9))
        self.n_gram = {b: (capture_policy.n_capture_body if b in WHOLE_BODY_BLOCKS else capture_policy.n_capture_element)
                       for b in BLOCKS}
        # Gradient-surgery control: the task gradient of these blocks' experts is the sum of the per-joint-type
        # gradients after PCGrad across types (every actor step, from all transitions of its chunk).
        self.pcgrad_blocks = parse_blocks(getattr(a, "morse_pcgrad_blocks", "")) if getattr(a, "morse_pcgrad_blocks", "") else []
        if any(b != "decoder" for b in self.pcgrad_blocks):
            raise ValueError("joint-level PCGrad is implemented for the decoder (morse_pcgrad_blocks=decoder)")
        self.mi_weight = float(getattr(a, "morse_mi_weight", 0.0))
        # The core's MI term may take its own weight (morse_mi_weight_core >= 0), and the blocks in
        # morse_mi_balance_blocks keep only the load-balancing half of their term, -H(E).
        mi_weight_core = float(getattr(a, "morse_mi_weight_core", -1.0))
        self.mi_block_weight = {b: (mi_weight_core if b == "core" and mi_weight_core >= 0.0 else self.mi_weight)
                                for b in BLOCKS}
        self.mi_blocks = [b for b in parse_blocks(getattr(a, "morse_mi_blocks", "decoder"))
                          if b in self.routed_blocks and self.mi_block_weight[b] > 0]
        self.mi_balance_blocks = parse_blocks(getattr(a, "morse_mi_balance_blocks", "") or "")
        if any(b not in self.mi_blocks for b in self.mi_balance_blocks):
            raise ValueError(f"morse_mi_balance_blocks {list(self.mi_balance_blocks)} must be routed MI blocks with a "
                             f"positive weight (MI blocks: {self.mi_blocks})")
        # MI groups: unit types (joint of a robot at element-level blocks and the joint-routed core, else the robot)
        # or whole robots; group prior: share of the units (robots with more joints weigh more at the decoder) or
        # equal per robot.
        self.mi_target = getattr(a, "morse_mi_target", "type")
        self.mi_weighting = getattr(a, "morse_mi_weighting", "units")
        if self.mi_target not in ("type", "robot") or self.mi_weighting not in ("units", "robot"):
            raise ValueError(f"Unknown morse_mi_target {self.mi_target} or morse_mi_weighting {self.mi_weighting}")
        # The body-routed core's MI groups: robots, or Embodiment Grouping's clusters of robots (uni_iql_eg).
        self.mi_core_groups = getattr(a, "morse_mi_core_groups", "robot")
        if self.mi_core_groups not in ("robot", "eg"):
            raise ValueError(f"Unknown morse_mi_core_groups {self.mi_core_groups}")
        self.core_group_of = None
        if self.mi_core_groups == "eg":
            if getattr(capture_policy, "core_routing", "body") != "body":
                raise ValueError("EG groups in the core's MI need morse_core_routing=body")
            self.core_group_of = np.array(eg_group_table(a, self.robot_names))
        # The core's group experts and its EG-group MI must use one grouping of exactly these robots.
        policy_groups = tuple(getattr(capture_policy, "core_group_of", ()) or ())
        if policy_groups:
            if tuple(eg_group_table(a, self.robot_names)) != policy_groups:
                raise ValueError(f"the policy's EG groups {policy_groups} differ from those of the training robots "
                                 f"{self.robot_names}")
            if self.core_group_of is not None and tuple(self.core_group_of) != policy_groups:
                raise ValueError("the core's group experts and its EG-group MI use different groupings")
        self.mi_cap = float(getattr(a, "morse_mi_cap", 0.0))
        self._mi_groups = {}       # per MI block: (robot of every group, group prior), from the routing record shapes
        self._mi_agg = {}          # per MI block: None, or the (groups, base groups) averaging of the core's EG groups
        self._cap_acc = None       # MI cap factors summed over the actor steps since the last diagnostic step
        self._routing_log = None   # per-robot expert selection and weights at every diagnostic step (routing.npz)
        self.lam = float(a.morse_lambda)
        self.eta = float(a.morse_eta)
        self.conflict_norm = a.morse_conflict_norm
        if self.conflict_norm not in ("pairs", "mass"):
            raise ValueError(f"Unknown morse_conflict_norm {self.conflict_norm}")
        self.unbiased_types = bool(a.morse_unbiased_types)
        self.zeta = float(a.morse_zeta)
        self.kappa = float(a.morse_kappa)
        self.calibrated = a.morse_threshold == "calibrated"
        self.relevance = bool(a.morse_relevance)
        self.evidence = a.morse_evidence
        if self.evidence not in ("unit", "type"):
            raise ValueError(f"Unknown morse_evidence {self.evidence}")
        if self.evidence == "type" and self.mode == "expert_mean":
            raise ValueError("expert_mean evidence is defined for units only")
        self.type_decay = float(a.morse_type_ema)
        self.conflict_every = max(1, int(a.morse_conflict_every))
        self.warmup = int(a.morse_warmup)
        self.ramp = int(a.morse_ramp)
        self.router_only = bool(a.morse_router_only)
        # A joint-routed core mixes whole-body experts per joint: its routing units (joints) are not the units of its
        # expert gradients (transitions), so it takes the MI objective only.
        self.joint_core = "core" in self.routed_blocks and getattr(capture_policy, "core_routing", "body") == "joint"
        if self.joint_core and (self.eta > 0.0 or (self.mode != "none" and "core" in self.conf_blocks)):
            raise ValueError("a joint-routed core takes only the MI objective: set morse_eta=0 and leave the core out of "
                             "the conflict objective (morse_conflict_mode=none or morse_conflict_blocks=decoder)")
        self.aux_blocks = [b for b in self.routed_blocks if not (b == "core" and self.joint_core)]
        # Compatibility routing: scoring policies (every expert of the scored block receives the unweighted output
        # gradient) capturing the scored transitions of every robot.
        self.compat_route = getattr(a, "morse_compat_route", "off")
        if self.compat_route not in ("off", "compat", "shuffled", "conflict_only", "uniform"):
            raise ValueError(f"Unknown morse_compat_route {self.compat_route}")
        self.compat_lambda = {"core": float(getattr(a, "morse_compat_lambda_core", 0.0)),
                              "decoder": float(getattr(a, "morse_compat_lambda_decoder", 0.0))}
        self.compat_blocks = [] if self.compat_route == "off" else [
            b for b in ("core", "decoder") if b in self.aux_blocks and self.compat_lambda[b] > 0.0]
        if self.compat_blocks and sum(self.core_shared):
            raise ValueError("compatibility routing is not defined with shared core experts")
        self.compat_tau = float(getattr(a, "morse_compat_tau", 0.05))
        self.compat_min_support = int(getattr(a, "morse_compat_min_support", 4))
        self.compat_n = {"core": int(getattr(a, "morse_compat_core_transitions", 64)),
                         "decoder": int(getattr(a, "morse_compat_decoder_transitions", 32))}
        self.score_policies = {
            b: capture_policy.clone(score_block=b, **({"n_capture_body": self.compat_n[b]} if b == "core"
                                                      else {"n_capture_decoder": self.compat_n[b]}))
            for b in self.compat_blocks}
        self.diag_frequency = int(a.morse_diag_frequency)
        self.snapshot_frequency = int(a.morse_snapshot_frequency)
        self.snapshot_dir = os.path.join(run_path, "morse_diagnostics") if run_path else None
        self.loss_fn = None  # set by the algorithm: loss_fn(out, obs, act, extra, key, env_type_id)
        self._step_fns = {}        # compiled updates per (kind, observation shapes)
        self._meta = None
        self._type_state = None    # accumulated type sums per block, (2, K, T, T): [0] the evidence (without
                                   # same-transition pairs if unbiased_types), [1] same-batch sums (diagnostic)
        self._type_count = 0
        # Robot-level diagnostics from full-batch per-robot gradients, accumulated over updates.
        self.robot_decay = float(a.morse_robot_ema)
        self._robot_acc = None     # EMA of the per-block robot Gram matrices, (R, R)
        self._robot_count = 0
        self._last_diag = None     # (wall time, update) of the last diagnostic step

    # ------------------------------------------------------------------ schedule
    def lambda_at(self, step):
        if self.mode == "none" or self.lam == 0.0 or step < self.warmup:
            return 0.0
        if self.ramp <= 0:
            return self.lam
        return self.lam * min(1.0, (step - self.warmup + 1) / self.ramp)

    def compat_factor_at(self, step):
        """Share of the compatibility weights in effect (0 before morse_warmup, then a linear morse_ramp)."""
        if not self.compat_blocks or step < self.warmup:
            return 0.0
        return 1.0 if self.ramp <= 0 else min(1.0, (step - self.warmup + 1) / self.ramp)

    def _routing_keys(self, step_sub, n_robots):
        """Routing-noise keys of every robot for update step_sub[0], actor step step_sub[1] (None without noise)."""
        if not self.router_noise:
            return [None] * n_robots
        base = jax.random.fold_in(jax.random.fold_in(self.routing_base, step_sub[0]), step_sub[1])
        return [jax.random.fold_in(base, r) for r in range(n_robots)]

    # ------------------------------------------------------------ per-robot pass
    def _robot_pass(self, env_type_id, obs):
        """Loss, parameter gradient, and captures of one robot's minibatch (traced into the update)."""
        def init(o):
            return self.capture_policy.init(jax.random.PRNGKey(0), *self.split_fn(o, env_type_id))
        template = jax.eval_shape(init, jax.ShapeDtypeStruct(obs.shape, obs.dtype))[CAPTURE_DELTA]

        def fn(params, extra, obs_b, act_b, rng, routing_key=None):
            zeros = jax.tree_util.tree_map(lambda s: jnp.zeros(s.shape, s.dtype), template)

            def loss_with_capture(prm, delta):
                variables = {"params": prm["params"], CAPTURE_DELTA: delta}
                out, state = self.capture_policy.apply(
                    variables, *self.split_fn(obs_b, env_type_id), robot_id=env_type_id,
                    mutable=[CAPTURE_IN, CAPTURE_ROUTER, ROUTING],
                    rngs=None if routing_key is None else {"routing": routing_key})
                return self.loss_fn(out, obs_b, act_b, extra, rng, env_type_id), state

            (loss, state), (g_params, g_delta) = jax.value_and_grad(
                loss_with_capture, argnums=(0, 1), has_aux=True)(params, zeros)
            return loss, g_params, extract_captures(state, g_delta, self.block_layers)

        return fn

    def _score_pass(self, block, env_type_id, obs):
        """Candidate update directions of a block's scored units, one robot: the scoring policy's layer inputs and
        backward signals (every expert receives the unweighted output gradient) on the first compat_n[block]
        transitions of the actor step's chunk, routed (noise included) exactly as in the training pass."""
        policy, n = self.score_policies[block], self.compat_n[block]

        def init(o):
            return policy.init(jax.random.PRNGKey(0), *self.split_fn(o, env_type_id))
        template = jax.eval_shape(init, jax.ShapeDtypeStruct((n,) + tuple(obs.shape[1:]), obs.dtype))[CAPTURE_DELTA]

        def fn(params, extra, obs_b, act_b, rng, routing_key=None):
            obs_s, act_s = obs_b[:n], act_b[:n]
            zeros = jax.tree_util.tree_map(lambda s: jnp.zeros(s.shape, s.dtype), template)

            def loss_with_capture(delta):
                out, state = policy.apply(
                    {"params": params["params"], CAPTURE_DELTA: delta}, *self.split_fn(obs_s, env_type_id),
                    robot_id=env_type_id, mutable=[CAPTURE_IN, CAPTURE_ROUTER],
                    rngs=None if routing_key is None else {"routing": routing_key})
                return self.loss_fn(out, obs_s, act_s, extra, rng, env_type_id), state

            (_, state), g_delta = jax.value_and_grad(loss_with_capture, has_aux=True)(zeros)
            return _block_units(state, g_delta, block, self.block_layers)

        return fn

    def _compat(self, params, extra, observations, actions, routing_keys, score_fns, key, factor):
        """Compatibility labels of every block's scored units and the router-only gradient of the compatibility
        objective (weights compat_lambda * factor), with label diagnostics."""
        inputs, metrics = {}, {}
        for block in self.compat_blocks:
            robots, recs, strata = [], [], []
            for r, (obs, act) in enumerate(zip(observations, actions)):
                layers, rec = score_fns[block][r](params, extra, obs, act, key, routing_keys[r])
                robots.append((layers, rec["p"]))
                recs.append(rec)
                n_t, n_j = rec["p"].shape[:2]
                strata.append(r * 1000 + np.tile(np.arange(n_j), n_t))     # robot, joint of a robot at the decoder
            scores, support = itf.compat_scores(robots, self.compat_min_support)
            labels = [itf.compat_labels(cos, valid, self.compat_tau) for cos, valid in scores]
            nonconf = jnp.concatenate([_units(nc, 1) for nc, _ in labels])
            conf = jnp.concatenate([_units(cf, 1) for _, cf in labels])
            if self.compat_route == "shuffled":
                key, k_shuffle = jax.random.split(key)
                nonconf, conf = itf.shuffle_within((nonconf, conf), jnp.asarray(np.concatenate(strata)), k_shuffle)
            inputs[block] = tuple(jnp.concatenate([_units(rec[k], 1) for rec in recs]) for k in ("morph", "state", "offset")) \
                + (nonconf, conf)
            valid2 = jnp.concatenate([_units(v[0] & v[1], 1) for _, v in scores])
            agree = jnp.concatenate([_units((c[0] >= -self.compat_tau) == (c[1] >= -self.compat_tau), 1) for c, _ in scores])
            n_a = nonconf.sum(axis=-1)
            metrics.update({
                f"morse/{block}/compat_nonconf_frac": nonconf.mean(), f"morse/{block}/compat_conf_frac": conf.mean(),
                f"morse/{block}/compat_valid_frac": valid2.mean(),
                f"morse/{block}/compat_agree_frac": jnp.sum(agree * valid2) / jnp.maximum(valid2.sum(), 1),
                f"morse/{block}/compat_mean_nonconf": n_a.mean(), f"morse/{block}/compat_one_nonconf_frac": (n_a == 1).mean(),
                f"morse/{block}/compat_no_nonconf_frac": (n_a == 0).mean(), f"morse/{block}/compat_support_min": support.min()})
        (_, loss_metrics), grads = jax.value_and_grad(self._compat_loss, has_aux=True)(
            params, inputs, {b: self.compat_lambda[b] * factor for b in self.compat_blocks})
        metrics.update(loss_metrics)
        metrics["morse/compat_factor"] = factor
        return grads, metrics

    def _compat_loss(self, params, inputs, lams):
        """The compatibility objective of every scored block on the router's noisy probabilities (offset fixed)."""
        total, metrics = jnp.zeros(()), {}
        for block, (morph, state, offset, nonconf, conf) in inputs.items():
            router = Router(self.router_experts[block], self.router_hidden, self.router_inputs)
            logits = router.apply({"params": params["params"][block]["router"]}, morph, state) + offset
            loss, parts = itf.compat_loss(jax.nn.softmax(logits, axis=-1), nonconf, conf, self.compat_route)
            total = total + lams[block] * loss
            metrics[f"morse/{block}/compat_loss"] = loss
            metrics.update({f"morse/{block}/compat_{k}": v for k, v in parts.items()})
        return total, metrics

    # ------------------------------------------------------------ unit metadata
    def _build_meta(self, captures):
        meta = {}
        for block in BLOCKS:
            element_level = block not in WHOLE_BODY_BLOCKS
            robot_ids, trans_ids, elem_ids, type_ids, type_robot, type_elem = [], [], [], [], [], []
            n_types = 0
            for r, cap in enumerate(captures):
                lead = cap[block]["p"].shape[:-1]           # (n,) or (n, E)
                n_trans = min(lead[0], self.n_gram[block])
                n_elem = lead[1] if len(lead) == 2 else 1
                idx = np.arange(n_trans * n_elem)
                robot_ids.append(np.full(idx.shape, r))
                trans_ids.append(r * 1_000_000 + idx // n_elem)
                elem_ids.append(idx % n_elem)
                # Unit type: element e of robot r (element level) or robot r (whole body).
                type_ids.append(n_types + idx % n_elem if element_level else np.full(idx.shape, r))
                n_new = n_elem if element_level else 1
                type_robot.append(np.full(n_new, r))
                type_elem.append(np.arange(n_new))
                n_types += n_new
            robot_ids = np.concatenate(robot_ids)
            trans_ids = np.concatenate(trans_ids)
            type_ids = np.concatenate(type_ids)
            meta[block] = {
                "robot_ids": robot_ids,
                "trans_ids": trans_ids,
                "elem_ids": np.concatenate(elem_ids),
                "onehot": np.eye(len(captures), dtype=np.float32)[robot_ids],
                "same_transition": trans_ids[:, None] == trans_ids[None, :],
                "cross_transition": (trans_ids[:, None] != trans_ids[None, :]).astype(np.float32),
                "same_robot": robot_ids[:, None] == robot_ids[None, :],
                "element_level": element_level,
                "type_ids": type_ids,
                "type_onehot": np.eye(n_types, dtype=np.float32)[type_ids],
                "type_robot": np.concatenate(type_robot),   # robot of every unit type
                "type_elem": np.concatenate(type_elem),     # element (joint or foot) index of every unit type
            }
        return meta

    def _combine(self, captures, block):
        """Units of all robots for unit Grams and routing objectives: the first n_gram captured transitions."""
        n = self.n_gram[block]
        layers = []
        for li, (kind, i) in enumerate(self.block_layers[block]):
            x_tail = 1 if (kind == "dense" and i == 0) else 2
            x = jnp.concatenate([_units(c[block]["x"][li][:n], x_tail) for c in captures])
            d = jnp.concatenate([_units(c[block]["d"][li][:n], 2) for c in captures])
            layers.append((kind, x, d))
        p = jnp.concatenate([_units(c[block]["p"][:n], 1) for c in captures])
        morph = jnp.concatenate([_units(c[block]["morph"][:n], 1) for c in captures])
        state = jnp.concatenate([_units(c[block]["state"][:n], 1) for c in captures])
        return layers, p, morph, state

    def _type_grad_sums(self, captures, block):
        """Per-type gradient contributions to the block's experts from every captured transition, (K, T, P)."""
        return jnp.concatenate([
            itf.type_grad_sums([(kind, c[block]["x"][li], c[block]["d"][li])
                                for li, (kind, _) in enumerate(self.block_layers[block])])
            for c in captures], axis=1)

    # --------------------------------------------------------- router objectives
    def _aux_loss(self, params, router_inputs, q_by_block, lam):
        """Conflict and utilization objectives on the captured units (first actor step of an update)."""
        total = jnp.zeros(())
        metrics = {}
        for block in self.aux_blocks:
            morph, state = router_inputs[block]
            router = Router(self.router_experts[block], self.router_hidden, self.router_inputs)
            logits = router.apply({"params": params["params"][block]["router"]}, morph, state)
            p_tilde = jax.nn.softmax(logits, axis=-1)
            loss_use = itf.utilization_loss(p_tilde)
            if q_by_block is None or self.mode == "none" or block not in self.conf_blocks:
                loss_conf = jnp.zeros(())
            elif self.mode == "expert_mean":
                loss_conf = itf.expert_mean_loss(p_tilde, q_by_block[block], self.conflict_norm)
            else:
                loss_conf = itf.conflict_loss(p_tilde, q_by_block[block], self.conflict_norm)
            total = total + lam * loss_conf + self.eta * loss_use
            metrics[f"morse/{block}/loss_conf"] = loss_conf
            metrics[f"morse/{block}/loss_use"] = loss_use
        return total, metrics

    def _mi_loss(self, params, routing):
        """Router-only -I(group; expert) summed over the MI blocks (Mod-Squad's MI) with each block's weight, from the
        routing records of the step's minibatch: each block's router re-applied to the recorded inputs of every
        transition of every robot. A group's routing is its mean soft routing over those transitions (``_mi_groups``
        holds the prior). A balance block takes only -H(E), the load-balancing half of -I = H(E|group) - H(E)."""
        total, metrics = jnp.zeros(()), {}
        for block in self.mi_blocks:
            router = Router(self.router_experts[block], self.router_hidden, self.router_inputs)
            rp = {"params": params["params"][block]["router"]}
            means = []
            for rec in routing:
                # The recorded routing noise (zero without noise) is held fixed: no gradient reaches its scale.
                logits = router.apply(rp, rec[block]["morph"], rec[block]["state"]) + rec[block]["offset"]
                p = jax.nn.softmax(logits, axis=-1)
                p = p.reshape((p.shape[0], -1, p.shape[-1]))                  # (transitions, elements or 1, K)
                means.append(p.mean(axis=0) if self.mi_target == "type" else p.mean(axis=(0, 1))[None])
            r = jnp.concatenate(means)
            if self._mi_agg.get(block) is not None:                          # the core's EG groups
                r = jnp.matmul(self._mi_agg[block], r, precision=itf.HIGHEST)
            neg_mi, h_cond, h_marg = itf.group_neg_mutual_information(r, self._mi_groups[block][1])
            term = -h_marg if block in self.mi_balance_blocks else neg_mi
            total = total + self.mi_block_weight[block] * term
            metrics[f"morse/{block}/mi"] = -neg_mi
            metrics[f"morse/{block}/h_expert_given_type"] = h_cond
            metrics[f"morse/{block}/h_expert"] = h_marg
        return total, metrics

    def _mi_groups_of(self, shapes):
        """(robot of every MI group, group prior) of a block from the shapes of its routing weights per robot,
        (transitions[, elements], K): joint types (target "type" with an element axis), else one group per robot."""
        robot_of, units = [], []
        for r, shape in enumerate(shapes):
            n, n_elem = shape[0], (shape[1] if len(shape) == 3 else 1)
            n_groups = n_elem if self.mi_target == "type" else 1
            robot_of.append(np.full(n_groups, r))
            units.append(np.full(n_groups, n * n_elem / n_groups))
        robot_of, units = np.concatenate(robot_of), np.concatenate(units)
        if self.mi_weighting == "robot":
            prior = 1.0 / (len(shapes) * np.bincount(robot_of)[robot_of])
        else:
            prior = units / units.sum()
        return robot_of, prior.astype(np.float32)

    def _eg_groups(self, block, shapes):
        """The core's EG groups (morse_mi_core_groups=eg) from its one-group-per-robot MI groups: the routing of a group
        is the prior-weighted mean of its robots' routing and its prior their summed prior. Returns (group of every
        group, group prior) and the (groups, robots) averaging matrix, or None for every other block."""
        robot_of, prior = self._mi_groups_of(shapes)
        if block != "core" or self.core_group_of is None:
            return (robot_of, prior), None
        if len(robot_of) != len(shapes):
            raise ValueError("EG groups in the core's MI need one MI group per robot (body-routed core, target type)")
        n_groups = int(self.core_group_of.max()) + 1
        member = np.eye(n_groups, dtype=np.float32)[self.core_group_of].T * prior[None]       # (groups, robots)
        group_prior = member.sum(axis=1)
        return (np.arange(n_groups), group_prior.astype(np.float32)), (member / group_prior[:, None]).astype(np.float32)

    def _routing_diagnostics(self, task_grads, routing):
        """The mean noise scale of every noisy router; for a core with shared experts also every branch's weighted output
        norm (global, group, routed mixture, and the output), the task gradient's norm on the global, group and routed
        experts (and on each group expert), and the routed experts' effective number and utilization (their
        renormalized top-k weights). Norms describe sizes, not causal contributions."""
        metrics = {}
        for block in self.routed_blocks:
            if "noise_scale" in routing[0][block]:
                metrics[f"morse/{block}/noise_scale"] = jnp.mean(jnp.concatenate(
                    [rec[block]["noise_scale"].reshape(-1) for rec in routing]))
        n_global, n_group = self.core_shared
        if "core" not in self.routed_blocks or not (n_global or n_group):
            return metrics
        norms = jnp.concatenate([rec["core"]["branch_norm"].reshape(-1, 4) for rec in routing])
        for i, name in enumerate(("global", "group", "routed", "out")):
            metrics[f"morse/core/branch_norm_{name}"] = norms[:, i].mean()
        p = jnp.concatenate([rec["core"]["p"].reshape(-1, rec["core"]["p"].shape[-1]) for rec in routing])
        entropy = -jnp.sum(jnp.where(p > 0, p * jnp.log(jnp.where(p > 0, p, 1.0)), 0.0), axis=-1)
        usage = p.mean(axis=0)
        metrics.update({"morse/core/routed_eff_experts": jnp.exp(entropy).mean(),
                        "morse/core/routed_usage_min": usage.min(), "morse/core/routed_usage_max": usage.max()})
        per_expert = 0.0                    # squared task-gradient norm of every core expert, (K,)
        for name, sub in task_grads["params"]["core"].items():
            if name in ("router", "router_noise"):
                continue
            for leaf in jax.tree_util.tree_leaves(sub):
                per_expert = per_expert + jnp.sum(jnp.square(leaf.reshape(leaf.shape[0], -1)), axis=1)
        shared = n_global + n_group
        metrics["morse/core/task_grad_global"] = jnp.sqrt(per_expert[:n_global].sum())
        metrics["morse/core/task_grad_group"] = jnp.sqrt(per_expert[n_global:shared].sum())
        metrics["morse/core/task_grad_routed"] = jnp.sqrt(per_expert[shared:].sum())
        for g in range(n_group):
            metrics[f"morse/core/task_grad_group_g{g}"] = jnp.sqrt(per_expert[n_global + g])
        return metrics

    def _routing_stats(self, routing):
        """Per routed block, (R, K): the share of every robot's routing units whose top-k includes each expert, and
        the mean routing weight of each expert."""
        stats = {}
        for block in self.routed_blocks:
            ps = [rec[block]["p"].reshape((-1, rec[block]["p"].shape[-1])) for rec in routing]
            stats[block] = (jnp.stack([(p > 0).mean(axis=0) for p in ps]), jnp.stack([p.mean(axis=0) for p in ps]))
        return stats

    def _interference(self, params, captures, rng, lam, type_state, type_count, with_snapshot):
        """Conflict evidence, router-only auxiliary gradients, diagnostics, and snapshot arrays."""
        meta = self._meta
        metrics, snaps, q_by_block, router_inputs, new_state = {}, {}, {}, {}, {}
        for block in BLOCKS:
            layers, p, morph, state = self._combine(captures, block)
            g = itf.gram(layers)
            ev = itf.conflict_evidence(g, p, self.zeta, self.kappa, self.calibrated, self.relevance)
            rng, k_shuffle, k_snap = jax.random.split(rng, 3)
            m = meta[block]
            # Type-level contributions accumulated over updates (also a diagnostic for unit evidence); [1] keeps
            # the same-batch sums, which include the noise covariance of units sharing a transition.
            if self.unbiased_types:
                s_main, s_same = itf.type_sums_split(g, m["type_onehot"], m["cross_transition"])
            else:
                s_main = s_same = itf.type_sums(g, m["type_onehot"])
            d = self.type_decay
            prev = type_state[block]
            acc = d * prev[:2] + (1.0 - d) * jnp.stack([s_main, s_same])
            if self.mode == "signed":
                # [2]/[3]: the same sums from even/odd updates only, to estimate the noise of the evidence.
                even = jnp.equal(jnp.mod(type_count, 2.0), 0.0)
                acc_even = jnp.where(even, d * prev[2] + (1.0 - d) * s_main, prev[2])
                acc_odd = jnp.where(even, prev[3], d * prev[3] + (1.0 - d) * s_main)
                acc = jnp.concatenate([acc, acc_even[None], acc_odd[None]])
            new_state[block] = acc
            acc_hat = acc / (1.0 - self.type_decay ** type_count)
            if with_snapshot:
                counts = jnp.maximum(m["type_onehot"].sum(axis=0), 1.0)
                snaps.setdefault(block, {}).update({
                    "type_gram_acc": acc_hat[0],
                    "type_gram_acc_samebatch": acc_hat[1],
                    "type_morph": jnp.matmul(m["type_onehot"].T, morph, precision=itf.HIGHEST) / counts[:, None]})
            if self.mode == "signed":
                if block in self.crossfit_blocks:
                    sev = self._crossfit_evidence(captures, block, type_state, type_count, new_state)
                    metrics[f"morse/{block}/compat_valid_frac"] = sev["valid_frac"]
                else:
                    sev = itf.signed_type_evidence(acc_hat[0], acc[2], acc[3], self.compat_z, self.compat_help)
                q_sup = itf.expand_type_q(sev["q"], m["type_ids"])
                # Is the term active? Routing overlap of flagged harmful / helpful type pairs on the expert where they
                # are flagged, relative to all pairs with evidence (1 = no effect; harmful < 1 < helpful if it works).
                counts = jnp.maximum(m["type_onehot"].sum(axis=0), 1.0)
                r_type = jnp.matmul(m["type_onehot"].T, p, precision=itf.HIGHEST) / counts[:, None]       # (T, K)
                overlap = jnp.einsum("ak,bk->kab", r_type, r_type)
                harm_m, help_m, pair_m = sev["q"] > 0, sev["q"] < 0, sev["pair"]
                base = (overlap * pair_m).sum() / jnp.maximum(pair_m.sum(), 1)
                metrics[f"morse/{block}/compat_overlap_harm"] = (overlap * harm_m).sum() / jnp.maximum(harm_m.sum(), 1) / base
                metrics[f"morse/{block}/compat_overlap_help"] = (overlap * help_m).sum() / jnp.maximum(help_m.sum(), 1) / base
                metrics[f"morse/{block}/compat_sigma"] = sev["sigma"].mean()
                metrics[f"morse/{block}/compat_reliability"] = sev["reliability"].mean()
                metrics[f"morse/{block}/compat_harm_frac"] = sev["harm_frac"]
                metrics[f"morse/{block}/compat_help_frac"] = sev["help_frac"]
                if with_snapshot:
                    snaps.setdefault(block, {})["type_cos"] = sev["cos"]
            elif self.evidence == "type":
                tev = itf.type_evidence(acc_hat[0], self.zeta, self.kappa, self.calibrated)
                q_type = itf.shuffle_units(tev["q"], tev["pair"], k_shuffle) if self.mode == "shuffled" else tev["q"]
                q_sup = itf.expand_type_q(q_type, m["type_ids"])
                n_type_pairs = jnp.maximum(tev["pair"].sum(), 1)
                metrics[f"morse/{block}/type_conflict_frac"] = (tev["q"] > 0).sum() / n_type_pairs
                metrics[f"morse/{block}/type_cos_mean"] = tev["cos_mean"]
                metrics[f"morse/{block}/type_kappa"] = tev["kappa"]
                if with_snapshot:
                    snaps.setdefault(block, {})["type_cos"] = tev["cos"]
            elif self.mode == "shuffled":
                q_sup = itf.shuffle_units(ev["q"], ev["pair"], k_shuffle)
            elif self.mode == "expert_mean":
                q_sup, _ = itf.expert_mean_evidence(g, ev["omega"], ev["valid"], self.zeta, self.kappa, self.calibrated)
            else:
                q_sup = ev["q"]
            q_by_block[block] = jax.lax.stop_gradient(q_sup)
            router_inputs[block] = (morph, state)
            diag = itf.block_diagnostics(g, p, ev, m["same_transition"], m["same_robot"], m["element_level"])
            metrics.update({f"morse/{block}/{k}": v for k, v in diag.items()})
            if with_snapshot:
                snaps.setdefault(block, {}).update(itf.block_snapshot(g, p, ev, layers, m["onehot"], k_snap))
        (_, aux_metrics), aux_grads = jax.value_and_grad(self._aux_loss, has_aux=True)(
            params, router_inputs, q_by_block, lam)
        metrics.update(aux_metrics)
        return aux_grads, metrics, snaps, new_state

    def _crossfit_evidence(self, captures, block, type_state, type_count, new_state):
        """Add this update's per-type gradient sums to its stream (update mod 4) and evaluate the four streams."""
        key = f"{block}_compat"
        prev = type_state[key]                                             # (4, K, T, P)
        i = jnp.mod(type_count - 1.0, 4.0).astype(jnp.int32)
        d = self.compat_decay
        cur = d * jax.lax.dynamic_index_in_dim(prev, i, 0, keepdims=False) + (1.0 - d) * self._type_grad_sums(captures, block)
        streams = jax.lax.dynamic_update_index_in_dim(prev, cur, i, 0)
        new_state[key] = streams
        n_upd = jnp.maximum(jnp.floor((type_count - 1.0 - jnp.arange(4.0)) / 4.0) + 1.0, 0.0)   # updates per stream
        corr = jnp.where(n_upd > 0, 1.0 - d ** n_upd, 1.0)                 # zero-initialized EMA bias correction
        return itf.crossfit_type_evidence(streams / corr[:, None, None, None], self.compat_z, self.compat_help)

    def _pcgrad(self, task_grads, captures, key, n_robots):
        """Replace the task gradient of each PCGrad block's experts by the mean over robots of the PCGrad-combined
        per-type gradients (the per-type sums add up exactly to the robots' gradients when nothing is projected)."""
        params = dict(task_grads["params"])
        metrics = {}
        for block in self.pcgrad_blocks:
            sums = self._type_grad_sums(captures, block)                     # (K, T, P)
            keys = jax.random.split(key, sums.shape[0] + 1)
            key = keys[0]
            proj, frac = jax.vmap(itf.pcgrad)(sums, keys[1:])
            flat = proj.sum(axis=1) / n_robots                              # (K, P)
            orig = sums.sum(axis=1) / n_robots
            blk = {m: dict(v) for m, v in params[block].items()}
            off = 0
            for kind, i in self.block_layers[block]:
                mod = f"dense_{i}" if kind == "dense" else f"ln_{i}"
                for leaf in (("kernel", "bias") if kind == "dense" else ("scale", "bias")):
                    shape = blk[mod][leaf].shape
                    n = int(np.prod(shape[1:]))
                    blk[mod][leaf] = flat[:, off:off + n].reshape(shape).astype(blk[mod][leaf].dtype)
                    off += n
            params[block] = blk
            metrics[f"morse/{block}/pcgrad_projected_frac"] = frac.mean()
            metrics[f"morse/{block}/pcgrad_norm_ratio"] = jnp.linalg.norm(flat) / jnp.maximum(jnp.linalg.norm(orig), 1e-30)
        return {**task_grads, "params": params}, metrics

    def _utilization(self, params, captures):
        """Router-only gradient of the utilization objective alone (steps without conflict evidence)."""
        router_inputs = {}
        for block in self.aux_blocks:
            _, _, morph, state = self._combine(captures, block)
            router_inputs[block] = (morph, state)
        (_, metrics), grads = jax.value_and_grad(self._aux_loss, has_aux=True)(
            params, router_inputs, None, jnp.zeros(()))
        return grads, metrics

    # -------------------------------------------------------------------- update
    def _step_fn(self, kind, observations, n_sub=1, with_compat=False):
        """The compiled update of one kind: none | util (utilization objective only) | interf (conflict
        evidence and diagnostics) | snap (interf plus snapshot arrays) | sub (a further actor step of the same
        update: task gradient only, no accumulators). The MI objective is part of every kind. With n_sub > 1 the
        step uses chunk ``sub`` of n_sub equal chunks of every robot's minibatch. Every robot's pass is traced in.
        with_compat adds the compatibility objective (first actor step of an update)."""
        key = (kind, n_sub, with_compat, tuple((o.shape, o.dtype) for o in observations))
        if self.pcgrad_blocks and self.capture_policy.n_capture_decoder < observations[0].shape[0] // n_sub:
            raise ValueError("joint-level PCGrad needs the decoder to capture every transition of a chunk "
                             f"(morse_compat_transitions >= {observations[0].shape[0] // n_sub})")
        if with_compat and any(self.compat_n[b] > observations[0].shape[0] // n_sub for b in self.compat_blocks):
            raise ValueError(f"compatibility scoring needs at most {observations[0].shape[0] // n_sub} transitions per "
                             f"robot and actor step, got {self.compat_n}")
        if key not in self._step_fns:
            robot_passes = [self._robot_pass(r, _chunk_struct(o, n_sub)) for r, o in enumerate(observations)]
            score_fns = {b: [self._score_pass(b, r, _chunk_struct(o, n_sub)) for r, o in enumerate(observations)]
                         for b in self.compat_blocks} if with_compat else {}

            def fn(policy_state, extra, observations, actions, rng, lam, type_state, type_count, robot_acc, chunk,
                   step_sub, compat_factor):
                if n_sub > 1:
                    observations = tuple(_chunk(o, chunk, n_sub) for o in observations)
                    actions = tuple(_chunk(a, chunk, n_sub) for a in actions)
                routing_keys = self._routing_keys(step_sub, len(robot_passes))
                losses, grads, captures = [], [], []
                for env_type_id, robot_pass in enumerate(robot_passes):
                    rng, sub = jax.random.split(rng)
                    loss, g, cap = robot_pass(policy_state.params, extra, observations[env_type_id],
                                              actions[env_type_id], sub, routing_keys[env_type_id])
                    losses.append(loss)
                    grads.append(g)
                    captures.append(cap)
                captures = tuple(captures)
                routing = tuple(c[ROUTING] for c in captures)
                task_grads = jax.tree_util.tree_map(lambda *gs: sum(gs) / len(gs), *grads)
                pc_metrics = {}
                if self.pcgrad_blocks:
                    rng, k_pc = jax.random.split(rng)
                    task_grads, pc_metrics = self._pcgrad(task_grads, captures, k_pc, len(grads))
                if kind == "sub":
                    robot_grams = {}
                else:
                    robot_grams, robot_acc = _accumulate_robot_grams(robot_acc, tuple(grads), self.robot_decay)
                metrics, snaps, aux_grads = {}, {}, None
                rng, sub = jax.random.split(rng)
                if kind in ("interf", "snap"):
                    aux_grads, metrics, snaps, type_state = self._interference(
                        policy_state.params, captures, sub, lam, type_state, type_count, kind == "snap")
                elif kind == "util":
                    aux_grads, metrics = self._utilization(policy_state.params, captures)

                total = task_grads if aux_grads is None else jax.tree_util.tree_map(jnp.add, task_grads, aux_grads)
                compat_grads = None
                if with_compat:
                    k_compat = jax.random.fold_in(jax.random.fold_in(self.routing_base, step_sub[0]), 0x434F4D)
                    compat_grads, compat_metrics = self._compat(policy_state.params, extra, observations, actions,
                                                                routing_keys, score_fns, k_compat, compat_factor)
                    total = jax.tree_util.tree_map(jnp.add, total, compat_grads)
                    metrics.update(compat_metrics)
                mi_grads, mi_raw, cap_factors = None, {}, {}
                if self.mi_blocks:
                    (_, mi_metrics), mi_grads = jax.value_and_grad(self._mi_loss, has_aux=True)(policy_state.params, routing)
                    mi_raw = {b: _router_norm(mi_grads, b) for b in self.mi_blocks}
                    if self.mi_cap > 0.0:
                        # At most mi_cap times the task gradient's norm on the same router (raw, before the optimizer).
                        cap_factors = {b: jnp.minimum(1.0, self.mi_cap * _router_norm(task_grads, b)
                                                      / jnp.maximum(mi_raw[b], 1e-30)) for b in self.mi_blocks}
                        mi_grads = jax.tree_util.tree_map_with_path(
                            lambda path, g: g * cap_factors[path[1].key] if _is_router(path) and getattr(
                                path[1], "key", None) in cap_factors else g, mi_grads)
                    total = jax.tree_util.tree_map(jnp.add, total, mi_grads)
                if kind in ("interf", "snap"):
                    # Routing diagnostics: MI terms, router gradients of the task and the MI objective per block, and
                    # every robot's expert selection (for the host's routing log).
                    for block in self.routed_blocks:
                        metrics[f"morse/{block}/router_grad_task"] = _router_norm(task_grads, block)
                        if mi_grads is not None and block in self.mi_blocks:
                            metrics[f"morse/{block}/router_grad_mi"] = mi_raw[block]
                            metrics[f"morse/{block}/router_grad_mi_ratio"] = (
                                metrics[f"morse/{block}/router_grad_mi"]
                                / jnp.maximum(metrics[f"morse/{block}/router_grad_task"], 1e-30))
                            if block in cap_factors:
                                metrics[f"morse/{block}/mi_cap_factor"] = cap_factors[block]
                        if compat_grads is not None and block in self.compat_blocks:
                            metrics[f"morse/{block}/router_grad_compat"] = _router_norm(compat_grads, block)
                    if mi_grads is not None:
                        metrics.update(mi_metrics)
                        metrics["morse/router_grad_mi"] = _router_norm(mi_grads)
                    metrics.update(self._routing_diagnostics(task_grads, routing))
                    stats = self._routing_stats(routing)
                    snaps = {**snaps, "_routing": stats}
                    for block, (select, weight) in stats.items():
                        for e in range(select.shape[1]):
                            metrics[f"morse/{block}/select_e{e}"] = select[:, e].mean()
                            metrics[f"morse/{block}/weight_e{e}"] = weight[:, e].mean()
                if self.router_only:
                    # Only routers adapt: mask the gradient (so clipping sees router gradients only) and keep
                    # every other parameter fixed even if the restored optimizer state still carries momentum.
                    total = jax.tree_util.tree_map_with_path(lambda path, g: g if _is_router(path) else jnp.zeros_like(g), total)
                    new_state = policy_state.apply_gradients(grads=total)
                    new_state = new_state.replace(params=jax.tree_util.tree_map_with_path(
                        lambda path, new, old: new if _is_router(path) else old, new_state.params, policy_state.params))
                else:
                    new_state = policy_state.apply_gradients(grads=total)
                metrics.update(pc_metrics)
                metrics.update(self._variance_metrics(total, policy_state.params, new_state.params))
                metrics["morse/router_grad_task"] = _router_norm(task_grads)
                if aux_grads is not None:
                    metrics["morse/router_grad_aux"] = _router_norm(aux_grads)
                return (new_state, jnp.mean(jnp.stack(losses)), metrics, snaps, type_state, robot_acc, robot_grams, rng,
                        cap_factors)

            self._step_fns[key] = jax.jit(fn)
        return self._step_fns[key]

    def _variance_metrics(self, grads, old_params, new_params):
        """Gradient norms before the optimizer's clip (all parameters, the log-std parameters, the descriptor
        embedding), the clip factors, and the norms of the parameter updates actually applied."""
        g_all = optax.global_norm(grads)
        out = {"morse/var/grad_norm": g_all, "morse/var/grad_norm_std": group_norm(grads, is_std_param),
               "morse/var/grad_norm_embedding": group_norm(grads, is_embedding)}
        if self.max_grad_norm > 0.0:
            clip = lambda n: jnp.minimum(1.0, self.max_grad_norm / jnp.maximum(n, 1e-30))
            if self.variance_mode == "protected":      # the log-std head is clipped on its own
                out["morse/var/clip_factor"] = clip(group_norm(grads, lambda p: not is_std_head(p)))
                out["morse/var/clip_factor_std"] = clip(group_norm(grads, is_std_head))
            elif self.freeze_core:                     # the clip sees the trainable gradient only; the other factor is S0's
                out["morse/var/clip_factor"] = clip(group_norm(grads, lambda p: not is_core(p)))
                out["morse/var/clip_factor_with_core"] = clip(g_all)
            else:
                out["morse/var/clip_factor"] = clip(g_all)
        delta = jax.tree_util.tree_map(jnp.subtract, new_params, old_params)
        out["morse/var/update_norm"] = optax.global_norm(delta)
        out["morse/var/update_norm_std"] = group_norm(delta, is_std_param)
        out["morse/var/update_norm_embedding"] = group_norm(delta, is_embedding)
        out["morse/var/grad_norm_core"] = group_norm(grads, is_core)
        out["morse/var/update_norm_core"] = group_norm(delta, is_core)
        return out

    def _add_cap(self, factors):
        """Accumulate the actor step's MI cap factors (device arrays, no host transfer)."""
        if not factors:
            return
        step = {b: jnp.stack([f, (f < 1.0).astype(f.dtype), jnp.ones_like(f)]) for b, f in factors.items()}
        self._cap_acc = step if self._cap_acc is None else {b: self._cap_acc[b] + v for b, v in step.items()}

    def _cap_metrics(self):
        """Mean MI cap factor and the share of actor steps in which the cap acted, since the last diagnostic step."""
        if self._cap_acc is None:
            return {}
        out = {}
        for block, (total, active, count) in jax.device_get(self._cap_acc).items():
            out[f"morse/{block}/mi_cap_factor_mean"] = float(total / count)
            out[f"morse/{block}/mi_cap_active_frac"] = float(active / count)
        self._cap_acc = None
        return out

    def _init_state(self, policy_state, extra, env_batches, rng, n_sub=1):
        """Unit metadata (from the capture shapes) and zero accumulators, before the first update."""
        captures = []
        for env_type_id, batch in enumerate(env_batches):
            obs, act = _chunk_struct(batch.observations, n_sub), _chunk_struct(batch.actions, n_sub)
            robot_pass = self._robot_pass(env_type_id, obs)
            captures.append(jax.eval_shape(robot_pass, policy_state.params, extra, obs, act, rng)[2])
        self._meta = self._build_meta(captures)
        n_acc = 4 if self.mode == "signed" else 2
        self._type_state = {b: jnp.zeros((n_acc, self.num_experts[b]) + (m["type_onehot"].shape[1],) * 2)
                            for b, m in self._meta.items()}
        for b in self.mi_blocks:
            self._mi_groups[b], self._mi_agg[b] = self._eg_groups(b, [c[ROUTING][b]["p"].shape for c in captures])
        for block in self.crossfit_blocks:                                 # four streams of per-type gradient sums
            shape = jax.eval_shape(lambda caps: self._type_grad_sums(caps, block), tuple(captures)).shape
            self._type_state[f"{block}_compat"] = jnp.zeros((4,) + tuple(shape))
        self._robot_acc = {b: jnp.zeros((len(env_batches), len(env_batches))) for b in ROBOT_LEVELS}
        logging.getLogger("rl_x").info(f"MORSE experts per block {self.num_experts}; "
                                        f"policy parameters {count_params(policy_state.params)}; "
                                        f"captured units per block { {b: len(m['robot_ids']) for b, m in self._meta.items()} }")
        if sum(self.core_shared):
            cats = count_core_categories(policy_state.params, *self.core_shared)
            self._param_counts = {f"morse/core/params_{k}": float(v) for k, v in cats.items()}
            logging.getLogger("rl_x").info(f"MORSE core experts: {self.core_shared[0]} global, {self.core_shared[1]} "
                                            f"group, {self.router_experts['core']} routed; parameters {cats}; EG groups "
                                            f"{dict(zip(self.robot_names, self.capture_policy.core_group_of))}")

    def update(self, policy_state, extra, env_batches, rng, step, sub=0, n_sub=1):
        """One MORSE policy step on chunk ``sub`` of n_sub chunks of the minibatches of update ``step``.
        Returns (policy_state, loss, metrics, rng); metrics only on diagnostic steps. The MI objective is part of
        every chunk's step; the conflict and utilization objectives, evidence, accumulators, and diagnostics belong to
        the first chunk."""
        t0 = time.time()
        if self._meta is None:
            self._init_state(policy_state, extra, env_batches, rng, n_sub)
        observations = tuple(batch.observations for batch in env_batches)
        actions = tuple(batch.actions for batch in env_batches)
        step_sub = jnp.asarray([step, sub], jnp.uint32)
        if sub > 0:
            policy_state, loss, _, _, _, _, _, rng, cap = self._step_fn("sub", observations, n_sub)(
                policy_state, extra, observations, actions, rng, jnp.float32(0.0), self._type_state,
                jnp.float32(self._type_count), self._robot_acc, jnp.int32(sub), step_sub, jnp.float32(0.0))
            self._add_cap(cap)
            return policy_state, loss, {}, rng
        lam = self.lambda_at(step)
        # Signed compatibility accumulates its evidence every update from the start (the loss waits for lam > 0).
        conflict_step = (lam > 0.0 or self.mode == "signed") and step % self.conflict_every == 0
        diag_step = self.diag_frequency > 0 and step % self.diag_frequency == 0
        snap_step = self.snapshot_dir is not None and self.snapshot_frequency > 0 and step % self.snapshot_frequency == 0
        if conflict_step or diag_step or snap_step:
            kind = "snap" if snap_step else "interf"
            self._type_count += 1
        elif self.eta > 0.0 and self.aux_blocks:
            kind = "util"
        else:
            kind = "none"
        self._robot_count += 1
        # Compatibility routing on the first actor step: after the warm-up, and on diagnostic steps for its labels.
        compat_factor = self.compat_factor_at(step)
        with_compat = bool(self.compat_blocks) and (compat_factor > 0.0 or diag_step)

        policy_state, loss, metrics, snaps, type_state, self._robot_acc, robot_grams, rng, cap = self._step_fn(
            kind, observations, n_sub, with_compat)(
            policy_state, extra, observations, actions, rng, jnp.float32(lam if conflict_step else 0.0),
            self._type_state, jnp.float32(self._type_count), self._robot_acc, jnp.int32(0), step_sub,
            jnp.float32(compat_factor))
        self._add_cap(cap)
        if kind in ("interf", "snap"):
            self._type_state = type_state
            snaps = dict(snaps)
            routing_stats = snaps.pop("_routing")
            if diag_step or snap_step:
                self._log_routing(step, routing_stats)
        if diag_step or snap_step:
            robot_metrics, robot_snaps = self._robot_diagnostics(robot_grams)
            metrics = {**metrics, **robot_metrics}
            if snap_step:
                self._save_snapshot(step, snaps, metrics, robot_snaps)
        if not diag_step:
            return policy_state, loss, {}, rng
        jax.block_until_ready(policy_state.params)
        out = {k: float(v) for k, v in jax.device_get(metrics).items()}
        out.update(self._cap_metrics())
        out.update(self._param_counts or {})
        out["morse/lambda"] = lam
        now = time.time()
        out["morse/update_s"] = now - t0
        # End-to-end cost: wall time per update since the last diagnostic step (critics, actor steps, evaluations and
        # logging included), and the device's peak memory so far.
        if self._last_diag is not None and step > self._last_diag[1]:
            out["morse/wall_per_update_s"] = (now - self._last_diag[0]) / (step - self._last_diag[1])
        self._last_diag = (now, step)
        try:
            peak = (jax.local_devices()[0].memory_stats() or {}).get("peak_bytes_in_use")
        except Exception:                                                   # no memory statistics on this backend
            peak = None
        if peak is not None:
            out["morse/peak_mem_gb"] = peak / 2 ** 30
        return policy_state, loss, out, rng

    def _log_routing(self, step, stats):
        """Append every robot's expert selection share and mean routing weight per block to the routing log, rewritten
        as ``morse_diagnostics/routing.npz`` (arrays <block>_select and <block>_weight of shape (steps, robots, K))."""
        stats = jax.device_get(stats)
        if self._routing_log is None:
            self._routing_log = {"step": []}
        self._routing_log["step"].append(step)
        for block, (select, weight) in stats.items():
            self._routing_log.setdefault(f"{block}_select", []).append(np.asarray(select))
            self._routing_log.setdefault(f"{block}_weight", []).append(np.asarray(weight))
        if self.snapshot_dir is None:
            return
        os.makedirs(self.snapshot_dir, exist_ok=True)
        path = os.path.join(self.snapshot_dir, "routing.npz")
        np.savez(path + ".tmp.npz", robot_names=np.array(self.robot_names),
                 **{k: np.asarray(v) for k, v in self._routing_log.items()})
        os.replace(path + ".tmp.npz", path)

    def _robot_diagnostics(self, robot_grams):
        """Robot-level statistics of the current update and of the accumulation over updates."""
        acc = {b: self._robot_acc[b] / (1.0 - self.robot_decay ** self._robot_count) for b in ROBOT_LEVELS}
        metrics, snaps = {}, {}
        for block in ROBOT_LEVELS:
            now, avg = itf.robot_level_stats(robot_grams[block]), itf.robot_level_stats(acc[block])
            for name in ("cos_mean", "neg_frac", "cancelled_fraction"):
                metrics[f"morse/{block}/robot_{name}"] = now[name]
                metrics[f"morse/{block}/robot_{name}_acc"] = avg[name]
            snaps[block] = {"robot_cos": now["cos"], "robot_cos_acc": avg["cos"], "robot_gram_acc": acc[block]}
        return metrics, snaps

    def _save_snapshot(self, step, snaps, metrics, robot_snaps):
        os.makedirs(self.snapshot_dir, exist_ok=True)
        arrays = {"robot_names": np.array(self.robot_names), "step": np.array(step),
                  "blocks": np.array(BLOCKS), "num_experts": np.array([self.num_experts[b] for b in BLOCKS]),
                  "unbiased_types": np.array(self.unbiased_types)}
        for block, snap in jax.device_get(snaps).items():
            for k, v in {**snap, **jax.device_get(robot_snaps[block])}.items():
                arrays[f"{block}__{k}"] = np.asarray(v)
            for k in ("robot_ids", "trans_ids", "elem_ids", "type_robot", "type_elem"):
                arrays[f"{block}__{k}"] = self._meta[block][k]
        for k, v in jax.device_get(robot_snaps["actor"]).items():
            arrays[f"actor__{k}"] = np.asarray(v)
        for k, v in jax.device_get(metrics).items():
            arrays["scalar__" + k.replace("/", ".")] = np.asarray(v)
        np.savez_compressed(os.path.join(self.snapshot_dir, f"step_{step:07d}.npz"), **arrays)
