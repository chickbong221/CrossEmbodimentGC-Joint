"""URMA policy whose local encoders, actor core, and action decoder are MoE blocks.

The block structure mirrors ``uni_iql_eg/iql/policy.py``. A block with one expert
is the original shared block (its routing weight is always 1), so
``num_experts=1`` reproduces URMA while still exposing the MORSE diagnostics.
"""
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
import flax.linen as nn
from flax.linen.initializers import constant, lecun_normal, orthogonal

from cross_embodiment_offline_rl.algorithms.morse.moe import MoEBlock
from cross_embodiment_offline_rl.algorithms.morse.variance import MODES as VARIANCE_MODES
from cross_embodiment_offline_rl.algorithms.utils import create_robot_groups, resolve_data_path


BLOCKS = ("joint_encoder", "foot_encoder", "core", "decoder")
JOINT_LEVEL_BLOCKS = ("joint_encoder", "decoder")
FOOT_LEVEL_BLOCKS = ("foot_encoder",)
WHOLE_BODY_BLOCKS = ("core",)
CORE_SHARED_MODES = ("none", "global", "group", "hierarchical")


def eg_group_table(a, robot_names):
    """Embodiment Grouping group of every robot, in the order of ``robot_names``: Ward clusters of the embodiment
    distances (``a.embodiment_distance_path``) into ``a.morse_eg_groups`` groups, as uni_iql_eg forms its M groups.
    Fails unless the distance matrix has exactly one row per robot and every group has a robot."""
    n_groups = int(getattr(a, "morse_eg_groups", 4))
    if n_groups < 2:
        raise ValueError(f"EG groups need morse_eg_groups >= 2, got {n_groups}")
    path = getattr(a, "embodiment_distance_path", "")
    if not path:
        raise ValueError("EG groups need algorithm.embodiment_distance_path")
    names = [str(n) for n in robot_names]
    shape = np.load(resolve_data_path(path)).shape
    if shape != (len(names), len(names)) or len(set(names)) != len(names):
        raise ValueError(f"EG groups: {path} has shape {shape}, the training robots are {len(names)} names "
                         f"({len(set(names))} distinct)")
    groups = create_robot_groups(path, names, n_groups)
    group_of = {name: g for g, members in enumerate(groups.values()) for name in members}
    table = tuple(int(group_of[name]) for name in names)
    if sorted(set(table)) != list(range(n_groups)):
        raise ValueError(f"EG groups: {n_groups} groups requested, robots fall into {sorted(set(table))}")
    return table


def parse_blocks(spec: str) -> Tuple[str, ...]:
    blocks = tuple(b.strip() for b in spec.split(",") if b.strip())
    unknown = set(blocks) - set(BLOCKS)
    if unknown:
        raise ValueError(f"Unknown MORSE blocks {sorted(unknown)}; choose from {BLOCKS}")
    return blocks


def load_routing_table(path):
    """Fixed decoder routing weights (robots, joints, experts) as a hashable nested tuple, or () if unset."""
    if not path:
        return ()
    table = np.asarray(np.load(path), dtype=np.float32)
    if table.ndim != 3 or not np.allclose(table.sum(-1), 1.0, atol=1e-5):
        raise ValueError(f"routing table {path}: expected (robots, joints, experts) rows summing to 1, got {table.shape}")
    return tuple(tuple(tuple(float(v) for v in row) for row in robot) for robot in table)


def get_morse_policy(config, capture=False, deterministic=False):
    a = config.algorithm
    table = load_routing_table(getattr(a, "morse_decoder_table", ""))
    n_decoder = int(getattr(a, "morse_num_experts_decoder", 0)) or a.morse_num_experts
    if table and len(table[0][0]) != n_decoder:
        raise ValueError(f"routing table has {len(table[0][0])} experts, the decoder has {n_decoder}")
    core_routing = getattr(a, "morse_core_routing", "body")
    if core_routing not in ("body", "joint"):
        raise ValueError(f"Unknown morse_core_routing {core_routing}")
    fixed_robot = parse_blocks(getattr(a, "morse_fixed_robot_blocks", "") or "")
    n_robots = len(config.environment.train_robot_types)
    for block in fixed_robot:
        n_block = n_decoder if block == "decoder" else a.morse_num_experts
        if block not in parse_blocks(a.morse_blocks) or block not in ("core", "decoder") or n_block != n_robots:
            raise ValueError(f"robot-fixed routing of {block} needs one {block} expert per training robot "
                             f"({n_robots}), got {n_block}")
        if block == "core" and core_routing != "body":
            raise ValueError("robot-fixed routing of the core needs morse_core_routing=body")
    if getattr(a, "morse_topk_ties", "exact") not in ("exact", "keep"):
        raise ValueError(f"Unknown morse_topk_ties {a.morse_topk_ties}")
    # Shared core experts: the EG group of every training robot (group experts), checked against the router's pool.
    core_shared = getattr(a, "morse_core_shared", "none")
    if core_shared not in CORE_SHARED_MODES:
        raise ValueError(f"Unknown morse_core_shared {core_shared}; choose from {CORE_SHARED_MODES}")
    core_group_of = ()
    if core_shared != "none":
        if "core" not in parse_blocks(a.morse_blocks) or core_routing != "body" or "core" in fixed_robot:
            raise ValueError("shared core experts need a routed body-level core (morse_blocks with core, "
                             "morse_core_routing=body, no robot-fixed core)")
        if getattr(a, "morse_compat_route", "off") != "off":
            raise ValueError("compatibility routing is not defined with shared core experts")
        if core_shared in ("group", "hierarchical"):
            core_group_of = eg_group_table(a, config.environment.train_robot_types)
        n_shared = int(core_shared in ("global", "hierarchical")) + (max(core_group_of) + 1 if core_group_of else 0)
        n_routed = a.morse_num_experts - n_shared
        topk = int(getattr(a, "morse_topk_core", 0))
        if n_routed < max(2, topk):
            raise ValueError(f"shared core experts: {n_routed} routed experts (of {a.morse_num_experts}) cannot support "
                             f"top-{max(2, topk)} routing")
        coef = float(getattr(a, "morse_core_shared_coef", 0.25))
        n_branches = 1 + int(core_shared == "hierarchical")
        if not 0.0 < coef or not n_branches * coef < 1.0:
            raise ValueError(f"morse_core_shared_coef {coef}: every shared branch needs a positive weight and the "
                             f"routed mixture the rest")
    spec = str(getattr(a, "morse_joint_encoder_hidden", "") or "").strip().lower()
    joint_hidden = () if spec in ("", "none") else tuple(int(w) for w in spec.split(",") if w.strip())
    core_depth = int(getattr(a, "morse_core_depth", 3))
    if core_depth < 2 or any(w <= 0 for w in joint_hidden):
        raise ValueError(f"morse_core_depth {core_depth} must be >= 2 and morse_joint_encoder_hidden "
                         f"{getattr(a, 'morse_joint_encoder_hidden', '')!r} positive widths")
    variance_mode = str(getattr(a, "morse_variance_mode", "original"))
    if variance_mode not in VARIANCE_MODES:
        raise ValueError(f"Unknown morse_variance_mode {variance_mode}")
    return MorsePolicy(
        std_dev=a.std_dev,
        softmax_temperature=a.softmax_temperature,
        softmax_temperature_min=a.softmax_temperature_min,
        stability_epsilon=a.stability_epsilon,
        policy_mean_abs_clip=a.policy_mean_abs_clip,
        policy_std_min_clip=a.policy_std_min_clip,
        policy_std_max_clip=a.policy_std_max_clip,
        deterministic=deterministic,
        max_action=a.max_action if deterministic else 1.0,
        moe_blocks=parse_blocks(a.morse_blocks),
        num_experts=a.morse_num_experts,
        num_experts_decoder=int(getattr(a, "morse_num_experts_decoder", 0)),
        topk_decoder=int(getattr(a, "morse_topk_decoder", 0)),
        topk_core=int(getattr(a, "morse_topk_core", 0)),
        core_routing=core_routing,
        expert_hidden=int(getattr(a, "morse_expert_hidden", 0)),
        router_init=float(getattr(a, "morse_router_init", 0.0)),
        router_hidden=a.morse_router_hidden,
        router_inputs=a.morse_router_inputs,
        detach_router_inputs=a.morse_detach_router_inputs,
        width_mult_core=a.morse_width_mult_core,
        width_mult_decoder=a.morse_width_mult_decoder,
        capture=capture,
        n_capture_element=a.morse_capture_transitions_element,
        n_capture_body=a.morse_capture_transitions_body,
        n_capture_decoder=(int(getattr(a, "morse_compat_transitions", 0))
                           if (getattr(a, "morse_compat_estimator", "products") == "crossfit"
                               or "decoder" in str(getattr(a, "morse_pcgrad_blocks", ""))) else 0),
        eval_topk=0 if capture else a.morse_eval_topk,
        decoder_table=table,
        identical_experts=bool(getattr(a, "morse_identical_experts", False)),
        calibration=bool(getattr(a, "morse_calibration", False)),
        n_robots=n_robots,
        router_noise=bool(getattr(a, "morse_router_noise", False)),
        noise_floor=float(getattr(a, "morse_router_noise_floor", 0.01)),
        topk_ties=getattr(a, "morse_topk_ties", "exact"),
        fixed_robot_blocks=fixed_robot,
        core_shared=core_shared,
        core_shared_coef=float(getattr(a, "morse_core_shared_coef", 0.25)),
        core_group_of=core_group_of,
        joint_encoder_hidden=joint_hidden,
        core_depth=core_depth,
        core_weight_norm=bool(getattr(a, "morse_core_weight_norm", False)),
        variance_mode=variance_mode,
    )


class MorsePolicy(nn.Module):
    std_dev: float
    softmax_temperature: float
    softmax_temperature_min: float
    stability_epsilon: float
    policy_mean_abs_clip: float
    policy_std_min_clip: float
    policy_std_max_clip: float
    deterministic: bool = False
    max_action: float = 1.0
    moe_blocks: Tuple[str, ...] = BLOCKS
    num_experts: int = 4
    num_experts_decoder: int = 0  # > 0: experts of the decoder (the other routed blocks keep num_experts)
    topk_decoder: int = 0         # > 0: the decoder routes every joint to its top-k experts, in training and evaluation
    topk_core: int = 0            # > 0: the same for the core (per joint with joint routing, else per transition)
    core_routing: str = "body"    # body: one core mixture per transition, broadcast to every joint | joint: every joint
                                  # mixes the whole-body core experts with its own routing weights (router inputs: the
                                  # joint's descriptor, its local state, and the whole-body core input)
    expert_hidden: int = 0        # > 0: width of every hidden layer of the core and decoder experts (core H -> H -> 128,
                                  # decoder H -> 1), instead of the width multipliers
    router_init: float = 0.0      # std of the routers' random output layers (0: zero layers)
    router_hidden: int = 64
    router_inputs: str = "both"
    detach_router_inputs: bool = False
    width_mult_core: float = 1.0
    width_mult_decoder: float = 1.0
    capture: bool = False
    n_capture_element: int = 8
    n_capture_body: int = 64
    n_capture_decoder: int = 0    # > 0: the decoder captures this many transitions (cross-fitted compatibility)
    eval_topk: int = 0
    # Analysis controls (Tests 1-2). ``decoder_table``: fixed decoder routing weights per (robot, joint), a
    # nested tuple (robots, joints, experts), empty for a learned router. ``calibration``: a per-joint output
    # scale and bias, mean = scale[r, j] * f + bias[r, j], initialized to 1 and 0 (the initial policy is
    # unchanged). Both need the robot index (``robot_id``), which every training and evaluation call passes.
    decoder_table: tuple = ()
    identical_experts: bool = False
    calibration: bool = False
    n_robots: int = 0
    max_joints: int = 32
    router_noise: bool = False    # learned routing noise in training (moe.MoEBlock)
    noise_floor: float = 0.01
    topk_ties: str = "exact"
    fixed_robot_blocks: Tuple[str, ...] = ()   # blocks routed one-hot by robot: expert r serves robot r only
    score_block: str = ""         # the block whose experts take the compatibility scoring pass (moe.MoEBlock.score_mode)
    core_shared: str = "none"     # always-on core experts: none | global | group (EG groups) | hierarchical (both)
    core_shared_coef: float = 0.25    # fixed weight of each always-on core branch
    core_group_of: Tuple[int, ...] = ()   # EG group of every training robot (group experts), by robot index
    joint_encoder_hidden: Tuple[int, ...] = ()   # hidden widths of the joint-state encoder (none: URMA's 4-unit layer)
    core_depth: int = 3                   # dense layers of every core expert, the last the 128-wide output
    core_weight_norm: bool = False        # WeightNorm on every dense layer of the core
    variance_mode: str = "original"       # log-std pathway (morse/variance.py): original | detach | private | protected

    def _joint_encoder_spec(self):
        """Widths, post-layer operations and initializers of the joint-state encoder f_psi: the hidden layers (the first
        with LayerNorm), then the 4-feature ELU output per joint that the attention aggregates."""
        hidden = tuple(self.joint_encoder_hidden)
        posts = (("ln_elu",) + ("elu",) * (len(hidden) - 1) if hidden else ()) + ("elu",)
        return hidden + (4,), posts, (lecun_normal(),) * (len(hidden) + 1)

    def _core_spec(self):
        """Widths, post-layer operations and initializers of every core expert: core_depth dense layers, the last the
        128-wide output; hidden widths 512 for the first half and 256 for the rest (URMA's 3 layers: 512, 256), scaled by
        width_mult_core, or all expert_hidden."""
        n_hidden = self.core_depth - 1
        if self.expert_hidden > 0:
            hidden = (self.expert_hidden,) * n_hidden
        else:
            n_wide = (n_hidden + 1) // 2
            hidden = tuple(int(w * self.width_mult_core) for w in (512,) * n_wide + (256,) * (n_hidden - n_wide))
        posts = ("ln_elu",) + ("elu",) * (n_hidden - 1) + ("none",)
        return hidden + (128,), posts, (orthogonal(np.sqrt(2)),) * self.core_depth

    def block_posts(self, block):
        """Post-layer operation of every dense layer of a block (moe.MoEBlock.layer_post)."""
        return {"joint_encoder": self._joint_encoder_spec()[1], "foot_encoder": ("elu",), "core": self._core_spec()[1],
                "decoder": ("ln_elu", "none")}[block]

    def block_layers(self, block):
        """Trainable layers of a block in forward order as (kind, index): every dense layer, followed by its LayerNorm
        if it has one (the capture layout of the updater)."""
        layers = []
        for i, post in enumerate(self.block_posts(block)):
            layers.append(("dense", i))
            if post == "ln_elu":
                layers.append(("ln", i))
        return tuple(layers)

    def num_experts_of(self, block):
        if block not in self.moe_blocks:
            return 1
        if block == "decoder" and self.num_experts_decoder > 0:
            return self.num_experts_decoder
        return self.num_experts

    def core_shared_counts(self):
        """(global, group) always-on experts of the core: the first experts of its expert axis, in that order."""
        if self.core_shared == "none" or "core" not in self.moe_blocks:
            return 0, 0
        if self.core_shared not in CORE_SHARED_MODES:
            raise ValueError(f"Unknown core_shared {self.core_shared}")
        n_group = 0
        if self.core_shared in ("group", "hierarchical"):
            if not self.core_group_of:
                raise ValueError("group experts need the EG group of every training robot (core_group_of)")
            n_group = max(self.core_group_of) + 1
        return int(self.core_shared in ("global", "hierarchical")), n_group

    def router_experts_of(self, block):
        """Experts a block's router chooses among: all of them, minus the core's always-on shared experts."""
        n = self.num_experts_of(block)
        return n - sum(self.core_shared_counts()) if block == "core" else n

    def _n_capture(self, block):
        if block == "decoder" and self.n_capture_decoder > 0:
            return self.n_capture_decoder
        return self.n_capture_body if block in WHOLE_BODY_BLOCKS else self.n_capture_element

    def _moe(self, block, features, posts, inits):
        n_global, n_group = self.core_shared_counts() if block == "core" else (0, 0)
        return MoEBlock(
            layer_features=tuple(features), layer_post=tuple(posts), kernel_inits=tuple(inits),
            num_experts=self.num_experts_of(block), router_hidden=self.router_hidden,
            router_inputs=self.router_inputs, detach_router_inputs=self.detach_router_inputs,
            capture=self.capture,
            n_capture=self._n_capture(block),
            eval_topk=self.eval_topk,
            train_topk={"decoder": self.topk_decoder, "core": self.topk_core}.get(block, 0),
            fixed_routing=bool((block == "decoder" and self.decoder_table) or block in self.fixed_robot_blocks),
            identical_init=self.identical_experts,
            router_init=self.router_init,
            joint_routing=(block == "core" and self.core_routing == "joint"),
            router_noise=self.router_noise,
            noise_floor=self.noise_floor,
            topk_ties=self.topk_ties,
            score_mode=(block == self.score_block),
            n_global=n_global,
            n_group=n_group,
            shared_coef=self.core_shared_coef,
            weight_norm=(block == "core" and self.core_weight_norm),
            name=block,
        )

    def _robot_p(self, block, robot_id):
        """One-hot routing of a robot-fixed block (None at init or without a robot index)."""
        if block not in self.fixed_robot_blocks or robot_id is None:
            return None
        return jax.nn.one_hot(robot_id, self.num_experts_of(block))

    def _group_p(self, robot_id):
        """One-hot EG group of the robot for the core's group experts (None without group experts or robot index; the
        block then fails outside init). A concrete index outside the training robots fails; a traced one (the jitted
        evaluation policy, whose indices come from the training list) is looked up on the device."""
        _, n_group = self.core_shared_counts()
        if n_group == 0 or robot_id is None:
            return None
        if isinstance(robot_id, (int, np.integer)):
            if not 0 <= int(robot_id) < len(self.core_group_of):
                raise ValueError(f"robot index {robot_id} has no EG group ({len(self.core_group_of)} training robots)")
            return jax.nn.one_hot(self.core_group_of[int(robot_id)], n_group)
        group = jnp.take(jnp.asarray(self.core_group_of), robot_id, mode="fill", fill_value=-1)
        return jax.nn.one_hot(group, n_group)

    def _aggregate(self, name, description, latent_state, slots):
        """Descriptor-conditioned aggregation of URMA (shared, unchanged)."""
        eps = self.stability_epsilon
        log_temperature = self.param(f"{name}_log_softmax_temperature",
                                     constant(jnp.log(self.softmax_temperature - self.softmax_temperature_min)), (1,))
        mask = nn.Dense(slots)(description)
        mask = nn.LayerNorm()(mask)
        mask = nn.elu(mask)
        mask = nn.Dense(slots)(mask)
        mask = jnp.clip(nn.tanh(mask), -1.0 + eps, 1.0 - eps)
        e_x = jnp.exp(mask / (jnp.exp(log_temperature) + self.softmax_temperature_min))
        mask = e_x / (e_x.sum(axis=-1, keepdims=True) + eps)
        mask = jnp.repeat(jnp.expand_dims(mask, axis=-1), latent_state.shape[-1], axis=-1)
        masked = mask * jnp.expand_dims(latent_state, axis=-2)
        masked = jnp.reshape(masked, masked.shape[:-2] + (masked.shape[-2] * masked.shape[-1],))
        return jnp.sum(masked, axis=-2)

    @nn.compact
    def __call__(self, dynamic_joint_description, dynamic_joint_state, dynamic_foot_description, dynamic_foot_state, general_state,
                 robot_id=None):
        jd, js = dynamic_joint_description, dynamic_joint_state
        fd, fs = dynamic_foot_description, dynamic_foot_state
        n_joints, n_feet = js.shape[-2], fs.shape[-2]

        # Morphology features: element descriptor plus a pooled descriptor of the embodiment.
        batch_shape = jd.shape[:-2]
        pooled = jnp.concatenate([
            jd.mean(axis=-2), fd.mean(axis=-2),
            jnp.full(batch_shape + (1,), n_joints / 32.0, jd.dtype),
            jnp.full(batch_shape + (1,), n_feet / 8.0, jd.dtype),
        ], axis=-1)
        pooled_joint = jnp.broadcast_to(pooled[..., None, :], batch_shape + (n_joints, pooled.shape[-1]))
        pooled_foot = jnp.broadcast_to(pooled[..., None, :], batch_shape + (n_feet, pooled.shape[-1]))
        joint_morph = jnp.concatenate([jd, pooled_joint], axis=-1)
        foot_morph = jnp.concatenate([fd, pooled_foot], axis=-1)

        # Local encoders (joint level and end-effector level).
        z_joint = self._moe("joint_encoder", *self._joint_encoder_spec())(js, joint_morph, js)
        z_foot = self._moe("foot_encoder", (4,), self.block_posts("foot_encoder"), (lecun_normal(),))(fs, foot_morph, fs)

        # Descriptor-conditioned aggregation (shared).
        joint_latent = self._aggregate("joint", jd, z_joint, 64)
        foot_latent = self._aggregate("foot", fd, z_foot, 32)

        # Whole-body actor core: its experts compute once per transition; the mixture is per transition (body) or
        # per joint (joint), giving every joint its own core representation.
        combined_input = jnp.concatenate([joint_latent, foot_latent, general_state], axis=-1)
        decoder_width = self.expert_hidden if self.expert_hidden > 0 else int(128 * self.width_mult_decoder)
        core = self._moe("core", *self._core_spec())
        if self.core_routing == "joint":
            body = jnp.broadcast_to(combined_input[..., None, :], batch_shape + (n_joints, combined_input.shape[-1]))
            z_b = core(combined_input, joint_morph, jnp.concatenate([body, js], axis=-1))        # (..., joints, 128)
        else:
            z = core(combined_input, pooled, combined_input, self._robot_p("core", robot_id),
                     group_p=self._group_p(robot_id))
            z_b = jnp.broadcast_to(jnp.expand_dims(z, axis=-2), batch_shape + (n_joints, z.shape[-1]))

        # Descriptor embedding for the decoder (shared).
        e = nn.Dense(128)(jd)
        e = nn.LayerNorm()(e)
        e = nn.elu(e)
        e = nn.Dense(128)(e)

        # Per-joint action decoder (joint level).
        decoder_input = jnp.concatenate([z_b, stop_gradient(z_joint), e], axis=-1)
        decoder_state = jnp.concatenate([z_b, stop_gradient(z_joint)], axis=-1)
        fixed_p = self._robot_p("decoder", robot_id)
        if self.decoder_table and robot_id is not None:
            fixed_p = jnp.asarray(self.decoder_table, dtype=jd.dtype)[robot_id, :n_joints]
        mean = self._moe("decoder", (decoder_width, 1), self.block_posts("decoder"),
                         (orthogonal(np.sqrt(2)), orthogonal(0.01)))(decoder_input, joint_morph, decoder_state, fixed_p)
        mean = mean.squeeze(-1)
        if self.calibration:
            scale = self.param("out_scale", constant(1.0), (self.n_robots, self.max_joints))
            bias = self.param("out_bias", constant(0.0), (self.n_robots, self.max_joints))
            if robot_id is not None:
                mean = mean * scale[robot_id, :n_joints] + bias[robot_id, :n_joints]

        if self.deterministic:
            return self.max_action * jnp.tanh(mean)

        mean = jnp.clip(mean, -self.policy_mean_abs_clip, self.policy_mean_abs_clip)
        head_input = stop_gradient(e) if self.variance_mode in ("detach", "protected") else e
        logstd = nn.Dense(1, kernel_init=orthogonal(0.1), bias_init=constant(np.log(self.std_dev)))(head_input).squeeze(-1)
        if self.variance_mode == "private":
            # The shared head's output is a fixed baseline; the learned part is a private offset per (robot, joint).
            offset = self.param("logstd_offset", constant(0.0), (self.n_robots, self.max_joints))
            if robot_id is None:
                if not self.is_initializing():
                    raise ValueError("private log-std offsets are indexed by robot: pass robot_id")
                robot_id = 0
            logstd = stop_gradient(logstd) + offset[robot_id, :n_joints]
        logstd = jnp.clip(logstd, np.log(self.policy_std_min_clip), np.log(self.policy_std_max_clip))
        return mean, logstd


def count_core_categories(params, n_global, n_group):
    """Core parameters by category: the global, group and routed experts (their slices of the expert axis), the router,
    and the noise head."""
    counts = {"global": 0, "group": 0, "routed": 0, "router": 0, "noise_head": 0}
    for name, sub in params["params"]["core"].items():
        for leaf in jax.tree_util.tree_leaves(sub):
            if name in ("router", "router_noise"):
                counts["router" if name == "router" else "noise_head"] += int(leaf.size)
                continue
            per_expert = int(np.prod(leaf.shape[1:]))
            counts["global"] += per_expert * n_global
            counts["group"] += per_expert * n_group
            counts["routed"] += per_expert * (leaf.shape[0] - n_global - n_group)
    return counts


def count_params(params):
    """Number of policy parameters per block (experts + router) and in shared modules."""
    counts, shared = {}, 0
    for name, sub in params["params"].items():
        n = sum(int(x.size) for x in jax.tree_util.tree_leaves(sub))
        if name in BLOCKS:
            counts[name] = n
        else:
            shared += n
    counts["shared"] = shared
    counts["total"] = sum(counts.values())
    return counts
