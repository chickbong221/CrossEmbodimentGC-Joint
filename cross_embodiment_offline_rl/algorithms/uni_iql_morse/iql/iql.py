import os
import json
import hashlib
import psutil
import logging
import time
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.training import checkpoints
import orbax.checkpoint
import optax
import wandb
from flax import serialization

from cross_embodiment_offline_rl.algorithms.uni_iql_eg.iql.general_properties import GeneralProperties
from cross_embodiment_offline_rl.algorithms.uni_iql_eg.iql.critic import get_valuecritic, get_critic
from cross_embodiment_offline_rl.algorithms.morse.branch import branch_state
from cross_embodiment_offline_rl.algorithms.morse.ckpt_analysis import CheckpointAnalysis, load_checkpoint
from cross_embodiment_offline_rl.algorithms.morse.config import LEGACY_ARCHITECTURE, RECORD_KEYS as MORSE_RECORD_KEYS
from cross_embodiment_offline_rl.algorithms.morse.coordination import CoordinationDump, load_end_flags
from cross_embodiment_offline_rl.algorithms.morse.policy import get_morse_policy
from cross_embodiment_offline_rl.algorithms.morse.probe import DecoderProbe
from cross_embodiment_offline_rl.algorithms.morse.conflict import ConflictAnalysis
from cross_embodiment_offline_rl.algorithms.morse.updater import MorseActorUpdater
from cross_embodiment_offline_rl.algorithms.morse.v8 import (build_joint_groups, get_v8_policy, load_joint_groups,
                                                             write_joint_groups)
from cross_embodiment_offline_rl.algorithms.morse.v8_updater import V8ActorUpdater
from cross_embodiment_offline_rl.algorithms.morse.variance import (frozen_clip_by_global_norm, grouped_clip_by_global_norm,
                                                                   is_core, is_std_head, logstd_state_spread, protected_loss,
                                                                   zero_first_moments)
from cross_embodiment_offline_rl.algorithms.utils import (Dataset, BatchPrefetcher, dataset_columns, resolve_data_path,
                                                          robot_batch_sizes)

rlx_logger = logging.getLogger("rl_x")


def params_sha1(tree):
    """SHA-1 of a parameter tree's values (leaves in tree order, as stored): equal hashes mean identical parameters."""
    h = hashlib.sha1()
    for leaf in jax.tree_util.tree_leaves(jax.device_get(tree)):
        a = np.ascontiguousarray(np.asarray(leaf))
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


class IQL:
    def __init__(self, config, env, run_path, writer) -> None:
        self.config = config
        self.env = env.train_env
        self.eval_env = env.eval_env
        self.writer = writer
        self.save_model = config.runner.save_model
        self.run_path = run_path
        self.save_path = os.path.join(run_path, "models")
        self.sync_path = os.path.join(run_path, "sync_hashes.jsonl")   # content hashes of minibatches and critics
        self.track_console = config.runner.track_console
        self.track_tb = config.runner.track_tb
        self.track_wandb = config.runner.track_wandb
        self.seed = config.environment.seed
        self.nr_envs = config.environment.nr_envs
        self.nr_eval_envs = config.environment.nr_eval_envs
        self.actor_lr = config.algorithm.actor_lr
        self.critic_lr = config.algorithm.critic_lr
        self.value_lr = config.algorithm.value_lr
        self.softmax_temperature_min = config.algorithm.softmax_temperature_min
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes
        self.save_latest_frequency = config.algorithm.save_latest_frequency
        self.determine_fastest_cpu_for_gpu = config.algorithm.determine_fastest_cpu_for_gpu
        self.multi_render = config.environment.multi_render
        self.batch_size = config.algorithm.batch_size * config.environment.nr_envs 
        self.nr_offline_updates = config.algorithm.nr_offline_updates
        self.nr_env_types = len(config.environment.train_robot_types)
        self.nr_eval_env_types = len(config.environment.eval_robot_types)
        self.robot_loss_weights = jnp.array([config.algorithm.robot_weights[robot_type] for robot_type in config.environment.train_robot_types])
        self.env_batch_sizes = robot_batch_sizes([config.algorithm.robot_weights[robot_type] for robot_type in config.environment.train_robot_types],
                                                 config.algorithm.batch_size, self.nr_env_types)
        # Actor steps per update, each on an equal chunk of every robot's minibatch (see morse/config.py).
        self.actor_steps = int(config.algorithm.actor_steps_per_update)
        if self.actor_steps < 1 or any(b % self.actor_steps for b in self.env_batch_sizes):
            raise ValueError(f"actor_steps_per_update={self.actor_steps} must divide every robot batch size {self.env_batch_sizes}")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        if self.determine_fastest_cpu_for_gpu and self.env.fastest_cpu_id is not None:
            p = psutil.Process()
            p.cpu_affinity([self.env.fastest_cpu_id,])
            rlx_logger.info(f"Using fastest CPU for GPU connection: {self.env.fastest_cpu_id}")
        
        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, value_key, critic_1_key, critic_2_key = jax.random.split(self.key, 5)

        self.os_shape = self.env.single_observation_space.shape
        self.as_shape = self.env.single_action_space.shape
        
        self.v8 = str(getattr(config.algorithm, "morse_arch", "urma_moe")) == "v8"
        self.policy = get_v8_policy(config) if self.v8 else get_morse_policy(config)
        self.get_processed_action = jax.jit(lambda action: action)
        self.value_critic = get_valuecritic(config, self.env)
        self.critic = get_critic(config, self.env)

        self.policy.apply = jax.jit(self.policy.apply)
        self.value_critic.apply = jax.jit(self.value_critic.apply)
        self.critic.apply = jax.jit(self.critic.apply)

        state = jnp.array([self.env.single_observation_space.sample() for _ in range(self.nr_envs)])

        # Create policy and critic state masks
        self.policy_general_state_mask = np.zeros(state.shape)
        self.critic_general_state_mask = np.zeros(state.shape)
        general_state_for_policy_names = [
            "trunk_roll_velocity", "trunk_pitch_velocity", "trunk_yaw_velocity",
            "goal_x_velocity", "goal_y_velocity", "goal_yaw_velocity",
            "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",
            "p_gain", "d_gain", "action_scaling_factor",
            "mass",
            "robot_length", "robot_width", "robot_height"
        ]
        general_state_for_critic_names = [
            "trunk_x_velocity", "trunk_y_velocity", "trunk_z_velocity",
            "trunk_roll_velocity", "trunk_pitch_velocity", "trunk_yaw_velocity",
            "goal_x_velocity", "goal_y_velocity", "goal_yaw_velocity",
            "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",
            "height_0",
            "p_gain", "d_gain", "action_scaling_factor",
            "mass",
            "robot_length", "robot_width", "robot_height"
        ]
        observation_name_to_ids = self.env.call("observation_name_to_id")
        for env_id in range(self.nr_envs):
            for name in general_state_for_policy_names:
                self.policy_general_state_mask[env_id, observation_name_to_ids[env_id][name]] = 1
            for name in general_state_for_critic_names:
                self.critic_general_state_mask[env_id, observation_name_to_ids[env_id][name]] = 1
        self.policy_general_state_mask = np.array(self.policy_general_state_mask, dtype=bool)
        self.critic_general_state_mask = np.array(self.critic_general_state_mask, dtype=bool)
        # eval
        if self.nr_eval_envs > 0:
            self.eval_policy_general_state_mask = np.zeros((self.nr_eval_envs, state.shape[-1]))
            eval_observation_name_to_ids = self.eval_env.call("observation_name_to_id")
            for env_id in range(self.nr_eval_envs):
                for name in general_state_for_policy_names:
                    self.eval_policy_general_state_mask[env_id, eval_observation_name_to_ids[env_id][name]] = 1
            self.eval_policy_general_state_mask = np.array(self.eval_policy_general_state_mask, dtype=bool)

        self.nr_envs_per_env_type = self.nr_envs // self.nr_env_types
        self.env_ids = np.array([self.nr_envs_per_env_type * i for i in range(self.nr_env_types + 1)])
        # eval
        if self.nr_eval_envs > 0:
            self.eval_nr_envs_per_env_type = self.nr_eval_envs // self.nr_eval_env_types
            self.eval_env_ids = np.array([self.eval_nr_envs_per_env_type * i for i in range(self.nr_eval_env_types + 1)])

        robot_types = self.env.call("robot_type")
        self.robot_types_list = np.array([robot_types[env_id] for env_id in self.env_ids[:-1]])
        # eval
        if self.nr_eval_envs > 0:
            eval_robot_types = self.eval_env.call("robot_type")
            self.eval_robot_types_list = np.array([eval_robot_types[env_id] for env_id in self.eval_env_ids[:-1]])

        self.nr_dynamic_joint_observations = np.array(self.env.call("nr_dynamic_joint_observations"))
        self.nr_dynamic_foot_observations = np.array(self.env.call("nr_dynamic_foot_observations"))
        self.dynamic_joint_observation_lengths = np.array(self.env.call("dynamic_joint_observation_length"))
        self.dynamic_foot_observation_lengths = np.array(self.env.call("dynamic_foot_observation_length"))
        self.single_dynamic_joint_observation_length = self.env.call("single_dynamic_joint_observation_length")[0]
        self.single_dynamic_foot_observation_length = self.env.call("single_dynamic_foot_observation_length")[0]
        self.dynamic_joint_description_size = self.env.call("dynamic_joint_description_size")[0]
        self.dynamic_foot_description_size = self.env.call("dynamic_foot_description_size")[0]
        # eval
        if self.nr_eval_envs > 0:
            self.eval_nr_dynamic_joint_observations = np.array(self.eval_env.call("nr_dynamic_joint_observations"))
            self.eval_nr_dynamic_foot_observations = np.array(self.eval_env.call("nr_dynamic_foot_observations"))
            self.eval_dynamic_joint_observation_lengths = np.array(self.eval_env.call("dynamic_joint_observation_length"))
            self.eval_dynamic_foot_observation_lengths = np.array(self.eval_env.call("dynamic_foot_observation_length"))
            self.eval_single_dynamic_joint_observation_length = self.eval_env.call("single_dynamic_joint_observation_length")[0]
            self.eval_single_dynamic_foot_observation_length = self.eval_env.call("single_dynamic_foot_observation_length")[0]
            self.eval_dynamic_joint_description_size = self.eval_env.call("dynamic_joint_description_size")[0]
            self.eval_dynamic_foot_description_size = self.eval_env.call("dynamic_foot_description_size")[0]

        self.missing_nr_of_actions = self.env.call("missing_nr_of_actions")
        self.all_missing_nr_of_actions = np.array([self.missing_nr_of_actions[env_id] for env_id in self.env_ids[:-1]])
        # eval
        if self.nr_eval_envs > 0:
            eval_missing_nr_of_actions = self.eval_env.call("missing_nr_of_actions")
            self.eval_all_missing_nr_of_actions = np.array([eval_missing_nr_of_actions[env_id] for env_id in self.eval_env_ids[:-1]])

        dummy_dynamic_joint_combined_state = state[:self.env_ids[1], :self.dynamic_joint_observation_lengths[self.env_ids[0]]].reshape((-1, self.nr_dynamic_joint_observations[self.env_ids[0]], self.single_dynamic_joint_observation_length))
        dummy_dynamic_joint_description = dummy_dynamic_joint_combined_state[:, :, :self.dynamic_joint_description_size]
        dummy_dynamic_joint_state = dummy_dynamic_joint_combined_state[:, :, self.dynamic_joint_description_size:]
        
        dummy_dynamic_foot_combined_state = state[:self.env_ids[1], self.dynamic_joint_observation_lengths[self.env_ids[0]]:self.dynamic_joint_observation_lengths[self.env_ids[0]] + self.dynamic_foot_observation_lengths[self.env_ids[0]]].reshape((-1, self.nr_dynamic_foot_observations[self.env_ids[0]], self.single_dynamic_foot_observation_length))
        dummy_dynamic_foot_description = dummy_dynamic_foot_combined_state[:, :, :self.dynamic_foot_description_size]
        dummy_dynamic_foot_state = dummy_dynamic_foot_combined_state[:, :, self.dynamic_foot_description_size:]

        dummy_general_policy_state = state[:self.env_ids[1], self.policy_general_state_mask[self.env_ids[0]]]
        dummy_general_critic_state = state[:self.env_ids[1], self.critic_general_state_mask[self.env_ids[0]]]

        dummy_action = jnp.array([self.env.single_action_space.sample() for _ in range(self.nr_envs)])
        dummy_action = dummy_action[:self.env_ids[1], :]

        schedule_fn = optax.cosine_decay_schedule(-self.actor_lr, self.nr_offline_updates * self.actor_steps)
        self.actor_schedule = schedule_fn
        # The protected-mean mode clips the log-std head's gradient on its own (morse/variance.py).
        self.variance_mode = str(getattr(config.algorithm, "morse_variance_mode", "original"))
        # A frozen core (morse_freeze_core): its gradient is zeroed before the clip, which sees the trainable gradient only.
        self.freeze_core = bool(getattr(config.algorithm, "morse_freeze_core", False))
        if self.freeze_core and self.variance_mode != "original":
            raise ValueError("morse_freeze_core is a run of its own: leave morse_variance_mode at original")
        policy_clip = (grouped_clip_by_global_norm(self.max_grad_norm, is_std_head) if self.variance_mode == "protected"
                       else frozen_clip_by_global_norm(self.max_grad_norm, is_core) if self.freeze_core
                       else optax.clip_by_global_norm(self.max_grad_norm))

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, dummy_dynamic_joint_description, dummy_dynamic_joint_state, dummy_dynamic_foot_description, dummy_dynamic_foot_state, dummy_general_policy_state),
            tx=optax.chain(
                policy_clip,
                optax.scale_by_adam(),
                optax.scale_by_schedule(schedule_fn),
            )
        )
        
        self.value_critic_state = TrainState.create(
            apply_fn=self.value_critic.apply,
            params=self.value_critic.init(value_key, dummy_dynamic_joint_description, dummy_dynamic_joint_state, dummy_dynamic_foot_description, dummy_dynamic_foot_state, dummy_general_critic_state),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=self.value_lr),
            )
        )
        
        self.critic_state_1 = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_1_key, dummy_dynamic_joint_description, dummy_dynamic_joint_state, dummy_dynamic_foot_description, dummy_dynamic_foot_state, dummy_general_critic_state, dummy_action),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=self.critic_lr),
            )
        )

        self.critic_state_2 = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_2_key, dummy_dynamic_joint_description, dummy_dynamic_joint_state, dummy_dynamic_foot_description, dummy_dynamic_foot_state, dummy_general_critic_state, dummy_action),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=self.critic_lr),
            )
        )

        self.target_critic_state_1 = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic_state_1.params,  # Copy initial parameters
            tx=optax.adam(learning_rate=self.critic_lr),
        )

        self.target_critic_state_2 = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic_state_2.params,  # Copy initial parameters
            tx=optax.adam(learning_rate=self.critic_lr),
        )

        self.critic_state = (self.critic_state_1, self.critic_state_2)
        self.target_critic_state = (self.target_critic_state_1, self.target_critic_state_2)

        # Branch from a checkpoint of another run (morse/branch.py): same starting policy, critics and optimizer
        # state; training continues from morse_branch_updates to nr_offline_updates.
        self.branch_updates = 0
        self.branch_raw = None
        if config.algorithm.morse_branch_from:
            raw = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath(config.algorithm.morse_branch_from), target=None)
            self.branch_raw = raw
            self.policy_state, tiled, kept = branch_state(self.policy_state, raw["policy"])
            if self.v8 and (tiled or kept):
                raise ValueError(f"morse_arch=v8 branch: policy leaves tiled {tiled} or missing in the checkpoint {kept}")
            self.value_critic_state = branch_state(self.value_critic_state, raw["value_critic"])[0]
            self.critic_state = tuple(branch_state(s, raw[k])[0] for s, k in zip(self.critic_state, ("critic_1", "critic_2")))
            self.target_critic_state = tuple(branch_state(s, raw[k])[0] for s, k in
                                             zip(self.target_critic_state, ("target_critic_1", "target_critic_2")))
            self.branch_updates = int(config.algorithm.morse_branch_updates)
            rlx_logger.info(f"branched from {config.algorithm.morse_branch_from} at {self.branch_updates} updates; "
                            f"tiled {len(tiled)} leaves into experts, kept {len(kept)} new leaves at initialization: {kept}")

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_eps_track_perf_perc_average = 0.0
            self.best_model_file_name = "model_best_jax"
            self.latest_model_file_name = "model_latest_jax"
            best_model_check_point_handler = orbax.checkpoint.PyTreeCheckpointHandler(aggregate_filename=self.best_model_file_name)
            latest_model_check_point_handler = orbax.checkpoint.PyTreeCheckpointHandler(aggregate_filename=self.latest_model_file_name)
            self.best_model_checkpointer = orbax.checkpoint.Checkpointer(best_model_check_point_handler)
            self.latest_model_checkpointer = orbax.checkpoint.Checkpointer(latest_model_check_point_handler)

        if self.track_wandb and "SLURM_JOB_ID" in os.environ:
            rlx_logger.info(f"Slurm job id: {os.environ['SLURM_JOB_ID']}")
            wandb.config["SLURM_JOB_ID"] = os.environ["SLURM_JOB_ID"]

        def split_obs(obs_batch, env_type_id):
            """Policy inputs of one robot: joint/foot descriptions and states, general state."""
            env_id = env_type_id * self.nr_envs_per_env_type
            djl = self.dynamic_joint_observation_lengths[env_id]
            nfl = self.dynamic_foot_observation_lengths[env_id]
            dj = obs_batch[:, :djl].reshape((-1, self.nr_dynamic_joint_observations[env_id], self.single_dynamic_joint_observation_length))
            df = obs_batch[:, djl:djl + nfl].reshape((-1, self.nr_dynamic_foot_observations[env_id], self.single_dynamic_foot_observation_length))
            return (dj[:, :, :self.dynamic_joint_description_size], dj[:, :, self.dynamic_joint_description_size:],
                    df[:, :, :self.dynamic_foot_description_size], df[:, :, self.dynamic_foot_description_size:],
                    obs_batch[:, self.policy_general_state_mask[env_id]])

        def awr_terms(policy_output, obs_batch, act_batch, extra, env_type_id):
            """Advantage weights (B, 1) and per-joint log-likelihoods (B, action_dim) of one robot's batch."""
            value_critic_params, critic1_params, critic2_params = extra
            env_id = env_type_id * self.nr_envs_per_env_type
            dj_desc, dj_state, df_desc, df_state, _ = split_obs(obs_batch, env_type_id)
            critic_general_state = obs_batch[:, self.critic_general_state_mask[env_id]]
            v = self.value_critic.apply(value_critic_params, dj_desc, dj_state, df_desc, df_state, critic_general_state).squeeze()
            q1 = self.critic.apply(critic1_params, dj_desc, dj_state, df_desc, df_state, critic_general_state, act_batch).squeeze()
            q2 = self.critic.apply(critic2_params, dj_desc, dj_state, df_desc, df_state, critic_general_state, act_batch).squeeze()
            q = jnp.minimum(q1, q2)
            exp_a = jnp.minimum(jnp.exp((q - v) * self.config.algorithm.beta), 100.0).reshape((-1, 1))

            mean, logstd = policy_output
            std = jnp.exp(logstd)
            action_dim = self.as_shape[0] - self.all_missing_nr_of_actions[env_type_id]
            lp = -0.5 * ((act_batch[:, :action_dim] - mean) / std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - logstd
            return exp_a, lp

        def policy_loss(policy_output, obs_batch, act_batch, extra, key, env_type_id):
            """IQL (AWR) policy loss of one robot, given the policy output on its batch."""
            exp_a, lp = awr_terms(policy_output, obs_batch, act_batch, extra, env_type_id)
            if self.variance_mode == "protected":
                mean, logstd = policy_output
                return protected_loss(exp_a, mean, logstd, act_batch[:, :mean.shape[-1]]) * self.robot_loss_weights[env_type_id]
            return -(exp_a * lp).mean() * self.robot_loss_weights[env_type_id]

        self.split_obs = split_obs

        def policy_loss_per_joint(policy_output, obs_batch, act_batch, extra, key, env_type_id):
            """The per-joint terms of policy_loss (they sum to it): the probe's unit-type losses."""
            exp_a, lp = awr_terms(policy_output, obs_batch, act_batch, extra, env_type_id)
            return -(exp_a * lp).mean(axis=0) / lp.shape[1] * self.robot_loss_weights[env_type_id]

        a = config.algorithm
        if self.v8:
            # morse_branch_from resumes a V8 run from a checkpoint of the same architecture: every leaf of the policy,
            # critics and optimizer state must be restored exactly (checked below and in check_branch).
            unsupported = {k: getattr(a, k) for k in ("morse_probe_at", "morse_ckpt_analysis") if getattr(a, k)}
            if unsupported or self.variance_mode != "original" or self.freeze_core or a.morse_router_only:
                raise ValueError(f"morse_arch=v8 does not support {unsupported or 'variance modes, core freezing or router-only updates'}")
            groups, n_groups = None, 0
            if a.morse_v8_direction == "mi_joint":
                groups, n_groups = self.v8_joint_groups(a)
            # A policy instance of its own: self.policy.apply is replaced by a jitted function above.
            self.morse = V8ActorUpdater(config, get_v8_policy(config), split_obs, list(self.robot_types_list), run_path,
                                        joint_groups=groups, n_groups=n_groups)
        else:
            self.morse = MorseActorUpdater(config, get_morse_policy(config, capture=True), split_obs,
                                           list(self.robot_types_list), run_path)
        self.morse.loss_fn = policy_loss

        # Analysis hooks: decoder probes (morse/probe.py) and checkpoints at chosen updates.
        self.probe_at = {int(x) for x in str(a.morse_probe_at).split(",") if x.strip()}
        self.save_at = {int(x) for x in str(a.morse_save_at).split(",") if x.strip()}
        self.probe = None
        if self.probe_at:
            self.probe = DecoderProbe(self.policy, split_obs, policy_loss_per_joint, list(self.robot_types_list),
                                      self.env_batch_sizes, a.morse_probe_batches,
                                      [float(s) for s in str(a.morse_probe_steps).split(",") if s.strip()],
                                      os.path.join(run_path, "morse_probe"), self.seed, a.morse_probe_lr)
        # Checkpoint analysis instead of training (morse/ckpt_analysis.py, morse/coordination.py).
        self.ckpt_analysis = self.coord_dump = self.conflict = None
        if a.morse_ckpt_analysis:
            self.ckpt_analysis = CheckpointAnalysis(get_morse_policy(config, capture=True), split_obs, awr_terms,
                                                    list(self.robot_types_list), self.env_batch_sizes, a, self.seed,
                                                    loss_fn=policy_loss)
            if int(getattr(a, "morse_conflict_batches", 0)) > 0:
                self.conflict = ConflictAnalysis(self.policy, split_obs, awr_terms, list(self.robot_types_list),
                                                 self.env_batch_sizes, self.robot_loss_weights, a, self.seed,
                                                 max(self.env.call("horizon")))
            if a.morse_coord_out:
                self.coord_dump = CoordinationDump(self.policy, split_obs, list(self.robot_types_list),
                                                   [observation_name_to_ids[i] for i in self.env_ids[:-1]],
                                                   max(self.env.call("horizon")), a, awr_terms=awr_terms)
        # Robot index of every evaluation robot in the training list (fixed routing tables and calibration).
        train_index = {str(rt): i for i, rt in enumerate(self.robot_types_list)}
        self.eval_robot_index = {}
        if self.nr_eval_envs > 0:
            self.eval_robot_index = {int(self.eval_env_ids[i]): train_index.get(str(rt))
                                     for i, rt in enumerate(self.eval_robot_types_list)}

    def v8_joint_groups(self, a, n_rows=4000):
        """The joint-group table of the J direction (morse/v8.py): loaded from morse_v8_joint_groups, or built there from
        the mean joint descriptions of every training robot over n_rows evenly spaced dataset rows."""
        path = str(a.morse_v8_joint_groups)
        if not path:
            raise ValueError("morse_v8_direction=mi_joint needs morse_v8_joint_groups (a JSON path)")
        if not os.path.isabs(path):                    # relative paths: next to morse/v8.py (the same on every machine)
            from cross_embodiment_offline_rl.algorithms.morse import v8 as v8_module
            path = os.path.join(os.path.dirname(os.path.abspath(v8_module.__file__)), path)
        names = [str(r) for r in self.robot_types_list]
        n_joints = [int(self.nr_dynamic_joint_observations[i]) for i in self.env_ids[:-1]]
        if not os.path.exists(path):
            root = resolve_data_path(a.dataset_path)
            with open(root / "metadata.json") as f:
                meta = json.load(f)
            columns = dataset_columns(self.config.environment.train_robot_types, int(meta["n_envs"]))
            obs = np.load(root / "observations.npy", mmap_mode="r")
            rows = np.unique(np.linspace(0, int(meta["n_samples"]) - 1, n_rows).astype(np.int64))
            length, size = self.single_dynamic_joint_observation_length, self.dynamic_joint_description_size
            descriptions = []
            for r, n in enumerate(n_joints):
                col = columns[r] if columns is not None else r
                block = np.asarray(obs[rows, col, :n * length], np.float64).reshape(len(rows), n, length)
                descriptions.append(block[:, :, :size].mean(axis=0))
            try:
                joint_names = {name: list(map(str, self.env.call("joint_names")[i]))
                               for name, i in zip(names, self.env_ids[:-1])}
            except Exception:
                joint_names = {}
            table = build_joint_groups(descriptions, names, joint_names, k=int(a.morse_v8_group_k))
            table["dataset_rows"] = int(len(rows))
            write_joint_groups(path, table)
            rlx_logger.info(f"V8 joint groups built and written to {path}: {table['groups']}")
        groups, k = load_joint_groups(path, names, n_joints)
        with open(path) as f:
            built_from = json.load(f).get("dataset", "")
        if built_from and built_from != os.path.basename(os.path.normpath(str(a.dataset_path))):
            rlx_logger.warning(f"V8 joint groups were built from {built_from}, this run trains on {a.dataset_path}")
        rlx_logger.info(f"V8 joint groups ({k}) from {path}, built from {built_from or 'the run dataset'}")
        return groups, k

    def check_branch(self, dataset):
        """Start of a run branched from a checkpoint: the branched policy is compared with the checkpoint's own on fixed
        minibatches of every robot (action means and log-stds, on the CPU), and every restored leaf of the policy, the
        critics and the optimizer state with the checkpoint's. The private variance mode first zeroes the Adam first
        moments of the head it freezes. Written to branch_check.json (with the state dependence of the checkpoint's
        log-std); raises if the policy differs from the checkpoint's or a leaf that should be restored does not match."""
        raw, a = self.branch_raw, self.config.algorithm
        batches = [dataset.sample(jax.random.PRNGKey(7_000_003 + i), self.env_batch_sizes)[0] for i in range(2)]
        inputs = [[self.split_obs(jnp.asarray(b[r].observations), r) for b in batches] for r in range(self.nr_env_types)]
        source = self.policy.clone() if self.v8 else self.policy.clone(variance_mode="original")
        variables = jax.tree_util.tree_map(jnp.asarray, raw["policy"]["params"])
        report = {"checkpoint": str(a.morse_branch_from), "branch_updates": self.branch_updates,
                  "variance_mode": self.variance_mode}
        select = None
        if self.variance_mode == "private":
            select = lambda names: "Dense_6" in names            # the shared head is frozen
        elif self.freeze_core:
            select = lambda names: "core" in names               # the core block is frozen
        if select is not None:
            # The restored Adam first moments of the frozen parameters are zeroed, so that with a zero gradient their
            # update is exactly zero and they stay bitwise where they are.
            self.policy_state = self.policy_state.replace(opt_state=zero_first_moments(self.policy_state.opt_state, select))
        if self.freeze_core:
            report["core_sha1_checkpoint"] = params_sha1(raw["policy"]["params"]["params"]["core"])
            report["core_sha1_start"] = params_sha1(self.policy_state.params["params"]["core"])
        # On the CPU: two evaluations of one function on the GPU can differ through near-tied top-k routing. The
        # checkpoint's own policy is evaluated twice to measure the noise of the evaluation itself.
        to_cpu = lambda t: jax.device_put(jax.device_get(t), jax.devices("cpu")[0])
        cpu_variables, cpu_params = to_cpu(variables), to_cpu(self.policy_state.params)
        inputs = [[to_cpu(inp) for inp in per_robot] for per_robot in inputs]
        n = self.nr_env_types
        mean_diff, logstd_diff, floor = np.zeros(n), np.zeros(n), np.zeros(n)
        for r in range(n):
            for inp in inputs[r]:
                m0, s0 = jax.device_get(source.apply(cpu_variables, *inp, robot_id=r))
                m0b, s0b = jax.device_get(source.apply(cpu_variables, *inp, robot_id=r))
                m1, s1 = jax.device_get(self.policy.apply(cpu_params, *inp, robot_id=r))
                mean_diff[r] = max(mean_diff[r], float(np.abs(m1 - m0).max()))
                logstd_diff[r] = max(logstd_diff[r], float(np.abs(s1 - s0).max()))
                floor[r] = max(floor[r], float(np.abs(m0b - m0).max()), float(np.abs(s0b - s0).max()))
        report["logstd_state_spread"] = logstd_state_spread(source, cpu_variables["params"], inputs).tolist()
        report["repeat_evaluation_noise"] = floor.tolist()
        report["robots"] = [str(r) for r in self.robot_types_list]
        report["mean_max_abs_diff"] = mean_diff.tolist()
        report["logstd_max_abs_diff"] = logstd_diff.tolist()

        def flat(tree):
            return {tuple(getattr(k, "key", str(k)) for k in p): np.asarray(v)
                    for p, v in jax.tree_util.tree_flatten_with_path(tree)[0]}
        states = {"policy": self.policy_state, "value_critic": self.value_critic_state, "critic_1": self.critic_state[0],
                  "critic_2": self.critic_state[1], "target_critic_1": self.target_critic_state[0],
                  "target_critic_2": self.target_critic_state[1]}
        leaves = {}
        for name, state in states.items():
            new, old = flat(serialization.to_state_dict(state)), flat(raw[name])
            common = [k for k in new if k in old and not (select is not None and "mu" in k and select(k))]
            diffs = [float(np.abs(new[k] - old[k]).max()) if new[k].size else 0.0 for k in common]
            leaves[name] = {"restored": len(common), "max_abs_diff": max(diffs + [0.0]),
                            "new_leaves": ["/".join(k) for k in new if k not in old],
                            "dropped_leaves": ["/".join(k) for k in old if k not in new]}
        report["leaves"] = leaves
        counts = {"/".join(k): int(v) for k, v in flat(self.policy_state.opt_state).items() if k[-1].endswith("count")}
        report["optimizer_counts"] = counts
        report["expected_count"] = self.branch_updates * self.actor_steps
        report["lr_at_count"] = float(-self.actor_schedule(self.branch_updates * self.actor_steps))
        report["policy_step"] = int(self.policy_state.step)
        with open(os.path.join(self.run_path, "branch_check.json"), "w") as f:
            json.dump(report, f, indent=1)
        rlx_logger.info(f"MORSE branch check ({self.variance_mode}): means {mean_diff.max():.2e}, log-stds "
                        f"{logstd_diff.max():.2e}, optimizer counts {sorted(set(counts.values()))}")
        bad = [f"{n}: {v['max_abs_diff']:.2e}" for n, v in leaves.items() if v["max_abs_diff"] > 0.0]
        if self.v8:
            bad += [f"{n}: new {v['new_leaves']}, dropped {v['dropped_leaves']}" for n, v in leaves.items()
                    if v["new_leaves"] or v["dropped_leaves"]]
        tol = max(1e-5, 10.0 * floor.max())
        if self.freeze_core and report["core_sha1_start"] != report["core_sha1_checkpoint"]:
            bad.append("core parameters differ from the checkpoint's")
        if mean_diff.max() > tol or logstd_diff.max() > tol or bad:
            raise RuntimeError(f"branch check failed: means {mean_diff.max():.2e}, log-stds {logstd_diff.max():.2e}, "
                               f"restored leaves {bad}")
        self.branch_raw = None

    def expectile_loss(self,diff, expectile=0.8) -> jnp.ndarray:
        weight = jnp.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def target_update(
        self, model: TrainState, target_model: TrainState, tau: float
    ) -> TrainState:
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau), model.params, target_model.params
        )
        return target_model.replace(params=new_target_params)
    
    def train(self):
        def train_loop():
            @jax.jit
            def get_action(policy_state, state, key, stochastic=True):
                @partial(jax.jit, static_argnums=(1,2))
                def get_single_deterministic_eval_action(key, env_id_0, env_id_1, stochastic):
                    state_b = state[env_id_0:env_id_1]

                    dynamic_joint_combined_state = state_b[:, :self.eval_dynamic_joint_observation_lengths[env_id_0]].reshape((-1, self.eval_nr_dynamic_joint_observations[env_id_0], self.eval_single_dynamic_joint_observation_length))
                    dynamic_joint_description = dynamic_joint_combined_state[:, :, :self.eval_dynamic_joint_description_size]
                    dynamic_joint_state = dynamic_joint_combined_state[:, :, self.eval_dynamic_joint_description_size:]

                    dynamic_foot_combined_state = state_b[:, self.eval_dynamic_joint_observation_lengths[env_id_0]:self.eval_dynamic_joint_observation_lengths[env_id_0] + self.eval_dynamic_foot_observation_lengths[env_id_0]].reshape((-1, self.eval_nr_dynamic_foot_observations[env_id_0], self.eval_single_dynamic_foot_observation_length))
                    dynamic_foot_description = dynamic_foot_combined_state[:, :, :self.eval_dynamic_foot_description_size]
                    dynamic_foot_state = dynamic_foot_combined_state[:, :, self.eval_dynamic_foot_description_size:]

                    action_mean, action_logstd = self.policy.apply(policy_state.params, dynamic_joint_description, dynamic_joint_state, dynamic_foot_description, dynamic_foot_state, state_b[:, self.eval_policy_general_state_mask[env_id_0]],
                                                                   robot_id=self.eval_robot_index.get(int(env_id_0)))
                    
                    action_std = jnp.exp(action_logstd)
                    key, subkey = jax.random.split(key)
                    action = jnp.where(stochastic, 
                                     action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape),
                                     action_mean)
                    action = jnp.concatenate([action, jnp.zeros((self.eval_nr_envs_per_env_type, self.eval_all_missing_nr_of_actions[env_id_0]))], axis=1)
                    processed_action = self.get_processed_action(action)

                    return processed_action, key
            
                processed_actions = []
                for i in range(self.eval_env_ids.shape[0] - 1):
                    env_id_0 = self.eval_env_ids[i]
                    env_id_1 = self.eval_env_ids[i + 1]
                    processed_action, key = get_single_deterministic_eval_action(key, env_id_0, env_id_1, stochastic)
                    processed_actions.append(processed_action)
                
                return jnp.concatenate(processed_actions), key
            

            def env_reset_callback():
                state, _ = self.env.reset()
                state = state.astype(np.float32)
                return state
            

            def reset_step_info_collection_callback():
                self.step_info_collection = {}
            

            def evaluate_callback(policy_state, old_state, key, force=False):
                self.evaluation_metrics = {}
                due = (self.current_nr_updates+1) % self.evaluation_frequency == 0 and self.evaluation_frequency != -1
                if self.nr_eval_envs > 0 and (due or force):
                    self.set_eval_mode()
                    eval_state, _ = self.eval_env.reset()
                    eval_nr_episodes = np.zeros(self.nr_eval_env_types, dtype=int)
                    for robot_type in self.eval_robot_types_list:
                        self.evaluation_metrics[f"eval/episode_return/{robot_type}"] = []
                        self.evaluation_metrics[f"eval/episode_length/{robot_type}"] = []
                        self.evaluation_metrics[f"eval/track_perf_perc/{robot_type}"] = []
                        self.evaluation_metrics[f"eval/eps_track_perf_perc/{robot_type}"] = []
                    while True:
                        eval_processed_action, key = get_action(policy_state, eval_state, key, self.config.algorithm.eval_stochastic)
                        eval_state, eval_reward, eval_terminated, eval_truncated, eval_info = self.eval_env.step(jax.device_get(eval_processed_action))
                        for eval_key, eval_info_value in self.eval_env.get_logging_info_dict(eval_info).items():
                            if "track_perf_perc" in eval_key:
                                name = eval_key.replace("env_info", "eval")
                                self.evaluation_metrics[name].extend(eval_info_value)
                        eval_done = eval_terminated | eval_truncated
                        for i, eval_single_done in enumerate(eval_done):
                            if eval_single_done:
                                for j, eval_env_id in enumerate(self.eval_env_ids):
                                    if i < eval_env_id and eval_nr_episodes[i] < self.evaluation_episodes:
                                        eval_nr_episodes[i] += 1  # works because only one env per type
                                        self.evaluation_metrics[f"eval/episode_return/{self.eval_robot_types_list[j-1]}"].append(
                                            self.eval_env.get_final_info_value_at_index(eval_info, f"episode_return_{self.eval_robot_types_list[j-1]}", i)
                                        )
                                        self.evaluation_metrics[f"eval/episode_length/{self.eval_robot_types_list[j-1]}"].append(
                                            self.eval_env.get_final_info_value_at_index(eval_info, f"episode_length_{self.eval_robot_types_list[j-1]}", i)
                                        )
                                        self.evaluation_metrics[f"eval/track_perf_perc/{self.eval_robot_types_list[j-1]}"].append(
                                            self.eval_env.get_final_info_value_at_index(eval_info, f"env_info/track_perf_perc/{self.eval_robot_types_list[j-1]}", i)
                                        )
                                        break
                        if np.all(eval_nr_episodes == self.evaluation_episodes):
                            break
                    self.evaluation_metrics = {key: np.mean(value) for key, value in self.evaluation_metrics.items()}
                    for metric in ["episode_return", "episode_length", "track_perf_perc"]:
                        vals = [
                            self.evaluation_metrics[f"eval/{metric}/{rt}"]
                            for rt in self.eval_robot_types_list
                        ]
                        # You can rename this key to "average", "all", etc. as needed
                        self.evaluation_metrics[f"eval/{metric}/average"] = np.mean(vals)
                    state, _ = self.env.reset()
                    self.set_train_mode()
                    return state.astype(np.float32), key
                else:
                    return old_state, key


            def evaluate_perturbed(policy_state, sigma, steps, seed):
                """One evaluation episode per robot as in evaluate_callback (mean actions), with N(0, sigma^2) noise added
                to every joint's action during the first ``steps`` steps; the noise is drawn from ``seed`` only, so two
                policies evaluated with the same seed see the same noise. Returns {robot: (return, length)}."""
                self.set_eval_mode()
                eval_state, _ = self.eval_env.reset()
                n_act = self.as_shape[0]
                mask = np.zeros((self.nr_eval_envs, n_act), np.float32)
                for i in range(self.nr_eval_envs):
                    mask[i, :n_act - self.eval_all_missing_nr_of_actions[i]] = 1.0
                rng, key, t, out = np.random.default_rng(seed), self.key, 0, {}
                while len(out) < self.nr_eval_env_types:
                    action, key = get_action(policy_state, eval_state, key, False)
                    action = np.asarray(jax.device_get(action), np.float32)
                    if t < steps and sigma > 0:
                        action = action + np.float32(sigma) * rng.standard_normal(action.shape).astype(np.float32) * mask
                    eval_state, _, eval_terminated, eval_truncated, eval_info = self.eval_env.step(action)
                    for i in np.flatnonzero(eval_terminated | eval_truncated):
                        name = str(self.eval_robot_types_list[np.searchsorted(self.eval_env_ids, i, side="right") - 1])
                        if name not in out:
                            out[name] = (float(self.eval_env.get_final_info_value_at_index(eval_info, f"episode_return_{name}", i)),
                                         float(self.eval_env.get_final_info_value_at_index(eval_info, f"episode_length_{name}", i)))
                    t += 1
                self.env.reset()
                self.set_train_mode()
                return out


            def save_and_log_callback(policy_state, value_critic_state, critic_state, target_critic_state, optimization_metrics, first_eval=False):
                # Logging (checkpoints are saved in train_offline, after the update counter has advanced)
                current_time = time.time()
                sps = int((self.batch_size * self.nr_envs) / (current_time - self.last_time))

                self.last_time = current_time
                time_metrics = {
                    "time/sps": sps
                }
                if prefetcher is not None:
                    time_metrics["time/data_wait_s"] = prefetcher.wait_s
                if not first_eval:
                    self.global_step += int(self.batch_size)
                    self.current_nr_updates += 1
                steps_metrics = {
                    "steps/nr_env_steps": self.global_step,
                    "steps/nr_env_type_steps": self.global_step // self.nr_env_types,
                    "steps/nr_updates": self.current_nr_updates,
                    "steps/nr_episodes": self.current_nr_episodes,
                }

                additional_metrics = {**self.evaluation_metrics, **steps_metrics, **time_metrics}

                # Written one update later, after the next update has been dispatched, with a single
                # device-to-host copy: the device never waits for the host's logging.
                flush_log()
                self.pending_log = (self.global_step, additional_metrics, optimization_metrics)

            def flush_log():
                if self.pending_log is None:
                    return
                step, additional_metrics, optimization_metrics = self.pending_log
                self.pending_log = None
                self.start_logging(step)
                for key, value in additional_metrics.items():
                    self.log(key, value, step)
                for key, value in jax.device_get(optimization_metrics).items():
                    self.log(key, value, step)
                self.end_logging()

            # Convert NumPy batch to JAX format
            def numpy_to_jax(batch):
                """
                Convert a NumPy batch to JAX arrays.
                """
                return jax.tree_util.tree_map(jnp.array, batch)


            @jax.jit
            def update_critic(carry, env_batches):
                def critic_loss_fn(critic_params_1, critic_params_2, value_critic_params, env_batches):
                    losses = []
                    q1s = []
                    q2s = []
                    
                    for env_type_id, batch in enumerate(env_batches):
                        observations = batch.observations
                        actions = batch.actions
                        next_observations = batch.next_observations
                        rewards = batch.rewards
                        dones = batch.dones
                        weight = self.robot_loss_weights[env_type_id]
                        
                        dynamic_joint_combined_state = observations[:, :self.dynamic_joint_observation_lengths[env_type_id*self.nr_envs_per_env_type]].reshape((-1, self.nr_dynamic_joint_observations[env_type_id*self.nr_envs_per_env_type], self.single_dynamic_joint_observation_length))
                        dynamic_joint_description = dynamic_joint_combined_state[:, :, :self.dynamic_joint_description_size]
                        dynamic_joint_state = dynamic_joint_combined_state[:, :, self.dynamic_joint_description_size:]
                        
                        dynamic_foot_combined_state = observations[:, self.dynamic_joint_observation_lengths[env_type_id*self.nr_envs_per_env_type]:self.dynamic_joint_observation_lengths[env_type_id*self.nr_envs_per_env_type] + self.dynamic_foot_observation_lengths[env_type_id*self.nr_envs_per_env_type]].reshape((-1, self.nr_dynamic_foot_observations[env_type_id*self.nr_envs_per_env_type], self.single_dynamic_foot_observation_length))
                        dynamic_foot_description = dynamic_foot_combined_state[:, :, :self.dynamic_foot_description_size]
                        dynamic_foot_state = dynamic_foot_combined_state[:, :, self.dynamic_foot_description_size:]
                        
                        next_dynamic_joint_combined_state = next_observations[:, :self.dynamic_joint_observation_lengths[env_type_id*self.nr_envs_per_env_type]].reshape((-1, self.nr_dynamic_joint_observations[env_type_id*self.nr_envs_per_env_type], self.single_dynamic_joint_observation_length))
                        next_dynamic_joint_description = next_dynamic_joint_combined_state[:, :, :self.dynamic_joint_description_size]
                        next_dynamic_joint_state = next_dynamic_joint_combined_state[:, :, self.dynamic_joint_description_size:]
                        
                        next_dynamic_foot_combined_state = next_observations[:, self.dynamic_joint_observation_lengths[env_type_id*self.nr_envs_per_env_type]:self.dynamic_joint_observation_lengths[env_type_id*self.nr_envs_per_env_type] + self.dynamic_foot_observation_lengths[env_type_id*self.nr_envs_per_env_type]].reshape((-1, self.nr_dynamic_foot_observations[env_type_id*self.nr_envs_per_env_type], self.single_dynamic_foot_observation_length))
                        next_dynamic_foot_description = next_dynamic_foot_combined_state[:, :, :self.dynamic_foot_description_size]
                        next_dynamic_foot_state = next_dynamic_foot_combined_state[:, :, self.dynamic_foot_description_size:]
                        
                        next_v = self.value_critic.apply(value_critic_params, next_dynamic_joint_description, next_dynamic_joint_state, next_dynamic_foot_description, next_dynamic_foot_state, next_observations[:, self.critic_general_state_mask[env_type_id*self.nr_envs_per_env_type]]).squeeze()

                        target_q = rewards + self.config.algorithm.discount * (1 - dones) * next_v

                        q1 = self.critic.apply(critic_params_1, dynamic_joint_description, dynamic_joint_state, dynamic_foot_description, dynamic_foot_state, observations[:, self.critic_general_state_mask[env_type_id*self.nr_envs_per_env_type]], actions).squeeze()
                        q2 = self.critic.apply(critic_params_2, dynamic_joint_description, dynamic_joint_state, dynamic_foot_description, dynamic_foot_state, observations[:, self.critic_general_state_mask[env_type_id*self.nr_envs_per_env_type]], actions).squeeze()
                        critic_loss = weight * ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
                        losses.append(critic_loss)
                        q1s.append(jnp.mean(q1))
                        q2s.append(jnp.mean(q2))
                    
                    loss_mean = jnp.mean(jnp.array(losses))
                    return loss_mean, (q1s, q2s)
                
                policy_state, value_critic_state, critic_state, target_critic_state, key = carry
                critic_state_1, critic_state_2 = critic_state
                
                grad_fn = jax.value_and_grad(critic_loss_fn, argnums=(0, 1), has_aux=True)
                (loss, (q1, q2)), (critic_1_gradients, critic_2_gradients) = grad_fn(
                    critic_state_1.params, critic_state_2.params, value_critic_state.params, env_batches
                )
                
                new_critic_state_1 = critic_state_1.apply_gradients(grads=critic_1_gradients)
                new_critic_state_2 = critic_state_2.apply_gradients(grads=critic_2_gradients)
                new_critic_state = (new_critic_state_1, new_critic_state_2)
                
                carry = policy_state, value_critic_state, new_critic_state, target_critic_state, key
                return carry, loss, q1, q2

            @jax.jit
            def update_targets(critic_state, target_critic_state):
                return tuple(self.target_update(c, t, self.config.algorithm.tau) for c, t in zip(critic_state, target_critic_state))

            @jax.jit
            def softmax_temperatures(policy_params, critic_params_1, critic_params_2):
                temps = {}
                for name, params in (("policy", policy_params), ("critic_1", critic_params_1), ("critic_2", critic_params_2)):
                    for part in ("joint", "foot"):
                        temps[f"{name}/{part}_softmax_temp"] = jnp.exp(params["params"][f"{part}_log_softmax_temperature"])[0] + self.softmax_temperature_min
                return temps

            @jax.jit
            def update_value_critic(carry, env_batches):
                def value_critic_loss_fn(value_critic_params, target_critic_params_1, target_critic_params_2, key, env_batches):
                    losses = []
                    vs = []
                    
                    for env_type_id, batch in enumerate(env_batches):
                        observations = batch.observations
                        actions = batch.actions
                        weight = self.robot_loss_weights[env_type_id]
                        
                        dynamic_joint_combined_state = observations[:, :self.dynamic_joint_observation_lengths[env_type_id*self.nr_envs_per_env_type]].reshape((-1, self.nr_dynamic_joint_observations[env_type_id*self.nr_envs_per_env_type], self.single_dynamic_joint_observation_length))
                        dynamic_joint_description = dynamic_joint_combined_state[:, :, :self.dynamic_joint_description_size]
                        dynamic_joint_state = dynamic_joint_combined_state[:, :, self.dynamic_joint_description_size:]
                        
                        dynamic_foot_combined_state = observations[:, self.dynamic_joint_observation_lengths[env_type_id*self.nr_envs_per_env_type]:self.dynamic_joint_observation_lengths[env_type_id*self.nr_envs_per_env_type] + self.dynamic_foot_observation_lengths[env_type_id*self.nr_envs_per_env_type]].reshape((-1, self.nr_dynamic_foot_observations[env_type_id*self.nr_envs_per_env_type], self.single_dynamic_foot_observation_length))
                        dynamic_foot_description = dynamic_foot_combined_state[:, :, :self.dynamic_foot_description_size]
                        dynamic_foot_state = dynamic_foot_combined_state[:, :, self.dynamic_foot_description_size:]
                        
                        q1 = self.critic.apply(
                            target_critic_params_1, dynamic_joint_description, dynamic_joint_state, dynamic_foot_description, dynamic_foot_state, observations[:, self.critic_general_state_mask[env_type_id*self.nr_envs_per_env_type]], actions
                        ).squeeze()
                        q2 = self.critic.apply(
                            target_critic_params_2, dynamic_joint_description, dynamic_joint_state, dynamic_foot_description, dynamic_foot_state, observations[:, self.critic_general_state_mask[env_type_id*self.nr_envs_per_env_type]], actions
                        ).squeeze()
                        q = jax.lax.stop_gradient(jnp.minimum(q1, q2))
                        v = self.value_critic.apply(value_critic_params, dynamic_joint_description, dynamic_joint_state, dynamic_foot_description, dynamic_foot_state, observations[:, self.critic_general_state_mask[env_type_id*self.nr_envs_per_env_type]]).squeeze()
                        value_loss = weight * self.expectile_loss(q - v, self.config.algorithm.expectile).mean()
                        losses.append(value_loss)
                        vs.append(jnp.mean(v))
                    
                    loss_mean = jnp.mean(jnp.array(losses))
                    v_mean = jnp.mean(jnp.array(vs))
                    return loss_mean, v_mean
                
                policy_state, value_critic_state, critic_state, target_critic_state, key = carry
                target_critic_state_1, target_critic_state_2 = target_critic_state
    
                key, subkey = jax.random.split(key)
                grad_fn = jax.value_and_grad(value_critic_loss_fn, argnums=(0), has_aux=True)
                (loss, v), grad = grad_fn(
                    value_critic_state.params, target_critic_state_1.params, target_critic_state_2.params, subkey, env_batches
                )
                
                new_value_state = value_critic_state.apply_gradients(grads=grad)
                carry = policy_state, new_value_state, critic_state, target_critic_state, key
                return carry, loss, v
            
            def update_policy_morse(carry, env_batches):
                policy_state, value_critic_state, critic_state, target_critic_state, key = carry
                extra = (value_critic_state.params, critic_state[0].params, critic_state[1].params)
                losses = []
                for sub in range(self.actor_steps):
                    policy_state, loss, metrics, key = self.morse.update(
                        policy_state, extra, env_batches, key, self.current_nr_updates, sub, self.actor_steps)
                    losses.append(loss)
                    if sub == 0:
                        morse_metrics = metrics
                loss = losses[0] if self.actor_steps == 1 else jnp.mean(jnp.stack(losses))
                carry = (policy_state, value_critic_state, critic_state, target_critic_state, key)
                return carry, loss, morse_metrics
            
            def update_with_variable_batch_sizes(policy_state, value_critic_state, critic_state, target_critic_state, env_batches, key):            
                # Execute the update process
                key, subkey = jax.random.split(key)
                carry = policy_state, value_critic_state, critic_state, target_critic_state, key
                
                # Update value critic
                carry, value_loss, v = update_value_critic(carry, env_batches)
                
                # Update policy
                carry, policy_loss, morse_metrics = update_policy_morse(carry, env_batches)
                
                # Update critic
                carry, critic_loss, q1, q2 = update_critic(carry, env_batches)
                
                # Get final states
                policy_state, value_critic_state, critic_state, target_critic_state, key = carry

                # Update target networks
                target_critic_state = update_targets(critic_state, target_critic_state)

                # Compute metrics
                mean_metrics = {
                    'value_loss': value_loss,
                    'critic_loss': critic_loss,
                    'policy_loss': policy_loss,
                    "v": v
                }
                mean_metrics.update(morse_metrics)

                # Add per-environment-type metrics
                for i in range(self.nr_env_types):
                    mean_metrics[f"{self.config.environment.train_robot_types[i * self.nr_envs_per_env_type]}_q1"] = q1[i]
                    mean_metrics[f"{self.config.environment.train_robot_types[i * self.nr_envs_per_env_type]}_q2"] = q2[i]
                # Add softmax temperature metrics
                mean_metrics.update(softmax_temperatures(policy_state.params, critic_state[0].params, critic_state[1].params))

                return policy_state, value_critic_state, critic_state, target_critic_state, mean_metrics, key

            def train_offline(carry, _):
                policy_state, value_critic_state, critic_state, target_critic_state, dataset, state, key = carry
                key, subkey = jax.random.split(key)

                # sampling (batch sizes proportional to the robot loss weights, self.env_batch_sizes)
                if prefetcher is not None:
                    env_batches = prefetcher.next()
                else:
                    env_batches, key = dataset.sample(subkey, self.env_batch_sizes)
                    env_batches = [numpy_to_jax(batch) for batch in env_batches]

                # model update
                env_batches = tuple(env_batches)
                n_before = self.current_nr_updates
                policy_state, value_critic_state, critic_state, target_critic_state, mean_metrics, key = update_with_variable_batch_sizes(
                    policy_state, value_critic_state, critic_state, target_critic_state, env_batches, key
                )
                # Synchronization check (diagnostic steps): runs with the same seed and data selection must take the same
                # minibatches and train the same critics, whatever their actor.
                diag = int(self.config.algorithm.morse_diag_frequency)
                if diag > 0 and n_before % diag == 0:
                    mean_metrics = {**mean_metrics, "sync/critic_norm": optax.global_norm(critic_state[0].params),
                                    "sync/value_norm": optax.global_norm(value_critic_state.params)}
                    if prefetcher is not None:
                        mean_metrics["sync/batch_checksum"] = float(prefetcher.checksum % 16777213)
                    # Content hashes (exact): the indices of every minibatch so far, and the critics' parameters after
                    # this update. Written to sync_hashes.jsonl; their first 24 bits are also logged.
                    hashes = {"update": int(n_before),
                              "batch": prefetcher.sha1.hexdigest() if prefetcher is not None else "",
                              "critic": params_sha1((critic_state[0].params, critic_state[1].params)),
                              "value": params_sha1(value_critic_state.params)}
                    if self.freeze_core:                     # the frozen core must hash the same at every logged update
                        hashes["core"] = params_sha1(policy_state.params["params"]["core"])
                    with open(self.sync_path, "a") as f:
                        f.write(json.dumps(hashes) + "\n")
                    for name in ("batch", "critic", "value", "core"):
                        if hashes.get(name):
                            mean_metrics[f"sync/{name}_hash24"] = float(int(hashes[name][:6], 16))
                
                # evaluation
                state, key = evaluate_callback(policy_state, state, key)
                evaluation = dict(self.evaluation_metrics)     # empty unless this update's policy was evaluated

                # save and log
                save_and_log_callback(policy_state, value_critic_state, critic_state, target_critic_state, mean_metrics)

                # Checkpoints after n updates hold exactly the policy after n updates; every one is recorded with
                # the evaluation of exactly these weights (models/checkpoints.jsonl, ckpt/... scalars).
                n = self.current_nr_updates
                saved = []
                if self.save_model and n % self.save_latest_frequency == 0:
                    self.save(policy_state, value_critic_state, critic_state, target_critic_state, "latest")
                    saved.append(f"tmp/checkpoint_{self.global_step}")
                if n in self.save_at:
                    self.save_at_updates(policy_state, value_critic_state, critic_state, target_critic_state, n)
                    saved.append(f"at_{n}")
                if saved and self.save_model:
                    if not evaluation:
                        state, _ = evaluate_callback(policy_state, state, jax.random.PRNGKey(0), force=True)
                        evaluation = dict(self.evaluation_metrics)
                    self.record_checkpoints(saved, n, evaluation)

                # analysis hooks (no effect on training)
                if self.probe is not None and n in self.probe_at:
                    t0 = time.time()
                    extra = (value_critic_state.params, critic_state[0].params, critic_state[1].params)
                    path = self.probe.run(policy_state.params, extra, dataset, n)
                    rlx_logger.info(f"MORSE probe at {n} updates: {path} ({time.time() - t0:.0f} s)")

                return (policy_state, value_critic_state, critic_state, target_critic_state, dataset, state, key), ()

            self.set_train_mode()

            state_shape = jax.ShapeDtypeStruct(self.np_state_shape, jnp.float32)

            state = jax.pure_callback(env_reset_callback, state_shape)

            rlx_logger.info(f"start loading dataset")
            # Data selection: rows [lo, hi) of every robot (morse_data_rows), and the dataset columns of the training
            # robots when they are not the dataset's own robots in its order (e.g. a single robot).
            a = self.config.algorithm
            rows = tuple(int(v) for v in str(a.morse_data_rows).split(",")) if getattr(a, "morse_data_rows", "") else None
            with open(resolve_data_path(a.dataset_path) / "metadata.json") as f:
                columns = dataset_columns(self.config.environment.train_robot_types, int(json.load(f)["n_envs"]))
            dataset = Dataset(a.dataset_path, a.normalize_state, mmap=a.dataset_mmap, rows=rows, columns=columns)
            # dataset = MTMemmapDataset(root=self.config.algorithm.dataset_path, normalize=self.config.algorithm.normalize_state)
            rlx_logger.info(f"finish loading dataset: {dataset.n_samples} rows of {dataset.n_envs} robots "
                            f"(rows {rows or 'all'}, columns {columns or 'all'})")
            if self.ckpt_analysis is not None:
                a = self.config.algorithm
                if a.morse_ckpt_eval:
                    def evaluate_checkpoint(variables):
                        """Deterministic evaluation of a policy, exactly as during training (one episode per robot)."""
                        evaluate_callback(self.policy_state.replace(params=variables), state, self.key, force=True)
                        return dict(self.evaluation_metrics)
                    t0 = time.time()
                    path = self.ckpt_analysis.evaluate(evaluate_checkpoint)
                    rlx_logger.info(f"MORSE checkpoint evaluation: {path} ({time.time() - t0:.0f} s)")
                if a.morse_ckpt_eval_noise:
                    t0 = time.time()
                    path = self.ckpt_analysis.evaluate_noise(
                        lambda variables, sigma, steps, seed: evaluate_perturbed(self.policy_state.replace(params=variables),
                                                                                 sigma, steps, seed),
                        a.morse_ckpt_eval_noise, int(a.morse_ckpt_eval_episodes))
                    rlx_logger.info(f"MORSE perturbed checkpoint evaluation: {path} ({time.time() - t0:.0f} s)")
                if self.coord_dump is not None:
                    t0 = time.time()
                    paths = self.ckpt_analysis.paths
                    (var_a, extra_a), (var_b, extra_b) = load_checkpoint(paths[0]), load_checkpoint(paths[1])
                    path = self.coord_dump.run(dataset, var_a, var_b, *load_end_flags(dataset.root), checkpoints=paths,
                                               extra_a=extra_a, extra_b=extra_b)
                    rlx_logger.info(f"MORSE coordination dump: {path} ({time.time() - t0:.0f} s)")
                if int(getattr(a, "morse_ckpt_grads", 0)) > 0:
                    t0 = time.time()
                    path = self.ckpt_analysis.gradient_split(dataset)
                    rlx_logger.info(f"MORSE gradient split: {path} ({time.time() - t0:.0f} s)")
                if int(getattr(a, "morse_ckpt_var", 0)) > 0:
                    t0 = time.time()
                    path = self.ckpt_analysis.variance_diagnostics(dataset)
                    rlx_logger.info(f"MORSE variance diagnostics: {path} ({time.time() - t0:.0f} s)")
                if self.conflict is not None:
                    t0 = time.time()
                    paths = self.conflict.run(dataset, self.ckpt_analysis.paths)
                    rlx_logger.info(f"MORSE gradient-conflict analysis: {paths} ({time.time() - t0:.0f} s)")
                if int(a.morse_ckpt_batches) > 0:
                    t0 = time.time()
                    path = self.ckpt_analysis.run(dataset)
                    rlx_logger.info(f"MORSE checkpoint analysis: {path} ({time.time() - t0:.0f} s)")
                return
            if self.branch_raw is not None:
                self.check_branch(dataset)
            prefetcher =BatchPrefetcher(dataset, self.env_batch_sizes, self.seed) if self.config.algorithm.data_prefetch else None
            carry = (self.policy_state, self.value_critic_state, self.critic_state, self.target_critic_state, dataset, state, self.key)

            reset_step_info_collection_callback()
            evaluate_callback(self.policy_state, state, self.key)
            save_and_log_callback(self.policy_state, self.value_critic_state, self.critic_state, self.target_critic_state, {}, first_eval=True)

            stop = int(a.morse_stop_at) or int(self.nr_offline_updates)      # the schedule keeps spanning nr_offline_updates
            for _ in range(stop - self.branch_updates):
                carry, _ = train_offline(carry, None)
            flush_log()
            if prefetcher is not None:
                prefetcher.close()
            ( self.policy_state,
            self.value_critic_state,
            self.critic_state,
            self.target_critic_state,
            _, _, _ ) = carry

        state, _ = self.env.reset()
        self.np_state_shape = state.shape
        self.evaluation_metrics = {}
        self.pending_log = None
        self.last_time = time.time()
        self.global_step = self.branch_updates * int(self.batch_size)
        self.current_nr_updates = self.branch_updates
        self.current_nr_episodes = 0
        train_loop()


    def log(self, name, value, step):
        if self.track_tb:
            if isinstance(value, jax.Array):
                value = np.array(value)
            # Convert NumPy array element type (compress as float16); MORSE diagnostics keep float32
            # because they include small gradient norms and bounded fractions.
            low = np.float32 if name.startswith(("morse/", "sync/")) else np.float16
            if isinstance(value, np.ndarray):
                if np.issubdtype(value.dtype, np.floating):
                    value = value.astype(low)
            else:
                if isinstance(value, float):
                    value = low(value)
            self.writer.add_scalar(name, value, step)
        if self.track_console:
            self.log_console(name, value)
    

    def log_console(self, name, value):
        value = np.format_float_positional(value, trim="-")
        rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │", flush=False)


    def start_logging(self, step):
        if self.track_console:
            rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐", flush=False)
        else:
            rlx_logger.info(f"Step: {step}")


    def end_logging(self):
        if self.track_console:
            rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")

    
    def save(self, policy_state, value_critic_state, critic_state, target_critic_state, type):
        critic_state_1, critic_state_2 = critic_state
        target_critic_state_1, target_critic_state_2 = target_critic_state
        checkpoint = {
            "config_algorithm": self.config.algorithm.to_dict(),
            "policy": policy_state,
            "value_critic": value_critic_state,
            "critic_1": critic_state_1,
            "critic_2": critic_state_2,
            "target_critic_1": target_critic_state_1,
            "target_critic_2": target_critic_state_2
        }
        if type == "best":
            checkpoints.save_checkpoint(ckpt_dir=f"{self.save_path}/tmp",target=checkpoint,step=self.global_step,keep=1)
        elif type == "latest":
            checkpoints.save_checkpoint(ckpt_dir=f"{self.save_path}/tmp",target=checkpoint,step=self.global_step,keep=1)


    def record_checkpoints(self, names, n, evaluation):
        """Append one line per checkpoint saved after n updates to models/checkpoints.jsonl: its folder, update count,
        architecture settings, and the deterministic evaluation of exactly its weights (one episode per robot, as
        during training); the returns are also logged as ckpt/episode_return/<robot>, apart from the eval/ curves."""
        robots = [str(rt) for rt in self.eval_robot_types_list] if self.nr_eval_envs > 0 else []
        returns = {rt: float(evaluation[f"eval/episode_return/{rt}"]) for rt in robots if f"eval/episode_return/{rt}" in evaluation}
        lengths = {rt: float(evaluation[f"eval/episode_length/{rt}"]) for rt in robots if f"eval/episode_length/{rt}" in evaluation}
        a = self.config.algorithm
        arch = {k: a[k] for k in MORSE_RECORD_KEYS if k in a}
        with open(os.path.join(self.save_path, "checkpoints.jsonl"), "a") as f:
            for name in names:
                f.write(json.dumps({"checkpoint": name, "updates": int(n), "global_step": int(self.global_step),
                                    "seed": int(self.seed), "returns": returns, "lengths": lengths,
                                    "mean_return": float(np.mean(list(returns.values()))) if returns else None,
                                    "settings": arch}) + "\n")
        if self.pending_log is not None and returns:
            self.pending_log[1].update({f"ckpt/episode_return/{rt}": v for rt, v in returns.items()})
            self.pending_log[1]["ckpt/episode_return/average"] = float(np.mean(list(returns.values())))
        rlx_logger.info(f"checkpoint {', '.join(names)} after {n} updates; exact evaluation mean return "
                        f"{np.mean(list(returns.values())) if returns else float('nan'):.1f}")

    def save_at_updates(self, policy_state, value_critic_state, critic_state, target_critic_state, n):
        """Checkpoint at a chosen number of updates (morse_save_at), kept next to the latest one."""
        checkpoint = {
            "config_algorithm": self.config.algorithm.to_dict(),
            "policy": policy_state,
            "value_critic": value_critic_state,
            "critic_1": critic_state[0],
            "critic_2": critic_state[1],
            "target_critic_1": target_critic_state[0],
            "target_critic_2": target_critic_state[1],
        }
        checkpoints.save_checkpoint(ckpt_dir=os.path.abspath(f"{self.save_path}/at_{n}"), target=checkpoint, step=n, keep=1)

    def load(config, env, run_path, writer, explicitly_set_algorithm_params):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = "/".join(splitted_path[:-1])
        checkpoint_file_name = splitted_path[-1]

        check_point_handler = orbax.checkpoint.PyTreeCheckpointHandler(aggregate_filename=checkpoint_file_name)
        checkpointer = orbax.checkpoint.Checkpointer(check_point_handler)

        loaded_algorithm_config = checkpointer.restore(checkpoint_dir)["config_algorithm"]
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params:
                config.algorithm[key] = value
        # Checkpoints saved before an architecture setting existed were trained with its legacy value.
        for key, value in LEGACY_ARCHITECTURE.items():
            if key not in loaded_algorithm_config and f"algorithm.{key}" not in explicitly_set_algorithm_params:
                config.algorithm[key] = value
        model = IQL(config, env, run_path, writer)

        # Keep only keys that exist in both saved checkpoint config_algorithm and current config
        saved_config = checkpointer.restore(checkpoint_dir)["config_algorithm"]
        filtered_config = {k: v for k, v in config.algorithm.to_dict().items() if k in saved_config}
        
        target = {
            "config_algorithm": filtered_config,
            "policy": model.policy_state,
            "value_critic": model.value_critic_state,
            "critic_1": model.critic_state[0],
            "critic_2":  model.critic_state[1],
            "target_critic_1":  model.target_critic_state[0],
            "target_critic_2":  model.target_critic_state[1]
        }
        checkpoint = checkpointer.restore(checkpoint_dir, item=target)

        model.policy_state = checkpoint["policy"]
        model.value_critic_state = checkpoint["value_critic"]
        model.critic_state = (checkpoint["critic_1"], checkpoint["critic_2"])
        model.target_critic_state = (checkpoint["target_critic_1"], checkpoint["target_critic_2"])

        if config.algorithm.reset_lr_scheduler:
            policy_optimizer = optax.chain(
                optax.clip_by_global_norm(config.algorithm.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=config.algorithm.actor_lr),
            )
            model.policy_state = model.policy_state.replace(tx=policy_optimizer, opt_state=policy_optimizer.init(model.policy_state.params))
        
        return model
    

    def test(self, episodes):
        @jax.jit
        def get_action(policy_state, state, key):
            @partial(jax.jit, static_argnums=(1,2))
            def get_action_single(state, env_id_0, env_id_1, key):
                state_b = state[env_id_0:env_id_1]

                dynamic_joint_combined_state_b = state_b[:, :self.dynamic_joint_observation_lengths[env_id_0]].reshape((-1, self.nr_dynamic_joint_observations[env_id_0], self.single_dynamic_joint_observation_length))
                dynamic_joint_description_b = dynamic_joint_combined_state_b[:, :, :self.dynamic_joint_description_size]
                dynamic_joint_state_b = dynamic_joint_combined_state_b[:, :, self.dynamic_joint_description_size:]

                dynamic_foot_combined_state_b = state_b[:, self.dynamic_joint_observation_lengths[env_id_0]:self.dynamic_joint_observation_lengths[env_id_0] + self.dynamic_foot_observation_lengths[env_id_0]].reshape((-1, self.nr_dynamic_foot_observations[env_id_0], self.single_dynamic_foot_observation_length))
                dynamic_foot_description_b = dynamic_foot_combined_state_b[:, :, :self.dynamic_foot_description_size]
                dynamic_foot_state_b = dynamic_foot_combined_state_b[:, :, self.dynamic_foot_description_size:]

                action_mean_b, action_logstd_b = self.policy.apply(policy_state.params, dynamic_joint_description_b, dynamic_joint_state_b, dynamic_foot_description_b, dynamic_foot_state_b, state_b[:, self.policy_general_state_mask[env_id_0]])

                if self.config.algorithm.eval_stochastic:
                    action_std = jnp.exp(action_logstd_b)
                    key, subkey = jax.random.split(key)
                    action = action_mean_b + action_std * jax.random.normal(subkey, shape=action_mean_b.shape)
                else:
                    action = action_mean_b
                action = jnp.concatenate([action, jnp.zeros((self.nr_envs_per_env_type, self.missing_nr_of_actions[env_id_0]))], axis=1)
                processed_action = self.get_processed_action(action)

                return processed_action, key


            processed_actions = []
            for i in range(self.env_ids.shape[0] - 1):
                env_id_0 = self.env_ids[i]
                env_id_1 = self.env_ids[i + 1]
                processed_action, key = get_action_single(state, env_id_0, env_id_1, key)
                processed_actions.append(processed_action)
            processed_action = jnp.concatenate(processed_actions, axis=0)

            return processed_action, key
        

        @partial(jax.jit, static_argnums=(2))
        def get_action_multi_render(policy_state, state, env_id, key):
            dynamic_joint_combined_state_b = state[:, :self.dynamic_joint_observation_lengths[env_id]].reshape((-1, self.nr_dynamic_joint_observations[env_id], self.single_dynamic_joint_observation_length))
            dynamic_joint_description_b = dynamic_joint_combined_state_b[:, :, :self.dynamic_joint_description_size]
            dynamic_joint_state_b = dynamic_joint_combined_state_b[:, :, self.dynamic_joint_description_size:]

            dynamic_foot_combined_state_b = state[:, self.dynamic_joint_observation_lengths[env_id]:self.dynamic_joint_observation_lengths[env_id] + self.dynamic_foot_observation_lengths[env_id]].reshape((-1, self.nr_dynamic_foot_observations[env_id], self.single_dynamic_foot_observation_length))
            dynamic_foot_description_b = dynamic_foot_combined_state_b[:, :, :self.dynamic_foot_description_size]
            dynamic_foot_state_b = dynamic_foot_combined_state_b[:, :, self.dynamic_foot_description_size:]

            action_mean_b, action_logstd_b = self.policy.apply(policy_state.params, dynamic_joint_description_b, dynamic_joint_state_b, dynamic_foot_description_b, dynamic_foot_state_b, state[:, self.policy_general_state_mask[env_id]])
            if self.config.algorithm.eval_stochastic:
                action_std = jnp.exp(action_logstd_b)
                key, subkey = jax.random.split(key)
                action = action_mean_b + action_std * jax.random.normal(subkey, shape=action_mean_b.shape)
            else:
                action = action_mean_b
            action = jnp.concatenate([action, jnp.zeros((self.nr_envs_per_env_type, self.missing_nr_of_actions[env_id]))], axis=1)
            processed_action = self.get_processed_action(action)

            return processed_action, key
        
        key  = self.key
        
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            episode_return = 0
            state, _ = self.env.reset()
            while True:
                if self.multi_render:
                    processed_action, key = get_action_multi_render(self.policy_state, state, self.env.active_env_id, key)
                else:
                    processed_action, key = get_action(self.policy_state, state, key)
                state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
                done = terminated | truncated
                episode_return += reward
                if done[0]:
                    print(f"terminated: {terminated[0]}, truncated: {truncated[0]}")
                    break
            rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")
        self.env.close()
            
    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...


    def general_properties():
        return GeneralProperties
