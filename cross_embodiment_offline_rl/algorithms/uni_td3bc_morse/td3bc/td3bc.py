import os
import pickle
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
from jax.flatten_util import ravel_pytree

from cross_embodiment_offline_rl.algorithms.uni_td3bc_eg.td3bc.general_properties import GeneralProperties
from cross_embodiment_offline_rl.algorithms.uni_td3bc_eg.td3bc.critic import get_critic
from cross_embodiment_offline_rl.algorithms.morse.policy import get_morse_policy
from cross_embodiment_offline_rl.algorithms.morse.updater import MorseActorUpdater
from cross_embodiment_offline_rl.algorithms.utils import Dataset, BatchPrefetcher, robot_batch_sizes

rlx_logger = logging.getLogger("rl_x")

class TD3BC:
    def __init__(self, config, env, run_path, writer) -> None:
        self.config = config
        self.env = env.train_env
        self.eval_env = env.eval_env
        self.writer = writer
        self.save_model = config.runner.save_model
        self.save_path = os.path.join(run_path, "models")
        self.track_console = config.runner.track_console
        self.track_tb = config.runner.track_tb
        self.track_wandb = config.runner.track_wandb
        self.seed = config.environment.seed
        self.nr_envs = config.environment.nr_envs
        self.nr_eval_envs = config.environment.nr_eval_envs
        self.actor_lr = config.algorithm.actor_lr
        self.critic_lr = config.algorithm.critic_lr
        self.policy_freq = config.algorithm.policy_freq  # TD3BC: policy update frequency
        self.policy_noise_std = config.algorithm.policy_noise_std  # TD3BC: target policy noise
        self.policy_noise_clip = config.algorithm.policy_noise_clip  # TD3BC: target policy noise clip
        self.alpha = config.algorithm.alpha  # TD3BC: BC loss weight
        self.nr_epochs = config.algorithm.nr_epochs
        self.softmax_temperature_min = config.algorithm.softmax_temperature_min
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes
        self.save_latest_frequency = config.algorithm.save_latest_frequency
        self.determine_fastest_cpu_for_gpu = config.algorithm.determine_fastest_cpu_for_gpu
        self.multi_render = config.environment.multi_render
        self.batch_size = config.algorithm.batch_size * config.environment.nr_envs 
        self.nr_offline_updates = config.algorithm.nr_offline_updates
        self.minibatch_size = config.algorithm.batch_size
        self.nr_minibatches = self.batch_size // self.minibatch_size
        self.nr_env_types = len(config.environment.train_robot_types)
        self.nr_eval_env_types = len(config.environment.eval_robot_types)
        self.robot_loss_weights = jnp.array([config.algorithm.robot_weights[robot_type] for robot_type in config.environment.train_robot_types])
        self.env_batch_sizes = robot_batch_sizes([config.algorithm.robot_weights[robot_type] for robot_type in config.environment.train_robot_types],
                                                 config.algorithm.batch_size, self.nr_env_types)


        rlx_logger.info(f"Using device: {jax.default_backend()}")

        if self.determine_fastest_cpu_for_gpu and self.env.fastest_cpu_id is not None:
            p = psutil.Process()
            p.cpu_affinity([self.env.fastest_cpu_id,])
            rlx_logger.info(f"Using fastest CPU for GPU connection: {self.env.fastest_cpu_id}")
        
        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_1_key, critic_2_key = jax.random.split(self.key, 4)

        self.os_shape = self.env.single_observation_space.shape
        self.as_shape = self.env.single_action_space.shape

        self.policy = get_morse_policy(config, deterministic=True)
        self.get_processed_action = jax.jit(lambda action: action)
        self.critic = get_critic(config, self.env)

        self.policy.apply = jax.jit(self.policy.apply)
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

        if int(config.algorithm.actor_steps_per_update) != 1:
            raise NotImplementedError("actor_steps_per_update > 1 is implemented for IQL only")
        schedule_fn = optax.cosine_decay_schedule(-self.actor_lr, self.nr_offline_updates)

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, dummy_dynamic_joint_description, dummy_dynamic_joint_state, dummy_dynamic_foot_description, dummy_dynamic_foot_state, dummy_general_policy_state),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.scale_by_adam(),
                optax.scale_by_schedule(schedule_fn),
            )
        )

        self.target_policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy_state.params,  # Copy initial parameters
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.scale_by_adam(),
                optax.scale_by_schedule(schedule_fn),
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

        def policy_loss(policy_output, obs_batch, act_batch, extra, key, env_type_id):
            """TD3+BC policy loss of one robot: -lambda * Q(s, pi(s)) + behavior cloning."""
            critic1_params, critic2_params = extra
            env_id = env_type_id * self.nr_envs_per_env_type
            dj_desc, dj_state, df_desc, df_state, _ = split_obs(obs_batch, env_type_id)
            critic_general_state = obs_batch[:, self.critic_general_state_mask[env_id]]
            mean = policy_output
            missing = self.all_missing_nr_of_actions[env_type_id]
            predicted_action = jnp.concatenate([mean, jnp.zeros((mean.shape[0], missing))], axis=1)
            q1 = self.critic.apply(jax.lax.stop_gradient(critic1_params), dj_desc, dj_state, df_desc, df_state, critic_general_state, predicted_action).squeeze()
            q2 = self.critic.apply(jax.lax.stop_gradient(critic2_params), dj_desc, dj_state, df_desc, df_state, critic_general_state, predicted_action).squeeze()
            q_value = jnp.minimum(q1, q2)
            loss_lambda = self.alpha / jax.lax.stop_gradient(jnp.abs(q_value).mean())
            bc_loss = jnp.square(mean - act_batch[:, :self.as_shape[0] - missing]).mean()
            return -q_value.mean() * loss_lambda + bc_loss

        self.morse = MorseActorUpdater(config, get_morse_policy(config, capture=True, deterministic=True), split_obs,
                                       list(self.robot_types_list), run_path)
        self.morse.loss_fn = policy_loss


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
            def get_action(policy_state, state, key):
                @partial(jax.jit, static_argnums=(1,2))
                def get_single_deterministic_eval_action(key, env_id_0, env_id_1):
                    state_b = state[env_id_0:env_id_1]

                    dynamic_joint_combined_state = state_b[:, :self.eval_dynamic_joint_observation_lengths[env_id_0]].reshape((-1, self.eval_nr_dynamic_joint_observations[env_id_0], self.eval_single_dynamic_joint_observation_length))
                    dynamic_joint_description = dynamic_joint_combined_state[:, :, :self.eval_dynamic_joint_description_size]
                    dynamic_joint_state = dynamic_joint_combined_state[:, :, self.eval_dynamic_joint_description_size:]

                    dynamic_foot_combined_state = state_b[:, self.eval_dynamic_joint_observation_lengths[env_id_0]:self.eval_dynamic_joint_observation_lengths[env_id_0] + self.eval_dynamic_foot_observation_lengths[env_id_0]].reshape((-1, self.eval_nr_dynamic_foot_observations[env_id_0], self.eval_single_dynamic_foot_observation_length))
                    dynamic_foot_description = dynamic_foot_combined_state[:, :, :self.eval_dynamic_foot_description_size]
                    dynamic_foot_state = dynamic_foot_combined_state[:, :, self.eval_dynamic_foot_description_size:]

                    action = self.policy.apply(policy_state.params, dynamic_joint_description, dynamic_joint_state, dynamic_foot_description, dynamic_foot_state, state_b[:, self.eval_policy_general_state_mask[env_id_0]])

                    action = jnp.concatenate([action, jnp.zeros((self.eval_nr_envs_per_env_type, self.eval_all_missing_nr_of_actions[env_id_0]))], axis=1)
                    processed_action = self.get_processed_action(action)

                    return processed_action, key
            
                processed_actions = []
                for i in range(self.eval_env_ids.shape[0] - 1):
                    env_id_0 = self.eval_env_ids[i]
                    env_id_1 = self.eval_env_ids[i + 1]
                    processed_action, key = get_single_deterministic_eval_action(key, env_id_0, env_id_1)
                    processed_actions.append(processed_action)
                
                return jnp.concatenate(processed_actions), key
            

            def env_reset_callback():
                state, _ = self.env.reset()
                state = state.astype(np.float32)
                return state
            

            def reset_step_info_collection_callback():
                self.step_info_collection = {}

            def evaluate_callback(policy_state, old_state, key):
                self.evaluation_metrics = {}
                if self.nr_eval_envs > 0 and (self.current_nr_updates+1) % self.evaluation_frequency == 0 and self.evaluation_frequency != -1:
                    self.set_eval_mode()
                    eval_state, _ = self.eval_env.reset()
                    eval_nr_episodes = np.zeros(self.nr_eval_env_types, dtype=int)
                    for robot_type in self.eval_robot_types_list:
                        self.evaluation_metrics[f"eval/episode_return/{robot_type}"] = []
                        self.evaluation_metrics[f"eval/episode_length/{robot_type}"] = []
                        self.evaluation_metrics[f"eval/track_perf_perc/{robot_type}"] = []
                        self.evaluation_metrics[f"eval/eps_track_perf_perc/{robot_type}"] = []
                    while True:
                        eval_processed_action, key = get_action(policy_state, eval_state, key)
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


            def save_and_log_callback(policy_state, target_policy_state, critic_state, target_critic_state, optimization_metrics, first_eval=False):
                if self.save_model and self.current_nr_updates % self.save_latest_frequency == 0 and not first_eval:
                    self.save(policy_state, target_policy_state, critic_state, target_critic_state, "latest")

                # Logging
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

                rollout_info_metrics = {}
                env_info_metrics = {}

                additional_metrics = {**rollout_info_metrics, **self.evaluation_metrics, **env_info_metrics, **steps_metrics, **time_metrics}

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
            

            # Modify update functions to handle different batch sizes per environment

            @jax.jit
            def update_critic(carry, env_batches):
                def critic_loss_fn(critic_params_1, critic_params_2, target_critic_params_1, target_critic_params_2, target_policy_params, env_batches, key):
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
                        
                        next_action = self.policy.apply(target_policy_params, next_dynamic_joint_description, next_dynamic_joint_state, next_dynamic_foot_description, next_dynamic_foot_state, next_observations[:, self.policy_general_state_mask[env_type_id*self.nr_envs_per_env_type]])


                        key, subk = jax.random.split(key)
                        policy_noise = jax.random.normal(subk, next_action.shape) * self.policy_noise_std

                        target_next_action = next_action + jnp.clip(policy_noise, -self.policy_noise_clip, self.policy_noise_clip)

                        target_next_action = jnp.concatenate([target_next_action, jnp.zeros((target_next_action.shape[0], self.all_missing_nr_of_actions[env_type_id]))], axis=1)

                        next_q1 = self.critic.apply(target_critic_params_1, next_dynamic_joint_description, next_dynamic_joint_state, next_dynamic_foot_description, next_dynamic_foot_state, next_observations[:, self.critic_general_state_mask[env_type_id*self.nr_envs_per_env_type]], target_next_action)
                        next_q2 = self.critic.apply(target_critic_params_2, next_dynamic_joint_description, next_dynamic_joint_state, next_dynamic_foot_description, next_dynamic_foot_state, next_observations[:, self.critic_general_state_mask[env_type_id*self.nr_envs_per_env_type]], target_next_action)
                        target_next_q = rewards + self.config.algorithm.discount * (1 - dones) * jnp.minimum(next_q1, next_q2).squeeze()

                        q1 = self.critic.apply(critic_params_1, dynamic_joint_description, dynamic_joint_state, dynamic_foot_description, dynamic_foot_state, observations[:, self.critic_general_state_mask[env_type_id*self.nr_envs_per_env_type]], actions).squeeze()
                        q2 = self.critic.apply(critic_params_2, dynamic_joint_description, dynamic_joint_state, dynamic_foot_description, dynamic_foot_state, observations[:, self.critic_general_state_mask[env_type_id*self.nr_envs_per_env_type]], actions).squeeze()
                        critic_loss = weight * ((q1 - target_next_q) ** 2 + (q2 - target_next_q) ** 2).mean()
                        losses.append(critic_loss)
                        q1s.append(jnp.mean(q1))
                        q2s.append(jnp.mean(q2))
                    
                    loss_mean = jnp.mean(jnp.array(losses))
                    return loss_mean, (q1s, q2s)

                policy_state, target_policy_state, critic_state, target_critic_state, key = carry
                critic_state_1, critic_state_2 = critic_state
                target_critic_state_1, target_critic_state_2 = target_critic_state
                key, subk = jax.random.split(key)
                grad_fn = jax.value_and_grad(critic_loss_fn, argnums=(0, 1), has_aux=True)
                (loss, (q1, q2)), (critic_1_gradients, critic_2_gradients) = grad_fn(
                    critic_state_1.params, critic_state_2.params, target_critic_state_1.params, target_critic_state_2.params, target_policy_state.params, env_batches, subk
                )
                
                new_critic_state_1 = critic_state_1.apply_gradients(grads=critic_1_gradients)
                new_critic_state_2 = critic_state_2.apply_gradients(grads=critic_2_gradients)
                new_critic_state = (new_critic_state_1, new_critic_state_2)

                carry = policy_state, target_policy_state, new_critic_state, target_critic_state, key
                return carry, loss, q1, q2
            
            def update_policy_morse(carry, env_batches):
                policy_state, target_policy_state, critic_state, target_critic_state, key = carry
                extra = (critic_state[0].params, critic_state[1].params)
                policy_state, loss, morse_metrics, key = self.morse.update(
                    policy_state, extra, env_batches, key, self.current_nr_updates)
                return (policy_state, target_policy_state, critic_state, target_critic_state, key), loss, morse_metrics

            @jax.jit
            def update_targets(policy_state, target_policy_state, critic_state, target_critic_state):
                tau = self.config.algorithm.tau
                return (self.target_update(policy_state, target_policy_state, tau),
                        tuple(self.target_update(c, t, tau) for c, t in zip(critic_state, target_critic_state)))

            @jax.jit
            def softmax_temperatures(policy_params, critic_params_1, critic_params_2):
                temps = {}
                for name, params in (("policy", policy_params), ("critic_1", critic_params_1), ("critic_2", critic_params_2)):
                    for part in ("joint", "foot"):
                        temps[f"{name}/{part}_softmax_temp"] = jnp.exp(params["params"][f"{part}_log_softmax_temperature"])[0] + self.softmax_temperature_min
                return temps

            def update_with_variable_batch_sizes(policy_state, target_policy_state, critic_state, target_critic_state, env_batches, key):
                # Execute the update process
                key, subkey = jax.random.split(key)
                carry = policy_state, target_policy_state, critic_state, target_critic_state, key

                # Update critic
                carry, critic_loss, q1, q2 = update_critic(carry, env_batches)

                mean_metrics = {
                    'critic_loss': critic_loss
                }
                policy_state, target_policy_state, critic_state, target_critic_state, key = carry

                if self.current_nr_updates % self.policy_freq  == 0:
                    # Update policy
                    carry, policy_loss, morse_metrics = update_policy_morse(carry, env_batches)
                    
                    policy_state, target_policy_state, critic_state, target_critic_state, key = carry

                    # Update target networks
                    target_policy_state, target_critic_state = update_targets(
                        policy_state, target_policy_state, critic_state, target_critic_state)

                    mean_metrics['policy_loss'] = policy_loss
                    mean_metrics.update(morse_metrics)

                # Add per-environment-type metrics
                for i in range(self.nr_env_types):
                    mean_metrics[f"{self.robot_types_list[i]}_q1"] = q1[i]
                    mean_metrics[f"{self.robot_types_list[i]}_q2"] = q2[i]
                # Add softmax temperature metrics
                mean_metrics.update(softmax_temperatures(policy_state.params, critic_state[0].params, critic_state[1].params))

                return policy_state, target_policy_state, critic_state, target_critic_state, mean_metrics, key

            def train_offline(carry, _):
                policy_state, target_policy_state, critic_state, target_critic_state, dataset, state, key = carry
                key, subkey = jax.random.split(key)

                # Sample per environment type (batch sizes proportional to the robot loss weights, self.env_batch_sizes)
                if prefetcher is not None:
                    env_batches = prefetcher.next()
                else:
                    env_batches, key = dataset.sample(subkey, self.env_batch_sizes)
                    env_batches = [numpy_to_jax(batch) for batch in env_batches]

                # Update model
                env_batches = tuple(env_batches)
                policy_state, target_policy_state, critic_state, target_critic_state, mean_metrics, key = update_with_variable_batch_sizes(
                    policy_state, target_policy_state, critic_state, target_critic_state, env_batches, key
                )
                
                # Evaluation
                state, key = evaluate_callback(policy_state, state, key)
                
                # Save and log
                save_and_log_callback(policy_state, target_policy_state, critic_state, target_critic_state, mean_metrics)

                return (policy_state, target_policy_state, critic_state, target_critic_state, dataset, state, key), ()

            self.set_train_mode()

            state_shape = jax.ShapeDtypeStruct(self.np_state_shape, jnp.float32)

            state = jax.pure_callback(env_reset_callback, state_shape)

            rlx_logger.info(f"start loading dataset")
            dataset = Dataset(self.config.algorithm.dataset_path, self.config.algorithm.normalize_state, mmap=self.config.algorithm.dataset_mmap)
            rlx_logger.info(f"finish loading dataset")
            prefetcher = BatchPrefetcher(dataset, self.env_batch_sizes, self.seed) if self.config.algorithm.data_prefetch else None
            carry = (self.policy_state, self.target_policy_state, self.critic_state, self.target_critic_state, dataset, state, self.key)

            reset_step_info_collection_callback()
            evaluate_callback(self.policy_state, state, self.key)
            save_and_log_callback(self.policy_state, self.target_policy_state, self.critic_state, self.target_critic_state, {}, first_eval=True)

            for _ in range(int(self.nr_offline_updates)):
                carry, _ = train_offline(carry, None)
            flush_log()
            if prefetcher is not None:
                prefetcher.close()
            ( self.policy_state,
             self.target_policy_state,
            self.critic_state,
            self.target_critic_state,
            _, _, _ ) = carry

        state, _ = self.env.reset()
        self.np_state_shape = state.shape
        self.evaluation_metrics = {}
        self.pending_log = None
        self.last_time = time.time()
        self.global_step = 0
        self.current_nr_updates = 0
        self.current_nr_episodes = 0
        train_loop()


    def log(self, name, value, step):
        if self.track_tb:
            if isinstance(value, jax.Array):
                value = np.array(value)
            # Convert NumPy array element type (compress as float16); MORSE diagnostics keep float32
            # because they include small gradient norms and bounded fractions.
            low = np.float32 if name.startswith("morse/") else np.float16
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

    
    def save(self, policy_state, target_policy_state, critic_state, target_critic_state, type):
        critic_state_1, critic_state_2 = critic_state
        target_critic_state_1, target_critic_state_2 = target_critic_state
        checkpoint = {
            "config_algorithm": self.config.algorithm.to_dict(),
            "policy": policy_state,
            "target_policy": target_policy_state,
            "critic_1": critic_state_1,
            "critic_2": critic_state_2,
            "target_critic_1": target_critic_state_1,
            "target_critic_2": target_critic_state_2
        }
        if type == "best":
            checkpoints.save_checkpoint(ckpt_dir=f"{self.save_path}/tmp",target=checkpoint,step=self.global_step,keep=1)
        elif type == "latest":
            checkpoints.save_checkpoint(ckpt_dir=f"{self.save_path}/tmp",target=checkpoint,step=self.global_step,keep=1)


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
        model = TD3BC(config, env, run_path, writer)

        # Keep only keys that exist in both saved checkpoint config_algorithm and current config
        saved_config = checkpointer.restore(checkpoint_dir)["config_algorithm"]
        filtered_config = {k: v for k, v in config.algorithm.to_dict().items() if k in saved_config}
        
        target = {
            "config_algorithm": filtered_config,
            "policy": model.policy_state,
            "target_policy": model.target_policy_state,
            "critic_1": model.critic_state[0],
            "critic_2":  model.critic_state[1],
            "target_critic_1":  model.target_critic_state[0],
            "target_critic_2":  model.target_critic_state[1]
        }
        checkpoint = checkpointer.restore(checkpoint_dir, item=target)

        model.policy_state = checkpoint["policy"]
        model.target_policy_state = checkpoint["target_policy"]
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

                action_mean_b = self.policy.apply(policy_state.params, dynamic_joint_description_b, dynamic_joint_state_b, dynamic_foot_description_b, dynamic_foot_state_b, state_b[:, self.policy_general_state_mask[env_id_0]])

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

            action_mean_b = self.policy.apply(policy_state.params, dynamic_joint_description_b, dynamic_joint_state_b, dynamic_foot_description_b, dynamic_foot_state_b, state[:, self.policy_general_state_mask[env_id]])

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
