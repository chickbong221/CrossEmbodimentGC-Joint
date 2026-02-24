import gymnasium as gym

from cross_embodiment_offline_rl.environments.anymal_b.environment import AnymalB, TrainAndEvalEnvironment
from cross_embodiment_offline_rl.environments.anymal_b.wrappers import RLXInfo, RecordEpisodeStatistics
from cross_embodiment_offline_rl.environments.anymal_b.general_properties import GeneralProperties


def create_env(config):
    def make_env(seed, mode="train"):
        def thunk():
            env = AnymalB(
                seed=seed,
                render=config.environment.render,
                mode=mode,
                control_type=config.environment.control_type,
                command_type=config.environment.command_type,
                command_sampling_type=config.environment.command_sampling_type,
                initial_state_type=config.environment.initial_state_type,
                reward_type=config.environment.reward_type,
                termination_type=config.environment.termination_type,
                domain_randomization_sampling_type=config.environment.domain_randomization_sampling_type,
                domain_randomization_action_delay_type=config.environment.domain_randomization_action_delay_type,
                domain_randomization_mujoco_model_type=config.environment.domain_randomization_mujoco_model_type,
                domain_randomization_seen_robot_type=config.environment.domain_randomization_seen_robot_type,
                domain_randomization_unseen_robot_type=config.environment.domain_randomization_unseen_robot_type,
                domain_randomization_perturbation_type=config.environment.domain_randomization_perturbation_type,
                domain_randomization_perturbation_sampling_type=config.environment.domain_randomization_perturbation_sampling_type,
                observation_noise_type=config.environment.observation_noise_type,
                observation_dropout_type=config.environment.observation_dropout_type,
                terrain_type=config.environment.terrain_type,
                missing_value=config.environment.missing_value,
                add_goal_arrow=config.environment.add_goal_arrow,
                timestep=config.environment.timestep,
                episode_length_in_seconds=config.environment.episode_length_in_seconds,
                total_nr_envs=config.environment.nr_envs,
                multi_robot_max_observation_size=config.environment.multi_robot_max_observation_size,
                multi_robot_max_action_size=config.environment.multi_robot_max_action_size,
                nr_envs_per_type=config.environment.nr_envs_per_type,
                cpu_id=config.environment.cpu_id
            )
            env = RecordEpisodeStatistics(env)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk
    
    fastest_cpu_id = None

    vector_environment_class = gym.vector.SyncVectorEnv if config.environment.nr_envs == 1 else gym.vector.AsyncVectorEnv
    env = vector_environment_class([make_env(config.environment.seed + i, mode=config.environment.mode) for i in range(config.environment.nr_envs)])
    env = RLXInfo(env)
    env.fastest_cpu_id = fastest_cpu_id
    env.general_properties = GeneralProperties

    env.reset(seed=config.environment.seed)

    eval_env = vector_environment_class([make_env(config.environment.seed + i, mode="test") for i in range(config.environment.nr_eval_envs)])
    eval_env = RLXInfo(eval_env)
    eval_env.general_properties = GeneralProperties

    eval_env.reset(seed=config.environment.seed)

    env = TrainAndEvalEnvironment(env, eval_env)

    return env
