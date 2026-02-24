from ml_collections import config_dict


def get_config(environment_name):
    config = config_dict.ConfigDict()

    config.name = environment_name

    config.nr_envs = 1
    config.nr_eval_envs = 1
    config.train_robot_types = (
        "anymal_b",
    )
    config.eval_robot_types = (
        "anymal_b",
    )

    config.seed = 1
    config.render = False
    config.mode = "train"
    config.control_type = "rudin2022"
    config.command_type = "random"
    config.command_sampling_type = "step_probability"
    config.initial_state_type = "default"
    config.reward_type = "rudin_own_var"
    config.termination_type = "rudin2022"
    config.domain_randomization_sampling_type = "step_probability"
    config.domain_randomization_action_delay_type = "default"
    config.domain_randomization_mujoco_model_type = "default"
    config.domain_randomization_control_type = "default"
    config.domain_randomization_perturbation_type = "default"
    config.domain_randomization_perturbation_sampling_type = "step_probability"
    config.domain_randomization_seen_robot_type = "default"
    config.domain_randomization_unseen_robot_type = "default"
    config.observation_noise_type = "default"
    config.observation_dropout_type = "default"
    config.terrain_type = "plane"
    config.missing_value = 0.0
    config.add_goal_arrow = False
    config.timestep = 0.005
    config.episode_length_in_seconds = 20

    config.multi_robot_max_observation_size = -1
    config.multi_robot_max_action_size = -1
    config.nr_envs_per_type = 1

    config.cpu_id = None
    config.multi_render = False

    return config
