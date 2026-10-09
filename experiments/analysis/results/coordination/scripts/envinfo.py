"""Observation layout, joint/foot names, nominal positions and action scaling of every robot (one env each, CPU)."""
import json, sys
import numpy as np
from cross_embodiment_offline_rl.environments.multi_robot_offline.default_config import get_config
from cross_embodiment_offline_rl.environments.multi_robot.robot_helper import ROBOTS

c = get_config("multi_robot_offline")
out = {}
for rt in c.train_robot_types:
    robot = [r for r in ROBOTS if r.long_name == rt][0]
    env = robot.cls(
        seed=0, render=False, mode="train", control_type=c.control_type, command_type="forward",
        command_sampling_type=c.command_sampling_type, initial_state_type=c.initial_state_type,
        reward_type=c.reward_type, termination_type=getattr(c, f"termination_type_{robot.short_name}"),
        domain_randomization_sampling_type=c.domain_randomization_sampling_type,
        domain_randomization_action_delay_type=c.domain_randomization_action_delay_type,
        domain_randomization_mujoco_model_type=c.domain_randomization_mujoco_model_type,
        domain_randomization_seen_robot_type=c.domain_randomization_seen_robot_type,
        domain_randomization_unseen_robot_type=c.domain_randomization_unseen_robot_type,
        domain_randomization_perturbation_type=c.domain_randomization_perturbation_type,
        domain_randomization_perturbation_sampling_type=c.domain_randomization_perturbation_sampling_type,
        observation_noise_type=c.observation_noise_type, observation_dropout_type=c.observation_dropout_type,
        terrain_type=c.terrain_type, missing_value=c.missing_value, add_goal_arrow=False, timestep=c.timestep,
        episode_length_in_seconds=c.episode_length_in_seconds, total_nr_envs=16,
        multi_robot_max_observation_size=668, multi_robot_max_action_size=24, nr_envs_per_type=1, cpu_id=None,
        test_command_path="command_forward.txt")
    ids = env.observation_name_to_id
    names = sorted(ids, key=ids.get)
    cf = env.control_function
    out[robot.short_name] = dict(
        long_name=rt, obs_names=names, joint_names=list(env.joint_names), foot_names=list(env.foot_names),
        nominal=[float(x) for x in env.nominal_joint_positions], scaling_factor=float(cf.scaling_factor),
        p_gain=float(cf.p_gain), d_gain=float(cf.d_gain), horizon=int(env.horizon),
        control_hz=float(env.control_frequency_hz), joint_desc=int(env.dynamic_joint_description_size),
        foot_desc=int(env.dynamic_foot_description_size))
    print(robot.short_name, len(env.joint_names), "joints", len(env.foot_names), "feet", "obs", len(names) - names.count("x"),
          "horizon", env.horizon, "scale", cf.scaling_factor, flush=True)
    env.close()
json.dump(out, open(sys.argv[1], "w"), indent=1)
