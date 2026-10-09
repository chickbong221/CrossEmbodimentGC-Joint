from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name
    
    config.device = "gpu"  # cpu, gpu
    config.nr_offline_updates = 1e5
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.value_lr = 3e-4
    config.batch_size = 1024
    config.softmax_temperature = 1.0
    config.softmax_temperature_min = 0.015
    config.stability_epsilon = 1e-8
    config.max_grad_norm = 0.5
    config.std_dev = 1.0
    config.policy_mean_abs_clip = 10.0
    config.policy_std_min_clip = 1e-8
    config.policy_std_max_clip = 2.0
    config.evaluation_frequency = 100
    config.evaluation_episodes = 10
    config.save_latest_frequency = 1000
    config.determine_fastest_cpu_for_gpu = False
    config.dataset_path='data/offline_data/all_robots_expert_forward_1m'
    config.dataset_mmap = False    # memory-map the dataset files instead of loading them into RAM
    config.data_prefetch = True    # sample minibatches in background threads (NumPy generator seeded with
                                   # environment.seed); False: the original in-loop sampler on the training key
    # IQL SPECIFIC
    config.normalize_state = False
    config.expectile = 0.7 
   
    config.beta = 3.0
    config.tau = 0.005
    config.discount = 0.99

    config.eval_stochastic = False

    config.reset_lr_scheduler = False

    config.robot_weights = {
        "unitree_a1": 1.0,
        "unitree_go1": 1.0,
        "unitree_go2": 1.0,
        "anymal_b": 1.0,
        "anymal_c": 1.0,
        "barkour_v0": 1.0,
        "barkour_vb": 1.0,
        "badger": 1.0,
        "bittle": 1.0,
        "unitree_h1": 1.0,
        "unitree_g1": 1.0,
        "talos": 1.0,
        "robotis_op3": 1.0,
        "nao_v5": 1.0,
        "cassie": 1.0,
        "hexapod": 1.0,
    }

    config.M = 4
    config.joint_encoder_hidden = 0   # > 0: EG variant with one hidden layer of this width in the policy's joint-state encoder (0: original)
    config.embodiment_distance_path = "data/meta/embodiment_distance.npy"
     
    return config
