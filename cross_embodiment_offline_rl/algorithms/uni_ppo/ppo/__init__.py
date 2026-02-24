from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from cross_embodiment_offline_rl.algorithms.uni_ppo.ppo.ppo import PPO
from cross_embodiment_offline_rl.algorithms.uni_ppo.ppo.default_config import get_config
from cross_embodiment_offline_rl.algorithms.uni_ppo.ppo.general_properties import GeneralProperties


PPO_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(PPO_FLAX, get_config, PPO, GeneralProperties)
