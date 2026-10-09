from cross_embodiment_offline_rl.algorithms.uni_iql_eg.iql.default_config import get_config as get_base_config
from cross_embodiment_offline_rl.algorithms.morse.config import add_morse_config


def get_config(algorithm_name):
    return add_morse_config(get_base_config(algorithm_name))
