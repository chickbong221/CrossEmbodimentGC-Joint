from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from cross_embodiment_offline_rl.algorithms.uni_bc_eg.bc.bc import BC
from cross_embodiment_offline_rl.algorithms.uni_bc_eg.bc.default_config import get_config
from cross_embodiment_offline_rl.algorithms.uni_bc_eg.bc.general_properties import GeneralProperties


BC_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(BC_FLAX, get_config, BC, GeneralProperties)
