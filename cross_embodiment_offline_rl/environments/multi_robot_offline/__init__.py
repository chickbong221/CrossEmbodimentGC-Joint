from rl_x.environments.environment_manager import (
    extract_environment_name_from_file,
    register_environment,
)
from cross_embodiment_offline_rl.environments.multi_robot_offline.create_env import (
    create_env,
)
from cross_embodiment_offline_rl.environments.multi_robot_offline.default_config import (
    get_config,
)
from cross_embodiment_offline_rl.environments.multi_robot_offline.general_properties import (
    GeneralProperties,
)


MULTI_ROBOT_OFFLINE = extract_environment_name_from_file(__file__)
register_environment(MULTI_ROBOT_OFFLINE, get_config, create_env, GeneralProperties)
