from cross_embodiment_offline_rl.environments.unitree_h1.domain_randomization.seen_robot_functions.default import DefaultDomainSeenRobotFunction
from cross_embodiment_offline_rl.environments.unitree_h1.domain_randomization.seen_robot_functions.none import NoneDomainSeenRobotFunction


def get_domain_randomization_seen_robot_function(name, env, **kwargs):
    if name == "default":
        return DefaultDomainSeenRobotFunction(env, **kwargs)
    elif name == "none":
        return NoneDomainSeenRobotFunction(env, **kwargs)
    else:
        raise NotImplementedError
