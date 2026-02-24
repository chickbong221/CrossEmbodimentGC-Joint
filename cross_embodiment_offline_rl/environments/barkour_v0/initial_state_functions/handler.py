from cross_embodiment_offline_rl.environments.barkour_v0.initial_state_functions.default import DefaultInitialState
from cross_embodiment_offline_rl.environments.barkour_v0.initial_state_functions.random import RandomInitialState


def get_initial_state_function(name, env, **kwargs):
    if name == "default":
        return DefaultInitialState(env, **kwargs)
    elif name == "random":
        return RandomInitialState(env, **kwargs)
    else:
        raise NotImplementedError
