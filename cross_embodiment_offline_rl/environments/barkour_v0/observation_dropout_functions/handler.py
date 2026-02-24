from cross_embodiment_offline_rl.environments.barkour_v0.observation_dropout_functions.default import DefaultObservationDropout
from cross_embodiment_offline_rl.environments.barkour_v0.observation_dropout_functions.none import NoneObservationDropout


def get_observation_dropout_function(name, env, **kwargs):
    if name == "default":
        return DefaultObservationDropout(env, **kwargs)
    elif name == "none":
        return NoneObservationDropout(env, **kwargs)
    else:
        raise NotImplementedError
