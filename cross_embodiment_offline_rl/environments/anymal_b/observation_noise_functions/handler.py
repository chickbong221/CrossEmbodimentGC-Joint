from cross_embodiment_offline_rl.environments.anymal_b.observation_noise_functions.default import DefaultObservationNoise
from cross_embodiment_offline_rl.environments.anymal_b.observation_noise_functions.none import NoneObservationNoise


def get_observation_noise_function(name, env, **kwargs):
    if name == "default":
        return DefaultObservationNoise(env, **kwargs)
    elif name == "none":
        return NoneObservationNoise(env, **kwargs)
    else:
        raise NotImplementedError
