from cross_embodiment_offline_rl.environments.unitree_a1.domain_randomization.mujoco_model_functions.default import DefaultDomainMuJoCoModel
from cross_embodiment_offline_rl.environments.unitree_a1.domain_randomization.mujoco_model_functions.none import NoneDomainMuJoCoModel


def get_domain_randomization_mujoco_model_function(name, env, **kwargs):
    if name == "default":
        return DefaultDomainMuJoCoModel(env, **kwargs)
    elif name == "none":
        return NoneDomainMuJoCoModel(env, **kwargs)
    else:
        raise NotImplementedError
