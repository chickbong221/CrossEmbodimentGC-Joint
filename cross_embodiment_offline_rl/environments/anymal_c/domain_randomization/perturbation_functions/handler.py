from cross_embodiment_offline_rl.environments.anymal_c.domain_randomization.perturbation_functions.default import DefaultDomainPerturbation
from cross_embodiment_offline_rl.environments.anymal_c.domain_randomization.perturbation_functions.none import NoneDomainPerturbation


def get_domain_randomization_perturbation_function(name, env, **kwargs):
    if name == "default":
        return DefaultDomainPerturbation(env, **kwargs)
    elif name == "none":
        return NoneDomainPerturbation(env, **kwargs)
    else:
        raise NotImplementedError
