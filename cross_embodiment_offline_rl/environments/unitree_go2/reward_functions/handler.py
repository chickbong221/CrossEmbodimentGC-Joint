from cross_embodiment_offline_rl.environments.unitree_go2.reward_functions.rudin_own_var import RudinOwnVarReward


def get_reward_function(name, env, **kwargs):
    if name == "rudin_own_var":
        return RudinOwnVarReward(env, **kwargs)
    else:
        raise NotImplementedError
