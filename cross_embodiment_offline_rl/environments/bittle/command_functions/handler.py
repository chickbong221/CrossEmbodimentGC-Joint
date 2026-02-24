from cross_embodiment_offline_rl.environments.bittle.command_functions.random import RandomCommands
from cross_embodiment_offline_rl.environments.bittle.command_functions.forward import ForwardCommands
from cross_embodiment_offline_rl.environments.bittle.command_functions.backward import BackwardCommands


def get_command_function(name, env, **kwargs):
    if name == "random":
        return RandomCommands(env, **kwargs)
    elif name == "forward":
        return ForwardCommands(env, **kwargs)
    elif name == "backward":
        return BackwardCommands(env, **kwargs)
    else:
        raise NotImplementedError
