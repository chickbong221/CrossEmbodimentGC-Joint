from cross_embodiment_offline_rl.environments.talos.terrain_functions.plane import PlaneTerrainGeneration


def get_terrain_function(name, env, **kwargs):
    if name == "plane":
        return PlaneTerrainGeneration(env, **kwargs)
    else:
        raise NotImplementedError
