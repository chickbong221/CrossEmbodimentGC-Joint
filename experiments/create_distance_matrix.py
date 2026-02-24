import numpy as np
import networkx as nx
import mujoco
import ot
import sys
from pathlib import Path
from ml_collections import config_dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cross_embodiment_offline_rl.environments.multi_robot.create_env import create_env
from cross_embodiment_offline_rl.environments.multi_robot.default_config import (
    get_config as get_env_config,
)


NUM_ROBOTS = 16
FGW_ALPHA = 0.5
FGW_EPSILON = 1e-2
OUTPUT_PATH = Path("data/meta/embodiment_distance_reproduce.npy")


def build_env_config() -> config_dict.ConfigDict:
    """Build the minimal config required by multi_robot.create_env."""
    config = config_dict.ConfigDict()

    # `create_env` requires full `environment` defaults.
    config.environment = get_env_config("multi_robot")
    config.environment.nr_envs = NUM_ROBOTS
    config.environment.nr_eval_envs = NUM_ROBOTS
    config.environment.async_skip_percentage = 0.0
    config.environment.cycle_cpu_affinity = True
    config.environment.seed = 0
    config.environment.command_type = "forward"

    # `create_env` only reads this field from `algorithm`.
    config.algorithm = config_dict.ConfigDict()
    config.algorithm.determine_fastest_cpu_for_gpu = False

    return config


def robot_to_graph(
    name_to_description_vector,
    joint_names,
    model,
    dynamic_joint_description_size,
    foot_names,
    robot_name,
    exception_robot_name,
):
    """Convert a robot embodiment to an undirected graph with node features."""
    graph = nx.Graph()

    for name, vec in name_to_description_vector.items():
        feature = np.asarray(vec, dtype=float)
        if feature.shape[0] < dynamic_joint_description_size:
            feature = np.pad(
                feature,
                (0, dynamic_joint_description_size - feature.shape[0]),
                mode="constant",
                constant_values=0,
            )
        graph.add_node(name, feature=feature)

    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        body_id = model.jnt_bodyid[joint_id]
        parent_body_id = model.body_parentid[body_id]

        parent_joint = None
        if robot_name == exception_robot_name and joint_name == "left-foot":
            parent_joint = "left-knee"
        elif robot_name == exception_robot_name and joint_name == "right-foot":
            parent_joint = "right-knee"
        else:
            for candidate_joint_name in joint_names:
                candidate_joint_id = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    candidate_joint_name,
                )
                if model.jnt_bodyid[candidate_joint_id] == parent_body_id:
                    parent_joint = candidate_joint_name
                    break

        if parent_joint is None:
            if "trunk" not in graph:
                graph.add_node(
                    "trunk",
                    feature=name_to_description_vector.get(
                        "trunk",
                        np.zeros(dynamic_joint_description_size),
                    ),
                )
            parent_joint = "trunk"

        graph.add_edge(parent_joint, joint_name)

    for foot_name in foot_names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, foot_name)
        if geom_id == -1:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{foot_name}_1")

        foot_body_id = model.geom_bodyid[geom_id]
        candidate_joint_ids = [
            joint_index for joint_index in range(model.njnt) if model.jnt_bodyid[joint_index] == foot_body_id
        ]
        if not candidate_joint_ids:
            parent_body_id = model.body_parentid[foot_body_id]
            candidate_joint_ids = [
                joint_index
                for joint_index in range(model.njnt)
                if model.jnt_bodyid[joint_index] == parent_body_id
            ]

        parent_joint_id = candidate_joint_ids[0]
        parent_joint_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            parent_joint_id,
        )
        graph.add_edge(parent_joint_name, foot_name)

    return graph


def build_matrices(graph, node_attr, metric="shortest"):
    nodes = list(graph.nodes())
    x = np.vstack([node_attr[node_name] for node_name in nodes])

    if metric == "shortest":
        dist = dict(nx.all_pairs_shortest_path_length(graph))
        n_nodes = len(nodes)
        c = np.zeros((n_nodes, n_nodes))
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                c[i, j] = dist[u][v]
    else:
        c = nx.to_numpy_array(graph, nodelist=nodes)

    return x, c


def compute_fgw_distance(graph1, graph2, alpha=FGW_ALPHA, epsilon=FGW_EPSILON):
    feat1 = nx.get_node_attributes(graph1, "feature")
    feat2 = nx.get_node_attributes(graph2, "feature")

    x1, c1 = build_matrices(graph1, feat1)
    x2, c2 = build_matrices(graph2, feat2)

    p = ot.unif(len(x1))
    q = ot.unif(len(x2))
    m = ot.dist(x1, x2, metric="sqeuclidean")

    _, log = ot.gromov.fused_gromov_wasserstein(
        m,
        c1,
        c2,
        p=p,
        q=q,
        alpha=alpha,
        epsilon=epsilon,
        armijo=True,
        log=True,
    )
    return log["loss"][-1]


def collect_robot_graphs(env, robot_names):
    name_to_description_vectors = env.train_env.call("name_to_description_vector")
    joint_names_list = env.train_env.call("joint_names")
    models = env.train_env.call("model")
    dynamic_joint_description_sizes = env.train_env.call("dynamic_joint_description_size")
    foot_names_list = env.train_env.call("foot_names")
    exception_robot_name = robot_names[14]

    robot_graphs = []
    for robot_id in range(NUM_ROBOTS):
        graph = robot_to_graph(
            name_to_description_vectors[robot_id],
            joint_names_list[robot_id],
            models[robot_id],
            dynamic_joint_description_sizes[robot_id],
            foot_names_list[robot_id],
            robot_names[robot_id],
            exception_robot_name,
        )
        print(graph)
        robot_graphs.append(graph)

    return robot_graphs


def compute_pairwise_distance_matrix(robot_graphs):
    num_graphs = len(robot_graphs)
    dist_matrix = np.zeros((num_graphs, num_graphs), dtype=float)

    for i in range(num_graphs):
        for j in range(i + 1, num_graphs):
            distance = compute_fgw_distance(robot_graphs[i], robot_graphs[j])
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance

    return dist_matrix


def main():
    config = build_env_config()
    env = create_env(config)

    try:
        robot_graphs = collect_robot_graphs(env, config.environment.train_robot_types)
    finally:
        env.close()

    dist_matrix = compute_pairwise_distance_matrix(robot_graphs)
    print("Pairwise FGW Distance Matrix:")
    print(dist_matrix)

    max_distance = dist_matrix.max()
    dist_map = dist_matrix / max_distance
    repo_root = Path(__file__).resolve().parents[1]
    output_path = repo_root / OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, dist_map)


if __name__ == "__main__":
    main()
