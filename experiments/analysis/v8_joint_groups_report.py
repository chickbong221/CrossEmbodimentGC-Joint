"""Rebuild the V8 joint-group table from a dataset and report it.

Usage: python experiments/analysis/v8_joint_groups_report.py TABLE.json DATASET_DIR [OUT.json]

The joint counts and robot order come from TABLE.json (as written by a training run); the joint descriptions are the
means over 4000 evenly spaced rows of DATASET_DIR. Prints the groups (members, robots, mean features), the normalized
mutual information between group and robot, and whether the rebuilt groups equal the table's. OUT.json (optional)
receives the rebuilt table.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cross_embodiment_offline_rl.algorithms.morse.v8 import GROUP_FEATURES, build_joint_groups, joint_features  # noqa: E402
from cross_embodiment_offline_rl.algorithms.utils import DATASET_ROBOTS  # noqa: E402

JOINT_OBS, DESC = 26, 23


def main(table_path, dataset_dir, out_path=None, n_rows=4000):
    table = json.load(open(table_path))
    names = list(table["groups"])
    n_joints = [len(table["groups"][n]) for n in names]
    root = Path(dataset_dir)
    meta = json.load(open(root / "metadata.json"))
    obs = np.load(root / "observations.npy", mmap_mode="r")
    rows = np.unique(np.linspace(0, int(meta["n_samples"]) - 1, n_rows).astype(np.int64))
    descs = []
    for r, (name, n) in enumerate(zip(names, n_joints)):
        # Long names map to the dataset's robot axis; short names (the environments' robot_type) are in dataset order.
        col = DATASET_ROBOTS.index(name) if name in DATASET_ROBOTS else r
        block = np.asarray(obs[rows, col, :n * JOINT_OBS], np.float64).reshape(len(rows), n, JOINT_OBS)
        descs.append(block[:, :, :DESC].mean(axis=0))
    rebuilt = build_joint_groups(descs, names, table.get("joint_names") or {}, k=int(table["k"]), seed=int(table["seed"]))
    same = rebuilt["groups"] == table["groups"]
    print(f"rebuilt from {root.name} ({len(rows)} rows): identical to the table: {same}")
    print(f"NMI(group, robot) table {table['nmi_group_robot']:.3f}, rebuilt {rebuilt['nmi_group_robot']:.3f}")
    per_robot = [joint_features(d) for d in descs]
    feats = np.concatenate(per_robot)
    # Feature audit: share of each feature's variance across all joints that lies within robots (1 = only joint
    # differences, 0 = only robot differences), and the number of robots in which the feature is constant.
    total = feats.var(axis=0)
    within = sum(((f - f.mean(0)) ** 2).sum(0) for f in per_robot) / len(feats)
    constant = sum((f.std(0) < 1e-6).astype(int) for f in per_robot)
    print("\nfeature audit (within-robot share of variance; robots where the feature is constant, of %d):" % len(names))
    for i, f in enumerate(GROUP_FEATURES):
        share = within[i] / total[i] if total[i] > 1e-12 else float("nan")
        print(f"  {f:18s} within {share:5.2f}   constant in {constant[i]:2d} robots   (std over all joints {np.sqrt(total[i]):.3f})")
    labels = np.concatenate([rebuilt["groups"][n] for n in names])
    owners = [(n, j) for n, c in zip(names, n_joints) for j in range(c)]
    jn = rebuilt.get("joint_names") or {}
    for g in range(int(table["k"])):
        idx = np.where(labels == g)[0]
        robots = sorted({owners[i][0] for i in idx})
        print(f"\ngroup {g}: {len(idx)} joints, {len(robots)} robots")
        print("  mean features: " + ", ".join(f"{f}={v:.2f}" for f, v in zip(GROUP_FEATURES, feats[idx].mean(0))))
        for r in robots:
            js = [owners[i][1] for i in idx if owners[i][0] == r]
            label = [jn[r][j] for j in js] if r in jn and len(jn[r]) > max(js) else js
            print(f"  {r}: {label}")
    if out_path:
        json.dump(rebuilt, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
