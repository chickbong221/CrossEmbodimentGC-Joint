"""Run on Hessian (CPU): core hashes of the freeze run's saved checkpoints and parameter distances per group."""
import hashlib, json, os
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np
import jax
from flax.training import checkpoints
from cross_embodiment_offline_rl.algorithms.morse.ckpt_analysis import param_group
R = "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/letuan_r2dreamergraph_2/morse"
SRC = f"{R}/repo/experiments/runs/morse/V7-CH/V7-CH-s0/models/at_75000"
F = "/tmp/var_runs/VARc/VARc-F0/models"
S0 = "/tmp/var_runs/VARb/VARb-S0/models/at_95000"
OUT = "/tmp/var_runs/frz_post"
os.makedirs(OUT, exist_ok=True)

def sha1(tree):
    h = hashlib.sha1()
    for leaf in jax.tree_util.tree_leaves(tree):
        a = np.ascontiguousarray(np.asarray(leaf))
        h.update(str((a.shape, a.dtype.str)).encode()); h.update(a.tobytes())
    return h.hexdigest()

def load(p):
    return checkpoints.restore_checkpoint(ckpt_dir=p, target=None)["policy"]["params"]["params"]

hashes = {"source": sha1(load(SRC)["core"])}
P = {}
for n in sorted(int(d[3:]) for d in os.listdir(F) if d.startswith("at_")):
    p = load(f"{F}/at_{n}")
    hashes[f"at_{n}"] = sha1(p["core"])
    P[n] = p
json.dump(hashes, open(f"{OUT}/core_hash_checkpoints.json", "w"), indent=1)
print(json.dumps(hashes, indent=1))
flat = lambda t: {tuple(getattr(x, "key", str(x)) for x in path): np.asarray(v, np.float64) for path, v in jax.tree_util.tree_flatten_with_path(t)[0]}
src, s0, f95 = flat(load(SRC)), flat(load(S0)), flat(P[max(P)])
groups = ["all", "desc_embedding", "std_head", "decoder_experts", "core_experts", "routers", "joint_state_encoder", "foot_state_encoder", "joint_attention", "foot_attention"]
def dist(a, b):
    sq = {}
    for k in a:
        if k in b:
            g = param_group(k); sq[g] = sq.get(g, 0.0) + float(np.sum((a[k] - b[k]) ** 2))
    sq["all"] = sum(sq.values())
    return {g: np.sqrt(v) for g, v in sq.items()}
lines = ["L2 distance of the policy parameters per group at +20k (source = V7-CH-s0 at 75k; core experts and routers are frozen in the freeze run)",
         f"{'pair':16s}" + "".join(f"{g:>20s}" for g in groups)]
for name, a, b in (("S0 - source", s0, src), ("freeze - source", f95, src), ("freeze - S0", f95, s0)):
    d = dist(a, b)
    lines.append(f"{name:16s}" + "".join(f"{d.get(g, 0.0):20.4f}" for g in groups))
open(f"{OUT}/param_distances.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
