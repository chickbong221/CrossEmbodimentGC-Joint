"""Parameter distances between the final branch checkpoints and to the source checkpoint, per parameter group (CPU only)."""
import os, sys
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np
import jax
from flax.training import checkpoints
from cross_embodiment_offline_rl.algorithms.morse.ckpt_analysis import param_group
R = "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/letuan_r2dreamergraph_2/morse"
paths = {"src": f"{R}/repo/experiments/runs/morse/V7-CH/V7-CH-s0/models/at_75000"}
for k in ("S0", "S1", "S2", "S3"):
    paths[k] = f"/tmp/var_runs/VARb/VARb-{k}/models/at_95000"
P = {}
for k, p in paths.items():
    raw = checkpoints.restore_checkpoint(ckpt_dir=p, target=None)
    flat = {tuple(getattr(x, "key", str(x)) for x in path): np.asarray(v, np.float64)
            for path, v in jax.tree_util.tree_flatten_with_path(raw["policy"]["params"]["params"])[0]}
    P[k] = flat
def dist(a, b):
    sq = {}
    for path in a:
        if path in b and path[0] != "logstd_offset":
            g = param_group(path)
            sq[g] = sq.get(g, 0.0) + float(np.sum((a[path] - b[path]) ** 2))
    sq["all"] = sum(sq.values())
    return {g: np.sqrt(v) for g, v in sq.items()}
groups = ["all", "desc_embedding", "std_head", "decoder_experts", "core_experts", "joint_state_encoder", "routers"]
print("L2 distance of parameters per group (final +20k checkpoints)")
print(f"{'pair':18s}" + "".join(f"{g:>20s}" for g in groups))
for name, a, b in (("S0 - source", "S0", "src"), ("S1 - S0", "S1", "S0"), ("S2 - S0", "S2", "S0"), ("S3 - S0", "S3", "S0"), ("S1 - source", "S1", "src"), ("S2 - source", "S2", "src"), ("S3 - source", "S3", "src")):
    d = dist(P[a], P[b])
    print(f"{name:18s}" + "".join(f"{d.get(g, 0.0):20.4f}" for g in groups))
off = P["S2"].get(("logstd_offset",))
if off is not None:
    print("S2 private offsets: max |offset| over valid entries", np.abs(off).max(), "mean |offset|", np.abs(off[off != 0]).mean() if (off != 0).any() else 0)
