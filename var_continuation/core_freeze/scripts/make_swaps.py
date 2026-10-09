"""Evaluation-only copies of two checkpoints with their core blocks swapped (CPU only). Core = every parameter under
params['params']['core'] (core experts, global/group-shared experts, router, router noise head). Everything else comes from
the 'rest' checkpoint (actor, critics and optimizer state of the file are kept as they are in it). The reference
configurations (core and rest from the same checkpoint) go through the same code and must reproduce the saved policy bitwise."""
import hashlib, json, os, sys
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np
import jax
from flax.training import checkpoints

R = "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/letuan_r2dreamergraph_2/morse"
EARLY = f"{R}/repo/experiments/runs/morse/V7-CH/V7-CH-s0/models/at_75000"
LATE = f"{R}/ckpt/VARb/VARb-S0/models/at_95000"
OUT = "/tmp/var_runs/swaps"


def sha1(tree):
    h = hashlib.sha1()
    for leaf in jax.tree_util.tree_leaves(tree):
        a = np.ascontiguousarray(np.asarray(leaf))
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def swap(core_src, rest_src):
    out = {k: v for k, v in rest_src.items()}
    pol = dict(rest_src["policy"])
    params = dict(rest_src["policy"]["params"])
    inner = dict(params["params"])
    inner["core"] = core_src["policy"]["params"]["params"]["core"]
    params["params"] = inner
    pol["params"] = params
    out["policy"] = pol
    return out


early = checkpoints.restore_checkpoint(ckpt_dir=EARLY, target=None)
late = checkpoints.restore_checkpoint(ckpt_dir=LATE, target=None)
steps = {"early": int(early["policy"]["step"]), "late": int(late["policy"]["step"])}
configs = {"ref75": (early, early, 75000), "ref95": (late, late, 95000),
           "core75_rest95": (early, late, 95000), "core95_rest75": (late, early, 75000)}
report = {"early": EARLY, "late": LATE, "steps": steps, "configs": {}}
core_h = {"early": sha1(early["policy"]["params"]["params"]["core"]), "late": sha1(late["policy"]["params"]["params"]["core"])}
rest_h = {k: sha1({n: v for n, v in c["policy"]["params"]["params"].items() if n != "core"}) for k, c in (("early", early), ("late", late))}
report["core_sha1"], report["rest_sha1"] = core_h, rest_h
report["core_keys"] = sorted(early["policy"]["params"]["params"]["core"].keys())
report["core_leaves"] = len(jax.tree_util.tree_leaves(early["policy"]["params"]["params"]["core"]))
report["core_params"] = int(sum(np.asarray(l).size for l in jax.tree_util.tree_leaves(early["policy"]["params"]["params"]["core"])))
report["core_differs"] = core_h["early"] != core_h["late"]
for name, (core_src, rest_src, step) in configs.items():
    new = swap(core_src, rest_src)
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    checkpoints.save_checkpoint(ckpt_dir=d, target=new, step=step, keep=1, overwrite=True)
    back = checkpoints.restore_checkpoint(ckpt_dir=d, target=None)
    report["configs"][name] = {"policy_sha1": sha1(back["policy"]["params"]),
                               "core_sha1": sha1(back["policy"]["params"]["params"]["core"]),
                               "rest_sha1": sha1({n: v for n, v in back["policy"]["params"]["params"].items() if n != "core"})}
for name, src in (("ref75", early), ("ref95", late)):
    report["configs"][name]["reproduces_saved_policy"] = report["configs"][name]["policy_sha1"] == sha1(src["policy"]["params"])
report["configs"]["core75_rest95"]["core_is_early"] = report["configs"]["core75_rest95"]["core_sha1"] == core_h["early"]
report["configs"]["core75_rest95"]["rest_is_late"] = report["configs"]["core75_rest95"]["rest_sha1"] == rest_h["late"]
report["configs"]["core95_rest75"]["core_is_late"] = report["configs"]["core95_rest75"]["core_sha1"] == core_h["late"]
report["configs"]["core95_rest75"]["rest_is_early"] = report["configs"]["core95_rest75"]["rest_sha1"] == rest_h["early"]
json.dump(report, open(f"{OUT}/swap_report.json", "w"), indent=1)
print(json.dumps(report, indent=1))
