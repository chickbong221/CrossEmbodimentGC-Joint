import os, sys
import numpy as np
from flax.training import checkpoints
R = "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/letuan_r2dreamergraph_2/morse/repo/experiments/runs/morse"
for p in ["T3-MIcompat/T3-MIcompat-s0/models/tmp", "T3-MIcompat-resume/T3-MIcompat-s0-from60k/models/tmp",
          "T1-URMA-A4-cal/T1-URMA-A4-cal-s0/models/at_2000", "T1-URMA-A4-cal/T1-URMA-A4-cal-s0/models/at_5000",
          "T1-URMA-A4-cal/T1-URMA-A4-cal-s0/models/tmp"]:
    d = os.path.join(R, p)
    raw = checkpoints.restore_checkpoint(ckpt_dir=d, target=None)
    pol = raw["policy"]
    counts = []
    def walk(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, path + "/" + str(k))
        elif "count" in path:
            counts.append((path, int(np.asarray(x))))
    walk(pol["opt_state"])
    print(p, os.listdir(d), "policy step", int(np.asarray(pol["step"])), "value step", int(np.asarray(raw["value_critic"]["step"])), "opt counts", sorted(set(c for _, c in counts)))
