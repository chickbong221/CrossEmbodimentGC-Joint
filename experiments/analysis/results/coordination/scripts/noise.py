import json, sys
import numpy as np
root, info = sys.argv[1], json.load(open(sys.argv[2]))
d = np.load(f"{root}/dones.npy"); act = np.load(f"{root}/actions.npy", mmap_mode="r")
for n in ["cas", "g1", "h1", "a1"]:
    r = list(info).index(n); D = len(info[n]["joint_names"])
    a = np.asarray(act[700_000:, r, :D], dtype=np.float64); dd = d[700_000:, r]
    ok = ~dd[:-1]                                   # consecutive steps inside one episode
    da = (a[1:] - a[:-1])[ok]
    white = da.var(0) / 2 / a.var(0)
    # lag-k autocorrelation of each joint inside episodes
    ac = []
    for k in (1, 2, 5):
        okk = np.ones(len(a) - k, bool)
        for j in range(k):
            okk &= ~dd[j:len(a) - k + j]
        x, y = a[:-k][okk] - a.mean(0), a[k:][okk] - a.mean(0)
        ac.append(np.round(np.median((x * y).mean(0) / a.var(0)), 3))
    C = np.corrcoef(a.T); Cd = np.corrcoef(da.T)
    off = ~np.eye(D, dtype=bool)
    print(f"{n}: D={D} per-joint action std {np.round(a.std(0), 2).tolist()}")
    print(f"     white-noise share var(da)/2var(a): median {np.median(white):.2f} range {white.min():.2f}-{white.max():.2f}; autocorr lag1/2/5 median {ac}")
    print(f"     mean |corr| actions {np.abs(C[off]).mean():.3f}, mean |corr| of step differences {np.abs(Cd[off]).mean():.3f}")
