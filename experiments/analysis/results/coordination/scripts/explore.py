import json, sys, time
import numpy as np
root, info = sys.argv[1], json.load(open(sys.argv[2]))
d = np.load(f"{root}/dones.npy"); te = np.load(f"{root}/terminateds.npy")[..., 0] > 0.5; tr = np.load(f"{root}/truncateds.npy")[..., 0] > 0.5
rw = np.load(f"{root}/rewards.npy")
print("done == term|trunc:", np.mean(d == (te | tr)), " term&trunc:", int((te & tr).sum()))
obs = np.load(f"{root}/observations.npy", mmap_mode="r")
rng = np.random.default_rng(0); sub = np.sort(rng.choice(1_000_000, 4000, replace=False))
O = np.asarray(obs[sub])            # (4000, 16, 668)
names = list(info)
for r, n in enumerate(names):
    ends = np.flatnonzero(d[:, r]); starts = np.concatenate([[0], ends[:-1] + 1])
    lens = ends - starts + 1
    cs = np.concatenate([[0], np.cumsum(rw[:, r], dtype=np.float64)])
    ret = cs[ends + 1] - cs[starts]
    full = tr[ends, r] & (lens == 1000); term = te[ends, r]
    q = lambda x: np.round(np.quantile(x, [0, .1, .5, .9, 1]), 1).tolist() if len(x) else []
    pos = starts[full]
    print(f"{n:5s} eps={len(ends)} full={full.sum()} term={term.sum()} other={(~full & ~term).sum()} | full ret q={q(ret[full])} | term ret q={q(ret[term])} len q={q(lens[term])}")
    print(f"      full-episode starts: min={pos.min() if len(pos) else -1} max={pos.max() if len(pos) else -1}  frac of transitions in full eps={lens[full].sum()/1e6:.3f}  "
          f"start-hist(5 bins)={np.histogram(pos, bins=5, range=(0, 1e6))[0].tolist()}")
    on = info[n]["obs_names"]; ix = {k: i for i, k in enumerate(on)}
    g = O[:, r, [ix["goal_x_velocity"], ix["goal_y_velocity"], ix["goal_yaw_velocity"]]]
    feet = info[n]["foot_names"]
    c = O[:, r, [ix[f + "_ground_contact"] for f in feet]]
    print(f"      goal vel unique rows: {np.unique(np.round(g, 3), axis=0)[:5].tolist()} ; contact values {np.unique(np.round(c, 2))[:6].tolist()} mean contact {np.round(c.mean(0), 2).tolist()}")
    a = np.load(f"{root}/actions.npy", mmap_mode="r")[sub, r, :len(info[n]['joint_names'])]
    print(f"      action abs max {np.abs(a).max():.2f}, std per joint (median) {np.median(a.std(0)):.3f}, padding nonzero {np.abs(np.load(f'{root}/actions.npy', mmap_mode='r')[sub, r, len(info[n]['joint_names']):]).max() if len(info[n]['joint_names']) < 24 else 0:.3f}")
