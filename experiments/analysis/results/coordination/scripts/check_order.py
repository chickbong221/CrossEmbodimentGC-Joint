import numpy as np, json, sys, time
root = sys.argv[1]
t0 = time.time()
d = np.load(f"{root}/dones.npy"); te = np.load(f"{root}/terminateds.npy"); tr = np.load(f"{root}/truncateds.npy"); rw = np.load(f"{root}/rewards.npy")
print("shapes", d.shape, d.dtype, te.shape, te.dtype, tr.shape, tr.dtype, rw.shape, rw.dtype)
print("done == term|trunc:", np.mean(d.astype(bool) == (te.astype(bool) | tr.astype(bool))))
names = ["a1","go1","go2","anyb","anyc","bkv0","bkvb","bg","bit","h1","g1","tl","op3","nao5","cas","hx"]
for r in range(16):
    dd = d[:, r].astype(bool); ends = np.flatnonzero(dd)
    lens = np.diff(np.concatenate([[-1], ends]))
    print(f"{names[r]:5s} episodes={len(ends):5d} len min/med/max={lens.min()}/{int(np.median(lens))}/{lens.max()} term={int(te[:, r].sum())} trunc={int(tr[:, r].sum())} last_done={int(ends[-1])}")
obs = np.load(f"{root}/observations.npy", mmap_mode="r"); nobs = np.load(f"{root}/next_observations.npy", mmap_mode="r")
rng = np.random.default_rng(0)
for r in [14, 10, 0, 15]:
    idx = np.sort(rng.choice(999_999, 300, replace=False))
    ok_cont = []; ok_reset = []
    for t in idx:
        same = np.array_equal(obs[t + 1, r], nobs[t, r])
        (ok_reset if d[t, r] else ok_cont).append(same)
    print(f"robot {names[r]}: continuity obs[t+1]==nobs[t] when not done: {np.mean(ok_cont):.3f} (n={len(ok_cont)}); when done: {np.mean(ok_reset) if ok_reset else float('nan'):.3f} (n={len(ok_reset)})")
print("elapsed", time.time() - t0)
