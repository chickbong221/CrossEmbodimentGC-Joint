"""Coordination tests on reference episodes (checkpoint review: does cross-joint coordination add information?).

Input: the per-robot dumps of morse/coordination.py (``morse_coord_out``): whole reference episodes (full length
without termination, high return) with the dataset actions and both checkpoints' action means (A earlier, B later).

Split: the reference episodes of every robot are split 60/20/20 into fit / calibration / test episodes (fixed seed).
Actions are scaled per joint by their standard deviation on the fit episodes. Contexts: foot-contact pattern (the
command is the same in every state of the dataset); for two feet, single support is split into early and late swing
by the swing foot's air time (median on the fit episodes). Contexts with little support are pooled into "other".

Test 1 (do reference actions have reproducible cross-joint structure?). Per context, mean m_c and covariance S_c of
the fit actions, and two models with the same mean and marginal variances:
    full:  Sigma = (1 - alpha) S_c + alpha diag(S_c) + eps I        diag:  Sigma = diag(S_c) + eps I
(alpha, eps) of the full model and eps of the diagonal one are chosen by the Gaussian NLL of the calibration actions
(one value per robot; alpha = 1 makes the full model diagonal). Reported: test NLL of both (with log-determinant),
paired bootstrap over test episodes; real test actions vs actions whose joints are shuffled independently within
their context (same marginals, no simultaneous coordination; the diagonal score of the two is identical by
construction); stability of the correlation structure across disjoint halves of the fit episodes and between fit
and test episodes; the support (transitions, episodes) of every context.

Test 2 (do the checkpoints depart from the reference coordination?). Both checkpoints' action means on the same test
states (same contexts): MSE against the dataset actions, D^2 = (mu - m_c)' Sigma^-1 (mu - m_c) under both models,
thresholds per model and context = 95th percentile of the calibration actions' D^2 (fixed before the checkpoints
are scored), exceedance rates, per context; paired bootstrap over test episodes for B - A.

Test 3 (beyond the size of the change). (A) drift delta = mu_B - mu_A (scaled): amplitude delta' diag(Sigma)^-1 delta
and coordination cost delta' Sigma^-1 delta of the full model; their ratio relative to the mean ratio of an
isotropic drift (1 = no preference, > 1 = the drift runs against the reference correlations), also for drifts of
random direction as a null; changes of D^2 within amplitude bins and among states whose MSE improved. (B) localization: in the context whose mean D^2_full increases most, the joint
pairs whose 2-D coordination excess (2-D Mahalanobis minus the two 1-D terms, under the reference 2x2 covariance)
increases most, with the data for a scatter plot of the top pair (held-out reference actions, both checkpoints).

Usage: python analysis/coordination_tests.py DUMP_DIR [--robots cas,g1] [--out DIR] [--boot 1000]
"""
import argparse
import json
import os

import numpy as np

LOG2PI = np.log(2.0 * np.pi)
ALPHAS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)
EPSILONS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
SPLIT = (0.6, 0.2, 0.2)


def regularized(S, alpha, eps):
    return (1.0 - alpha) * S + alpha * np.diag(np.diag(S)) + eps * np.eye(len(S))


class Gauss:
    """Gaussian N(mean, cov): squared Mahalanobis distance and negative log-likelihood of rows."""

    def __init__(self, mean, cov):
        self.mean, self.cov = mean, cov
        self.prec = np.linalg.inv(cov)
        self.logdet = np.linalg.slogdet(cov)[1]

    def d2(self, x):
        e = x - self.mean
        return np.einsum("ni,ij,nj->n", e, self.prec, e)

    def nll(self, x):
        return 0.5 * (len(self.mean) * LOG2PI + self.logdet + self.d2(x))


def split_episodes(episodes, seed):
    """Fit / calibration / test episode ids (a fixed permutation of the robot's reference episodes)."""
    ep = np.random.default_rng(seed).permutation(np.unique(episodes))
    n_fit, n_cal = int(round(SPLIT[0] * len(ep))), int(round(SPLIT[1] * len(ep)))
    return ep[:n_fit], ep[n_fit:n_fit + n_cal], ep[n_fit + n_cal:]


def foot_side(name):
    n = name.lower()
    return "left" if ("left" in n or n.startswith("l_") or n.startswith("fl") or n.startswith("rl")) else \
        "right" if ("right" in n or n.startswith("r_") or n.startswith("fr") or n.startswith("rr")) else name


def contexts(contact, air, feet, fit, min_fit, min_eps, min_held, episode, cal, test):
    """Context label of every transition: contact pattern (+ early/late swing in single support for two feet)."""
    c = contact > 0
    n, F = c.shape
    if F == 2:
        sides = [foot_side(f) for f in feet]
        label = np.where(c.all(1), "double support", np.where(~c.any(1), "flight", ""))
        for k in range(2):
            single = c[:, k] & ~c[:, 1 - k]
            swing_air = air[:, 1 - k]
            if (single & fit).sum() == 0:
                continue
            thr = np.median(swing_air[single & fit])
            name = f"{sides[k]} stance"
            label = np.where(single & (swing_air <= thr), name + ", early swing", label)
            label = np.where(single & (swing_air > thr), name + ", late swing", label)
    else:
        label = np.array(["contact " + "".join("1" if b else "0" for b in row) for row in c])
    label = label.astype(object)
    # Pool contexts with little support (fit transitions and episodes, held-out transitions) into "other".
    for _ in range(2):
        for lab in np.unique(label):
            sel = label == lab
            ok = ((sel & fit).sum() >= min_fit and len(np.unique(episode[sel & fit])) >= min_eps
                  and (sel & cal).sum() >= min_held and (sel & test).sum() >= min_held)
            if not ok and lab != "other":
                label[sel] = "other"
    sel = label == "other"
    if sel.any() and not ((sel & fit).sum() >= min_fit and (sel & cal).sum() >= min_held and (sel & test).sum() >= min_held):
        label[sel] = "excluded"
    return label


def boot_ci(stat, groups, n_boot, seed, alpha=0.05):
    """Percentile interval of ``stat(indices)`` over resamples of the episode groups (lists of transition indices)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        vals.append(stat(np.concatenate([groups[i] for i in pick])))
    vals = np.asarray(vals, dtype=np.float64)
    return [float(np.nanpercentile(vals, 100 * alpha / 2, axis=0)), float(np.nanpercentile(vals, 100 * (1 - alpha / 2), axis=0))]


def fmt(x, d=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)


def analyze_robot(path, seed, n_boot, shuffles):
    z = np.load(path, allow_pickle=False)
    robot = str(z["robot"])
    joints = [str(j) for j in z["joint_names"]]
    feet = [str(f) for f in z["foot_names"]]
    D = len(joints)
    act = z["act"].astype(np.float64)
    mu = {"A": z["mean_A"].astype(np.float64), "B": z["mean_B"].astype(np.float64)}
    episode, t = z["episode"], z["t"]
    out = {"robot": robot, "joints": joints, "feet": feet, "n_joints": D}

    # Episodes of the dataset and the reference set.
    term, trunc, ref = z["ep_terminated"], z["ep_truncated"], z["ep_reference"]
    out["episodes"] = {"all": int(len(term)), "terminated": int(term.sum()), "full_length": int((trunc & ~term).sum()),
                       "reference": int(ref.sum()),
                       "reference_return": [fmt(np.min(z["ep_return"][ref]), 1), fmt(np.median(z["ep_return"][ref]), 1),
                                            fmt(np.max(z["ep_return"][ref]), 1)] if ref.any() else None,
                       "reference_first_sample": int(z["ep_start"][ref].min()) if ref.any() else None,
                       "reference_transitions": int(len(act))}
    if len(act) == 0:
        return out, None

    ep_fit, ep_cal, ep_test = split_episodes(episode, seed)
    fit, cal, test = np.isin(episode, ep_fit), np.isin(episode, ep_cal), np.isin(episode, ep_test)
    out["split_episodes"] = {"fit": int(len(ep_fit)), "calibration": int(len(ep_cal)), "test": int(len(ep_test))}

    # Per-joint scale from the fit episodes.
    scale = np.maximum(act[fit].std(axis=0), 1e-3)
    za = act / scale
    zm = {k: v / scale for k, v in mu.items()}

    # Noise share of the reference actions: var of step differences / (2 var), inside fit episodes.
    same = np.r_[False, (episode[1:] == episode[:-1]) & (t[1:] == t[:-1] + 1)] & fit
    da = act[same] - act[np.flatnonzero(same) - 1]
    white = da.var(axis=0) / (2.0 * act[fit].var(axis=0))
    out["action_noise"] = {"white_share_median": fmt(np.median(white)), "white_share_range": [fmt(white.min()), fmt(white.max())],
                           "action_std_median": fmt(np.median(act[fit].std(axis=0))),
                           "policy_std_A_median": fmt(np.median(np.exp(z["logstd_A"]))),
                           "policy_std_B_median": fmt(np.median(np.exp(z["logstd_B"])))}

    # Contexts.
    min_fit, min_eps, min_held = max(20 * D, 1000), 20, 200
    label = contexts(z["contact"].astype(np.float32), z["air_time"].astype(np.float32), feet, fit, min_fit, min_eps,
                     min_held, episode, cal, test)
    ctxs = [c for c in np.unique(label) if c != "excluded"]
    out["excluded_transitions"] = int((label == "excluded").sum())
    support = {}
    for c in ctxs:
        s = label == c
        row = {}
        for name, m in (("fit", fit), ("calibration", cal), ("test", test)):
            eps_c, cnt = np.unique(episode[s & m], return_counts=True)
            row[name] = {"transitions": int((s & m).sum()), "episodes": int(len(eps_c)),
                         "episodes_ge10": int((cnt >= 10).sum()),
                         "max_episode_share": fmt(cnt.max() / cnt.sum()) if len(cnt) else None}
        support[c] = row
    out["contexts"] = support

    # Test 1: fit, select regularization on calibration NLL, evaluate on test.
    fits = {}
    for c in ctxs:
        x = za[(label == c) & fit]
        fits[c] = (x.mean(axis=0), np.cov(x, rowvar=False, bias=True).reshape(D, D))
    cal_x = {c: za[(label == c) & cal] for c in ctxs}

    def cal_nll(alpha, eps):
        tot = n = 0
        for c in ctxs:
            g = Gauss(fits[c][0], regularized(fits[c][1], alpha, eps))
            tot += g.nll(cal_x[c]).sum()
            n += len(cal_x[c])
        return tot / n

    grid_full = {(a_, e_): cal_nll(a_, e_) for a_ in ALPHAS for e_ in EPSILONS}
    alpha, eps_full = min(grid_full, key=grid_full.get)
    eps_diag = min(EPSILONS, key=lambda e_: grid_full[(1.0, e_)])
    out["regularization"] = {"alpha": alpha, "eps_full": eps_full, "eps_diag": eps_diag,
                             "cal_nll_full": fmt(grid_full[(alpha, eps_full)], 4),
                             "cal_nll_diag": fmt(grid_full[(1.0, eps_diag)], 4),
                             "cal_nll_full_alpha0": fmt(min(grid_full[(0.0, e_)] for e_ in EPSILONS), 4)}
    full = {c: Gauss(fits[c][0], regularized(fits[c][1], alpha, eps_full)) for c in ctxs}
    diag = {c: Gauss(fits[c][0], regularized(fits[c][1], 1.0, eps_diag)) for c in ctxs}

    n = len(za)
    nll_f, nll_d = np.full(n, np.nan), np.full(n, np.nan)
    d2 = {key: np.full(n, np.nan) for key in ("ref_full", "ref_diag", "A_full", "A_diag", "B_full", "B_diag")}
    thr_f, thr_d = np.full(n, np.nan), np.full(n, np.nan)
    rho_iso = {}
    for c in ctxs:
        s = label == c
        nll_f[s], nll_d[s] = full[c].nll(za[s]), diag[c].nll(za[s])
        d2["ref_full"][s], d2["ref_diag"][s] = full[c].d2(za[s]), diag[c].d2(za[s])
        for k in ("A", "B"):
            d2[f"{k}_full"][s], d2[f"{k}_diag"][s] = full[c].d2(zm[k][s]), diag[c].d2(zm[k][s])
        thr_f[s] = np.percentile(d2["ref_full"][s & cal], 95)
        thr_d[s] = np.percentile(d2["ref_diag"][s & cal], 95)
        rho_iso[c] = float(np.mean(np.diag(full[c].prec) * np.diag(full[c].cov)))

    test_eps = [np.flatnonzero((episode == e) & (label != "excluded")) for e in ep_test]
    tidx = np.concatenate(test_eps)

    gain = lambda i: float(np.mean(nll_d[i] - nll_f[i]))
    out["test1"] = {"test_nll_full": fmt(np.mean(nll_f[tidx]), 4), "test_nll_diag": fmt(np.mean(nll_d[tidx]), 4),
                    "gain_nats_per_transition": fmt(gain(tidx), 4),
                    "gain_ci": [fmt(v, 4) for v in boot_ci(gain, test_eps, n_boot, seed + 1)],
                    "gain_nats_per_joint": fmt(gain(tidx) / D, 4),
                    "episodes_full_better": fmt(np.mean([gain(g) > 0 for g in test_eps]), 3),
                    "by_context": {}}
    for c in ctxs:
        g_c = [g[label[g] == c] for g in test_eps]
        g_c = [g for g in g_c if len(g)]
        i_c = np.concatenate(g_c)
        R = full[c].cov / np.sqrt(np.outer(np.diag(full[c].cov), np.diag(full[c].cov)))
        S = fits[c][1]
        Rraw = S / np.sqrt(np.outer(np.diag(S), np.diag(S)) + 1e-12)
        off = ~np.eye(D, dtype=bool)
        out["test1"]["by_context"][c] = {
            "test_transitions": int(len(i_c)), "test_episodes": int(len(g_c)),
            "gain_nats_per_transition": fmt(gain(i_c), 4),
            "gain_ci": [fmt(v, 4) for v in boot_ci(gain, g_c, max(200, n_boot // 5), seed + 2)],
            "mean_abs_corr_fit": fmt(np.abs(Rraw[off]).mean()),
            "total_correlation_nats": fmt(-0.5 * np.linalg.slogdet(R)[1], 3)}

    # Shuffle control on the test actions: each joint permuted independently within its context.
    rng = np.random.default_rng(seed + 3)
    real_f, shuf_f, real_nll, shuf_nll, auc, diag_check = [], [], [], [], [], []
    for _ in range(shuffles):
        sf, sn = np.full(n, np.nan), np.full(n, np.nan)
        dd = 0.0
        for c in ctxs:
            i_c = tidx[label[tidx] == c]
            x = za[i_c].copy()
            for j in range(D):
                x[:, j] = x[rng.permutation(len(x)), j]
            sf[i_c], sn[i_c] = full[c].d2(x), full[c].nll(x)
            dd += abs(diag[c].d2(x).sum() - d2["ref_diag"][i_c].sum())
        real_f.append(np.mean(d2["ref_full"][tidx])), shuf_f.append(np.mean(sf[tidx]))
        real_nll.append(np.mean(nll_f[tidx])), shuf_nll.append(np.mean(sn[tidx]))
        a_ = d2["ref_full"][tidx]
        b_ = sf[tidx]
        ranks = np.argsort(np.argsort(np.r_[a_, b_]))
        auc.append(float((ranks[len(a_):].sum() - len(b_) * (len(b_) - 1) / 2) / (len(a_) * len(b_))))
        diag_check.append(dd)

    def shuffle_gap(i):          # bootstrap over test episodes of mean D2_full(shuffled) - mean D2_full(real), 1 shuffle
        r = np.random.default_rng(len(i))
        tot_s = 0.0
        for c in ctxs:
            k = i[label[i] == c]
            if len(k) == 0:
                continue
            x = za[k].copy()
            for j in range(D):
                x[:, j] = x[r.permutation(len(x)), j]
            tot_s += full[c].d2(x).sum()
        return tot_s / len(i) - float(np.mean(d2["ref_full"][i]))

    out["test1"]["shuffle"] = {"repeats": shuffles, "mean_d2_full_real": fmt(np.mean(real_f)), "mean_d2_full_shuffled": fmt(np.mean(shuf_f)),
                               "mean_nll_full_real": fmt(np.mean(real_nll), 4), "mean_nll_full_shuffled": fmt(np.mean(shuf_nll), 4),
                               "auc_real_below_shuffled": fmt(np.mean(auc)),
                               "gap_ci": [fmt(v) for v in boot_ci(shuffle_gap, test_eps, max(200, n_boot // 5), seed + 4)],
                               "diag_sum_identical": bool(max(diag_check) < 1e-6 * max(1.0, float(np.sum(d2["ref_diag"][tidx]))))}

    # Stability: correlation matrices from disjoint halves of the fit episodes, and fit vs test.
    def corr(x):
        S_ = np.cov(x, rowvar=False, bias=True).reshape(D, D)
        return S_ / np.sqrt(np.outer(np.diag(S_), np.diag(S_)) + 1e-12)

    def pcorr(x):
        P = np.linalg.inv(regularized(np.cov(x, rowvar=False, bias=True).reshape(D, D), alpha, eps_full))
        return -P / np.sqrt(np.outer(np.diag(P), np.diag(P)))

    iu = np.triu_indices(D, 1)
    stab = {}
    rng = np.random.default_rng(seed + 5)
    for c in ctxs:
        s = label == c
        sims, psims, signs = [], [], []
        for _ in range(20):
            perm = rng.permutation(ep_fit)
            h1, h2 = np.isin(episode, perm[:len(perm) // 2]) & s, np.isin(episode, perm[len(perm) // 2:]) & s
            if h1.sum() < 2 * D or h2.sum() < 2 * D:
                continue
            r1, r2 = corr(za[h1])[iu], corr(za[h2])[iu]
            sims.append(np.corrcoef(r1, r2)[0, 1])
            psims.append(np.corrcoef(pcorr(za[h1])[iu], pcorr(za[h2])[iu])[0, 1])
            strong = np.abs(r1) > 0.3
            signs.append(np.mean(np.sign(r1[strong]) == np.sign(r2[strong])) if strong.any() else np.nan)
        rf, rt = corr(za[s & fit])[iu], corr(za[s & test])[iu]
        top = np.argsort(-np.abs(rf))[:3]
        stab[c] = {"halves_corr_similarity": fmt(np.median(sims)) if sims else None,
                   "halves_partial_corr_similarity": fmt(np.median(psims)) if psims else None,
                   "strong_pairs_same_sign": fmt(np.nanmedian(signs)) if signs else None,
                   "fit_vs_test_corr_similarity": fmt(np.corrcoef(rf, rt)[0, 1]),
                   "top_pairs": [{"pair": [joints[iu[0][k]], joints[iu[1][k]]], "corr_fit": fmt(rf[k]), "corr_test": fmt(rt[k])}
                                 for k in top]}
    out["test1"]["stability"] = stab

    # Test 2: checkpoints on the test states.
    sq = {k: ((mu[k] - act) ** 2).mean(axis=1) for k in ("A", "B")}

    def t2(i):
        r = {}
        for k in ("A", "B"):
            r[f"mse_{k}"] = float(np.mean(sq[k][i]))
            r[f"full_median_{k}"] = float(np.median(d2[f"{k}_full"][i]))
            r[f"full_p95_{k}"] = float(np.percentile(d2[f"{k}_full"][i], 95))
            r[f"diag_median_{k}"] = float(np.median(d2[f"{k}_diag"][i]))
            r[f"diag_p95_{k}"] = float(np.percentile(d2[f"{k}_diag"][i], 95))
            r[f"exceed_full_{k}"] = float(np.mean(d2[f"{k}_full"][i] > thr_f[i]))
            r[f"exceed_diag_{k}"] = float(np.mean(d2[f"{k}_diag"][i] > thr_d[i]))
        return r

    keys = list(t2(tidx[:10]).keys())
    diffs = ["mse", "full_median", "full_p95", "diag_median", "diag_p95", "exceed_full", "exceed_diag"]

    def t2_with_ci(groups, nb, sd):
        idx = np.concatenate(groups)
        base = t2(idx)
        res = {k: fmt(v, 5) for k, v in base.items()}
        rng_ = np.random.default_rng(sd)
        samples = []
        for _ in range(nb):
            pick = rng_.integers(0, len(groups), len(groups))
            r_ = t2(np.concatenate([groups[p] for p in pick]))
            samples.append([r_[f"{m}_B"] - r_[f"{m}_A"] for m in diffs])
        samples = np.asarray(samples)
        for j, m in enumerate(diffs):
            res[f"{m}_diff"] = fmt(base[f"{m}_B"] - base[f"{m}_A"], 5)
            res[f"{m}_diff_ci"] = [fmt(np.percentile(samples[:, j], 2.5), 5), fmt(np.percentile(samples[:, j], 97.5), 5)]
        res["transitions"], res["episodes"] = int(len(idx)), int(len(groups))
        return res

    out["test2"] = t2_with_ci(test_eps, n_boot, seed + 6)
    out["test2"]["reference_test_actions_exceed_full"] = fmt(np.mean(d2["ref_full"][tidx] > thr_f[tidx]), 4)
    out["test2"]["reference_test_actions_exceed_diag"] = fmt(np.mean(d2["ref_diag"][tidx] > thr_d[tidx]), 4)
    out["test2"]["reference_test_actions_full_median"] = fmt(np.median(d2["ref_full"][tidx]))
    out["test2"]["by_context"] = {}
    for c in ctxs:
        g_c = [g[label[g] == c] for g in test_eps]
        g_c = [g for g in g_c if len(g)]
        out["test2"]["by_context"][c] = t2_with_ci(g_c, max(200, n_boot // 5), seed + 7)

    # Test 3A: drift amplitude vs coordination cost.
    delta = zm["B"] - zm["A"]
    amp, cost, iso = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for c in ctxs:
        s = label == c
        dvar = np.diag(full[c].cov)
        amp[s] = (delta[s] ** 2 / dvar).sum(axis=1)
        cost[s] = np.einsum("ni,ij,nj->n", delta[s], full[c].prec, delta[s])
        iso[s] = rho_iso[c]
    ratio = cost / np.maximum(amp, 1e-300) / iso
    dfull = d2["B_full"] - d2["A_full"]
    ddiag = d2["B_diag"] - d2["A_diag"]
    dmse = sq["B"] - sq["A"]
    amp_j = amp / D
    edges = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, np.inf]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        i = tidx[(amp_j[tidx] >= lo) & (amp_j[tidx] < hi)]
        if len(i) < 20:
            continue
        bins.append({"amp_per_joint": [lo, None if hi == np.inf else hi], "n": int(len(i)),
                     "mean_dD2_full": fmt(np.mean(dfull[i])), "mean_dD2_diag": fmt(np.mean(ddiag[i])),
                     "ratio_median": fmt(np.median(ratio[i])), "ratio_p90": fmt(np.percentile(ratio[i], 90)),
                     "frac_ratio_gt2": fmt(np.mean(ratio[i] > 2)),
                     "frac_full_up_diag_down": fmt(np.mean((dfull[i] > 0) & (ddiag[i] <= 0)))})
    from_rank = lambda x: np.argsort(np.argsort(x))
    improved = tidx[dmse[tidx] < 0]
    # Null: drifts of random direction (isotropic after scaling by the model's marginal std), same contexts.
    rng = np.random.default_rng(seed + 11)
    null = []
    by_ctx = {}
    for c in ctxs:
        k = int((label[tidx] == c).sum())
        u = rng.normal(size=(k, D)) * np.sqrt(np.diag(full[c].cov))
        a_u = (u ** 2 / np.diag(full[c].cov)).sum(axis=1)
        r_u = np.einsum("ni,ij,nj->n", u, full[c].prec, u) / a_u / rho_iso[c]
        null.append(r_u)
        i_c = tidx[label[tidx] == c]
        by_ctx[c] = {"ratio_median": fmt(np.median(ratio[i_c])), "ratio_p99": fmt(np.percentile(ratio[i_c], 99)),
                     "null_ratio_median": fmt(np.median(r_u)), "null_ratio_p99": fmt(np.percentile(r_u, 99)),
                     "frac_ratio_gt2": fmt(np.mean(ratio[i_c] > 2)), "null_frac_ratio_gt2": fmt(np.mean(r_u > 2))}
    null = np.concatenate(null)
    out["test3a"] = {"spearman_amp_cost": fmt(np.corrcoef(from_rank(amp[tidx]), from_rank(cost[tidx]))[0, 1], 4),
                     "ratio_median": fmt(np.median(ratio[tidx])), "ratio_p90": fmt(np.percentile(ratio[tidx], 90)),
                     "ratio_p99": fmt(np.percentile(ratio[tidx], 99)), "frac_ratio_gt2": fmt(np.mean(ratio[tidx] > 2)),
                     "null_ratio_median": fmt(np.median(null)), "null_ratio_p90": fmt(np.percentile(null, 90)),
                     "null_ratio_p99": fmt(np.percentile(null, 99)), "null_frac_ratio_gt2": fmt(np.mean(null > 2)),
                     "by_context": by_ctx,
                     "amp_per_joint_median": fmt(np.median(amp_j[tidx]), 6), "amp_per_joint_p95": fmt(np.percentile(amp_j[tidx], 95), 6),
                     "by_amplitude": bins,
                     "mse_improved_frac": fmt(len(improved) / len(tidx)),
                     "mse_improved_dD2_full_up": fmt(np.mean(dfull[improved] > 0)) if len(improved) else None,
                     "mse_improved_dD2_diag_up": fmt(np.mean(ddiag[improved] > 0)) if len(improved) else None,
                     "mse_improved_mean_dD2_full": fmt(np.mean(dfull[improved])) if len(improved) else None,
                     "mse_improved_mean_dD2_diag": fmt(np.mean(ddiag[improved])) if len(improved) else None,
                     "ratio_ci_median": [fmt(v) for v in boot_ci(lambda i: float(np.median(ratio[i])), test_eps, max(200, n_boot // 5), seed + 8)]}

    # Test 3B: localization in the context with the largest increase of mean D2_full (predefined ranking).
    inc = {c: float(np.mean(dfull[tidx[label[tidx] == c]])) for c in ctxs}
    top_c = max(inc, key=inc.get)
    i_c = tidx[label[tidx] == top_c]
    g_c = [g[label[g] == top_c] for g in test_eps]
    g_c = [g for g in g_c if len(g)]
    m, Sg, P = full[top_c].mean, full[top_c].cov, full[top_c].prec
    pairs = []
    for a_i, b_i in zip(*iu):
        S2 = Sg[np.ix_([a_i, b_i], [a_i, b_i])]
        P2 = np.linalg.inv(S2)
        rho = S2[0, 1] / np.sqrt(S2[0, 0] * S2[1, 1])

        def excess(x):
            e = x[:, [a_i, b_i]] - m[[a_i, b_i]]
            return np.einsum("ni,ij,nj->n", e, P2, e) - (e[:, 0] ** 2 / S2[0, 0] + e[:, 1] ** 2 / S2[1, 1])

        ex_a, ex_b = excess(zm["A"][i_c]), excess(zm["B"][i_c])
        prec_term = lambda x: 2 * P[a_i, b_i] * (x[:, a_i] - m[a_i]) * (x[:, b_i] - m[b_i])
        pairs.append({"pair": [joints[a_i], joints[b_i]], "ij": [int(a_i), int(b_i)], "ref_corr": fmt(rho),
                      "mean_excess_A": fmt(np.mean(ex_a)), "mean_excess_B": fmt(np.mean(ex_b)),
                      "d_excess": float(np.mean(ex_b - ex_a)),
                      "d_precision_term": fmt(np.mean(prec_term(zm["B"][i_c]) - prec_term(zm["A"][i_c])))})
    pairs.sort(key=lambda p: -p["d_excess"])
    for p in pairs[:5]:
        a_i, b_i = p["ij"]
        S2 = Sg[np.ix_([a_i, b_i], [a_i, b_i])]
        P2 = np.linalg.inv(S2)

        def dex(i, a_i=a_i, b_i=b_i, S2=S2, P2=P2):
            out_ = 0.0
            for k, sgn in (("B", 1.0), ("A", -1.0)):
                e = zm[k][i][:, [a_i, b_i]] - m[[a_i, b_i]]
                out_ += sgn * np.mean(np.einsum("ni,ij,nj->n", e, P2, e) - (e[:, 0] ** 2 / S2[0, 0] + e[:, 1] ** 2 / S2[1, 1]))
            return out_
        p["d_excess_ci"] = [fmt(v) for v in boot_ci(dex, g_c, max(200, n_boot // 5), seed + 9)]
        p["d_excess"] = fmt(p["d_excess"])
    for p in pairs[5:]:
        p["d_excess"] = fmt(p["d_excess"])
    out["test3b"] = {"context_increase_mean_dD2_full": {c: fmt(v) for c, v in sorted(inc.items(), key=lambda kv: -kv[1])},
                     "top_context": top_c, "top_pairs": pairs[:5],
                     "pairs_by_precision_term": sorted(({"pair": p["pair"], "d_precision_term": p["d_precision_term"]} for p in pairs),
                                                       key=lambda p: -(p["d_precision_term"] or 0))[:5]}
    # Plot data for the top pair: held-out reference actions of the context and both checkpoints' means (raw units).
    a_i, b_i = pairs[0]["ij"]
    rng = np.random.default_rng(seed + 10)
    pick = rng.choice(len(i_c), min(3000, len(i_c)), replace=False)
    S2 = Sg[np.ix_([a_i, b_i], [a_i, b_i])] * np.outer(scale[[a_i, b_i]], scale[[a_i, b_i]])
    plot = {"pair": np.array([joints[a_i], joints[b_i]]), "context": np.array(top_c),
            "ref": act[i_c[pick]][:, [a_i, b_i]], "A": mu["A"][i_c[pick]][:, [a_i, b_i]], "B": mu["B"][i_c[pick]][:, [a_i, b_i]],
            "mean": m[[a_i, b_i]] * scale[[a_i, b_i]], "cov": S2}
    return out, plot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dump_dir")
    p.add_argument("--robots", default="")
    p.add_argument("--out", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--shuffles", type=int, default=10)
    a = p.parse_args()
    out_dir = a.out or os.path.join(a.dump_dir, "tests")
    os.makedirs(out_dir, exist_ok=True)
    names = [r for r in a.robots.split(",") if r] or sorted(f[:-4] for f in os.listdir(a.dump_dir) if f.endswith(".npz"))
    results = {}
    for r in names:
        res, plot = analyze_robot(os.path.join(a.dump_dir, f"{r}.npz"), a.seed, a.boot, a.shuffles)
        results[r] = res
        if plot is not None:
            np.savez(os.path.join(out_dir, f"pair_{r}.npz"), **plot)
        print(f"{r}: done", flush=True)
        with open(os.path.join(out_dir, "coordination_tests.json"), "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
