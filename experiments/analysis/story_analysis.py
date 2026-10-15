"""Motivation analysis for the MORSE story, on the dense URMAv2 gradient files written by morse/conflict.py.

python story_analysis.py TABLE.json DIST.npy FILE.npz [FILE.npz ...]

All cosines are persistent (split-half, noise-corrected; conflict_report.persistent) for the IQL actor loss.
Units are (robot, joint); a robot's gradient is the sum of its units. Parameter blocks: enc, core, dec.
  T1  joint level vs body level: cross-robot unit pairs split by same/different joint group (TABLE) x same/different EG
      group; plus a within-robot split by joint group. Permutation null: joint labels shuffled inside every robot.
  T2  robot level: mean cosine and % negative of robot pairs, inside vs across EG groups, and the Spearman correlation
      of the robot-pair cosine with the embodiment distance (DIST, EG's FGW matrix).
  T3  EG's grouped step on the same gradients (conflict_grouping.robot_part).
  T4  within a robot, success vs fall episodes (conflict_grouping.within_robot), for context.
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conflict_report import EG, GROUPS, POSITIONAL, load_table, membership, persistent  # noqa: E402
from conflict_grouping import robot_part, within_robot  # noqa: E402

BLOCKS = ("enc", "core", "dec")
VI = 0  # iql


def rank(x):
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return np.nan
    return float(np.corrcoef(rank(a[ok]), rank(b[ok]))[0, 1])


def unit_stats(C, robot, jg, egu, rng, n_null=200):
    ok = np.isfinite(np.diag(C))
    iu = np.triu_indices(len(robot), 1)
    good = ok[iu[0]] & ok[iu[1]]
    v = C[iu]
    cross = good & (robot[iu[0]] != robot[iu[1]])
    within = good & (robot[iu[0]] == robot[iu[1]])
    se = egu[iu[0]] == egu[iu[1]]

    def cell(mask):
        x = v[mask]
        return (float(np.mean(x)), float(np.mean(x < 0)), int(mask.sum())) if mask.any() else (np.nan, np.nan, 0)

    def effects(lab):
        sj = lab[iu[0]] == lab[iu[1]]
        return (np.mean(v[cross & sj]) - np.mean(v[cross & ~sj]),            # joint-type effect across robots
                np.mean(v[cross & sj & ~se]) - np.mean(v[cross & ~sj & se]))  # same type/other body minus other type/same body

    sj = jg[iu[0]] == jg[iu[1]]
    out = {"units": float(ok.mean()),
           "x_all": cell(cross), "x_sj": cell(cross & sj), "x_dj": cell(cross & ~sj),
           "x_sj_se": cell(cross & sj & se), "x_sj_de": cell(cross & sj & ~se),
           "x_dj_se": cell(cross & ~sj & se), "x_dj_de": cell(cross & ~sj & ~se),
           "w_sj": cell(within & sj), "w_dj": cell(within & ~sj),
           "x_se": cell(cross & se), "x_de": cell(cross & ~se)}
    e_obs = effects(jg)
    nulls = []
    for _ in range(n_null):
        nl = jg.copy()
        for r in np.unique(robot):
            idx = np.flatnonzero(robot == r)
            nl[idx] = jg[rng.permutation(idx)]
        nulls.append(effects(nl))
    nulls = np.array(nulls)
    out["eff_type"], out["eff_type_null"], out["p_type"] = e_obs[0], float(nulls[:, 0].mean()), float(np.mean(nulls[:, 0] >= e_obs[0]))
    out["eff_tvb"], out["eff_tvb_null"], out["p_tvb"] = e_obs[1], float(nulls[:, 1].mean()), float(np.mean(nulls[:, 1] >= e_obs[1]))
    return out


def robot_stats(z, gi, eg, dist):
    robot = z["unit_robot"].astype(int)
    R = len(eg)
    M = membership(robot, R)
    Gc = z["half_cross"][gi, VI].astype(np.float64)
    S1, S2 = z["half_same1"][gi, VI].astype(np.float64), z["half_same2"][gi, VI].astype(np.float64)
    C = persistent(M.T @ Gc @ M, M.T @ S1 @ M, M.T @ S2 @ M)
    iu = np.triu_indices(R, 1)
    v = C[iu]
    fin = np.isfinite(v)
    same = eg[iu[0]] == eg[iu[1]]
    m = lambda mask: (float(np.nanmean(v[mask & fin])), float(np.mean(v[mask & fin] < 0))) if (mask & fin).any() else (np.nan, np.nan)
    return {"rel": int(np.isfinite(np.diag(C)).sum()), "all": m(np.ones_like(same)), "in": m(same), "out": m(~same),
            "rho": spearman(v, dist[iu])}


def main():
    table, dist_path, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    dist = np.load(dist_path)
    rng = np.random.default_rng(0)
    rows = []
    for f in sorted(files, key=lambda p: (int(re.search(r"at_(\d+)", p).group(1)), p)):
        upd = int(re.search(r"at_(\d+)", f).group(1))
        seed = int(re.search(r"/s(\d+)_conflict", f).group(1))
        z = np.load(f, allow_pickle=True)
        names = [str(n) for n in z["robot_names"]]
        eg = np.array([EG.get(n, EG[POSITIONAL[i]]) for i, n in enumerate(names)])
        robot, joint = z["unit_robot"].astype(int), z["unit_joint"].astype(int)
        tab, _ = load_table(table, names)
        jg = np.array([tab[r][j] for r, j in zip(robot, joint)])
        res = {"upd": upd, "seed": seed}
        for b in BLOCKS:
            gi = GROUPS.index(b)
            Gc = z["half_cross"][gi, VI].astype(np.float64)
            C = persistent(Gc, z["half_same1"][gi, VI].astype(np.float64), z["half_same2"][gi, VI].astype(np.float64))
            res[b] = unit_stats(C, robot, jg, eg[robot], rng)
            res[b + "_rob"] = robot_stats(z, gi, eg, dist)
            res[b + "_eg"] = robot_part(z, gi, VI, eg)
        res["quality"] = within_robot(z, GROUPS.index("dec"), VI)
        rows.append(res)
        print(f"done {seed} {upd}", file=sys.stderr, flush=True)

    P = lambda x: f"{100 * x:3.0f}%"
    for b in BLOCKS:
        print(f"\n==================== block: {b} ====================")
        print("T1 cross-robot unit pairs: mean cos (% neg)   SJ/DJ = same/different joint group, SE/DE = same/different EG group")
        print(f"{'upd':>6s} s units | {'all':>11s} | {'SJ':>11s} {'DJ':>11s} | {'SJ&DE':>11s} {'DJ&SE':>11s} | "
              f"{'type eff':>8s} p     | {'SJDE-DJSE':>9s} p     | within: {'SJ':>11s} {'DJ':>11s}")
        for r in rows:
            u = r[b]
            c = lambda k: f"{u[k][0]:+.3f}({P(u[k][1])})"
            print(f"{r['upd']:6d} {r['seed']} {100 * u['units']:4.0f}% | {c('x_all')} | {c('x_sj')} {c('x_dj')} | "
                  f"{c('x_sj_de')} {c('x_dj_se')} | {u['eff_type']:+.3f} {u['p_type']:.3f} | {u['eff_tvb']:+.3f}  {u['p_tvb']:.3f} | "
                  f"        {c('w_sj')} {c('w_dj')}")
        print("T2 robot pairs (persistent): rel robots | mean cos (% neg) all / in EG / across EG | Spearman(cos, body distance)")
        for r in rows:
            q = r[b + "_rob"]
            print(f"{r['upd']:6d} {r['seed']} {q['rel']:3d} | {q['all'][0]:+.2f}({P(q['all'][1])}) {q['in'][0]:+.2f}({P(q['in'][1])})"
                  f" {q['out'][0]:+.2f}({P(q['out'][1])}) | rho {q['rho']:+.2f}")
        print("T3 EG step on the same gradients: % robots opposing the step that moves them (plain / EG own / EG round);"
              " % negative robot pairs in / across EG groups")
        for r in rows:
            e = r[b + "_eg"]
            print(f"{r['upd']:6d} {r['seed']} | step {P(e['step_plain'])} {P(e['step_own'])} {P(e['step_round'])} |"
                  f" pairs in {P(e['pair_in'])} across {P(e['pair_out'])} | mean cos in {e['cos_in']:+.2f} across {e['cos_out']:+.2f}")
        # means over the cells 2k-75k (the 100k core is unreliable)
        sel = [r for r in rows if r["upd"] <= 75000]
        mean = lambda f: np.nanmean([f(r) for r in sel])
        print(f"MEAN 2k-75k ({len(sel)} cells): cross all {mean(lambda r: r[b]['x_all'][0]):+.3f} ({P(mean(lambda r: r[b]['x_all'][1]))}),"
              f" SJ {mean(lambda r: r[b]['x_sj'][0]):+.3f} ({P(mean(lambda r: r[b]['x_sj'][1]))}),"
              f" DJ {mean(lambda r: r[b]['x_dj'][0]):+.3f} ({P(mean(lambda r: r[b]['x_dj'][1]))}),"
              f" SJ&DE {mean(lambda r: r[b]['x_sj_de'][0]):+.3f}, DJ&SE {mean(lambda r: r[b]['x_dj_se'][0]):+.3f},"
              f" SE {mean(lambda r: r[b]['x_se'][0]):+.3f}, DE {mean(lambda r: r[b]['x_de'][0]):+.3f};"
              f" within SJ {mean(lambda r: r[b]['w_sj'][0]):+.3f} ({P(mean(lambda r: r[b]['w_sj'][1]))}),"
              f" within DJ {mean(lambda r: r[b]['w_dj'][0]):+.3f} ({P(mean(lambda r: r[b]['w_dj'][1]))})")
        print(f"   type effect > null in {sum(r[b]['p_type'] < 0.05 for r in sel)}/{len(sel)} cells;"
              f" SJ&DE > DJ&SE in {sum(r[b]['eff_tvb'] > 0 for r in sel)}/{len(sel)} cells (p<0.05 in {sum(r[b]['p_tvb'] < 0.05 for r in sel)})")
        print(f"   robots: mean cos {mean(lambda r: r[b + '_rob']['all'][0]):+.2f} ({P(mean(lambda r: r[b + '_rob']['all'][1]))} neg),"
              f" in EG {mean(lambda r: r[b + '_rob']['in'][0]):+.2f}, across {mean(lambda r: r[b + '_rob']['out'][0]):+.2f},"
              f" Spearman with body distance {mean(lambda r: r[b + '_rob']['rho']):+.2f}"
              f" (negative in {sum(r[b + '_rob']['rho'] < 0 for r in sel)}/{len(sel)})")
        print(f"   EG: pairs negative in {P(mean(lambda r: r[b + '_eg']['pair_in']))} vs across {P(mean(lambda r: r[b + '_eg']['pair_out']))};"
              f" robots opposing their step plain {P(mean(lambda r: r[b + '_eg']['step_plain']))},"
              f" EG own {P(mean(lambda r: r[b + '_eg']['step_own']))}, EG round {P(mean(lambda r: r[b + '_eg']['step_round']))}")
    print("\nT4 within a robot, decoder: success vs fall episodes, episode cos / class-mean cos (robots)")
    for r in rows:
        q = r["quality"]
        print(f"{r['upd']:6d} {r['seed']} | {q[0]:+.2f} / {q[1]:+.2f} ({q[2]})")


if __name__ == "__main__":
    main()
