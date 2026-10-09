"""The 57 run (CAP-8x64-top4-s0) next to the MORSE-v6 runs, from cached TensorBoard curves (tb_scalars.npz).

Prints markdown tables:
  A  per-robot return at 100k of the 57 run and every finished v6 run, and where the gap comes from
  B  contested robots: peak and final return of every finished run (learned at some time? still learned at the end?)
  C  the hexapod over every cached run
  D  seed-paired comparison: runs with the same seed share the minibatch stream and the critic (the critics never see
     the policy), so differences within a seed come from the actor
  E  v6 3/4 vs 1/2 width, and robot-type vs free core, seed by seed
  F  decoder-router statistics of the 57 run and of v6
  G  early actor loss by seed (the evidence for the pairing in D)
  H  the 57-run family at 100k with and without the hexapod
  I  decoder sparsity x core capacity, as seed-adjusted deviations (descriptive, not a causal test)
  J  hexapod and Cassie over training: mean return per 5k updates
  K  v6 results by horizon, and means over seeds
  L  v6 router statistics at the end of training (core and decoder)
  M  v6 core-router drift after 50k (from routing.npz): are the late collapses a routing change?
  N  do robots with an identical core expert set to another robot get lost more often?
  O  the late collapse in v6 3/4 s0: badger, nao5 and go2 over training, next to the same seed's other runs
Runs that have not reached a horizon are left out of the table for that horizon.

Run from the repo root:  python experiments/analysis/compare_57run_v6.py TBCACHE THRESHOLDS_JSON
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_morse as am  # noqa: E402
from returns_table import window  # noqa: E402

C, TH = sys.argv[1], json.load(open(sys.argv[2]))
SH = [am.SHORT_NAMES[r] for r in am.ROBOTS]
THS = {am.SHORT_NAMES[r]: TH[r] for r in am.ROBOTS}
NOHX = [r for r in SH if r != "hx"]
CONTESTED = ["go2", "g1", "tl", "nao5", "cas", "hx"]
THS_C = dict(THS, hx=20.0)      # the frozen hexapod threshold (3.0) is at the noise level; the 57 run walks at ~90

RUNS = {"57 run": "CAP-8x64-top4/CAP-8x64-top4-s0", "dec top-2": "CAP-8x64-top2/CAP-8x64-top2-s0",
        "dec 8x64 dense": "CAP-8x64/CAP-8x64-s0", "dec 4x64 dense": "CAP-4x64/CAP-4x64-s0", "EG": "EG-IQL-M4/1790757709_oghlb",
        "URMA s0": "URMA-IQL-A4/URMA-IQL-A4-s0", "URMA s1": "T1-URMA-A4/T1-URMA-A4-s1", "URMA s2": "T1-URMA-A4/T1-URMA-A4-s2"}
for g, exp, seeds in (("MoE", "MORSE-IQL-v2-noconf-A4", 3), ("v3", "MORSE-v3", 3), ("v4", "MORSE-v4", 3), ("v5", "MORSE-v5", 3),
                      ("v6 1/2", "MORSE-v6-w50", 5), ("v6 3/4", "MORSE-v6-w75", 5), ("v6 1/2 robot-core", "MORSE-v6-w50-robotMI", 2)):
    for s in range(seeds):
        RUNS[f"{g} s{s}"] = f"{exp}/{exp}-s{s}"
V6 = [f"{g} s{s}" for g in ("v6 3/4", "v6 1/2", "v6 1/2 robot-core") for s in range(5)]
_tb = {}


def tb(k):
    if k not in _tb:
        p = os.path.join(C, RUNS[k]) if k in RUNS else None
        _tb[k] = am.load_tb(p) if p and os.path.exists(os.path.join(p, "tb_scalars.npz")) else None
    return _tb[k]


def series(t, tag):
    """(updates, values) of a tag: logged steps are mapped to updates through steps/nr_updates."""
    s, v = t[tag]
    su, u = t["steps/nr_updates"]
    o = np.argsort(su)
    return np.interp(s, su[o], u[o]), np.asarray(v)


def reached(t):
    return int(t["steps/nr_updates"][1].max())


def ok(k, h):
    """The run is cached and has reached update h (same tolerance as returns_table.window)."""
    return tb(k) is not None and reached(tb(k)) >= h - 100


def finished(k):
    return ok(k, 100000)


def mean_win(t, h, robots=SH):
    return float(np.mean([window(t, r, h) for r in robots]))


def mean_avg(t, a, b, robots=SH):
    """Mean over robots of the average return over all evaluations with a <= update <= b."""
    out = []
    for r in robots:
        u, v = series(t, f"eval/episode_return/{r}")
        out.append(v[(u >= a) & (u <= b)].mean())
    return float(np.mean(out))


def n_learned(t, h, robots=SH):
    return sum(window(t, r, h) >= THS[r] for r in robots)


# ------------------------------------------------------------------ A
print("## A. Per-robot return at 100k (mean of the last 10 evaluations); * = learned (frozen threshold)\n")
cols = [c for c in ["57 run"] + V6 if finished(c)]
means = {c: mean_win(tb(c), 100000) for c in cols}
best_v6 = max((c for c in cols if c != "57 run"), key=lambda c: means[c])
same = max((c for c in cols if c.endswith(" s0") and c != "57 run"), key=lambda c: means[c])
print("| robot | " + " | ".join(cols) + f" | 57 run minus {same} (best v6 on its seed) | 57 run minus {best_v6} (best v6 overall) |")
print("|---|" + "---|" * (len(cols) + 2))
for r in SH:
    v = {c: window(tb(c), r, 100000) for c in cols}
    cells = " | ".join(f"{v[c]:.0f}{'*' if v[c] >= THS[r] else ''}" for c in cols)
    d1, d2 = (v["57 run"] - v[same]) / 16, (v["57 run"] - v[best_v6]) / 16
    print(f"| {r} | {cells} | {d1:+.2f} | {d2:+.2f} |")
print("| **mean (16 robots)** | " + " | ".join(f"**{means[c]:.1f}** ({n_learned(tb(c), 100000)})" for c in cols)
      + f" | {means['57 run'] - means[same]:+.1f} | {means['57 run'] - means[best_v6]:+.1f} |")
print("| mean without hexapod | " + " | ".join(f"{mean_win(tb(c), 100000, NOHX):.1f}" for c in cols) + " | | |")
best = {r: max(window(tb(c), r, 100000) for c in cols) for r in SH}
stable, contested = sum(best[r] for r in SH if r not in CONTESTED), sum(best[r] for r in CONTESTED)
print(f"\nThe ten non-contested robots add {stable / 16:.1f} to the 16-robot mean in every run here (a1 and go1 stay at 14-15). "
      f"The six contested robots add {contested / 16:.1f} if all are kept at their best return, so the ceiling is "
      f"{(stable + contested) / 16:.1f}.\n")

# ------------------------------------------------------------------ B
print("## B. Contested robots: peak (20-evaluation average, at k updates) -> final (last 10 evaluations)\n")
print("`*` still learned at the end, `x` learned at some time and lost, `.` never learned.\n")
print("| run | " + " | ".join(CONTESTED) + " | kept |")
print("|---|" + "---|" * (len(CONTESTED) + 1))
order = (["57 run", "dec top-2", "dec 8x64 dense", "dec 4x64 dense"] + [f"v6 3/4 s{s}" for s in range(5)] + [f"v6 1/2 s{s}" for s in range(5)]
         + [f"v6 1/2 robot-core s{s}" for s in range(2)] + [f"v5 s{s}" for s in range(3)] + [f"v4 s{s}" for s in range(3)]
         + [f"MoE s{s}" for s in range(3)] + [f"URMA s{s}" for s in range(3)] + ["EG"])
tally = {r: {"*": 0, "x": 0, ".": 0} for r in CONTESTED}
kept_by, sym_by = {}, {}
for k in order:
    if not finished(k):
        continue
    t, cells, kept = tb(k), [], 0
    sym_by[k] = {}
    for r in CONTESTED:
        u, v = series(t, f"eval/episode_return/{r}")
        ma = np.convolve(v, np.ones(20) / 20, mode="valid")
        peak, at, fin = float(ma.max()), int(u[int(np.argmax(ma)) + 19]), float(v[-10:].mean())
        sym = "*" if fin >= THS_C[r] else ("x" if peak >= THS_C[r] else ".")
        tally[r][sym] += 1
        sym_by[k][r] = sym
        kept += sym == "*"
        cells.append(f"{peak:.0f}@{at // 1000}k->{fin:.0f}{sym}")
    kept_by[k] = kept
    print(f"| {k} | " + " | ".join(cells) + f" | {kept} |")
print("\n| robot | runs that kept it | learned then lost | never learned |\n|---|---|---|---|")
for r in CONTESTED:
    print(f"| {r} | {tally[r]['*']} | {tally[r]['x']} | {tally[r]['.']} |")
print("\nContested robots kept at the end (of 6), by configuration:\n")
print("| configuration | runs | kept per run | mean |\n|---|---|---|---|")
for name, keys in (("57 run", ["57 run"]), ("57 family siblings (dec top-2, dense 8x64, dense 4x64)", ["dec top-2", "dec 8x64 dense", "dec 4x64 dense"]),
                   ("v6 3/4", [f"v6 3/4 s{s}" for s in range(5)]), ("v6 1/2", [f"v6 1/2 s{s}" for s in range(5)]),
                   ("v6 1/2 robot-type core", [f"v6 1/2 robot-core s{s}" for s in range(2)]), ("v5", [f"v5 s{s}" for s in range(3)]),
                   ("v4", [f"v4 s{s}" for s in range(3)]), ("MoE", [f"MoE s{s}" for s in range(3)]),
                   ("URMA", [f"URMA s{s}" for s in range(3)]), ("EG", ["EG"])):
    ks = [k for k in keys if k in kept_by]
    if ks:
        print(f"| {name} | {len(ks)} | {', '.join(str(kept_by[k]) for k in ks)} | {np.mean([kept_by[k] for k in ks]):.1f} |")
print("\nPlateau = go2, g1, tl and nao5 all kept at the end (with the ten non-contested robots this is 54.1 on the 15 robots "
      "without the hexapod). Bonus = Cassie and/or hexapod kept on top.\n")
print("| configuration | runs | plateau reached | of the four kept, per run | bonus kept, per run |\n|---|---|---|---|---|")
PLATEAU = ("go2", "g1", "tl", "nao5")
for name, keys in (("57 run", ["57 run"]), ("57 family siblings", ["dec top-2", "dec 8x64 dense", "dec 4x64 dense"]),
                   ("v6 3/4", [f"v6 3/4 s{s}" for s in range(5)]), ("v6 1/2", [f"v6 1/2 s{s}" for s in range(5)]),
                   ("v6 1/2 robot-type core", [f"v6 1/2 robot-core s{s}" for s in range(2)]), ("v5", [f"v5 s{s}" for s in range(3)]),
                   ("v4", [f"v4 s{s}" for s in range(3)]), ("MoE", [f"MoE s{s}" for s in range(3)]),
                   ("URMA", [f"URMA s{s}" for s in range(3)])):
    ks = [k for k in keys if k in sym_by]
    if not ks:
        continue
    four = [sum(sym_by[k][r] == "*" for r in PLATEAU) for k in ks]
    bonus = ["+".join(r for r in ("cas", "hx") if sym_by[k][r] == "*") or "-" for k in ks]
    print(f"| {name} | {len(ks)} | {sum(f == 4 for f in four)} of {len(ks)} | {', '.join(map(str, four))} | {', '.join(bonus)} |")
print("\nThe same by seed (every finished run of the table above that has this seed; seed 0 also holds the 57 run and its three siblings):\n")
print("| seed | runs | plateau reached | bonus robot kept |\n|---|---|---|---|")
for sd in (0, 1, 2):
    ks = [k for k in sym_by if k.endswith(f" s{sd}") or (sd == 0 and k in ("57 run", "dec top-2", "dec 8x64 dense", "dec 4x64 dense"))]
    n_p = sum(all(sym_by[k][r] == "*" for r in PLATEAU) for k in ks)
    n_b = sum(any(sym_by[k][r] == "*" for r in ("cas", "hx")) for k in ks)
    print(f"| {sd} | {len(ks)} | {n_p} of {len(ks)} | {n_b} |")

# ------------------------------------------------------------------ C
print("\n## C. Hexapod, every cached run with a hexapod curve\n")
rows = []
for p in sorted(glob.glob(os.path.join(C, "*", "*", "tb_scalars.npz"))):
    t = am.load_tb(os.path.dirname(p))
    try:
        u, v = series(t, "eval/episode_return/hx")
    except KeyError:
        continue
    if len(v) < 50:
        continue
    ma = np.convolve(v, np.ones(20) / 20, mode="valid")
    j = int(np.argmax(ma))
    rows.append((os.path.relpath(os.path.dirname(p), C).replace("\\", "/"), float(ma.max()), int(u[j + 19]), float(v[-10:].mean())))
rows.sort(key=lambda r: -r[1])
print("| run | best 20-eval average | at update | last 10 evaluations |\n|---|---|---|---|")
for r in rows[:8]:
    print(f"| {r[0]} | {r[1]:.1f} | {r[2] // 1000}k | {r[3]:.1f} |")
print(f"\n{len(rows)} cached runs; peak above 20 at some time: {sum(r[1] > 20 for r in rows)}; still above 20 at the end: "
      f"{sum(r[3] > 20 for r in rows)}. (Episode length is 1000 for the hexapod in every run: it never falls, it stands still.)")
v6_hx = [r for r in rows if "MORSE-v6" in r[0]]
print(f"Of the {len(v6_hx)} cached v6 runs, the best hexapod 20-evaluation average is {max(r[1] for r in v6_hx):.1f} "
      f"({max(v6_hx, key=lambda r: r[1])[0]}).")

# ------------------------------------------------------------------ D
H1, H2 = 65000, 100000
print(f"\n## D. Seed-paired comparison at {H1 // 1000}k and {H2 // 1000}k (16 robots / without hexapod), and averages over the later part of training\n")
pairs = [("URMA", "URMA s{}"), ("MoE (4 full dense core, dec 4x128 dense)", "MoE s{}"), ("v3", "v3 s{}"), ("v4", "v4 s{}"), ("v5", "v5 s{}"),
         ("v6 1/2", "v6 1/2 s{}"), ("v6 3/4", "v6 3/4 s{}"), ("v6 1/2 robot-type core", "v6 1/2 robot-core s{}")]
seed0_only = [("57 run", "57 run"), ("dec 8x64 top-2", "dec top-2"), ("dec 8x64 dense", "dec 8x64 dense"),
              ("dec 4x64 dense", "dec 4x64 dense"), ("EG", "EG")]


def pair_cell(k, h, a, b):
    """'at h, 16 / 15 robots' and 'average a..b, 16 / 15 robots' for run k, or '-' when the run has not reached h."""
    if not ok(k, h):
        return "-", "-"
    t = tb(k)
    return (f"{mean_win(t, h):.1f} / {mean_win(t, h, NOHX):.1f}", f"{mean_avg(t, a, b):.1f} / {mean_avg(t, a, b, NOHX):.1f}")


print("| seed | run | at 65k, 16 / 15 robots | average 40k-65k | at 100k, 16 / 15 robots | average 50k-100k |\n|---|---|---|---|---|---|")
for s in (0, 1, 2):
    rows = [(n, p.format(s)) for n, p in pairs] + (seed0_only if s == 0 else [])
    for n, k in rows:
        if tb(k) is None:
            continue
        a1, a2 = pair_cell(k, H1, 40000, H1)
        b1, b2 = pair_cell(k, H2, 50000, H2)
        print(f"| {s} | {n} | {a1} | {a2} | {b1} | {b2} |")
print("\nDeviation of each finished run from the mean of the five reference configs (URMA, MoE, v3, v4, v5) at the same seed;")
print("average return over all evaluations from 50k to 100k, 16 robots.\n")
ref = {s: np.mean([mean_avg(tb(p.format(s)), 50000, 100000) for _, p in pairs[:5]]) for s in (0, 1, 2)}
print("| run | s0 | s1 | s2 |\n|---|---|---|---|")
for n, p in pairs + seed0_only:
    cells = []
    for s in (0, 1, 2):
        k = p.format(s) if "{}" in p else (p if s == 0 else None)
        cells.append(f"{mean_avg(tb(k), 50000, 100000) - ref[s]:+.1f}" if k and finished(k) else "")
    if any(cells):
        print(f"| {n} | " + " | ".join(cells) + " |")
print("| *reference mean (absolute)* | " + " | ".join(f"{ref[s]:.1f}" for s in (0, 1, 2)) + " |")

# ------------------------------------------------------------------ E
print("\n## E. v6 3/4 vs 1/2 width, and robot-type vs free core (both 1/2 width), seed by seed (mean return, 16 robots)\n")
print("Cells: first run's value vs second's, and the difference.\n")
print("| seed | comparison | at 65k | average 40k-65k | at 100k | average 50k-100k |\n|---|---|---|---|---|---|")
diffs = {}
for name, fa, fb, seeds in (("3/4 minus 1/2", "v6 3/4 s{}", "v6 1/2 s{}", range(3)),
                            ("robot-type core minus free core", "v6 1/2 robot-core s{}", "v6 1/2 s{}", range(2))):
    for s in seeds:
        ka, kb = fa.format(s), fb.format(s)
        cells = []
        for j, (h, a) in enumerate(((H1, 40000), (H2, 50000))):
            if ok(ka, h) and ok(kb, h):
                x, y = mean_win(tb(ka), h), mean_win(tb(kb), h)
                cells.append(f"{x:.1f} vs {y:.1f} ({x - y:+.1f})")
                diffs.setdefault((name, 2 * j), []).append(x - y)
                x, y = mean_avg(tb(ka), a, h), mean_avg(tb(kb), a, h)
                cells.append(f"{x:.1f} vs {y:.1f} ({x - y:+.1f})")
                diffs.setdefault((name, 2 * j + 1), []).append(x - y)
            else:
                cells += ["-", "-"]
        print(f"| {s} | {name} | " + " | ".join(cells) + " |")
for name in ("3/4 minus 1/2", "robot-type core minus free core"):
    cells = [f"{np.mean(diffs[(name, i)]):+.1f} ({len(diffs[(name, i)])} seeds)" if (name, i) in diffs else "-" for i in range(4)]
    print(f"| mean | {name} | " + " | ".join(cells) + " |")

# ------------------------------------------------------------------ F
print("\n## F. Decoder-router statistics (mean within +-1.5k updates of the point)\n")


def at(t, tag, u0, half=1500):
    u, v = series(t, tag)
    m = (u >= u0 - half) & (u <= u0 + half)
    return float(v[m].mean()) if m.any() else float("nan")


print("| run | tag | 5k | 25k | 50k | 95k |\n|---|---|---|---|---|---|")
for k in ("57 run", "dec top-2", "dec 8x64 dense", "v6 3/4 s0", "v6 3/4 s1", "v6 3/4 s2", "v6 1/2 s0", "v6 1/2 s1", "v6 1/2 s2",
          "v6 1/2 robot-core s0", "v5 s2"):
    t = tb(k)
    if t is None:
        continue
    for tag in ("eff_experts", "router_entropy", "usage_min", "usage_max"):
        print(f"| {k} | {tag} | " + " | ".join(f"{at(t, 'morse/decoder/' + tag, u0):.3f}" for u0 in (5000, 25000, 50000, 95000)) + " |")

# ------------------------------------------------------------------ G
print("\n## G. Actor loss over the first 500 updates, by seed\n")
print("| seed | runs | min | max |\n|---|---|---|---|")
for s, keys in ((0, ["URMA s0", "MoE s0", "57 run", "dec top-2", "dec 8x64 dense", "dec 4x64 dense", "v3 s0", "v4 s0", "v5 s0", "v6 1/2 s0",
                     "v6 3/4 s0", "v6 1/2 robot-core s0"]),
                (1, [f"{g} s1" for g in ("URMA", "MoE", "v3", "v4", "v5", "v6 1/2", "v6 3/4", "v6 1/2 robot-core")]),
                (2, [f"{g} s2" for g in ("URMA", "MoE", "v3", "v4", "v5", "v6 1/2", "v6 3/4")])):
    vals = []
    for k in keys:
        t = tb(k)
        if t is None:
            continue
        u, v = series(t, "policy_loss")
        vals.append(float(v[u < 500].mean()))
    print(f"| {s} | {len(vals)} | {min(vals):.3f} | {max(vals):.3f} |")

# ------------------------------------------------------------------ H
print("\n## H. The 57-run family at 100k (seed 0): what the sparse decoder adds\n")
print("| decoder (all: 4 full-width dense core experts) | 16 robots | 15 robots (no hexapod) | Cassie | hexapod |\n|---|---|---|---|---|")
for n, k in (("4 x 128 dense (plain MoE s0)", "MoE s0"), ("4 x 64 dense", "dec 4x64 dense"), ("8 x 64 dense", "dec 8x64 dense"),
             ("8 x 64 top-2", "dec top-2"), ("8 x 64 top-4 (the 57 run)", "57 run")):
    t = tb(k)
    print(f"| {n} | {mean_win(t, 100000):.1f} | {mean_win(t, 100000, NOHX):.1f} | {window(t, 'cas', 100000):.0f} | {window(t, 'hx', 100000):.0f} |")

# ------------------------------------------------------------------ I
print("\n## I. Decoder sparsity x core capacity: deviation from the seed's reference mean (average over 50k-100k, 16 robots)\n")
print("Core capacity = active core parameters per routing unit relative to URMA's core: URMA 1.0x, v3 0.41x, v5 1.37x, v4 and v6 1/2 1.65x, "
      "v6 3/4 2.74x, plain MoE and the 57-run family 4.0x. A decoder counts as sparse when top-k < K. The v6 robot-type core "
      "(1.65x, one expert per robot) is left out: it is a different core mechanism.\n")
cells = {("dense decoder", "core < 2x"): [f"URMA s{s}" for s in range(3)],
         ("sparse decoder", "core < 2x"): [f"{g} s{s}" for g in ("v3", "v4", "v5") for s in range(3)] + [f"v6 1/2 s{s}" for s in range(3)],
         ("dense decoder", "core >= 2.7x"): [f"MoE s{s}" for s in range(3)] + ["dec 4x64 dense", "dec 8x64 dense"],
         ("sparse decoder", "core >= 2.7x"): ["57 run", "dec top-2"] + [f"v6 3/4 s{s}" for s in range(3)]}
print("| decoder | core | runs | mean deviation | per run |\n|---|---|---|---|---|")
for (d, c), keys in cells.items():
    devs = []
    for k in keys:
        seed = int(k[-1]) if re.search(r" s\d$", k) else 0
        devs.append(mean_avg(tb(k), 50000, 100000) - ref[seed])
    print(f"| {d} | {c} | {len(keys)} | {np.mean(devs):+.1f} | " + ", ".join(f"{k} {x:+.1f}" for k, x in zip(keys, devs)) + " |")
k_nov3 = [k for k in cells[("sparse decoder", "core < 2x")] if not k.startswith("v3")]
print(f"\nWithout v3 (its core is only 0.41x), the sparse-decoder, core < 2x cell has {len(k_nov3)} runs and mean deviation "
      f"{np.mean([mean_avg(tb(k), 50000, 100000) - ref[int(k[-1])] for k in k_nov3]):+.1f}.")

# ------------------------------------------------------------------ J
print("\n## J. Hexapod and Cassie over training: mean return per 5k-update bin (right edge shown)\n")
edges = list(range(0, 100001, 5000))
print("| robot | run | " + " | ".join(f"{e // 1000}k" for e in edges[1:]) + " |\n|---|---|" + "---|" * (len(edges) - 1))
for r in ("hx", "cas"):
    for k in ("57 run", "dec top-2", "dec 8x64 dense", "v5 s2", "v6 3/4 s0", "v6 3/4 s1", "v6 3/4 s2", "v6 1/2 s0", "v6 1/2 robot-core s0",
              "v6 1/2 robot-core s1"):
        t = tb(k)
        if t is None:
            continue
        u, v = series(t, f"eval/episode_return/{r}")
        print(f"| {r} | {k} | " + " | ".join(f"{v[(u > a) & (u <= b)].mean():.0f}" if ((u > a) & (u <= b)).any() else "-"
                                              for a, b in zip(edges[:-1], edges[1:])) + " |")

# ------------------------------------------------------------------ K
print("\n## K. v6 results by horizon, and means over seeds (cells: mean return over 16 robots, robots learned in brackets)\n")
HZ = (25000, 50000, 65000, 75000, 100000)
print("| run | " + " | ".join(f"{h // 1000}k" for h in HZ) + " | 100k without hexapod |\n|---|" + "---|" * (len(HZ) + 1))


def k_row(name, keys):
    cells = []
    for h in HZ:
        if all(ok(k, h) for k in keys):
            cells.append(f"{np.mean([mean_win(tb(k), h) for k in keys]):.1f} ({np.mean([n_learned(tb(k), h) for k in keys]):.1f})")
        else:
            cells.append("-")
    nh = f"{np.mean([mean_win(tb(k), 100000, NOHX) for k in keys]):.1f}" if all(ok(k, 100000) for k in keys) else "-"
    print(f"| {name} | " + " | ".join(cells) + f" | {nh} |")


for k in V6:
    if tb(k) is not None:
        k_row(k, [k])
print("| *means over seeds* | | | | | | |")
for name, keys in (("57 run (seed 0)", ["57 run"]), ("v6 3/4, s0-s2", [f"v6 3/4 s{s}" for s in range(3)]),
                   ("v6 1/2, s0-s2", [f"v6 1/2 s{s}" for s in range(3)]),
                   ("v6 1/2 robot-type core, s0-s1", [f"v6 1/2 robot-core s{s}" for s in range(2)]),
                   ("v5, s0-s2", [f"v5 s{s}" for s in range(3)]), ("v4, s0-s2", [f"v4 s{s}" for s in range(3)]),
                   ("MoE, s0-s2", [f"MoE s{s}" for s in range(3)]), ("URMA, s0-s2", [f"URMA s{s}" for s in range(3)]),
                   ("EG (one run)", ["EG"])):
    k_row(name, keys)
print("| *seed 0 only* | | | | | | |")
for name, keys in (("57 run", ["57 run"]), ("v6 3/4 s0", ["v6 3/4 s0"]), ("v6 1/2 s0", ["v6 1/2 s0"]),
                   ("v6 1/2 robot-core s0", ["v6 1/2 robot-core s0"]), ("v5 s0", ["v5 s0"]), ("v4 s0", ["v4 s0"]),
                   ("MoE s0", ["MoE s0"]), ("URMA s0", ["URMA s0"])):
    k_row(name, keys)

# ------------------------------------------------------------------ L
print("\n## L. v6 router statistics (mean of the last 5 logged values)\n")
print("H(E) = entropy of the marginal expert usage (max 2.08 for 8 experts). H(E|group) = entropy given the group: the robot for the "
      "core, the joint type for the decoder. MI = H(E) - H(E|group). Ratio = router-gradient norm of the MI term over the task's.\n")
print("| run | core H(E) | core H(E\\|robot) | core MI | core MI/task | decoder H(E) | decoder H(E\\|joint) | decoder MI | decoder MI/task |")
print("|---|---|---|---|---|---|---|---|---|")
for k in V6:
    t = tb(k)
    if t is None:
        continue
    cells = []
    for b in ("core", "decoder"):
        for tag in ("h_expert", "h_expert_given_type", "mi", "router_grad_mi_ratio"):
            tg = f"morse/{b}/{tag}"
            if tg in t:
                s_, v_ = t[tg]
                cells.append(f"{np.asarray(v_)[np.argsort(s_)][-5:].mean():.2f}")
            else:
                cells.append("-")
    print(f"| {k} | " + " | ".join(cells) + " |")

# ------------------------------------------------------------------ M
def routing(k):
    p = RUNS.get(k)
    f = os.path.join(C, p, "morse_diagnostics", "routing.npz") if p else None
    return np.load(f, allow_pickle=True) if f and os.path.exists(f) else None


print("\n## M. v6 core-router drift after 50k (routing.npz, one record per 500 updates)\n")
print("Selection pattern = share of a robot's states in which each core expert is in the top-4 (4 slots per state). Drift = slots (of 4) "
      "that differ from the same robot's pattern near 50k (average over 47.5k-52.5k), measured over 90k-99.5k. Freeze = last update at "
      "which a robot's pattern was more than 0.1 slot from its final one.\n")
print("| run | median robot freezes at | last robot to freeze | robots drifting > 0.1 slot after 50k | drift nao5 | drift bg | mean drift |")
print("|---|---|---|---|---|---|---|")
for k in V6:
    z = routing(k)
    if z is None:
        continue
    names, st, sel = [str(n) for n in z["robot_names"]], z["step"], z["core_select"]
    ref = sel[(st >= 47500) & (st <= 52500)].mean(0)
    drift = np.abs(sel[st >= 90000].mean(0) - ref).sum(-1) / 2
    dist = np.abs(sel - sel[-4:].mean(0)).sum(-1) / 2
    freeze = []
    for i in range(len(names)):
        bad = np.where(dist[:, i] > 0.1)[0]
        freeze.append(0 if not len(bad) else int(st[min(bad.max() + 1, len(st) - 1)]))
    print(f"| {k} | {int(np.median(freeze))} | {names[int(np.argmax(freeze))]} ({max(freeze)}) | {int((drift > 0.1).sum())} | "
          f"{drift[names.index('nao5')]:.2f} | {drift[names.index('bg')]:.2f} | {drift.mean():.2f} |")

# ------------------------------------------------------------------ N
print("\n## N. Robots with an identical core expert set to another robot: kept or lost? (free-core v6 runs, end of training)\n")
print("Overlap of two robots = share of the 4 top-k slots they have in common (sum over experts of the smaller selection frequency, over 4). "
      "Identical = overlap above 0.95. Outcome as in table B.\n")
print("| run | robot | outcome | robots with an identical core expert set |\n|---|---|---|---|")
n_ident = {"kept": 0, "lost": 0}
for k in [f"v6 3/4 s{s}" for s in range(3)] + [f"v6 1/2 s{s}" for s in range(3)]:
    z = routing(k)
    if z is None or k not in sym_by:
        continue
    names, sel = [str(n) for n in z["robot_names"]], z["core_select"][-4:].mean(0)
    ov = np.minimum(sel[:, None, :], sel[None, :, :]).sum(-1) / 4.0
    for r in ("go2", "nao5", "tl"):
        i = names.index(r)
        same = [names[j] for j in range(len(names)) if j != i and ov[i, j] > 0.95]
        out = {"*": "kept", "x": "lost", ".": "never"}[sym_by[k][r]]
        if same and out in n_ident:
            n_ident[out] += 1
        print(f"| {k} | {r} | {out} | {', '.join(same) or '-'} |")
print(f"\nRobot-run pairs (go2, nao5, tl) with an identical set: {n_ident['kept']} kept, {n_ident['lost']} lost.")

# ------------------------------------------------------------------ O
print("\n## O. Late collapse in v6 3/4 s0: mean return per 5k-update bin (right edge shown), same seed in the other rows\n")
print("| robot | run | " + " | ".join(f"{e // 1000}k" for e in edges[1:]) + " |\n|---|---|" + "---|" * (len(edges) - 1))
for r in ("bg", "nao5", "go2"):
    for k in ("v6 3/4 s0", "v6 1/2 s0", "57 run"):
        t = tb(k)
        u, v = series(t, f"eval/episode_return/{r}")
        print(f"| {r} | {k} | " + " | ".join(f"{v[(u > a) & (u <= b)].mean():.0f}" for a, b in zip(edges[:-1], edges[1:])) + " |")
