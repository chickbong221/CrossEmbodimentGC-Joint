"""Final read of the V8 batch at 100k updates: mean return over ALL 16 robots, per-robot table, gate robots.

Reads tb_scalars.npz caches under runs/morse (analysis/tb_cache.py). The four MI runs on Hessian were restarted from
their 50k checkpoint, so their curve is the original run up to 50k and the resumed run after.
Run from experiments/:  python analysis/v8b_final.py [OUT.md]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_morse as am
from returns_table import window

H = "runs/morse"
STITCHED = {
    "J50": ("V8b-J/V8b-J50-s0", "V8b-J-resume/V8b-J50-s0-from50k"),
    "J75": ("V8b-J/V8b-J75-s0", "V8b-J-resume/V8b-J75-s0-from50k"),
    "R50": ("V8b-R/V8b-R50-s0", "V8b-R-resume/V8b-R50-s0-from50k"),
    "R75": ("V8b-R/V8b-R75-s0", "V8b-R-resume/V8b-R75-s0-from50k"),
}
SINGLE = {
    "J25": "V8b-J25/V8b-J25-s0", "R25": "V8b-R25/V8b-R25-s0", "G25": "V8b-G25/V8b-G25-s0",
    "G50": "V8b-G/V8b-G50-s0", "G75": "V8b-G/V8b-G75-s0",
    "EG-s1": "EG-IQL-M4/EG-IQL-M4-s1", "EG-s2": "EG-IQL-M4/EG-IQL-M4-s2", "57cfg-s1": "V8-57cfg/V8-57cfg-s1",
}
ORDER = ["J25", "J50", "J75", "R25", "R50", "R75", "G25", "G50", "G75", "EG-s1", "EG-s2", "57cfg-s1"]
# robots whose final return is all-or-nothing between runs, with the return that counts as learned
GATE = {"unitree_a1": 30, "unitree_go1": 30, "unitree_go2": 30, "badger": 30, "talos": 50, "nao_v5": 50, "cassie": 40,
        "hexapod": 10}
POINTS = (25000, 50000, 75000, 100000)


def load():
    tb = {}
    for k, (orig, resumed) in STITCHED.items():
        tb[k] = (am.load_tb(f"{H}/{orig}"), am.load_tb(f"{H}/{resumed}"))
    for k, path in SINGLE.items():
        if os.path.exists(f"{H}/{path}/tb_scalars.npz") or os.path.isdir(f"{H}/{path}"):
            tb[k] = (am.load_tb(f"{H}/{path}"), None)
    return tb


def main():
    tb = load()
    robots = list(am.ROBOTS)
    names = [k for k in ORDER if k in tb]

    def get(k, robot, n):
        orig, resumed = tb[k]
        return window(resumed if (resumed is not None and n > 50000) else orig, robot, n)

    def reach(k):
        orig, resumed = tb[k]
        return float(np.max((resumed if resumed is not None else orig)["steps/nr_updates"][1]))

    def mean_all(k, n, robs=robots):
        return float(np.nanmean([get(k, r, n) for r in robs])) if reach(k) >= n - 100 else float("nan")

    def fmt(x, nd=1):
        return "n/a" if not np.isfinite(x) else f"{x:.{nd}f}"

    out = []
    P = out.append
    P(f"Robots in every average ({len(robots)}): " + ", ".join(am.SHORT_NAMES.get(r, r) for r in robots))
    P("")
    P("## Mean return over all 16 robots (window = mean of the last 10 evals up to N updates)")
    P("")
    P("`avg 80-100k` is the mean of the 80k, 90k and 100k values; a single window is unstable for the all-or-nothing "
      "robots (Cassie, hexapod, A1/Go1/Go2, NAO, badger).")
    P("")
    P("| run | 25k | 50k | 75k | 100k | avg 80-100k | last update |")
    P("|---|---:|---:|---:|---:|---:|---:|")
    for k in names:
        avg = float(np.nanmean([mean_all(k, n) for n in (80000, 90000, 100000)])) if reach(k) >= 99900 else float("nan")
        P(f"| {k} | " + " | ".join(fmt(mean_all(k, n)) for n in POINTS) + f" | {fmt(avg)} | {reach(k):,.0f} |")
    P("")
    last = 100000
    P("## Per-robot return at 100k (mean of the last 10 evals)")
    P("")
    P("| robot | " + " | ".join(names) + " |")
    P("|---|" + "---:|" * len(names))
    for r in robots:
        P(f"| {am.SHORT_NAMES.get(r, r)} | " + " | ".join(
            fmt(get(k, r, last), 0) if reach(k) >= last - 100 else "n/a" for k in names) + " |")
    P("| **mean (16)** | " + " | ".join(f"**{fmt(mean_all(k, last))}**" for k in names) + " |")
    for cls, members in am.CLASSES.items():
        P(f"| [{cls}] | " + " | ".join(fmt(mean_all(k, last, members)) for k in names) + " |")
    P("")
    P("## Gate robots learned at 100k (X = return at or above the threshold)")
    P("")
    P("Thresholds: " + ", ".join(f"{am.SHORT_NAMES[r]} {t}" for r, t in GATE.items()) + ".")
    P("")
    P("| run | " + " | ".join(am.SHORT_NAMES[r] for r in GATE) + " | learned |")
    P("|---|" + ":---:|" * len(GATE) + "---:|")
    for k in names:
        if reach(k) < last - 100:
            continue
        flags = [get(k, r, last) >= t for r, t in GATE.items()]
        P(f"| {k} | " + " | ".join("X" if f else "." for f in flags) + f" | {sum(flags)}/{len(GATE)} |")
    P("")
    grid = list(range(10000, 100001, 10000))
    P("## All-robot mean every 10k updates")
    P("")
    P("| run | " + " | ".join(f"{g // 1000}k" for g in grid) + " |")
    P("|---|" + "---:|" * len(grid))
    for k in names:
        P(f"| {k} | " + " | ".join(fmt(mean_all(k, g)) if reach(k) >= g - 100 else "" for g in grid) + " |")
    text = "\n".join(out)
    print(text)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
