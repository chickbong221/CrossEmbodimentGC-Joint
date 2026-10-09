"""Per-robot evaluation return of several runs (mean of the last 10 evaluations up to N updates), class means,
and robots learned: a robot counts as learned in a run when it reaches at least half of the best return any run
in the table (or in --reference runs) reaches on it. Cassie and the hexapod, which no run learns, then count
for nobody. With --thresholds FILE the thresholds are frozen: read from FILE, or (if FILE does not exist) set
once to half of the best return of these runs at any of the horizons and written to FILE, so that "learned"
does not change with the runs shown or the horizon.
Run from experiments/:  python analysis/returns_table.py LABEL=RUN_DIR [...] [--upto 100000,50000] [--thresholds F]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_morse as am


def window(tb, robot, upto, last=10):
    tag = f"eval/episode_return/{am.SHORT_NAMES.get(robot, robot)}"
    if tag not in tb or "steps/nr_updates" not in tb:
        return np.nan
    s, v = tb[tag]
    su, u = tb["steps/nr_updates"]
    o = np.argsort(su)
    upd = np.interp(s, su[o], u[o])
    if upd.max() < upto - 100:
        return np.nan
    sel = v[upd <= upto + 1][-last:]
    return float(sel.mean()) if len(sel) else np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--upto", default="100000")
    parser.add_argument("--reference", nargs="*", default=[], help="LABEL=RUN_DIR runs that only set the best return")
    parser.add_argument("--thresholds", default="", help="JSON of frozen per-robot 'learned' thresholds")
    args = parser.parse_args()
    ups = [int(u) for u in args.upto.split(",")]
    tabs = {}
    for spec in args.runs + args.reference:
        label, pattern = spec.split("=", 1)
        dirs = glob.glob(pattern)
        if not dirs:
            print(f"missing {spec}")
            continue
        tb = am.load_tb(dirs[0])
        tabs[label] = {n: {r: window(tb, r, n) for r in am.ROBOTS} for n in ups}
    shown = [s.split("=", 1)[0] for s in args.runs if s.split("=", 1)[0] in tabs]
    frozen = None
    if args.thresholds:
        if os.path.exists(args.thresholds):
            frozen = json.load(open(args.thresholds))
        else:
            frozen = {r: 0.5 * float(np.nanmax([tabs[l][n][r] for l in tabs for n in ups])) for r in am.ROBOTS}
            json.dump(frozen, open(args.thresholds, "w"), indent=1)
        print("frozen thresholds: " + ", ".join(f"{am.SHORT_NAMES[r]} {frozen[r]:.1f}" for r in am.ROBOTS))
    for n in ups:
        best = {r: np.nanmax([tabs[l][n][r] for l in tabs]) for r in am.ROBOTS}
        if frozen is not None:
            best = {r: 2.0 * frozen[r] for r in am.ROBOTS}
        print(f"\n== return at {n // 1000}k updates (mean of the last 10 evaluations)")
        print("robot".ljust(8) + "".join(f"{l[:13]:>14}" for l in shown))
        for r in am.ROBOTS:
            print(am.SHORT_NAMES[r].ljust(8) + "".join(f"{tabs[l][n][r]:14.1f}" for l in shown))
        for cls, members in list(am.CLASSES.items()) + [("ALL", am.ROBOTS)]:
            print(f"[{cls}]".ljust(8)[:8] + "".join(f"{np.nanmean([tabs[l][n][r] for r in members]):14.1f}" for l in shown))
        learned = {l: [r for r in am.ROBOTS if tabs[l][n][r] >= 0.5 * best[r] and best[r] > 5.0] for l in shown}
        print("learned".ljust(8) + "".join(f"{len(learned[l]):14d}" for l in shown))
        for l in shown:
            missing = [am.SHORT_NAMES[r] for r in am.ROBOTS if r not in learned[l] and best[r] > 5.0]
            print(f"   {l}: not learned {', '.join(missing) if missing else '-'}")


if __name__ == "__main__":
    main()
