"""Success rates of two checkpoints under paired action noise (morse_ckpt_eval_noise, <out>_noise.json).

Success = the episode reaches the horizon (length >= 999). Episodes with the same index used the same noise for both
checkpoints, so the comparison is paired: "A only" / "B only" count episodes that only one checkpoint survived, and
p is the exact two-sided sign (McNemar) test on those discordant episodes.

Usage: python analysis/noise_report.py NOISE_JSON [--robots cas,nao5,g1] [--out report.md]
"""
import argparse
import json
from math import comb


def sign_test(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("noise")
    p.add_argument("--robots", default="cas,nao5,g1,h1,tl,op3,go2,anyb,anyc,bkv0,bkvb,bg,bit,a1,go1,hx")
    p.add_argument("--labels", default="60k,90k")
    p.add_argument("--out", default="")
    a = p.parse_args()
    d = json.load(open(a.noise))
    (pa, ra), (pb, rb) = list(d.items())
    la, lb = a.labels.split(",")
    lines = [f"| Robot | Noise σ @ steps | Episodes | Success {la} | Success {lb} | {la} only | {lb} only | p (paired) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in [x for x in a.robots.split(",") if x]:
        for cond in ra:
            if r not in ra[cond]:
                continue
            sa = [l >= 999 for l in ra[cond][r]["length"]]
            sb = [l >= 999 for l in rb[cond][r]["length"]]
            only_a = sum(x and not y for x, y in zip(sa, sb))
            only_b = sum(y and not x for x, y in zip(sa, sb))
            lines.append(f"| {r} | {cond} | {len(sa)} | {100 * sum(sa) / len(sa):.0f}% | {100 * sum(sb) / len(sb):.0f}% | "
                         f"{only_a} | {only_b} | {sign_test(only_a, only_b):.3g} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
