"""Write a compact cache of a run's TensorBoard scalars (<run>/tb_scalars.npz), for comparing runs from different
servers without copying the full event files. analyze_morse.load_tb reads the cache when a folder has no event files.

Usage: python analysis/tb_cache.py RUN_DIR [RUN_DIR ...] [--prefixes eval/,steps/,policy_loss]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_morse as am


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--prefixes", default="eval/,steps/,policy_loss,morse/decoder/,morse/router_grad")
    args = parser.parse_args()
    prefixes = tuple(p for p in args.prefixes.split(",") if p)
    for run in args.runs:
        tb = am.load_tb(run)
        arrays = {}
        for tag, (s, v) in tb.items():
            if tag.startswith(prefixes):
                arrays["s::" + tag], arrays["v::" + tag] = np.asarray(s), np.asarray(v)
        np.savez_compressed(os.path.join(run, "tb_scalars.npz"), **arrays)
        print(f"{run}: {len(arrays) // 2} tags cached")


if __name__ == "__main__":
    main()
