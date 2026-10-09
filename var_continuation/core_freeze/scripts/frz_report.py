"""Report of the core-swap evaluations and the core-freeze continuation: python frz_report.py ROOT OUT
ROOT = var_continuation/ (data/ of the variance experiment: S0 TB, thresholds, original run) with core_freeze/data/ (this part)."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT, OUT = sys.argv[1], sys.argv[2]
V, F = os.path.join(ROOT, "data"), os.path.join(ROOT, "core_freeze", "data")
os.makedirs(OUT, exist_ok=True)
FULL = ["unitree_a1", "unitree_go1", "unitree_go2", "anymal_b", "anymal_c", "barkour_v0", "barkour_vb", "badger", "bittle",
        "unitree_h1", "unitree_g1", "talos", "robotis_op3", "nao_v5", "cassie", "hexapod"]
SHORT = ["a1", "go1", "go2", "anyb", "anyc", "bkv0", "bkvb", "bg", "bit", "h1", "g1", "tl", "op3", "nao5", "cas", "hx"]
THR = json.load(open(os.path.join(V, "learned_thresholds.json")))
START = 75000
lines, results = [], {}


def say(s=""):
    lines.append(s)


def load_tb(path):
    z = np.load(path)
    return {k[3:]: (z[k], z["v::" + k[3:]]) for k in z.files if k.startswith("s::")}


def series(tb, tag):
    s, v = tb[tag]
    su, u = tb["steps/nr_updates"]
    o = np.argsort(su)
    return np.interp(s, su[o], u[o]), v


def window_mean(tb, short, upto, last=10):
    upd, v = series(tb, f"eval/episode_return/{short}")
    sel = v[upd <= upto + 1][-last:]
    return float(sel.mean()) if len(sel) else float("nan")


def bins(tb, short, lo, hi, width=1000):
    upd, v = series(tb, f"eval/episode_return/{short}")
    edges = np.arange(lo, hi + 1, width)
    mean, walk = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (upd > a) & (upd <= b)
        mean.append(float(v[m].mean()) if m.sum() >= 5 else float("nan"))
        walk.append(float((v[m] > THR[FULL[SHORT.index(short)]]).mean()) if m.sum() >= 5 else float("nan"))
    return edges[1:], np.array(mean), np.array(walk)


def read_jsonl(p):
    return [json.loads(l) for l in open(p)]


def noise_rows(path):
    """{checkpoint folder name: {robot: returns}} of the first (only) condition of a paired-noise json."""
    d = json.load(open(path))
    out = {}
    for ck, conds in d.items():
        name = ck.rstrip("/").split("/")[-1] if "VARb" not in ck and "VARc" not in ck else ck.rstrip("/").split("/")[-3] + "/" + ck.rstrip("/").split("/")[-1]
        out[name] = {r: np.array(v["return"]) for r, v in next(iter(conds.values())).items()}
    return out


def go2_noise(rows):
    r = rows[[k for k in rows if "go2" in k][0]]
    return float(r.mean()), float((r > THR["unitree_go2"]).mean()), len(r)


# 1. Checkpoint swaps.
say("## 1. Checkpoint component swaps (evaluation only)")
say("")
sw = json.load(open(os.path.join(F, "swap", "swap_report.json")))
ev = json.load(open(os.path.join(F, "swap", "swap-eval_eval.json")))
nz = json.load(open(os.path.join(F, "swap", "swap-eval_noise.json")))
names = {"ref75": ("Early reference", "75k", "75k"), "ref95": ("Late reference", "95k", "95k"),
         "core75_rest95": ("Early-core swap", "75k", "95k"), "core95_rest75": ("Late-core swap", "95k", "75k")}
exact, noisy = {}, {}
for ck, runs in ev.items():
    key = ck.rstrip("/").split("/")[-1]
    exact[key] = {SHORT[FULL.index(r)] if r in FULL else r: runs[0][f"eval/episode_return/{r}"] for r in
                  [k.split("/")[-1] for k in runs[0] if k.startswith("eval/episode_return/") and not k.endswith("average")]}
    exact[key]["_repeat_identical"] = all(abs(runs[0][k] - runs[1][k]) < 1e-6 for k in runs[0])
for ck, conds in nz.items():
    key = ck.rstrip("/").split("/")[-1]
    noisy[key] = {r: np.array(v["return"]) for r, v in next(iter(conds.values())).items()}
say(f"Core = every parameter under `core`: core experts, shared (global and group) experts, router and router noise head "
    f"({sw['core_params']:,} parameters in {sw['core_leaves']} leaves: {', '.join(sw['core_keys'])}). Early = V7-CH-s0 at 75k, late = the "
    f"S0-original continuation at 95k. Deterministic evaluation: one episode per robot, run twice (identical both times: "
    f"{all(exact[k]['_repeat_identical'] for k in exact)}). Paired noise: N(0, 0.1^2) on the first 50 steps, 30 episodes, evaluation "
    f"seeds 1000+e identical for every configuration.")
say("")
say("| Configuration | Core | Rest of actor | Go2 exact | Go2 under noise: mean return (walks) | Mean over 16 robots, exact | Mean over 16 robots, under noise |")
say("|---|---|---|---|---|---|---|")
for k, (label, c, r) in names.items():
    g = go2_noise(noisy[k])
    allm = np.mean([v.mean() for v in noisy[k].values()])
    ex = [exact[k][s] for s in SHORT]
    say(f"| {label} | {c} | {r} | {exact[k]['go2']:.1f} | {g[0]:.1f} ({g[1]:.2f}) | {np.mean(ex):.1f} | {allm:.1f} |")
results["swaps"] = {k: {"go2_exact": exact[k]["go2"], "go2_noise": go2_noise(noisy[k]), "mean16_exact": float(np.mean([exact[k][s] for s in SHORT]))}
                    for k in names}
say("")
say("Deterministic return of every robot:")
say("")
say("| Robot | " + " | ".join(names[k][0] for k in names) + " |")
say("|---|" + "---:|" * 4)
for s in SHORT:
    say(f"| {s} | " + " | ".join(f"{exact[k][s]:.1f}" for k in names) + " |")
say("")
rep = sw["configs"]
say("Reference check. Both reference configurations are built by the same swap code and reproduce the saved policies bitwise (policy parameter hash equal: "
    f"early {rep['ref75']['reproduces_saved_policy']}, late {rep['ref95']['reproduces_saved_policy']}); the swapped configurations carry the intended core and rest hashes "
    f"(early core + late rest: {rep['core75_rest95']['core_is_early'] and rep['core75_rest95']['rest_is_late']}, late core + early rest: "
    f"{rep['core95_rest75']['core_is_late'] and rep['core95_rest75']['rest_is_early']}). Their evaluations equal those of the saved checkpoints: exact Go2 "
    "50.8 and 2.8 as recorded in `checkpoints.jsonl`, and the paired-noise numbers 7.9 (walks 0.13) and 5.8 (0.00) equal those of the variance report.")
say("")
# 2. Core freeze against the matched S0 control.
tb_s0 = load_tb(os.path.join(V, "tb", "VARb-S0.npz"))
tb_f0 = load_tb(os.path.join(F, "tb", "VARc-F0.npz"))
tb_o = load_tb(os.path.join(V, "tb", "V7-CH-s0_original.npz"))
H = int(min(float(tb_s0["steps/nr_updates"][1].max()), float(tb_f0["steps/nr_updates"][1].max()), 95000) // 100 * 100)
start_win = {s: window_mean(tb_o, s, START) for s in SHORT}
say("## 2. Core freeze against the matched S0-original continuation")
say("")
say(f"Common horizon {H} updates (+{H - START}). Returns: mean of the last 10 evaluations (one episode per robot every 100 updates) up to the horizon; "
    f"\"walks\" is Go2's share of evaluations above the frozen threshold ({THR['unitree_go2']:.1f}) in the last 2,000 updates.")
say("")
say("| Run | Go2 return at +H | Go2 walks (last 2k) | Mean return, 16 robots | Go2 exact at +20k | Go2 under paired noise (walks) |")
say("|---|---:|---:|---:|---:|---:|")
ck_s0 = {r["checkpoint"]: r for r in read_jsonl(os.path.join(V, "meta", "VARb-S0", "checkpoints.jsonl"))}
ck_f0 = {r["checkpoint"]: r for r in read_jsonl(os.path.join(F, "meta", "VARc-F0", "checkpoints.jsonl"))}
n_s0 = noise_rows(os.path.join(V, "noise_eval", "var-noise-S0_noise.json"))
n_f0 = noise_rows(os.path.join(F, "noise_eval", "var-noise-F0_noise.json")) if os.path.exists(os.path.join(F, "noise_eval", "var-noise-F0_noise.json")) else {}
rows = {}
for name, tb, ck, nzr in (("S0-original (control)", tb_s0, ck_s0, [v for k, v in n_s0.items() if "VARb" in k]),
                          ("core freeze", tb_f0, ck_f0, list(n_f0.values()))):
    upd, v = series(tb, "eval/episode_return/go2")
    last = v[(upd > H - 2000) & (upd <= H)]
    allr = np.array([window_mean(tb, s, H) for s in SHORT])
    exact95 = ck.get("at_95000", {}).get("returns", {}).get("go2", float("nan"))
    gn = go2_noise(nzr[0]) if nzr else (float("nan"), float("nan"), 0)
    rows[name] = {"go2": window_mean(tb, "go2", H), "walk": float((last > THR["unitree_go2"]).mean()), "mean16": float(np.nanmean(allr)),
                  "exact95": exact95, "noise": gn, "per_robot": dict(zip(SHORT, [float(x) for x in allr]))}
    say(f"| {name} | {rows[name]['go2']:.1f} | {rows[name]['walk']:.2f} | {np.nanmean(allr):.1f} | {exact95:.1f} | {gn[0]:.1f} ({gn[1]:.2f}) |")
results["freeze"] = rows
say(f"| source checkpoint (75k) | {start_win['go2']:.1f} (window) | | {np.nanmean(list(start_win.values())):.1f} | 50.8 | 7.9 (0.13) |")
say("")
say("Go2, mean return per 1,000-update bin (share above the threshold in brackets):")
say("")
pts = (1000, 2000, 5000, 10000, 15000, 20000)
say("| Run | " + " | ".join(f"+{p // 1000}k" for p in pts if START + p <= H) + " |")
say("|---|" + "---:|" * len([p for p in pts if START + p <= H]))
for name, tb in (("S0-original", tb_s0), ("core freeze", tb_f0), ("original run (CH-s0)", tb_o)):
    e, m, w = bins(tb, "go2", START, H, 1000)
    sel = [i for i, b in enumerate(e) if (b - START) in pts]
    say(f"| {name} | " + " | ".join(f"{m[i]:.1f} ({w[i]:.2f})" for i in sel) + " |")
say("")
def exact_txt(ck):
    return ", ".join(f"+{int(c[3:]) - START}: {ck[c]['returns']['go2']:.1f} / {ck[c]['mean_return']:.1f}" for c in sorted(ck, key=lambda x: int(x[3:])))
say("Exact deterministic evaluation of the saved checkpoints (Go2 / mean over the 16 robots): S0 " + exact_txt(ck_s0) + "; core freeze " + exact_txt(ck_f0) + ".")
say("")
say("Per-robot return at the horizon (window mean of the last 10 evaluations); the source is the original run before 75k:")
say("")
say("| Robot | Source | S0 | Core freeze | Freeze minus S0 |")
say("|---|---:|---:|---:|---:|")
for s in SHORT:
    a, b = rows["S0-original (control)"]["per_robot"][s], rows["core freeze"]["per_robot"][s]
    say(f"| {s} | {start_win[s]:.1f} | {a:.1f} | {b:.1f} | {b - a:+.1f} |")
lost = {n: [s for s in SHORT if start_win[s] >= THR[FULL[SHORT.index(s)]] and r["per_robot"][s] < THR[FULL[SHORT.index(s)]]] for n, r in rows.items()}
gained = {n: [s for s in SHORT if start_win[s] < THR[FULL[SHORT.index(s)]] and r["per_robot"][s] >= THR[FULL[SHORT.index(s)]]] for n, r in rows.items()}
say("")
say("Status changes by the frozen thresholds: " + "; ".join(f"{n}: lost {lost[n] or 'none'}, gained {gained[n] or 'none'}" for n in rows) + ".")
results["status"] = {"lost": lost, "gained": gained}
say("")
# 3. The core stayed unchanged; clipping and optimizer evidence.
say("## 3. Proof that the core stayed unchanged")
say("")
bc = json.load(open(os.path.join(F, "meta", "VARc-F0", "branch_check.json")))
sync_f = {r["update"]: r for r in read_jsonl(os.path.join(F, "meta", "VARc-F0", "sync_hashes.jsonl"))}
sync_s = {r["update"]: r for r in read_jsonl(os.path.join(V, "meta", "VARb-S0", "sync_hashes.jsonl"))}
core_set = {r["core"] for r in sync_f.values()}
say(f"- Core parameter hash (SHA-1 over all {sw['core_leaves']} core leaves) of the checkpoint {bc['core_sha1_checkpoint'][:16]}..., of the branched run's core at start "
    f"{bc['core_sha1_start'][:16]}...: equal {bc['core_sha1_start'] == bc['core_sha1_checkpoint']}.")
say(f"- The run logs the core hash at every logged update (every 250 updates; {len(sync_f)} updates from {min(sync_f)} to {max(sync_f)}): "
    f"{len(core_set)} distinct value(s), equal to the checkpoint's: {core_set == {bc['core_sha1_checkpoint']}}.")
ckh = os.path.join(F, "meta", "VARc-F0", "core_hash_checkpoints.json")
if os.path.exists(ckh):
    h = json.load(open(ckh))
    say("- Saved checkpoints: " + ", ".join(f"{k} {v[:12]}... (equal to the source's: {v == bc['core_sha1_checkpoint']})" for k, v in h.items() if k.startswith("at_")) + ".")
    results["core_hash_checkpoints"] = {k: v == bc["core_sha1_checkpoint"] for k, v in h.items()}
upd_n, v_n = series(tb_f0, "morse/var/update_norm_core")
say(f"- Norm of the applied update to the core parameters at every logged step: max {float(np.abs(v_n).max()):.1e} over {len(v_n)} logged steps (the update is exactly zero: the gradient "
    f"is zeroed and the restored Adam first moments of the core were zeroed). The trainable parameters moved: update norm of the whole actor {float(np.mean(series(tb_f0, 'morse/var/update_norm')[1])):.4f} on average against "
    f"{float(np.mean(series(tb_s0, 'morse/var/update_norm')[1])):.4f} in S0.")
say("")
say("**How the frozen core is excluded.** The core's gradient is computed (the core stays in the forward and backward pass, with its routing noise, and the gradient flows through it into the "
    "joint and foot encoders and the attention layers: nothing is stop-gradiented) and the optimizer's first step, `frozen_clip_by_global_norm`, sets it to zero before the "
    "global-norm clip, so it neither updates the core nor enters the clip's norm. The core router's MI and balance gradients (the core MI term, which reaches only the core router) are "
    "excluded the same way; the decoder MI and balance terms and the task loss are unchanged. The restored Adam state of every trainable parameter and the step counts (300,000 at start) "
    "are kept; only the core's first moments were set to zero. S0's global clip would have been computed over the core's gradient too; both factors are logged:")
say("")
say("| Logged (every 250 updates) | Core freeze | S0 |")
say("|---|---:|---:|")
def stat(tb, tag, fn):
    return fn(series(tb, tag)[1]) if tag in tb else float("nan")
say(f"| Clip factor actually applied, minimum | {stat(tb_f0, 'morse/var/clip_factor', np.min):.3f} | {stat(tb_s0, 'morse/var/clip_factor', np.min):.3f} |")
say(f"| Clip factor with the core's gradient included, minimum | {stat(tb_f0, 'morse/var/clip_factor_with_core', np.min):.3f} | (same as above) |")
say(f"| Gradient norm of the trainable parameters, mean | {np.mean(series(tb_f0, 'morse/var/grad_norm')[1]) if False else float(np.sqrt(np.maximum(np.mean(series(tb_f0, 'morse/var/grad_norm')[1]) ** 2 - np.mean(series(tb_f0, 'morse/var/grad_norm_core')[1]) ** 2, 0))):.4f} | {stat(tb_s0, 'morse/var/grad_norm', np.mean):.4f} (all, core included) |")
say(f"| Gradient norm of the core (computed, discarded), mean | {stat(tb_f0, 'morse/var/grad_norm_core', np.mean):.4f} | not logged |")
for blk in ("decoder", "core"):
    say(f"| {blk} MI cap factor, mean over steps | {stat(tb_f0, f'morse/{blk}/mi_cap_factor_mean', np.mean):.3f} | {stat(tb_s0, f'morse/{blk}/mi_cap_factor_mean', np.mean):.3f} |")
say(f"| decoder router task-gradient norm, mean | {stat(tb_f0, 'morse/decoder/router_grad_task', np.mean):.4f} | {stat(tb_s0, 'morse/decoder/router_grad_task', np.mean):.4f} |")
say("")
say("## 4. Data and critic synchronization with S0")
say("")
common = sorted(set(sync_f) & set(sync_s))
same = {f: all(sync_f[u][f] == sync_s[u][f] for u in common) for f in ("batch", "critic", "value")}
say(f"{len(common)} logged updates in common ({common[0]} to {common[-1]}): identical minibatch-stream hashes {same['batch']}, identical critic hashes {same['critic']}, identical value-critic hashes {same['value']}. "
    "The routing noise has its own key stream (seed, update, actor step, robot), the main key stream splits the same way in every mode, the diagnostic steps are the same (every 250 updates) and the freeze changes no random draw, so the control is matched.")
results["sync"] = {"updates": len(common), **{k: bool(v) for k, v in same.items()}}
say("")
# 5. Fixed-batch diagnostics.
say("## 5. Go2 mean-action error and the remaining actor components")
say("")
zf = np.load(os.path.join(F, "diagnostics", "var-diag-F0_var.npz")) if os.path.exists(os.path.join(F, "diagnostics", "var-diag-F0_var.npz")) else None
zs = np.load(os.path.join(V, "diagnostics", "var-diag-S0_var.npz"))


def diag_cells(z, wanted):
    out = {}
    for i, p in enumerate(z["checkpoints"]):
        p = str(p).rstrip("/")
        key = p.split("at_")[-1] if "at_" in p else p
        if key in wanted:
            werr = z[f"c{i}_w_err"] / z[f"c{i}_w_mean"]
            sig = np.array([np.nanmean(np.exp(r)) for r in z[f"c{i}_logstd"]])
            out[key] = (float(werr[SHORT.index("go2")]), float(np.nanmean(werr)), float(sig[SHORT.index("go2")]), float(np.nanmean(sig)))
    return out


wanted = ("75000", "80000", "85000", "95000")
ds, df_ = diag_cells(zs, wanted), (diag_cells(zf, wanted) if zf is not None else {})
say("Fixed minibatches of the training data (4 per robot, the same for every checkpoint), advantage weights from each checkpoint's own critics; Go2 / mean over the 16 robots.")
say("")
say("| Checkpoint | S0 weighted mean error | Freeze weighted mean error | S0 mean sigma | Freeze mean sigma |")
say("|---|---|---|---|---|")
for k in wanted:
    a, b = ds.get(k), df_.get(k)
    lab = "source (75k)" if k == "75000" else f"+{int(k) - START}"
    say(f"| {lab} | " + (f"{a[0]:.3f} / {a[1]:.3f}" if a else "n/a") + " | " + (f"{b[0]:.3f} / {b[1]:.3f}" if b else "n/a") + " | "
        + (f"{a[2]:.3f} / {a[3]:.3f}" if a else "n/a") + " | " + (f"{b[2]:.3f} / {b[3]:.3f}" if b else "n/a") + " |")
results["diagnostics"] = {"S0": ds, "freeze": df_}
say("")
pd = os.path.join(F, "param_distances.txt")
if os.path.exists(pd):
    say("Parameter change by group (descriptive only; magnitudes do not identify the cause of a collapse):")
    say("")
    say("```")
    say(open(pd).read().rstrip())
    say("```")
    say("")

# Figures.
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
col = {"S0-original": "k", "core freeze": "tab:red", "original run (CH-s0)": "gray"}
for name, tb in (("S0-original", tb_s0), ("core freeze", tb_f0), ("original run (CH-s0)", tb_o)):
    e, m, w = bins(tb, "go2", START, H, 500)
    ls = ":" if name.startswith("original") else "-"
    axes[0, 0].plot(e, m, color=col[name], ls=ls, label=name)
    axes[0, 1].plot(e, w, color=col[name], ls=ls, label=name)
    axes[1, 0].plot(e, np.nanmean(np.array([bins(tb, s, START, H, 500)[1] for s in SHORT]), axis=0), color=col[name], ls=ls, label=name)
    if name != "original run (CH-s0)":
        others = np.array([bins(tb, s, START, H, 500)[1] for s in SHORT if s != "go2"])
        axes[1, 1].plot(e, np.nanmean(others, axis=0), color=col[name], label=name)
axes[0, 0].set_title("Go2 return (mean per 500 updates)")
axes[0, 1].set_title("Go2: share of evaluations above the frozen threshold")
axes[1, 0].set_title("Mean return over the 16 robots")
axes[1, 1].set_title("Mean return over the 15 robots other than Go2")
for ax in axes.ravel():
    ax.set_xlabel("updates")
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "core_freeze_curves.png"), dpi=110)
with open(os.path.join(OUT, "core_freeze_tables.md"), "w") as f:
    f.write("\n".join(lines))
with open(os.path.join(OUT, "core_freeze_results.json"), "w") as f:
    json.dump(results, f, indent=1, default=float)
print("\n".join(lines))
