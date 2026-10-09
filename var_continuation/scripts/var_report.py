"""Report of the variance-continuation experiment: python var_report.py DATA_DIR OUT_DIR [PREFIX=VAR]
DATA_DIR holds tb/<run>.npz (tb_extract, all tags), meta/<run>/{branch_check.json,sync_hashes.jsonl,checkpoints.jsonl},
var/<run>_var.npz (variance diagnostics), and tb/CH-s0.npz (the original run). Writes OUT_DIR/report.md, results.json,
and curve figures."""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA, OUT = sys.argv[1], sys.argv[2]
PFX = sys.argv[3] if len(sys.argv) > 3 else "VAR"
THR_PATH = sys.argv[4] if len(sys.argv) > 4 else os.path.join(DATA, "learned_thresholds.json")
os.makedirs(OUT, exist_ok=True)
FULL = ["unitree_a1", "unitree_go1", "unitree_go2", "anymal_b", "anymal_c", "barkour_v0", "barkour_vb", "badger", "bittle",
        "unitree_h1", "unitree_g1", "talos", "robotis_op3", "nao_v5", "cassie", "hexapod"]
SHORT = ["a1", "go1", "go2", "anyb", "anyc", "bkv0", "bkvb", "bg", "bit", "h1", "g1", "tl", "op3", "nao5", "cas", "hx"]
THR = json.load(open(THR_PATH))
BRANCHES = [("S0", "S0-original"), ("S1", "S1-detach"), ("S2", "S2-private-std"), ("S3", "S3-protected-mean")]
START, TARGET = 75000, "go2"
lines, results = [], {}


def say(s=""):
    lines.append(s)


def load_tb(path):
    z = np.load(path)
    return {k[3:]: (z[k], z["v::" + k[3:]]) for k in z.files if k.startswith("s::")}


def eval_series(tb, short):
    """(update, return) of every evaluation of one robot (one episode per evaluation, every 100 updates)."""
    s, v = tb[f"eval/episode_return/{short}"]
    su, u = tb["steps/nr_updates"]
    o = np.argsort(su)
    return np.interp(s, su[o], u[o]), v


def window_mean(tb, short, upto, last=10):
    """Mean of the last ``last`` evaluations up to ``upto`` updates (the frozen-threshold protocol)."""
    upd, v = eval_series(tb, short)
    sel = v[upd <= upto + 1][-last:]
    return float(sel.mean()) if len(sel) else float("nan")


def bins(tb, short, lo, hi, width=1000):
    """Per bin of ``width`` updates in (lo, hi]: mean return and share of evaluations above the robot's threshold."""
    upd, v = eval_series(tb, short)
    edges = np.arange(lo, hi + 1, width)
    mean, walk = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (upd > a) & (upd <= b)
        mean.append(float(v[m].mean()) if m.sum() >= 5 else float("nan"))
        walk.append(float((v[m] > THR[FULL[SHORT.index(short)]]).mean()) if m.sum() >= 5 else float("nan"))
    return edges[1:], np.array(mean), np.array(walk)


def last_update(tb):
    return float(tb["steps/nr_updates"][1].max())


def eval_series_tag(tb, tag):
    """(update, value) of any logged scalar."""
    s, v = tb[tag]
    su, u = tb["steps/nr_updates"]
    o = np.argsort(su)
    return np.interp(s, su[o], u[o]), v


def read_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def read_jsonl(p):
    try:
        return [json.loads(l) for l in open(p)]
    except Exception:
        return []


tb = {}
for k, name in BRANCHES:
    p = os.path.join(DATA, "tb", f"{PFX}-{k}.npz")
    if os.path.exists(p):
        tb[k] = load_tb(p)
orig = load_tb(os.path.join(DATA, "tb", "CH-s0.npz")) if os.path.exists(os.path.join(DATA, "tb", "CH-s0.npz")) else None

# 1. Starting checkpoint and the original run's trajectory.
say("## 1. Starting checkpoint")
say("")
start_rec = next((r for r in read_jsonl(os.path.join(DATA, "meta", "CH-s0", "checkpoints.jsonl")) if r["checkpoint"] == "at_75000"), None)
say("- Run V7-CH-s0 (C-H: hierarchical shared core + MI), seed 0, checkpoint `models/at_75000` (75,000 updates), architecture of V7 "
    "(joint encoder 4 units, 3-layer core, no WeightNorm; passed as explicit legacy flags). Actor, value and both critics with "
    "target critics and optimizer state are in the checkpoint.")
if start_rec:
    ret = start_rec["returns"]
    say(f"- Exact deterministic evaluation of this checkpoint (one episode per robot): Go2 {ret['go2']:.1f}, "
        f"mean over 16 robots {start_rec['mean_return']:.1f}; Cassie {ret['cas']:.1f} and hexapod {ret['hx']:.1f} were already lost.")
if orig is not None:
    e, m, w = bins(orig, TARGET, START - 15000, 100000, 5000)
    say("- Original run, Go2 (evaluation every 100 updates): mean return per 5k bin to " + ", ".join(
        f"{int(a) // 1000}k: {b:.1f} (walks {c:.2f})" for a, b, c in zip(e, m, w)) + ".")
say("")

# 3/4. Completed updates, equivalence and synchronization checks.
say("## 2. Completed updates and starting checks")
say("")
say("| Branch | Last update | Mean diff (start, vs checkpoint) | Log-std diff | Restored leaves (max diff) | Optimizer count | Evaluation noise floor |")
say("|---|---:|---:|---:|---|---|---:|")
for k, name in BRANCHES:
    bc = read_json(os.path.join(DATA, "meta", f"{PFX}-{k}", "branch_check.json"))
    last = last_update(tb[k]) if k in tb else float("nan")
    if bc is None:
        say(f"| {name} | {last:.0f} | no branch check | | | | |")
        continue
    leaves = ", ".join(f"{n}:{v['max_abs_diff']:.0e}" for n, v in bc["leaves"].items())
    say(f"| {name} | {last:.0f} | {max(bc['mean_max_abs_diff']):.1e} | {max(bc['logstd_max_abs_diff']):.1e} | {leaves} | "
        f"{sorted(set(bc['optimizer_counts'].values()))} (expected {bc['expected_count']}) | {max(bc['repeat_evaluation_noise']):.1e} |")
bcs = {k: read_json(os.path.join(DATA, "meta", f"{PFX}-{k}", "branch_check.json")) for k, _ in BRANCHES}
if bcs.get("S0"):
    sp = bcs["S0"]["logstd_state_spread"]
    say("")
    say("Spread of the checkpoint's log-std over states, per robot (max over joints; a state-independent per-joint table would be "
        "off by this much): " + ", ".join(f"{s}: {x:.3f}" for s, x in zip(SHORT, sp)) + ". "
        f"Learning-rate factor at the restored count: {bcs['S0']['lr_at_count']:.2e}.")
for k in ("S2",):
    if bcs.get(k):
        say(f"New leaves of {k}: " + ", ".join(bcs[k]["leaves"]["policy"]["new_leaves"][:4]) + "; the head's Adam first moments were zeroed (frozen head).")
# Synchronization: minibatch and critic hashes at every logged update.
sync = {k: {r["update"]: r for r in read_jsonl(os.path.join(DATA, "meta", f"{PFX}-{k}", "sync_hashes.jsonl"))} for k, _ in BRANCHES}
common = sorted(set.intersection(*[set(v) for v in sync.values() if v])) if any(sync.values()) else []
if common:
    same = {f: all(len({sync[k][u][f] for k in sync if sync[k]}) == 1 for u in common) for f in ("batch", "critic", "value")}
    say("")
    say(f"Synchronization: {len(common)} updates with content hashes in all branches ({common[0]}..{common[-1]}); identical minibatch "
        f"hashes: {same['batch']}, identical critic hashes: {same['critic']}, identical value-critic hashes: {same['value']}.")
results["sync"] = {"updates": len(common)}
say("")
# 5. Comparison.
H = int(min([last_update(tb[k]) for k in tb] + [95000])) if tb else START
H = int(H // 100 * 100)
say("## 3. Comparison at the common completed horizon")
say("")
say(f"Common horizon: {H} updates (+{H - START}). Returns are means of the last 10 evaluations (one episode per robot every 100 "
    f"updates) up to the horizon; \"walks\" is the share of Go2's evaluations above the frozen threshold ({THR['unitree_go2']:.1f}) in "
    f"the last 2,000 updates before it. Collapse = Go2 learned (window mean >= frozen threshold) at the start and not at the horizon.")
say("")
var = {k: np.load(p) for k, _ in BRANCHES for p in [os.path.join(DATA, "var", f"{PFX.lower()}-diag-{k}_var.npz")] if os.path.exists(p)}


def var_summary(z, i):
    """Weighted mean error and std statistics of checkpoint i of a variance-diagnostics file: Go2 and the robot mean."""
    ws = z[f"c{i}_w_err"] / z[f"c{i}_w_mean"]
    ls = z[f"c{i}_logstd"]
    std = np.array([np.nanmean(np.exp(r)) for r in ls])
    return ws, std, z[f"c{i}_frac_lo"] + z[f"c{i}_frac_hi"]


start_win = {s: (window_mean(orig, s, START) if orig is not None else float("nan")) for s in SHORT}
say("| Branch | Go2 return at +H | Go2 walks (last 2k) | Mean return, 16 robots | Weighted mean error (Go2 / robot mean) | Std behavior (mean sigma, Go2 / robots) | Collapse observed |")
say("|---|---:|---:|---:|---|---|---|")
rows = {}
refs = [("original run (CH-s0, same updates)", orig)] if orig is not None else []
for name, t in [(n, tb[k]) for k, n in BRANCHES if k in tb] + refs:
    g = window_mean(t, TARGET, H)
    upd, v = eval_series(t, TARGET)
    last2k = v[(upd > H - 2000) & (upd <= H)]
    walk = float((last2k > THR["unitree_go2"]).mean()) if len(last2k) else float("nan")
    allr = np.array([window_mean(t, s, H) for s in SHORT])
    learned = g >= THR["unitree_go2"]
    collapse = "yes" if (start_win["go2"] >= THR["unitree_go2"] and not learned) else "no"
    key = next((k for k, n in BRANCHES if n == name), None)
    werr = std_txt = "n/a"
    if key in var and var[key] is not None:
        z = var[key]
        i = len([p for p in z["checkpoints"]]) - 1
        ws, std, fb = var_summary(z, i)
        werr = f"{ws[SHORT.index('go2')]:.4f} / {np.nanmean(ws):.4f}"
        std_txt = f"{std[SHORT.index('go2')]:.3f} / {np.nanmean(std):.3f} (at bounds {np.nanmean(fb):.3f})"
    rows[name] = {"go2": g, "walk_last2k": walk, "mean16": float(np.nanmean(allr)), "collapse": collapse,
                  "per_robot": dict(zip(SHORT, [float(x) for x in allr]))}
    say(f"| {name} | {g:.1f} | {walk:.2f} | {np.nanmean(allr):.1f} | {werr} | {std_txt} | {collapse} |")
results["comparison"] = rows
say("")
say(f"Go2's window mean at the start (original run, last 10 evaluations before {START}): {start_win['go2']:.1f}.")
say("")
say("Go2, mean return per bin after the start (walks in parentheses):")
say("")
say("| Branch | " + " | ".join(f"+{(b - START) // 1000}k" for b in range(START + 1000, H + 1, 1000) if (b - START) in (1000, 2000, 5000, 10000, 15000, 20000)) + " |")
say("|---|" + "---:|" * len([b for b in range(START + 1000, H + 1, 1000) if (b - START) in (1000, 2000, 5000, 10000, 15000, 20000)]))
for name, t in [(n, tb[k]) for k, n in BRANCHES if k in tb] + refs:
    e, m, w = bins(t, TARGET, START, H, 1000)
    sel = [i for i, b in enumerate(e) if (b - START) in (1000, 2000, 5000, 10000, 15000, 20000)]
    say(f"| {name} | " + " | ".join(f"{m[i]:.1f} ({w[i]:.2f})" for i in sel) + " |")
say("")
# Robots that improved, deteriorated, or collapsed (frozen thresholds).
say("Robots that change status by the horizon (learned = window mean >= frozen threshold; start = original run at 75k):")
say("")
for name, t in [(n, tb[k]) for k, n in BRANCHES if k in tb] + refs:
    lost = [s for s in SHORT if start_win[s] >= THR[FULL[SHORT.index(s)]] and window_mean(t, s, H) < THR[FULL[SHORT.index(s)]]]
    gained = [s for s in SHORT if start_win[s] < THR[FULL[SHORT.index(s)]] and window_mean(t, s, H) >= THR[FULL[SHORT.index(s)]]]
    moved = [f"{s} {window_mean(t, s, H) - start_win[s]:+.0f}" for s in SHORT if abs(window_mean(t, s, H) - start_win[s]) >= 10]
    say(f"- {name}: lost {lost or 'none'}, gained {gained or 'none'}; changes of 10 or more: {', '.join(moved) or 'none'}.")
say("")
# 6. Mean and variance behavior at the saved checkpoints (fixed minibatches; clean routing, and with the router's training noise).
say("## 4. Mean and variance diagnostics at the saved checkpoints")
say("")
say("Fixed minibatches of the training data (the checkpoint's own training set: no held-out episodes exist for it), the same for every "
    "checkpoint; advantage weights from each checkpoint's own critics. Per cell: Go2 / mean over 16 robots.")
say("")


def cp_name(path):
    return os.path.basename(os.path.dirname(str(path).rstrip("/"))) + "/" + os.path.basename(str(path).rstrip("/")) if "/models" not in str(path) else str(path).rstrip("/").split("/")[-1]


def cell(a):
    a = np.asarray(a, float)
    return f"{a[SHORT.index('go2')]:.3g} / {np.nanmean(a):.3g}"


results["variance"] = {}
for k, name in BRANCHES:
    if k not in var:
        continue
    z = var[k]
    say(f"**{name}**")
    say("")
    say("| Checkpoint | weighted mean err | mean err | sigma | at bounds | (a-mu)^2/sigma^2 | w/sigma^2 mean | w/sigma^2 p99 | w mean | w p99 | w clipped | ESS share |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, path in enumerate(z["checkpoints"]):
        for noisy in ("", "n"):
            if f"c{i}{noisy}_w_err" not in z.files:
                continue
            t = f"c{i}{noisy}_"
            wm = z[t + "w_mean"]
            sigma = np.array([np.nanmean(np.exp(r)) for r in z[t + "logstd"]])
            ess = wm ** 2 / z[t + "w_sq_mean"]
            label = os.path.basename(str(path).rstrip("/")).replace("at_", "+") if "at_" in str(path) else os.path.basename(str(path).rstrip("/"))
            if "at_" in str(path):
                label = f"at_{int(str(path).rstrip('/').split('at_')[-1])}"
            say(f"| {label}{' (training noise)' if noisy else ''} | {cell(z[t + 'w_err'] / wm)} | {cell(z[t + 'err'])} | {cell(sigma)} | "
                f"{cell(z[t + 'frac_lo'] + z[t + 'frac_hi'])} | {cell(z[t + 'z2'])} | {cell(z[t + 'eff_mean'])} | "
                f"{cell(z[t + 'eff_q'][:, 2])} | {cell(wm)} | {cell(z[t + 'w_q'][:, 2])} | {cell(z[t + 'w_clipped'])} | {cell(ess)} |")
    say("")
    say("Gradient norms of the robot-averaged loss gradient through the action mean and through the log-std, by parameter group, "
        "and the cosine between the two pathways' gradients:")
    say("")
    say("| Checkpoint | total (mean path) | total (std path) | embedding (mean path) | embedding (std path) | std head (std path) | cos total | cos embedding |")
    say("|---|---|---|---|---|---|---|---|")
    for i, path in enumerate(z["checkpoints"]):
        for noisy in ("", "n"):
            t = f"c{i}{noisy}_"
            if t + "avg_gn_mean_total" not in z.files:
                continue
            label = f"at_{int(str(path).rstrip('/').split('at_')[-1])}" if "at_" in str(path) else os.path.basename(str(path).rstrip("/"))
            say(f"| {label}{' (training noise)' if noisy else ''} | {float(z[t + 'avg_gn_mean_total']):.4f} | {float(z[t + 'avg_gn_std_total']):.4f} | "
                f"{float(z[t + 'avg_gn_mean_desc_embedding']):.4f} | {float(z[t + 'avg_gn_std_desc_embedding']):.4f} | "
                f"{float(z[t + 'avg_gn_std_std_head']):.4f} | {float(z[t + 'cos_mean_std_total']):.3f} | {float(z[t + 'cos_mean_std_embedding']):.3f} |")
    say("")

# 7. Training-time diagnostics (every 250 updates).
say("## 5. Training-time gradient diagnostics (logged every 250 updates; mean over the period)")
say("")
TAGS = [("morse/var/grad_norm", "grad norm"), ("morse/var/grad_norm_std", "std-param grad"), ("morse/var/grad_norm_embedding", "embedding grad"),
        ("morse/var/clip_factor", "clip factor (min)"), ("morse/var/update_norm", "update norm"), ("morse/var/update_norm_std", "std update"),
        ("morse/var/update_norm_embedding", "embedding update"), ("morse/core/router_grad_task", "core router task grad"),
        ("morse/decoder/router_grad_task", "decoder router task grad"), ("morse/core/mi_cap_factor_mean", "core MI cap factor"),
        ("morse/decoder/mi_cap_factor_mean", "decoder MI cap factor")]
periods = [(START, START + 5000), (START + 5000, START + 10000), (START + 10000, START + 20000)]
say("| Branch | " + " | ".join(f"{lab} {int((a - START) / 1000)}-{int((b - START) / 1000)}k" for lab in ("grad norm", "std-param grad", "embedding grad") for a, b in periods) + " | clip factor min | core router task grad | decoder router task grad | core MI cap | decoder MI cap |")
say("|---|" + "---:|" * 14)
for k, name in BRANCHES:
    if k not in tb:
        continue
    t = tb[k]
    def per(tag, a, b, fn=np.mean):
        if tag not in t:
            return float("nan")
        upd, v = eval_series_tag(t, tag)
        m = (upd > a) & (upd <= b)
        return float(fn(v[m])) if m.any() else float("nan")
    cells = [f"{per(tag, a, b):.4f}" for tag in ("morse/var/grad_norm", "morse/var/grad_norm_std", "morse/var/grad_norm_embedding") for a, b in periods]
    say(f"| {name} | " + " | ".join(cells) + f" | {per('morse/var/clip_factor', START, H, np.min):.3f} | {per('morse/core/router_grad_task', START, H):.4f} | "
        f"{per('morse/decoder/router_grad_task', START, H):.4f} | {per('morse/core/mi_cap_factor_mean', START, H):.3f} | {per('morse/decoder/mi_cap_factor_mean', START, H):.3f} |")
say("")
# Figures and files.
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
colors = {"S0": "k", "S1": "tab:blue", "S2": "tab:green", "S3": "tab:red"}
for k, name in BRANCHES:
    if k not in tb:
        continue
    e, m, w = bins(tb[k], TARGET, START, H, 500)
    axes[0, 0].plot(e, m, color=colors[k], label=name)
    axes[0, 1].plot(e, w, color=colors[k], label=name)
    means = np.nanmean(np.array([bins(tb[k], s, START, H, 500)[1] for s in SHORT]), axis=0)
    axes[1, 0].plot(e, means, color=colors[k], label=name)
    if "morse/var/grad_norm_std" in tb[k]:
        u, v = eval_series_tag(tb[k], "morse/var/grad_norm_std")
        axes[1, 1].plot(u, v, color=colors[k], label=name + " std-param grad")
        if "morse/var/grad_norm_embedding" in tb[k]:
            u, v = eval_series_tag(tb[k], "morse/var/grad_norm_embedding")
            axes[1, 1].plot(u, v, color=colors[k], ls="--", label=name + " embedding grad")
if orig is not None:
    e, m, w = bins(orig, TARGET, START, H, 500)
    axes[0, 0].plot(e, m, color="gray", ls=":", label="original run (CH-s0)")
    axes[0, 1].plot(e, w, color="gray", ls=":", label="original run (CH-s0)")
    axes[1, 0].plot(e, np.nanmean(np.array([bins(orig, s, START, H, 500)[1] for s in SHORT]), axis=0), color="gray", ls=":", label="original run (CH-s0)")
axes[0, 0].set_title("Go2 return (mean per 500 updates)")
axes[0, 1].set_title("Go2: share of evaluations above the frozen threshold")
axes[1, 0].set_title("Mean return over the 16 robots")
axes[1, 1].set_title("Gradient norms before clipping (logged every 250 updates)")
axes[1, 1].set_yscale("log")
for ax in axes.ravel():
    ax.set_xlabel("updates")
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "curves.png"), dpi=110)

# Std and weighted error along the saved checkpoints.
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for k, name in BRANCHES:
    if k not in var:
        continue
    z = var[k]
    ups, werr, sig, g2 = [], [], [], []
    for i, path in enumerate(z["checkpoints"]):
        if "at_" not in str(path) or f"c{i}_w_err" not in z.files:
            continue
        ups.append(int(str(path).rstrip("/").split("at_")[-1]))
        werr.append(np.nanmean(z[f"c{i}_w_err"] / z[f"c{i}_w_mean"]))
        sig.append(np.nanmean([np.nanmean(np.exp(r)) for r in z[f"c{i}_logstd"]]))
        g2.append(float(z[f"c{i}_avg_gn_std_std_head"]))
    o = np.argsort(ups)
    axes[0].plot(np.array(ups)[o], np.array(werr)[o], "o-", color=colors[k], label=name)
    axes[1].plot(np.array(ups)[o], np.array(sig)[o], "o-", color=colors[k], label=name)
    axes[2].plot(np.array(ups)[o], np.array(g2)[o], "o-", color=colors[k], label=name)
axes[0].set_title("advantage-weighted mean-action squared error")
axes[1].set_title("mean sigma over robots and joints")
axes[2].set_title("log-std-pathway gradient norm of the std head")
for ax in axes:
    ax.set_xlabel("updates")
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "variance.png"), dpi=110)

with open(os.path.join(OUT, "report_tables.md"), "w") as f:
    f.write("\n".join(lines))
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=1, default=float)
print("\n".join(lines))
