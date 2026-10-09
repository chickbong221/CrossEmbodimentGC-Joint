"""Checkpoint analysis: how every robot's policy changes between two checkpoints of one run (no training).

Both checkpoints (A earlier, B later) are evaluated on the same fixed data: ``n_batches`` training-size minibatches
of every robot, drawn once with a fixed seed. For every robot, and per joint where an array has a joint axis, it
writes (``morse_ckpt_out``, .npz):

1. Prediction losses. ``nll_A``/``nll_B``: negative log-likelihood of the dataset actions under each policy.
   ``awr_<policy>_w<critics>``: the IQL (AWR) policy loss with the advantage weights of either checkpoint's critics,
   so awr_A_wA -> awr_B_wA is the change of the policy alone with the weights frozen at A, and awr_B_wB the loss the
   later policy minimizes. ``w_corr``: correlation of A's and B's advantage weights over the same transitions.
2. Policy-output and routing changes. ``out_change``: |mean_B - mean_A| of the action mean, ``out_change_std`` the
   same in units of A's policy standard deviation, ``logstd_change``: |logstd_B - logstd_A|. ``dec_tv``: total
   variation between A's and B's decoder routing weights of a joint, ``dec_switch``: fraction of states whose top
   decoder expert changes, ``core_tv``: the same total variation for the core (one routing per transition).
   ``dec_p_A``/``dec_p_B``: mean decoder routing weights per joint, ``core_p_A``/``core_p_B`` per robot.
3. Where the decoder's output change comes from. The decoder's output (before the clip) is
       m(F, R, E) = sum_k p_k(router R; morphology, state_F) h_k(experts E; x_F),
   with upstream features F (the decoder input x and the router's state input, both computed by the encoders and
   the core), router parameters R, and expert parameters E, each taken from A or B. The eight combinations give
   the Shapley contribution phi_i of each factor (phi_features + phi_router + phi_experts = m(B,B,B) - m(A,A,A) for
   every input); ``share_<i>`` = E[phi_i dm] / E[dm^2] (the shares sum to 1) and ``abs_<i>`` = E|phi_i|.
   ``expert_drift``: change of the experts a joint used at A on A's own inputs, sum_k p_k^A |h_k^B(x_A) - h_k^A(x_A)|
   (learned expert computation overwritten under fixed inputs); ``feature_drift``: the same experts (A's) on B's
   inputs instead of A's, sum_k p_k^A |h_k^A(x_B) - h_k^A(x_A)| (change from the shared features alone).
   ``recon_err``: largest |clip(m(A,A,A)) - mean_A|, a check of the decoder recomputation (0 without calibration).

``evaluate_noise`` (``morse_ckpt_eval_noise``) repeats that evaluation with paired action noise during the first steps.

``gradient_split`` (``morse_ckpt_grads``) locates the conflict between robots: for each checkpoint, every robot's
gradient of the IQL policy loss on fixed minibatches, the robots' Gram matrix of these gradients per parameter group
(the state encoders, the attention MLPs, the core and decoder experts, the decoder's descriptor embedding, the action
std head, the routers), and every robot's mean routing weights and top-k selection rates.

``variance_diagnostics`` (``morse_ckpt_var``) takes any number of checkpoints and measures how the policy's learned
standard deviation relates to its mean fit and its gradients (see the method).

``evaluate`` (``morse_ckpt_eval``) evaluates both checkpoints exactly as during training. Note that in runs started
before 2026-10-04 a "latest" checkpoint labelled N updates holds the policy after N + 1 updates (it was saved before
the update counter was incremented), while the evaluation logged at N updates is that of the policy after N updates.
Since then a "latest" checkpoint labelled N holds the policy after N updates, and ``models/checkpoints.jsonl`` records
every saved checkpoint with its update count and the evaluation of exactly those weights. ``models/at_N`` checkpoints
(``morse_save_at``) always hold the policy after N updates.
"""
import json
import os

import numpy as np
import jax
import jax.numpy as jnp
from flax.training import checkpoints

from cross_embodiment_offline_rl.algorithms.morse.moe import CAPTURE_DELTA, CAPTURE_IN, CAPTURE_ROUTER, ROUTING, route, symlog

FACTORS = ("features", "router", "experts")
DECODER_POSTS = ("ln_elu", "none")     # the decoder's layers (morse/policy.py): width-128 LN-ELU layer, linear output
# Policy parameters outside the expert blocks, by role (MorsePolicy creates them in this order).
SHARED_GROUPS = {"joint_attention": ("Dense_0", "LayerNorm_0", "Dense_1", "joint_log_softmax_temperature"),
                 "foot_attention": ("Dense_2", "LayerNorm_1", "Dense_3", "foot_log_softmax_temperature"),
                 "desc_embedding": ("Dense_4", "LayerNorm_2", "Dense_5"),
                 "std_head": ("Dense_6", "logstd_offset")}
BLOCK_GROUPS = {"joint_encoder": "joint_state_encoder", "foot_encoder": "foot_state_encoder", "core": "core_experts",
                "decoder": "decoder_experts"}


def param_group(path):
    """Role of a policy parameter from its path in params['params']: an expert block's experts, a router (with its
    noise head), one of SHARED_GROUPS, or other."""
    if path[0] in BLOCK_GROUPS:
        return "routers" if len(path) > 1 and path[1] in ("router", "router_noise") else BLOCK_GROUPS[path[0]]
    return next((g for g, names in SHARED_GROUPS.items() if path[0] in names), "other")


def router_weights(rp, morph, state, inputs="both", topks=()):
    """Routing weights of a MoE block's router (moe.Router, then moe.route with the block's top-k restrictions)."""
    logits = 0.0
    if inputs in ("both", "morph"):
        h = jax.nn.elu(morph @ rp["morph_hidden"]["kernel"] + rp["morph_hidden"]["bias"])
        logits = logits + h @ rp["morph_out"]["kernel"] + rp["morph_out"]["bias"]
    if inputs in ("both", "state"):
        h = jax.nn.elu(symlog(state) @ rp["state_hidden"]["kernel"] + rp["state_hidden"]["bias"])
        logits = logits + h @ rp["state_out"]["kernel"] + rp["state_out"]["bias"]
    return route(logits, *topks)[1]


def expert_outputs(ep, x, posts=DECODER_POSTS, epsilon=1e-6):
    """Output of every expert of a MoE block, (..., K, out), for an input without an expert axis (moe.MoEBlock)."""
    h = x
    for i, post in enumerate(posts):
        d = ep[f"dense_{i}"]
        kernel = d["kernel"]
        if "wn_scale" in d:                    # WeightNorm (moe.ExpertDense): w = g v / ||v|| per output unit
            kernel = kernel * (d["wn_scale"] / jnp.sqrt(jnp.sum(jnp.square(kernel), axis=-2) + 1e-12))[:, None, :]
        h = jnp.einsum("...i,kio->...ko" if i == 0 else "...ki,kio->...ko", h, kernel) + d["bias"]
        if post == "ln_elu":
            ln = ep[f"ln_{i}"]
            mu = h.mean(axis=-1, keepdims=True)
            var = jnp.square(h - mu).mean(axis=-1, keepdims=True)
            h = jax.nn.elu((h - mu) * jax.lax.rsqrt(var + epsilon) * ln["scale"] + ln["bias"])
        elif post == "elu":
            h = jax.nn.elu(h)
    return h


def shapley3(v):
    """Shapley contributions of three factors; ``v[(a, b, c)]`` is the value with factor i at B where entry i is 1."""
    out = []
    for i in range(3):
        j, k = [n for n in range(3) if n != i]
        def at(*on):
            return v[tuple(int(n in on) for n in range(3))]
        out.append((at(i) - at()) / 3 + (at(i, j) - at(j)) / 6 + (at(i, k) - at(k)) / 6 + (at(i, j, k) - at(j, k)) / 3)
    return out


def load_checkpoint(path):
    """Policy variables and (value critic, critic 1, critic 2) parameters of a checkpoint folder (latest in it)."""
    raw = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath(path), target=None)
    if raw is None:
        raise FileNotFoundError(f"no checkpoint in {path}")
    to_jnp = lambda t: jax.tree_util.tree_map(jnp.asarray, t)
    return to_jnp(raw["policy"]["params"]), tuple(to_jnp(raw[k]["params"]) for k in ("value_critic", "critic_1", "critic_2"))


class CheckpointAnalysis:
    def __init__(self, capture_policy, split_fn, awr_terms, robot_names, batch_sizes, a, seed, loss_fn=None):
        self.paths = [p.strip() for p in str(a.morse_ckpt_analysis).split(",") if p.strip()]
        if not self.paths:
            raise ValueError("morse_ckpt_analysis needs checkpoint folders: EARLIER,LATER (variance diagnostics take any number)")
        self.out = a.morse_ckpt_out or os.path.join(os.path.abspath(self.paths[1]), "ckpt_analysis.npz")
        self.n_batches = int(a.morse_ckpt_batches)
        self.n_grads = int(getattr(a, "morse_ckpt_grads", 0))
        self.n_var = int(getattr(a, "morse_ckpt_var", 0))
        self.var_noise = bool(getattr(a, "morse_ckpt_var_noise", False))
        self.loss_fn = loss_fn
        self.dump = {n.strip() for n in str(getattr(a, "morse_ckpt_dump", "")).split(",") if n.strip()}
        everything = 10 ** 9
        self.cap = capture_policy.clone(n_capture_element=everything, n_capture_body=everything,
                                        n_capture_decoder=everything)
        self.topks = (capture_policy.topk_decoder, capture_policy.eval_topk)
        self.router_inputs = capture_policy.router_inputs
        self.calibration = bool(capture_policy.calibration)
        self.clip = float(capture_policy.policy_mean_abs_clip)
        self.split_fn, self.awr_terms = split_fn, awr_terms
        self.robot_names, self.batch_sizes = list(robot_names), list(batch_sizes)
        self.seed = int(seed)
        self._fns = {}

    def _robot_fn(self, r, obs):
        """Jitted per-robot statistics (sums over the transitions of one minibatch)."""
        return self._build(r, obs)[0]

    def _build(self, r, obs):
        if r in self._fns:
            return self._fns[r]
        cap, split_fn = self.cap, self.split_fn
        template = jax.eval_shape(lambda o: cap.init(jax.random.PRNGKey(0), *split_fn(o, r)),
                                  jax.ShapeDtypeStruct(obs.shape, obs.dtype))[CAPTURE_DELTA]

        def forward(variables, obs_b):
            zeros = jax.tree_util.tree_map(lambda s: jnp.zeros(s.shape, s.dtype), template)
            (mean, logstd), st = cap.apply({"params": variables["params"], CAPTURE_DELTA: zeros}, *split_fn(obs_b, r),
                                           robot_id=r, mutable=[CAPTURE_IN, CAPTURE_ROUTER])
            rt = st[CAPTURE_ROUTER]["decoder"]
            return dict(mean=mean, logstd=logstd, x=st[CAPTURE_IN]["decoder"]["dense_0"]["x"][0],
                        morph=rt["morph"][0], state=rt["state"][0], p=rt["p"][0], core_p=st[CAPTURE_ROUTER]["core"]["p"][0])

        def per_state(var_a, var_b, extra_a, extra_b, obs_b, act_b):
            fa, fb = forward(var_a, obs_b), forward(var_b, obs_b)
            w_a, lp_a = self.awr_terms((fa["mean"], fa["logstd"]), obs_b, act_b, extra_a, r)
            w_b, lp_b = self.awr_terms((fb["mean"], fb["logstd"]), obs_b, act_b, extra_b, r)
            # Decoder output m(F, R, E) for the eight combinations of A and B.
            dec = (var_a["params"]["decoder"], var_b["params"]["decoder"])
            feats = (fa, fb)
            experts = [{k: v for k, v in d.items() if k != "router"} for d in dec]
            h = {(f, e): expert_outputs(experts[e], feats[f]["x"])[..., 0] for f in (0, 1) for e in (0, 1)}  # (B, J, K)
            routed = "router" in dec[0]
            p = {(f, rr): (router_weights(dec[rr]["router"], feats[f]["morph"], feats[f]["state"], self.router_inputs,
                                          self.topks) if routed else feats[0]["p"]) for f in (0, 1) for rr in (0, 1)}
            m = {(f, rr, e): (p[(f, rr)] * h[(f, e)]).sum(axis=-1) for f in (0, 1) for rr in (0, 1) for e in (0, 1)}
            return dict(fa=fa, fb=fb, w_a=w_a[:, 0], w_b=w_b[:, 0], lp_a=lp_a, lp_b=lp_b, h=h, m=m,
                        dm=m[(1, 1, 1)] - m[(0, 0, 0)], phi=shapley3(m))

        def dump(var_a, var_b, extra_a, extra_b, obs_b, act_b):
            """Per-transition arrays of one robot (dataset action, both policies, routing, Shapley parts)."""
            ps = per_state(var_a, var_b, extra_a, extra_b, obs_b, act_b)
            fa, fb = ps["fa"], ps["fb"]
            return {"act": act_b[:, :fa["mean"].shape[1]], "mean_A": fa["mean"], "mean_B": fb["mean"],
                    "logstd_A": fa["logstd"], "logstd_B": fb["logstd"], "w_A": ps["w_a"], "w_B": ps["w_b"],
                    "dm": ps["dm"], "phi": jnp.stack(ps["phi"], axis=-1), "p_A": fa["p"], "p_B": fb["p"]}

        def stats(var_a, var_b, extra_a, extra_b, obs_b, act_b):
            ps = per_state(var_a, var_b, extra_a, extra_b, obs_b, act_b)
            fa, fb, w_a, w_b, lp_a, lp_b = (ps[k] for k in ("fa", "fb", "w_a", "w_b", "lp_a", "lp_b"))
            h, m, dm, phi = ps["h"], ps["m"], ps["dm"], ps["phi"]
            s = {"n": jnp.float32(obs_b.shape[0]),
                 "nll_A": -lp_a.mean(axis=0), "nll_B": -lp_b.mean(axis=0),
                 "awr_A_wA": -(w_a[:, None] * lp_a).mean(axis=0), "awr_A_wB": -(w_b[:, None] * lp_a).mean(axis=0),
                 "awr_B_wA": -(w_a[:, None] * lp_b).mean(axis=0), "awr_B_wB": -(w_b[:, None] * lp_b).mean(axis=0),
                 "w_A": w_a.mean(), "w_B": w_b.mean(), "w_AB": (w_a * w_b).mean(),
                 "w_AA": (w_a * w_a).mean(), "w_BB": (w_b * w_b).mean()}
            sd_a = jnp.exp(fa["logstd"])
            s["out_change"] = jnp.abs(fb["mean"] - fa["mean"]).mean(axis=0)
            s["out_change_std"] = (jnp.abs(fb["mean"] - fa["mean"]) / sd_a).mean(axis=0)
            s["logstd_change"] = jnp.abs(fb["logstd"] - fa["logstd"]).mean(axis=0)
            s["dec_tv"] = 0.5 * jnp.abs(fb["p"] - fa["p"]).sum(axis=-1).mean(axis=0)
            s["dec_switch"] = (jnp.argmax(fb["p"], -1) != jnp.argmax(fa["p"], -1)).astype(jnp.float32).mean(axis=0)
            s["core_tv"] = 0.5 * jnp.abs(fb["core_p"] - fa["core_p"]).sum(axis=-1).mean()
            s["dec_p_A"], s["dec_p_B"] = fa["p"].mean(axis=0), fb["p"].mean(axis=0)
            s["core_p_A"], s["core_p_B"] = fa["core_p"].mean(axis=0), fb["core_p"].mean(axis=0)
            s["dm_abs"], s["dm_sq"] = jnp.abs(dm).mean(axis=0), jnp.square(dm).mean(axis=0)
            for name, ph in zip(FACTORS, phi):
                s[f"proj_{name}"] = (ph * dm).mean(axis=0)
                s[f"abs_{name}"] = jnp.abs(ph).mean(axis=0)
            s["expert_drift"] = (fa["p"] * jnp.abs(h[(0, 1)] - h[(0, 0)])).sum(axis=-1).mean(axis=0)
            s["feature_drift"] = (fa["p"] * jnp.abs(h[(1, 0)] - h[(0, 0)])).sum(axis=-1).mean(axis=0)
            recon = jnp.clip(m[(0, 0, 0)], -self.clip, self.clip)
            s["recon_err"] = jnp.where(self.calibration, 0.0, jnp.abs(recon - fa["mean"]).max())
            return s

        self._fns[r] = (jax.jit(stats), jax.jit(dump))
        return self._fns[r]

    def _grad_fn(self, r, obs):
        """Jitted gradient of robot r's IQL policy loss on one minibatch (clean routing), with the robot's mean routing
        weights per block (every expert) and top-k selection rates per routed block (its router's experts)."""
        if ("grad", r) in self._fns:
            return self._fns[("grad", r)]
        cap, split_fn = self.cap, self.split_fn
        template = jax.eval_shape(lambda o: cap.init(jax.random.PRNGKey(0), *split_fn(o, r)),
                                  jax.ShapeDtypeStruct(obs.shape, obs.dtype))[CAPTURE_DELTA]

        def fn(params, extra, obs_b, act_b):
            zeros = jax.tree_util.tree_map(lambda s: jnp.zeros(s.shape, s.dtype), template)

            def loss(prm):
                (mean, logstd), st = cap.apply({"params": prm, CAPTURE_DELTA: zeros}, *split_fn(obs_b, r), robot_id=r,
                                               mutable=[CAPTURE_IN, CAPTURE_ROUTER, ROUTING])
                w, lp = self.awr_terms((mean, logstd), obs_b, act_b, extra, r)
                return -(w * lp).mean(), st

            (_, st), grads = jax.value_and_grad(loss, has_aux=True)(params)
            units = lambda p: p.reshape(-1, p.shape[-1])
            weights = {b: units(rec["p"][0]).mean(axis=0) for b, rec in st[CAPTURE_ROUTER].items()}
            select = {b: (units(rec["p"][0]) > 0).mean(axis=0) for b, rec in st.get(ROUTING, {}).items()}
            return grads, weights, select

        self._fns[("grad", r)] = jax.jit(fn)
        return self._fns[("grad", r)]

    def gradient_split(self, dataset):
        """For both checkpoints: per parameter group (param_group), the robots' Gram matrix of their policy-loss
        gradients averaged over the n_grads fixed minibatches (``<ckpt>_gram_<group>``) and the Gram of their mean
        gradients (``<ckpt>_gram_mean_<group>``); the group sizes; every robot's mean routing weights
        (``<ckpt>_p_<block>``, (robots, experts)) and top-k selection rates (``<ckpt>_sel_<block>``). Written to
        <morse_ckpt_out without .npz>_grads.npz."""
        batches = [dataset.sample(jax.random.PRNGKey(self.seed * 1_000_003 + 104_729 + i), self.batch_sizes)[0]
                   for i in range(self.n_grads)]
        out = {"robot_names": np.array(self.robot_names), "checkpoints": np.array(self.paths),
               "n_batches": np.array(self.n_grads)}
        for tag, path in zip(("A", "B"), self.paths):
            variables, extra = load_checkpoint(path)
            params = variables["params"]
            leaf_paths = [tuple(getattr(k, "key", str(k)) for k in p)
                          for p, _ in jax.tree_util.tree_leaves_with_path(params)]
            groups = [param_group(p) for p in leaf_paths]
            names = sorted(set(groups))
            for g in names:
                out[f"{tag}_params_{g}"] = np.array(sum(int(np.prod(l.shape)) for l, gg in
                                                        zip(jax.tree_util.tree_leaves(params), groups) if gg == g))
            gram = {g: 0.0 for g in names}
            mean_grad = {g: 0.0 for g in names}
            weights, select = {}, {}
            for batch in batches:
                vecs = {g: [] for g in names}
                for r, rb in enumerate(batch):
                    obs, act = jnp.asarray(rb.observations), jnp.asarray(rb.actions)
                    grads, w, s = self._grad_fn(r, obs)(params, extra, obs, act)
                    leaves = jax.tree_util.tree_leaves(grads)
                    for g in names:
                        vecs[g].append(jnp.concatenate([l.ravel() for l, gg in zip(leaves, groups) if gg == g]))
                    for k, v in w.items():
                        weights.setdefault(k, [np.zeros_like(np.asarray(v)) for _ in batch])[r] += np.asarray(v)
                    for k, v in s.items():
                        select.setdefault(k, [np.zeros_like(np.asarray(v)) for _ in batch])[r] += np.asarray(v)
                for g in names:
                    v = jnp.stack(vecs[g])
                    gram[g] = gram[g] + np.asarray(jnp.matmul(v, v.T, precision=jax.lax.Precision.HIGHEST))
                    mean_grad[g] = mean_grad[g] + v / self.n_grads
            for g in names:
                out[f"{tag}_gram_{g}"] = gram[g] / self.n_grads
                out[f"{tag}_gram_mean_{g}"] = np.asarray(jnp.matmul(mean_grad[g], mean_grad[g].T,
                                                                    precision=jax.lax.Precision.HIGHEST))
            for k, rows in weights.items():
                out[f"{tag}_p_{k}"] = np.stack(rows) / self.n_grads
            for k, rows in select.items():
                out[f"{tag}_sel_{k}"] = np.stack(rows) / self.n_grads
        path = os.path.splitext(self.out)[0] + "_grads.npz"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, **out)
        return path

    def _var_fn(self, r, obs, noisy):
        """Jitted variance diagnostics of robot r on one minibatch: the policy's outputs against the dataset actions,
        the advantage weights, and the loss gradient split by pathway (through the action mean, through the log-std),
        by exact output cotangents of the training loss (``loss_fn``, with the robot's loss weight)."""
        if ("var", r, noisy) in self._fns:
            return self._fns[("var", r, noisy)]
        plain, split_fn = self.cap.clone(capture=False), self.split_fn
        lo, hi = float(np.log(self.cap.policy_std_min_clip)), float(np.log(self.cap.policy_std_max_clip))

        def fn(params, extra, obs_b, act_b, rkey):
            def forward(prm):
                return plain.apply({"params": prm}, *split_fn(obs_b, r), robot_id=r, rngs={"routing": rkey} if noisy else None)

            (mean, logstd), vjp = jax.vjp(forward, params)
            w = self.awr_terms((mean, logstd), obs_b, act_b, extra, r)[0]                      # (B, 1)
            act = act_b[:, :mean.shape[1]]
            err2, var = jnp.square(act - mean), jnp.exp(2.0 * logstd)
            eff = w / var                                                                         # mean-learning weights
            q = jnp.asarray([0.5, 0.9, 0.99])
            s = {"logstd": logstd.mean(axis=0), "logstd_spread": logstd.max(axis=0) - logstd.min(axis=0),
                 "w_mean": w.mean(), "w_sq_mean": jnp.square(w).mean(), "w_clipped": (w >= 99.99).mean(),
                 "w_q": jnp.quantile(w[:, 0], q), "err": err2.mean(), "w_err": (w * err2.mean(axis=1, keepdims=True)).mean(),
                 "z2": (err2 / var).mean(), "w_z2": (w * err2 / var).mean(),
                 "eff_mean": eff.mean(), "eff_q": jnp.quantile(eff.ravel(), q), "eff_max": eff.max(),
                 "frac_lo": (logstd <= lo + 1e-6).mean(), "frac_hi": (logstd >= hi - 1e-6).mean(),
                 "mean_abs": jnp.abs(mean).mean()}
            g_mean, g_logstd = jax.grad(lambda m, sd: self.loss_fn((m, sd), obs_b, act_b, extra, None, r),
                                        argnums=(0, 1))(mean, logstd)
            g_by_mean = vjp((g_mean, jnp.zeros_like(logstd)))[0]
            g_by_std = vjp((jnp.zeros_like(mean), g_logstd))[0]
            return s, g_by_mean, g_by_std

        self._fns[("var", r, noisy)] = jax.jit(fn)
        return self._fns[("var", r, noisy)]

    def variance_diagnostics(self, dataset):
        """For every checkpoint folder listed (any number), on ``n_var`` fixed minibatches of every robot: the
        advantage-weighted mean-action squared error, the log-std per joint and its share at the clipping bounds, the
        standardized squared residual (a - mu)^2 / sigma^2, the mean-learning weights w / sigma^2 and the advantage
        weights w (mean, quantiles, share clipped, effective sample size), and the gradient norms of the training loss
        through the mean and through the log-std, per parameter group, for each robot and for the robot-averaged
        gradient the optimizer sees. Clean routing, and (``morse_ckpt_var_noise``) the same with the router's
        training noise drawn from the same keys for every checkpoint. Written to <morse_ckpt_out without .npz>_var.npz."""
        batches = [dataset.sample(jax.random.PRNGKey(self.seed * 1_000_003 + 31_337 + i), self.batch_sizes)[0]
                   for i in range(self.n_var)]
        out = {"robot_names": np.array(self.robot_names), "checkpoints": np.array(self.paths),
               "n_batches": np.array(self.n_var)}
        n_robots = len(self.robot_names)
        groups = ("total", "desc_embedding", "std_head", "decoder_experts", "core_experts", "joint_state_encoder",
                  "foot_state_encoder", "joint_attention", "foot_attention", "routers", "other")

        def group_norms(tree):
            sq = {}
            for p, leaf in jax.tree_util.tree_leaves_with_path(tree):
                g = param_group(tuple(getattr(k, "key", str(k)) for k in p))
                sq[g] = sq.get(g, 0.0) + jnp.sum(jnp.square(leaf))
            sq["total"] = sum(sq.values())
            return {g: jnp.sqrt(sq.get(g, 0.0)) for g in groups}

        group_norms = jax.jit(group_norms)
        add = jax.jit(lambda a, b: jax.tree_util.tree_map(jnp.add, a, b))
        flat = lambda t: jnp.concatenate([l.ravel() for l in jax.tree_util.tree_leaves(t)])
        cosine = lambda u, v: float(jnp.vdot(u, v) / (jnp.linalg.norm(u) * jnp.linalg.norm(v) + 1e-30))
        embedding = lambda t: {k: v for k, v in t.items() if k in SHARED_GROUPS["desc_embedding"]}
        for i, path in enumerate(self.paths):
            variables, extra = load_checkpoint(path)
            params = variables["params"]
            for noisy in ((False, True) if self.var_noise else (False,)):
                tag = f"c{i}{'n' if noisy else ''}"
                acc = [None] * n_robots
                avg = {name: {g: 0.0 for g in groups} for name in ("mean", "std")}
                cos_total = cos_embedding = 0.0
                for b, batch in enumerate(batches):
                    sums = {"mean": None, "std": None}
                    for r, rb in enumerate(batch):
                        obs, act = jnp.asarray(rb.observations), jnp.asarray(rb.actions)
                        rkey = jax.random.fold_in(jax.random.PRNGKey(self.seed * 1_000_003 + 555), b * 1000 + r)
                        s, g_mean, g_std = self._var_fn(r, obs, noisy)(params, extra, obs, act, rkey)
                        s = jax.device_get(s)
                        for name, g in (("mean", g_mean), ("std", g_std)):
                            s.update({f"gn_{name}_{k}": v for k, v in jax.device_get(group_norms(g)).items()})
                            sums[name] = g if sums[name] is None else add(sums[name], g)
                        acc[r] = s if acc[r] is None else {k: acc[r][k] + v for k, v in s.items()}
                    # The optimizer's task gradient is the mean over the robots' gradients (morse/updater.py).
                    for name in ("mean", "std"):
                        sums[name] = jax.tree_util.tree_map(lambda x: x / n_robots, sums[name])
                        for g, v in jax.device_get(group_norms(sums[name])).items():
                            avg[name][g] += float(v) / len(batches)
                    cos_total += cosine(flat(sums["mean"]), flat(sums["std"])) / len(batches)
                    cos_embedding += cosine(flat(embedding(sums["mean"])), flat(embedding(sums["std"]))) / len(batches)
                for k in acc[0]:
                    vals = [np.asarray(a[k]) / len(batches) for a in acc]
                    if vals[0].ndim == 0 or k in ("w_q", "eff_q"):
                        out[f"{tag}_{k}"] = np.array(vals)
                    else:                                             # per joint: padded to the widest robot
                        n = max(len(v) for v in vals)
                        out[f"{tag}_{k}"] = np.array([np.pad(v, (0, n - len(v)), constant_values=np.nan) for v in vals])
                for name in ("mean", "std"):
                    for g in groups:
                        out[f"{tag}_avg_gn_{name}_{g}"] = np.asarray(avg[name][g])
                out[f"{tag}_cos_mean_std_total"], out[f"{tag}_cos_mean_std_embedding"] = np.asarray(cos_total), np.asarray(cos_embedding)
        path = os.path.splitext(self.out)[0] + "_var.npz"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, **out)
        return path

    def evaluate(self, evaluate_fn, repeats=2):
        """Evaluation metrics of both checkpoints (``evaluate_fn(policy variables)``), each evaluated ``repeats`` times
        to check that the evaluation is deterministic; written to <morse_ckpt_out without .npz>_eval.json."""
        rows = {}
        for path in self.paths:
            variables = load_checkpoint(path)[0]
            rows[path] = [{k: float(v) for k, v in evaluate_fn(variables).items()} for _ in range(repeats)]
        out = os.path.splitext(self.out)[0] + "_eval.json"
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump(rows, f, indent=1)
        return out

    def evaluate_noise(self, evaluate_fn, conditions, episodes):
        """Perturbed evaluations of both checkpoints: for every condition "SIGMA@STEPS" and episode e, one episode per
        robot with action noise drawn from seed e (the same noise for both checkpoints, so outcomes are paired);
        SIGMA = 0 runs two episodes (determinism check). ``evaluate_fn(variables, sigma, steps, seed)`` returns
        {robot: (return, length)}. Written to <morse_ckpt_out without .npz>_noise.json."""
        conds = [(float(c.split("@")[0]), int(c.split("@")[1])) for c in str(conditions).split(",") if c.strip()]
        rows = {}
        for path in self.paths:
            variables = load_checkpoint(path)[0]
            rows[path] = {}
            for sigma, steps in conds:
                runs = [evaluate_fn(variables, sigma, steps, 1000 + e) for e in range(2 if sigma == 0 else episodes)]
                rows[path][f"{sigma:g}@{steps}"] = {r: {"return": [x[r][0] for x in runs], "length": [x[r][1] for x in runs]}
                                                     for r in runs[0]}
        out = os.path.splitext(self.out)[0] + "_noise.json"
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump(rows, f, indent=1)
        return out

    def run(self, dataset):
        var_a, extra_a = load_checkpoint(self.paths[0])
        var_b, extra_b = load_checkpoint(self.paths[1])
        acc = [None] * len(self.batch_sizes)
        dumps = {r: [] for r, name in enumerate(self.robot_names) if str(name) in self.dump}
        for i in range(self.n_batches):
            env_batches, _ = dataset.sample(jax.random.PRNGKey(self.seed * 1_000_003 + 7919 + i), self.batch_sizes)
            for r, b in enumerate(env_batches):
                obs, act = jnp.asarray(b.observations), jnp.asarray(b.actions)
                stats_fn, dump_fn = self._build(r, obs)
                s = jax.device_get(stats_fn(var_a, var_b, extra_a, extra_b, obs, act))
                acc[r] = s if acc[r] is None else {k: (np.maximum(acc[r][k], v) if k == "recon_err" else acc[r][k] + v)
                                                   for k, v in s.items()}
                if r in dumps:
                    dumps[r].append(jax.device_get(dump_fn(var_a, var_b, extra_a, extra_b, obs, act)))
        n_joint = max(len(np.atleast_1d(a["nll_A"])) for a in acc)
        out = {"robot_names": np.array(self.robot_names), "checkpoints": np.array(self.paths),
               "n_batches": np.array(self.n_batches), "factors": np.array(FACTORS),
               "n_joints": np.array([len(np.atleast_1d(a["nll_A"])) for a in acc])}
        for k in acc[0]:
            if k in ("n", "recon_err"):
                out[k] = np.array([a[k] for a in acc])
                continue
            vals = [np.asarray(a[k]) / self.n_batches for a in acc]
            if vals[0].ndim == 0 or k.startswith("core_p"):      # per robot (core routing: one weight per expert)
                out[k] = np.array(vals)
            else:
                shape = (len(vals), n_joint) + vals[0].shape[1:]
                arr = np.full(shape, np.nan)
                for r, v in enumerate(vals):
                    arr[r, :v.shape[0]] = v
                out[k] = arr
        for r, parts in dumps.items():
            for k in parts[0]:
                out[f"dump_{self.robot_names[r]}_{k}"] = np.concatenate([np.asarray(d[k]) for d in parts], axis=0)
        os.makedirs(os.path.dirname(os.path.abspath(self.out)), exist_ok=True)
        np.savez_compressed(self.out, **out)
        return self.out
