"""Reference-episode dump for the coordination tests: both checkpoints' action means on whole episodes (no training).

The offline dataset keeps every robot's transitions in time order: observation t+1 equals next observation t unless
the episode ended at t, so the episodes are the runs of transitions up to a done flag. For every robot this writes
``<morse_coord_out>/<robot>.npz`` with

1. The episode table of the whole dataset: ``ep_start`` (first sample), ``ep_len``, ``ep_return``, ``ep_terminated``
   (ended by the termination condition, a fall), ``ep_truncated`` (reached the horizon), ``ep_complete`` (False only
   for a last episode cut by the end of the data) and ``ep_reference``.
2. Every transition of the reference episodes, in order. Reference episodes are complete, reach the horizon without
   termination, and have a return of at least ``morse_coord_min_return`` times the best such return of the robot.
   ``index`` (dataset sample), ``episode`` (row of the episode table), ``t`` (step in the episode), ``act`` (dataset
   action), ``mean_A``/``mean_B`` (the action means of the checkpoints EARLIER and LATER of ``morse_ckpt_analysis``,
   i.e. the actions of their deterministic evaluation) and observation features for context definitions (float16,
   as observed): ``joint_pos``, ``joint_vel`` (per joint), ``contact``, ``air_time`` (per foot) and ``general``
   (``general_names``). ``logstd_A``/``logstd_B``: the policies' per-joint log-std averaged over these transitions
   (it depends on the joint descriptor only, not on the state).

With ``morse_coord_startup=N`` > 0 the transitions are instead the first N steps of every complete episode (reference
or not; ``reference`` marks the reference ones), and ``w_A``/``w_B`` add the IQL advantage weight of the dataset
action under each checkpoint's critics (the weight of that transition in the policy loss).
"""
import os

import numpy as np
import jax
import jax.numpy as jnp

GENERAL = ("trunk_x_velocity", "trunk_y_velocity", "trunk_z_velocity", "trunk_roll_velocity", "trunk_pitch_velocity",
           "trunk_yaw_velocity", "goal_x_velocity", "goal_y_velocity", "goal_yaw_velocity", "projected_gravity_x",
           "projected_gravity_y", "projected_gravity_z", "height_0")


def load_end_flags(root):
    """Terminated and truncated flags (samples, robots) of a dataset folder (the trainer's Dataset loads only dones)."""
    def flag(name):
        x = np.load(os.path.join(str(root), name), mmap_mode="r")
        return np.asarray(x).reshape(x.shape[0], x.shape[1]) > 0.5
    return flag("terminateds.npy"), flag("truncateds.npy")


def episode_table(done, terminated, truncated, reward):
    """Episodes of one robot's time-ordered transitions: the runs up to and including a done flag."""
    done = np.asarray(done, bool).reshape(-1)
    ends = np.flatnonzero(done)
    if len(done) and not done[-1]:
        ends = np.append(ends, len(done) - 1)           # the last episode, cut by the end of the data
    starts = np.concatenate([[0], ends[:-1] + 1]).astype(np.int64)
    csum = np.concatenate([[0.0], np.cumsum(np.asarray(reward, np.float64).reshape(-1))])
    return dict(ep_start=starts, ep_len=ends - starts + 1, ep_return=csum[ends + 1] - csum[starts],
                ep_terminated=np.asarray(terminated, bool).reshape(-1)[ends],
                ep_truncated=np.asarray(truncated, bool).reshape(-1)[ends], ep_complete=done[ends])


def reference_mask(table, horizon, min_return_frac):
    """Complete episodes that reach the horizon without termination, with at least a fraction of the best return."""
    full = table["ep_complete"] & table["ep_truncated"] & ~table["ep_terminated"] & (table["ep_len"] == horizon)
    if not full.any():
        return full
    best = table["ep_return"][full].max()
    return full & (table["ep_return"] >= min_return_frac * best) if best > 0 else full


def feature_indices(name_to_id):
    """Joint and foot names in action order and the observation indices of the dumped features."""
    names = sorted(name_to_id, key=name_to_id.get)
    joints = [n[:-len("_position")] for n in names if n.endswith("_position") and n[:-len("_position")] + "_velocity" in name_to_id]
    feet = [n[:-len("_ground_contact")] for n in names
            if n.endswith("_ground_contact") and not n.endswith("_cycles_since_last_ground_contact")]
    idx = dict(joint_pos=[name_to_id[j + "_position"] for j in joints], joint_vel=[name_to_id[j + "_velocity"] for j in joints],
               contact=[name_to_id[f + "_ground_contact"] for f in feet],
               air_time=[name_to_id[f + "_cycles_since_last_ground_contact"] for f in feet],
               general=[name_to_id[g] for g in GENERAL])
    return joints, feet, idx


class CoordinationDump:
    def __init__(self, policy, split_fn, robot_names, name_to_ids, horizon, a, chunk=8192, awr_terms=None):
        self.policy, self.split_fn, self.awr_terms = policy, split_fn, awr_terms
        self.robot_names, self.name_to_ids = list(robot_names), list(name_to_ids)
        self.horizon, self.chunk = int(horizon), int(chunk)
        self.out = str(a.morse_coord_out)
        self.min_return = float(a.morse_coord_min_return)
        self.startup = int(getattr(a, "morse_coord_startup", 0))
        self._fns = {}

    def predict(self, variables, obs, r, n_joints, act=None, extra=None):
        """Policy mean and log-std of robot ``r`` on the rows of ``obs`` (fixed-size chunks, the last one padded), and
        with ``act`` and critic parameters ``extra`` the advantage weight of each row's action."""
        if r not in self._fns:
            self._fns[r] = jax.jit(lambda v, o: self.policy.apply(v, *self.split_fn(o, r), robot_id=r))
            if self.awr_terms is not None:
                self._fns[("w", r)] = jax.jit(lambda out, o, ac, ex: self.awr_terms(out, o, ac, ex, r)[0][:, 0])
        empty = np.zeros((0, n_joints), np.float32)
        means, logstds, weights = [empty], [empty], [np.zeros(0, np.float32)]
        for i in range(0, len(obs), self.chunk):
            x = np.asarray(obs[i:i + self.chunk])
            n = len(x)
            pad = lambda y: np.concatenate([y, np.zeros((self.chunk - n,) + y.shape[1:], y.dtype)]) if n < self.chunk else y
            x = jnp.asarray(pad(x))
            m, s = self._fns[r](variables, x)
            if extra is not None:
                weights.append(np.asarray(jax.device_get(self._fns[("w", r)]((m, s), x, jnp.asarray(pad(np.asarray(act[i:i + self.chunk]))),
                                                                          extra))[:n], np.float32))
            m, s = jax.device_get((m, s))
            means.append(np.asarray(m[:n], np.float32))
            logstds.append(np.asarray(s[:n], np.float32))
        return np.concatenate(means), np.concatenate(logstds), np.concatenate(weights)

    def run(self, dataset, var_a, var_b, terminated, truncated, checkpoints=(), extra_a=None, extra_b=None):
        os.makedirs(self.out, exist_ok=True)
        for r, name in enumerate(self.robot_names):
            table = episode_table(dataset.done[:, r], terminated[:, r], truncated[:, r], dataset.rew[:, r])
            table["ep_reference"] = reference_mask(table, self.horizon, self.min_return)
            if self.startup > 0:            # the first steps of every complete episode
                eps = np.flatnonzero(table["ep_complete"])
                lens = np.minimum(table["ep_len"][eps], self.startup)
            else:                           # whole reference episodes
                eps = np.flatnonzero(table["ep_reference"])
                lens = table["ep_len"][eps]
            starts = table["ep_start"][eps]
            index = (np.concatenate([np.arange(s, s + n) for s, n in zip(starts, lens)]) if len(eps)
                     else np.zeros(0, np.int64))
            joints, feet, fidx = feature_indices(self.name_to_ids[r])
            obs = np.asarray(dataset.obs[index, r, :])
            act = np.asarray(dataset.act[index, r, :])
            weighted = self.startup > 0 and self.awr_terms is not None and extra_a is not None
            mean_a, logstd_a, w_a = self.predict(var_a, obs, r, len(joints), act, extra_a if weighted else None)
            mean_b, logstd_b, w_b = self.predict(var_b, obs, r, len(joints), act, extra_b if weighted else None)
            if mean_a.shape[1] != len(joints):
                raise ValueError(f"{name}: {mean_a.shape[1]} policy outputs but {len(joints)} joints in the observation")
            episode = np.repeat(eps, lens).astype(np.int32)
            out = dict(robot=np.array(name), checkpoints=np.array(list(checkpoints)), horizon=np.array(self.horizon),
                       min_return_frac=np.array(self.min_return), startup=np.array(self.startup),
                       joint_names=np.array(joints), foot_names=np.array(feet), general_names=np.array(GENERAL), **table,
                       index=index.astype(np.int64), episode=episode, t=(index - np.repeat(starts, lens)).astype(np.int32),
                       reference=table["ep_reference"][episode], act=act[:, :len(joints)].astype(np.float32),
                       mean_A=mean_a, mean_B=mean_b, logstd_A=logstd_a.mean(axis=0), logstd_B=logstd_b.mean(axis=0))
            if weighted:
                out["w_A"], out["w_B"] = w_a, w_b
            for k, cols in fidx.items():
                out[k] = obs[:, cols].astype(np.float16)
            np.savez(os.path.join(self.out, f"{name}.npz"), **out)
        return self.out
