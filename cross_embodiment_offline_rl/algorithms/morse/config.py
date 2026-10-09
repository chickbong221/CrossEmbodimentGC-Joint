# Settings added on 2026-10-04 and the values that reproduce the architecture of earlier runs. A checkpoint whose saved
# config lacks one of these keys gets this value when it is loaded (iql.py ``IQL.load``).
LEGACY_ARCHITECTURE = {"morse_topk_core": 0, "morse_core_routing": "body", "morse_expert_hidden": 0,
                       "morse_router_init": 0.0, "morse_core_shared": "none", "morse_joint_encoder_hidden": "",
                       "morse_core_depth": 3, "morse_core_weight_norm": False,
                       "morse_variance_mode": "original", "morse_freeze_core": False, "morse_arch": "urma_moe"}
# Defaults before 2026-10-04 of the settings whose default changed then. Run definitions written before that date rely
# on them; rerunning one needs them as explicit flags (together with LEGACY_ARCHITECTURE).
PREVIOUS_DEFAULTS = {"morse_num_experts": 4, "morse_num_experts_decoder": 0, "morse_topk_decoder": 0,
                     "morse_conflict_mode": "pairwise", "morse_lambda": 1.0, "morse_eta": 0.01, "morse_mi_weight": 0.0,
                     "morse_mi_blocks": "decoder", "morse_mi_weighting": "units"}
# Settings written next to every saved checkpoint (models/checkpoints.jsonl); the checkpoint holds the whole config.
RECORD_KEYS = ("morse_blocks", "morse_num_experts", "morse_num_experts_decoder", "morse_topk_core", "morse_topk_decoder",
               "morse_eval_topk", "morse_core_routing", "morse_expert_hidden", "morse_width_mult_core",
               "morse_width_mult_decoder", "morse_router_hidden", "morse_router_inputs", "morse_router_init",
               "morse_detach_router_inputs", "morse_decoder_table", "morse_identical_experts", "morse_calibration",
               "morse_mi_weight", "morse_mi_weight_core", "morse_mi_blocks", "morse_mi_balance_blocks", "morse_mi_target",
               "morse_mi_weighting", "morse_eta", "morse_lambda",
               "morse_conflict_mode", "actor_steps_per_update", "actor_lr", "batch_size", "nr_offline_updates",
               "morse_router_noise", "morse_router_noise_floor", "morse_topk_ties", "morse_fixed_robot_blocks",
               "morse_mi_core_groups", "morse_mi_cap", "morse_compat_route", "morse_compat_lambda_core",
               "morse_compat_lambda_decoder", "morse_compat_tau", "morse_data_rows", "morse_core_shared",
               "morse_core_shared_coef", "morse_eg_groups", "morse_joint_encoder_hidden", "morse_core_depth",
               "morse_core_weight_norm", "morse_variance_mode", "morse_freeze_core", "morse_arch",
               "morse_v8_direction", "morse_v8_width", "morse_v8_experts", "morse_v8_balance", "morse_v8_mi_weight",
               "morse_v8_cap", "morse_v8_compat_lambda", "morse_v8_joint_groups")


def add_morse_config(config):
    """Add MORSE hyperparameters to an algorithm config (defaults = main method).

    Main method (2026-10-04): experts in the core and the decoder, 8 per block with top-4 routing; the core's experts
    are mixed per joint (joint-conditioned core routing); joint-type MI with equal robot weights on both blocks; no
    utilization or compatibility objectives.
    """
    # MORSE replaces Embodiment Grouping; one group means every robot updates together.
    config.M = 1

    # Expert blocks: any subset of joint_encoder, foot_encoder, core, decoder. The joint and foot encoders
    # hold a few dozen parameters (attention logits), so experts there add routing but no capacity.
    config.morse_blocks = "core,decoder"
    config.morse_num_experts = 8
    config.morse_num_experts_decoder = 8       # > 0: experts of the decoder; the core keeps morse_num_experts
    config.morse_topk_core = 4                 # > 0: every routing unit (a joint with joint routing) uses exactly its
    config.morse_topk_decoder = 4              # top-k experts of the block (weights renormalized), in training and
                                               # evaluation; 0 = dense routing
    config.morse_core_routing = "joint"        # joint: every joint mixes the whole-body core experts with its own router
                                               # (descriptor, joint state, whole-body input) and feeds its own decoder |
                                               # body: one core mixture per transition, broadcast to every joint
    config.morse_expert_hidden = 64            # > 0: hidden width of every core and decoder expert layer (core 64 -> 64
                                               # -> 128, decoder 64 -> 1); 0 = the width multipliers below
    config.morse_width_mult_core = 1.0        # hidden-width multipliers, used for the capacity-matched
    config.morse_width_mult_decoder = 1.0     # dense baseline (K=1; core 2.57, decoder 4.0 match K=4)
    config.morse_joint_encoder_hidden = "256,256"  # hidden widths of the joint-state encoder f_psi (URMAv2: 2 x 256; "" =
                                                   # URMA's single 4-unit layer); its output stays 4 features per joint
    config.morse_core_depth = 5                # dense layers of every core expert, the last the 128-wide output (URMA 3)
    config.morse_core_weight_norm = True       # WeightNorm on every dense layer of the core (URMAv2)
    config.morse_router_hidden = 64
    config.morse_router_inputs = "both"        # both | morph | state
    config.morse_router_init = 0.01            # std of the routers' random output layers: breaks the ties of uniform
                                               # initial routing, so exact top-k starts from varied expert sets
    config.morse_detach_router_inputs = False

    # Interference-guided routing supervision.
    config.morse_conflict_mode = "none"        # pairwise | shuffled | expert_mean | signed | none. signed: penalize
                                               # routing overlap of harmful type pairs and reward it for helpful ones,
                                               # on cosines beyond morse_compat_z noise std (even/odd split estimate);
                                               # evidence accumulates every update from the start
    config.morse_lambda = 0.0                  # weight of L_conf
    config.morse_eta = 0.0                     # weight of L_use
    config.morse_conflict_norm = "pairs"       # pairs: L_conf is a mean over unit pairs, so its pressure grows with
                                               # the amount of conflict | mass: normalized by the total conflict mass
    config.morse_threshold = "fixed"           # fixed (kappa) | calibrated (zeta * std of alignments)
    config.morse_zeta = 1.0
    config.morse_kappa = 0.05
    config.morse_relevance = True
    config.morse_evidence = "type"             # type: alignments of per-type contributions accumulated over updates
                                               # (element of a robot, or robot) | unit: alignments of unit pairs
    config.morse_unbiased_types = True         # type sums without pairs of units from the same transition, whose
                                               # shared sample makes their noise correlated (the diagonal keeps them)
    config.morse_type_ema = 0.99               # decay of the type-level accumulation
    config.morse_conflict_every = 1            # apply the conflict objective every N policy updates
    config.morse_warmup = 5000                 # policy updates before conflict supervision starts
    config.morse_ramp = 5000                   # linear ramp of lambda after warm-up
    config.morse_capture_transitions_element = 8   # transitions per robot captured at joint/foot-level blocks
    config.morse_capture_transitions_body = 64     # transitions per robot captured at whole-body blocks
    config.morse_conflict_blocks = ""          # blocks that take the conflict objective ("" = every routed block)
    config.morse_compat_z = 2.0                # signed mode: cosines within z noise std count as no evidence
    config.morse_compat_help = 1.0             # signed mode: weight of the helpful (shared-routing) reward
    config.morse_compat_estimator = "products" # signed mode: products = type-sum products of the captured transitions,
                                               # accumulated over updates (morse_type_ema) | crossfit = per-type
                                               # gradient sums of morse_compat_transitions transitions per robot,
                                               # accumulated in four interleaved streams (update mod 4, decay
                                               # morse_compat_ema per stream update); cosines are products of
                                               # different streams, i.e. of independent data
    config.morse_compat_ema = 0.9
    config.morse_compat_transitions = 256      # crossfit / PCGrad: transitions per robot and step (256 = a whole actor chunk)
    config.morse_pcgrad_blocks = ""            # gradient-surgery control: "decoder" replaces the decoder's task gradient by
                                               # the sum of per-joint-type gradients after PCGrad across joint types

    # Mod-Squad-style mutual information between unit type and expert, -I(A;E) = sum_a pi_a H(r_a) - H(rbar),
    # where r_a is the mean soft routing of type a over every transition of the actor step's minibatch (joint of a
    # robot at the decoder and the joint-routed core). Router-only; every actor step; the blocks' losses add up.
    config.morse_mi_weight = 0.005
    config.morse_mi_blocks = "core,decoder"
    config.morse_mi_target = "type"            # type: joint of a robot (element-level blocks) | robot: -I(R;E), one
                                               # routing preference per robot (I(R,J;E) = I(R;E) + I(J;E|R))
    config.morse_mi_weighting = "robot"        # units: groups weighted by their share of units (at the decoder a
                                               # robot weighs by its number of joints) | robot: every robot equal
    config.morse_mi_weight_core = -1.0         # >= 0: weight of the core's MI term (negative: morse_mi_weight)
    config.morse_mi_balance_blocks = ""        # MI blocks whose term keeps only its load-balancing half, -H(E): every
                                               # expert used equally on average, no pull toward group-specific experts
    config.morse_mi_core_groups = "robot"      # body-routed core with morse_mi_target=type: the MI groups are the robots
                                               # (robot) or their Embodiment Grouping clusters (eg: morse_eg_groups
                                               # clusters by embodiment distance, as in uni_iql_eg); a group's routing
                                               # is the mean of its robots' routing, its prior the sum of their priors
    config.morse_eg_groups = 4                 # number of Embodiment Grouping clusters (Ward clustering of
                                               # embodiment_distance_path, as uni_iql_eg with M clusters) for the core's
                                               # EG-group MI and its group experts; at least 2 (config.M stays 1: MORSE
                                               # updates every robot together)
    config.morse_mi_cap = 0.0                  # > 0: in every actor step and block, the MI gradient on the router is
                                               # scaled down to at most this multiple of the task gradient's norm on
                                               # the same router (raw gradients, before the optimizer)

    # Routing exploration (2026-10-06).
    config.morse_router_noise = False          # learned routing noise in training (Mod-Squad): logits + eps * sigma(x),
                                               # sigma = softplus(noise head) + morse_router_noise_floor, eps ~ N(0, 1)
                                               # from a stream of its own; evaluation routes on the clean logits. The MI
                                               # and compatibility objectives use the noisy probabilities with the noise
                                               # held fixed (no gradient to its scale); load balancing the clean ones
    config.morse_router_noise_floor = 0.01
    config.morse_topk_ties = "exact"           # exact: ties go to the lower expert index | keep: every weight >= the
                                               # k-th largest is kept (the rule before 2026-10-04: a uniform start is
                                               # dense)
    config.morse_fixed_robot_blocks = ""       # blocks routed one-hot by robot (robot-private experts: one expert per
                                               # training robot); they have no router and take no routing objectives

    # Compatibility routing (2026-10-06): per routing unit (a transition at the body-routed core, a joint of a transition
    # at the decoder), expert e is non-conflicting when the unit's task-gradient direction on e agrees, cosine >=
    # -morse_compat_tau, with the expert's actual update from the other transitions, in both halves of the scored
    # sample (even and odd transitions); conflicting when both cosines are below -tau; unknown otherwise. The router's
    # noisy probabilities, renormalized over the assessed experts, are trained toward the uniform distribution over the
    # non-conflicting ones (KL). First actor step of an update; lambda after morse_warmup with a morse_ramp ramp.
    config.morse_compat_route = "off"          # off | compat | shuffled (label rows permuted among units of the same
                                               # robot, robot-joint at the decoder) | conflict_only (-log of the mass on
                                               # non-conflicting experts) | uniform (target uniform over the assessed
                                               # experts, labels ignored)
    config.morse_compat_lambda_core = 0.01
    config.morse_compat_lambda_decoder = 0.01
    config.morse_compat_tau = 0.05
    config.morse_compat_core_transitions = 64      # scored transitions per robot at the core
    config.morse_compat_decoder_transitions = 32   # scored transitions per robot at the decoder (all their joints)
    config.morse_compat_min_support = 4            # transitions routed to an expert that each reference half needs

    # Shared core experts (2026-10-06; DeepSeekMoE's shared-expert isolation, extended by group experts). The core's
    # morse_num_experts experts split into always-on shared experts and routed ones: global = one expert every robot
    # uses; group = one expert per EG group (morse_eg_groups), used by the robots of that group; hierarchical = both.
    # The router covers the routed experts only (top morse_topk_core of them, weights renormalized), and so do the
    # core's MI and balance terms; the shared experts take the task gradient only. Core output = c * global + c * group
    # + (1 - c * number of shared branches) * routed mixture, with c = morse_core_shared_coef (fixed, no learned gate).
    # Body-routed cores only; not with compatibility routing or robot-fixed routing.
    config.morse_core_shared = "none"          # none | global | group | hierarchical
    config.morse_core_shared_coef = 0.25       # weight of each always-on branch (global, group)

    # V8 (2026-10-07, morse/v8.py and morse/v8_updater.py): URMAv2 layer widths with a Mod-Squad-style MoE layer in
    # every second layer (joint encoder, core twice, decoder bank). The other morse_* architecture and objective keys
    # do not apply, except the router settings (morse_router_*), the compatibility keys (morse_compat_tau,
    # morse_compat_min_support, morse_compat_core_transitions, morse_compat_decoder_transitions), morse_warmup,
    # morse_ramp and morse_diag_frequency.
    config.morse_arch = "urma_moe"             # urma_moe: the MorsePolicy family above | v8
    config.morse_v8_direction = "none"         # mi_joint | mi_robot: DeepSeekMoE layers (1 shared + top-3 of 15
                                               # routed, softmax scores not renormalized) with -I(label; expert) |
                                               # compat: top-4 of 16 routed (renormalized) with compatibility routing |
                                               # none: the compat layer form with balancing only
    config.morse_v8_width = 0.5                # expert width: feed-forward experts d -> width*d/2 -> d (1 = the
                                               # parameters of the dense layer they replace); decoder hidden width*128
    config.morse_v8_experts = 16               # experts per MoE layer (shared included)
    config.morse_v8_balance = 0.01             # weight of KL(pbar || uniform) per MoE layer, equal robot weights
    config.morse_v8_mi_weight = 0.0005         # weight of -I(label; expert) per MoE layer (MI directions)
    config.morse_v8_cap = 0.0                  # > 0: two-part cap of the routing objective (MI or compatibility):
                                               # per router, and one factor for the non-router parameters it reaches,
                                               # each at most this multiple of the task gradient's norm there; it
                                               # needs three backward passes per actor step. 0: off, one backward pass
                                               # (the objective waits morse_warmup updates and ramps in over morse_ramp)
    config.morse_v8_compat_lambda = 0.005      # weight of the compatibility objective per MoE layer (compat)
    config.morse_v8_joint_groups = ""          # mi_joint: JSON joint-group table (morse/v8.py; a relative path is
                                               # taken next to morse/v8.py); built from the dataset and written there
                                               # if the file does not exist
    config.morse_v8_group_k = 8                # joint groups when the table is built

    # Data selection (2026-10-06).
    config.morse_data_rows = ""                # "lo,hi": train on dataset rows [lo, hi) of every robot (empty: all rows);
                                               # robots outside the full training list use their own dataset column

    # Analysis controls (Tests 1-2).
    config.morse_decoder_table = ""            # .npy (robots, joints, experts): fixed decoder routing, no router
    config.morse_identical_experts = False     # all experts of a block start from the same parameters
    config.morse_calibration = False           # per-joint output scale and bias (initialized to 1 and 0)
    config.morse_probe_at = ""                 # updates at which to measure joint-type interactions in the decoder
                                               # on frozen parameters with independent data (morse/probe.py)
    config.morse_probe_batches = 4             # minibatches per probe data subset (5 subsets)
    config.morse_probe_steps = "0.002,0.02"    # norms of the probe's lookahead steps along a joint's gradient
    config.morse_probe_lr = 3e-4               # the probe's Adam-like lookahead step, -lr * sign(gradient)
    config.morse_variance_mode = "original"    # log-std pathway (morse/variance.py): original | detach | private | protected
    config.morse_freeze_core = False           # True: every parameter of the core block is frozen (morse/variance.py); the rest of the actor and the critics train as usual
    config.morse_stop_at = 0                   # > 0: end training after this many updates in total (the learning-rate schedule still spans nr_offline_updates)
    config.morse_save_at = ""                  # updates at which to save a checkpoint (models/at_<updates>)
    config.morse_branch_from = ""              # checkpoint folder to start from (morse/branch.py: a shared block
                                               # becomes identical experts), e.g. another run's models/at_2000
    config.morse_branch_updates = 0            # updates of that checkpoint; training continues to nr_offline_updates
    config.morse_ckpt_analysis = ""            # "EARLIER,LATER": two checkpoint folders of one run; instead of
                                               # training, compare them on fixed data (morse/ckpt_analysis.py)
    config.morse_ckpt_out = ""                 # .npz written by the checkpoint analysis
    config.morse_ckpt_batches = 8              # training-size minibatches of every robot in its fixed data (0 = skip)
    config.morse_ckpt_var = 0                  # > 0: variance diagnostics of every listed checkpoint on this many fixed minibatches (ckpt_analysis.variance_diagnostics)
    config.morse_ckpt_var_noise = False        # also with the router's training noise (same keys for every checkpoint)
    config.morse_ckpt_grads = 0                # > 0: also split the robots' policy gradients by parameter group on this
                                               # many minibatches, with their routing (ckpt_analysis.gradient_split)
    config.morse_ckpt_dump = ""                # robots (comma-separated names) whose per-transition arrays are also
                                               # written: dataset action, both policies' mean/logstd, routing, Shapley parts
    config.morse_ckpt_eval = False             # also evaluate both checkpoints exactly as during training (deterministic,
                                               # one episode per robot, twice), written to <morse_ckpt_out>_eval.json
    config.morse_coord_out = ""                # folder: per-robot dump of whole reference episodes with both
                                               # checkpoints' action means (morse/coordination.py)
    config.morse_coord_min_return = 0.5        # reference episodes: full length without termination and at least this
                                               # fraction of the robot's best such return
    config.morse_coord_startup = 0             # > 0: dump the first N steps of every complete episode instead, with
                                               # both checkpoints' advantage weights of the dataset actions
    config.morse_ckpt_eval_noise = ""          # "SIGMA@STEPS,...": also evaluate both checkpoints with N(0, SIGMA^2) noise
                                               # on every joint's mean action during the first STEPS steps (paired noise
                                               # across checkpoints), written to <morse_ckpt_out>_noise.json
    config.morse_ckpt_eval_episodes = 40       # episodes per robot, condition and checkpoint for morse_ckpt_eval_noise
    config.morse_conflict_batches = 0          # > 0: gradient-conflict analysis (morse/conflict.py) of every checkpoint in
                                               # morse_ckpt_analysis on this many fixed minibatches of every robot
    config.morse_conflict_episodes = 64        # episodes per class (success, fall, low return) and robot for its trajectory-level part

    # Actor optimizer steps per update, each on 1/n of every robot's minibatch (critics still update once on the
    # whole minibatch). 4 matches Embodiment Grouping with M=4, which takes one actor step per group: same actor
    # steps and samples. The MI objective applies at every actor step; the conflict and utilization objectives,
    # evidence, and diagnostics stay once per update (first actor step).
    config.actor_steps_per_update = 1

    # Deployment and adaptation.
    config.morse_eval_topk = 0                 # 0 = the blocks' own routing (their top-k); >0 additionally keeps the
                                               # top-k experts in the evaluation policy (runner.mode=test: TD3+BC
                                               # also uses this policy for target actions during training)
    config.morse_router_only = False           # update only router parameters (adaptation to new robots)

    # Diagnostics for the paper and appendix.
    config.morse_diag_frequency = 500          # log scalar diagnostics every N updates (0 = off)
    config.morse_snapshot_frequency = 10000    # save matrices for the appendix every N updates (0 = off)
    config.morse_robot_ema = 0.99              # decay of the accumulated robot-level gradient Gram (diagnostics)
    return config
