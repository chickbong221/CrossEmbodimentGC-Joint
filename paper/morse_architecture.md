# MORSE paper architecture (v2, 2026-10-13)

**Title:** Learning Morphology-Aware Sharing and Specialization for Cross-Embodiment Offline Reinforcement Learning
**Method:** MORSE, Morphology-aware Routing of Shared Experts

**Thesis.** This sentence appears in the abstract, the introduction, the opening of the method and the conclusion:

> A cross-embodiment policy should share computation according to morphological relationships at the level where
> that computation operates: individual joints for local processing, whole embodiments for body-wide coordination.

## Style rules (from the two reference papers)

- **DiDiCM (CVPR'26).**
  - The introduction follows the arc: context, then "Yet ... remains open", then *To this end, we introduce X, a
    framework that ...*, then an instantiation that keeps a well-known backbone for a fair comparison, then
    contributions (1)–(3) in one paragraph.
  - The method is defined independently of the backbone. The concrete network appears in the experiments (DiDiRN
    corresponds to MORSE on URMA), with a figure that colours the original modules and the new ones differently.
  - Each related-work paragraph has a bold run-in heading and ends with one contrast sentence.
- **EG (ICLR'26).**
  - The problem setting uses a generic morphology descriptor, not URMA.
  - The backbone enters only in the setup ("we adopt URMA").
  - The method is something that "integrates with standard offline RL methods".
- **Credit.**
  - The MI form (Mod-Squad) and the shared expert (DeepSeekMoE, PLE) are credited once, in Related Work.
  - The method section states each design choice with **our own reason**. It never says "following X", "as in X",
    or "imported from X".

## Page budget (8 pages)

Abstract · 1 Intro (1.1 p) · 2 Related Work (0.6 p) · 3 Problem Formulation (0.5 p) · 4 MORSE (2 p) ·
5 Experiments (3 p) · 6 Conclusion (0.2 p) · Supplementary

---

## Abstract (~170 words, 5 sentences)

1. **Setting and challenge.** Reuse control knowledge across robots while accommodating their different bodies.
2. **Insight.** These relationships exist at more than one level.
   - Joints can share local control structure across otherwise different robots.
   - Coordinated actions need whole-body context.
3. **MORSE.**
   - Expert layers combine an always-active shared expert with routed experts that are shared across robots.
   - Joint-conditioned routers select experts for local encoding and action prediction.
   - Embodiment-conditioned routing processes the aggregated body representation.
4. **Objective.** A mutual-information objective is aligned with the two routing levels: joint groups at joint
   level, embodiments at body level. It uses embodiment-balanced statistics.
5. **Results.**
   - 16 robots, six datasets, three offline RL learners: [X] over robot-level grouping.
   - Analysis: experts are shared by corresponding joints across robots.

## 1 Introduction

- **P1 Context.**
  - One policy for many robots. Offline RL makes cheap suboptimal data usable.
  - Cross-embodiment offline RL (the EG benchmark).
  - Morphology-agnostic policies make this possible: they process every joint with shared modules conditioned on
    descriptors.
- **P2 Problem: two existing levers, each acting at one level.**
  - Joint-based architectures share every module across every joint. Representing joints individually does not
    decide *which* joints should share *which* parameters.
  - Robot-level grouping (EG) changes how whole embodiments take part in the update. All parts of one robot still
    receive the same decision.
  - Close with the question, in italics: *How can one policy selectively share computation across heterogeneous
    robots while preserving both local specialization and whole-body coordination?*
- **P3 Observation (Fig. 1).** Morphological similarity is partial and exists at two levels.
  - Concrete examples from our joint groups:
    - the calf joints of A1, Go1, Go2 and Barkour form one group;
    - the hip-abduction joints of Unitree and ANYmal form another;
    - the yaw joints of H1, G1 and Talos form a third;
    - yet the robots as wholes differ.
  - Within one robot, joints play different roles.
  - Locomotion is coordinated: a joint's action depends on the whole body.
  - Leads to the principle, stated once in italics: *specialize local processing according to joint morphology, and
    organize whole-body processing according to embodiment morphology.*
- **P4 "To this end, we introduce MORSE ..."** One mechanism with two placements:
  - **Expert layers.** One always-active shared expert plus a pool of routed experts shared across robots.
  - **Joint-level layers (encoder, decoder).** A router reads the joint descriptor, the robot descriptor and the
    state, and picks k experts per joint.
  - **Body-level layers (core).** A router reads the robot descriptor and the body state.
  - The body representation feeds every joint decoder and its router.
  - The MI objective is aligned with both routing levels.
  - **What MORSE leaves unchanged.** MORSE changes only the policy network. The offline RL objective stays the
    same, and there is no update schedule and no simulator feedback.
  - Therefore MORSE applies to BC, TD3+BC and IQL alike, and it is complementary to update-level methods such as EG.
- **P5 Instantiation and headline.**
  - MORSE is defined on the joint–body–joint structure that morphology-agnostic policies share.
  - We instantiate it on URMA, the benchmark's backbone. The aggregation, the std branch and the critics are kept.
  - This allows a direct comparison with the reported baselines.
  - Headline results [X]: gains across six datasets and three learners, largest with suboptimal data. Robots with
    few similar embodiments (NAO, Cassie) gain most.
- **P6 Contributions (1)(2)(3), inline.**
  1. **Selective sharing at complementary morphological levels.**
     - Sparse joint-level experts for encoding and action prediction.
     - Embodiment-level experts for whole-body context, given to every joint.
  2. **Morphology-aligned routing regularization.**
     - Joint groups from joint-local properties at joint level.
     - Embodiment identity at body level.
     - Embodiment-balanced statistics.
  3. **An empirical study.**
     - The 16-robot benchmark.
     - Controlled comparisons of routing granularity, layer design, grouping variable, capacity and robot
       reweighting.
     - Analyses of what the experts share and how updates interact.
- **Novelty claim.**
  - Do not claim "first joint-level MoE": M2oE exists.
  - Safe form: *to our knowledge, the first cross-embodiment policy that learns which computation to share at both
    the joint and the body level within one network*. Check this before submission.

**Fig. 1 (teaser).**
- **Left:** robot-level grouping. Whole robots are grouped, and one network is shared.
- **Middle:** MORSE.
  - Joints of a quadruped, a humanoid and Cassie, coloured by joint group, route into a pool of shared and routed
    experts.
  - One body-level route per robot.
- **Right:** [X] results bar or per-class gain.

## 2 Related Work (three paragraphs, bold run-in heads)

Written 2026-10-13 in `morse_cvpr_revised.tex` (about 470 words). The "To the best of our knowledge … first"
paragraph cut from the intro now closes the first paragraph. Rule: the intro gives the high-level positioning with
2–3 citations per claim; Related Work gives the detailed contrasts. Citations may appear in both sections, but the
discussion must not repeat.

- **Cross-embodiment and morphology-agnostic control.**
  - OXE, Octo, π0: embodiment-specific heads.
  - NerveNet, SMP, Amorpheus, MetaMorph, AnyMorph, SWAT, URMA: modules shared over all joints or limbs.
  - ModuMorph, HyperDistill: limb-specific parameters from a hypernetwork. They depend on morphology but not on
    state, and come from one shared generator.
  - M2oE: experts per module for reconfigurable modular robots.
  - EG: cross-embodiment offline RL with robot-level grouping in the update.
  - *Contrast:* MORSE learns sparse, state-dependent sharing at two morphological levels inside one policy. It works
    on the architecture side, so it is complementary to EG's update-side grouping.
- **Parameter sharing in multi-task learning.**
  - Gradient manipulation: PCGrad, CAGrad, Nash-MTL.
  - Task grouping: TAG, SEL.
  - Task-specific parameters: Recon, HarmoDT.
  - *Contrast:* there, the units are tasks identified by an index. In MORSE, the units are joints and bodies
    identified by morphology descriptors, so sharing follows similarity. It is learned end to end from the policy
    loss.
- **Mixture of experts.**
  - Sparse MoE: Shazeer, GShard, Switch, V-MoE.
  - Shared plus specialized experts: PLE, DeepSeekMoE.
  - Task-conditioned experts in vision (MMoE, M³ViT, Mod-Squad) and in RL (soft modularization, CARE, PaCo, MOORE,
    MoE-Loco, Obando-Ceron et al.).
  - **Credit the MI form here, once:** "the dependence between a task variable and expert selection has been used to
    obtain task-specific experts [Mod-Squad]".
  - *Contrast:* in MORSE, both the routing unit and the dependence variable follow morphology: joints with joint
    groups, bodies with embodiments, balanced over embodiments.

## 3 Problem Formulation

- **Setting.** Keep the draft's notation: R, J_r, F_r, d_j, d_f, m_r, o_j, o_f, o^g, π_θ(a | s, m_r).
  - The pooled dataset D has mixed quality.
  - **Eq. 1** is the actor loss with equal robot weights. The per-sample loss ℓ_i comes from the base learner (BC,
    TD3+BC or IQL).
- **Joint–body–joint policies.** **Eq. 2:**
  - z^loc_j = F(o_j, d_j)
  - x = A({(z^loc_j, d_j)}, {(z^loc_f, d_f)}, o^g)
  - z = C(x)
  - μ_j = D(z, c_j), with c_j = (z^loc_j, d_j)
  - F and D are *joint-level*: they are applied to every joint with the same parameters.
  - C is *body-level*: it is applied once per state.
  - Attention aggregation (URMA) fits this form directly. Transformer and graph policies interleave the joint-wise
    and body-wise maps.
- **The sharing question, made precise.**
  - Every joint of every robot passes through the same F and D, and every robot through the same C.
  - What should decide which joints, and which bodies, use the same parameters? MORSE learns this.
- The gradient decomposition (draft Eq. 3) **leaves this section** and moves to §5.5.

## 4 MORSE

**Opening paragraph and Fig. 2.**
- The paragraph goes from the principle to the two placements to the regularizer.
- Fig. 2 shows the generic framework, not URMA:
  - joint-level expert layers in F and D, and body-level expert layers in C;
  - the shared expert, and routers with their inputs;
  - MI arrows labelled "joint groups" and "embodiments".

### 4.1 Expert layers with shared and routed experts

- **Eq. 3:** y = f^b_0(x) + Σ_{e=1}^{K−1} G^e_b(t) f^b_e(x).
- **Shared expert f_0 (always active).** It holds what all robots need, so the routed experts specialize instead of
  each re-learning the common part.
- **Routed weights.** The softmax scores of the top-k routed experts, *not renormalized*.
  - While the router is uncertain, the shared expert dominates.
  - A routed expert gains weight only as the router becomes confident.
  - The result is a smooth transition from shared to specialized computation.
- **Eq. 4 (router):**
  - h_b(t) = ψ^m_b(d_j, d̄_r) + ψ^s_b(s^b_j)
  - G_b(t) = TopK(Softmax(h_b(t)))
- **Morphology branch.**
  - All robots share one descriptor format, so corresponding joints receive similar preferences.
  - d̄_r (mean joint and foot descriptors plus counts) tells the joint which body it belongs to.
- **State branch.** Adapts the selection to the situation (stance or swing).
- **Learned routing noise** during training: one sentence here, details in the supplement.

### 4.2 Joint-level experts: local encoding and action prediction

- **Encoder.** An expert layer inside F. The router's state input is o_j.
- **Decoder.** Every decoder expert D_e is a full action head with its own descriptor embedding.
  - **Eq. 5:** μ_j = Σ_e G^e_D(t) D_e(z, c_j).
  - Each group can therefore learn its own mapping from descriptor to action.
- **Decoder router.** Its state is [z, z^loc_j], so the choice of local action expert is informed by the whole body.
- **Separate encoder and decoder routers.** Joints can share how they read observations but act through different
  heads, and the reverse.

### 4.3 Body-level experts: whole-body coordination

- **Routing.** The core's expert layers route once per (robot, state).
  - The morphology branch reads d̄_r.
  - The state branch reads the layer input.
  - Dissimilar bodies can use different experts, and similar bodies share them.
- **Coordination channel.** z is broadcast to every joint decoder and to every decoder router.
- **Anchor sentence:** "The two placements use the same layer; only the routing unit changes, and it follows the
  scope of the computation."

### 4.4 Morphology-aligned routing objective

- **Why it is needed.** The task loss trains the routing, but it does not encourage persistent sharing tied to
  morphology.
- **Joint groups.**
  - **Features.** 13 joint-local properties:
    - |axis|, number of child joints, |nominal position|;
    - torque and velocity limits relative to the robot's largest;
    - damping, armature, stiffness, friction;
    - range width, |range centre|.
  - **Mirror-invariant.** Left and right counterparts get the same features.
  - **Excluded.** Torso-relative position, robot-wide gains, mass and dimensions, so that a group describes a
    joint's *function*, not its robot.
  - **Clustering.** Standardized over all joints of all robots, then k-means with k = 8, once before training.
  - **Groups cross robot boundaries.** NMI with robot identity is 0.44. The full table is in the supplement. Give 2–3
    examples here.
- **Eq. 6 (embodiment-balanced statistics).**
  - P_b(e | t) is averaged over the robot's transitions, and P(t) = 1 / (|R| |J_r|).
  - Talos (30 joints) and Bittle (8) count equally.
- **Eq. 7 (MI).** Write it as I_b(Y; E) = H(E) − H(E | Y).
  - Read it as: use all experts, and let each group concentrate on a few.
  - At body level, Y is the robot, with P(r) = 1/|R|. The same equation covers both levels.
- **What it does not do.**
  - It fixes no expert per joint.
  - Selection stays state-dependent, because the objective acts on routing probabilities averaged over states.
  - The groups act as a prior, not as an assignment.

### 4.5 Training

- **Eq. 8:** L_train = L − λ Σ_b I_b + β Σ_b KL(p̄_b ‖ u).
- **What stays the same.**
  - The base learner and the critics are unchanged.
  - λ starts after a warm-up, so routes first form from the task.
  - No stage, scheduler or extra data is added.
- **Optional:** a half-column Algorithm 1.

## 5 Experiments

### 5.1 Setup

- **Benchmark.**
  - 16 robots (9 quadrupeds, 6 bipeds, 1 hexapod) and six datasets, with the EG protocol.
  - Metric: mean return over all 16 robots.
- **Baselines.**
  - BC, TD3+BC, IQL.
  - IQL+PCGrad, IQL+SEL.
  - +EG on all three learners: the reported numbers, plus our reproduction on the same code.
- **Instantiating MORSE on URMA.** A DiDiRN-style paragraph with Fig. 3, where URMA modules are grey and MORSE
  modules are orange.
  - URMA has three stages:
    1. per-joint and per-foot encoders with descriptor-conditioned attention aggregation;
    2. an MLP core;
    3. a per-joint decoder.
  - **Kept unchanged:** the aggregation, the foot encoder, the std branch and the critics.
  - **Replaced:**
    - the middle layer of the joint encoder, with a joint-level expert layer;
    - every second core layer, with a body-level expert layer;
    - the decoder, with a bank of decoder experts.
  - **Settings.** K = 16 (1 shared, 15 routed), k = 3, plus λ, β and the warm-up.
  - Four actor steps per update, to match EG's step count.

### 5.2 Main results

- **Table 1.** Methods × six datasets, with ± SE.
- **Table 2.** Per robot, or per class.
- **Paragraphs:**
  - comparison with robot-level grouping;
  - across base learners;
  - which robots benefit.

### 5.3 Controlled comparisons (Table 3: IQL, 70% Sub Fwd plus one more dataset)

| Block | Question | Rows |
|---|---|---|
| (a) Granularity | Are both levels needed? | dense · body-level only · joint-level only · both |
| (b) Layer design | Do the design choices matter? | no shared expert · renormalized top-k · decoder router without body context |
| (c) Grouping variable | Does morphology-aligned MI matter? | no MI · robot identity at joint layers · random groups (same sizes) · joint groups · per-joint (unbalanced) weighting |
| (d) Capacity and reweighting | Is it only capacity or robot reweighting? | dense with matched active parameters (details in the supplement) · dense + EG's implied robot weights · MORSE + EG schedule (complementarity) |

### 5.4 What do the experts share? (Fig. 4)

- A heatmap of joint group × expert per layer.
- Overlap of expert sets between corresponding joints of different robots, compared with non-corresponding joints.
- Body-level routing clusters robots, compared with EG's FGW groups.
- Within-robot diversity, I(J; E | R).

### 5.5 How updates interact (supporting diagnostic)

- The decomposition and localized equations move here from the draft's §3 and §5.3.
- Measured robot-gradient alignment on dense URMA, compared with MORSE's experts.
- In the dense model, robot gradients in the core are mostly aligned, which supports the shared expert.
- Success-vs-fall conflict sits in the decoder.
- Presented as diagnostics, not as the cause of the gains.

## 6 Conclusion

- **Restate the principle.**
- **Limitations.**
  - The groups are fixed by k-means.
  - The experiments are state-only locomotion.
  - The hexapod is not learned.
- **Future work.**
  - Learned groups.
  - Transfer to new robots: routing extends to an unseen robot through its descriptors, without new parameters.
  - Combination with update-level grouping.

## Supplementary

- A. Architecture and hyperparameters, including widths and capacity.
- B. Joint-group table: the full k = 8 table and the k = 16 variant.
- C. Full per-robot tables.
- D. Conflict-analysis details.
- E. EG reweighting analysis.
- F. Evaluation protocol.

---

## Notation

- **Kept from the draft:**
  - R, J_r, F_r, d_j, d_f, d̄_r, m_r, o_j, o^g;
  - z, z^loc_j, c_j;
  - t = (r, j), b (layer);
  - K, k, G_b, h_b, ψ^m_b, ψ^s_b;
  - D_e, y_t, P(t), I_b(Y; E), λ, β.
- **New, only two:** F (the joint encoder in Eq. 2) and f^b_e (the experts of layer b, with e = 0 the shared
  expert).
- **Leave the preliminaries** (they go to §5.5 with the decomposition): U_r, g_u, φ.

## Changes from the current draft

- **Remove the conflict-driven framing** from the abstract, intro P3 and contribution 1. Conflict becomes a §5.5
  diagnostic.
- **Remove the attributions in the method**: "Following Mod-Squad", "As in DeepSeekMoE", and "as in Mod-Squad" for
  the noise. Credit these sources in §2 instead.
- **Move** "We build on URMA because ..." from Related Work to the §5.1 instantiation.
- **Cut repeated sentences:**
  - "every joint has its own path" (4×);
  - "corresponding joints of different robots can share ... while joints of one robot use different ones" (4×);
  - "end to end without estimating gradients" (2×).
- **Add:**
  - the generic template;
  - the body-level role, and z feeding the decoder router;
  - the reasons for the shared expert and for not renormalizing;
  - the joint-group construction;
  - the reason for embodiment balancing;
  - what the MI objective does not do;
  - the instantiation paragraph;
  - Table 3 and the sharing analysis.

## Open items for the writing

- **Capacity row.** An earlier rule says model size goes only in the appendix. Keep only a one-line pointer in
  Table 3(d)?
- **Section order.** DiDiCM puts Problem Formulation before Related Work. Use the CVPR-standard order (above) unless
  you prefer DiDiCM's.
