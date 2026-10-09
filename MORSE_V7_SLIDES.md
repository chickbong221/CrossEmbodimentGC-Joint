# MORSE v7: slide content for Claude Design

## Design brief (for Claude Design)

- **Audience:** research meeting (ML and robotics).
- **Story:** what EG did → where EG falls short → what our runs show → two proposed methods → how we test them.
- **Format:**
  - 16:9, clean academic style, about 26 slides plus an appendix.
  - Each slide's headline is a full sentence at the top.
  - Supporting runs go in a small footnote at the bottom of each slide.
- **Colors:**
  - EG: gray.
  - Shared-core runs: orange; MoE-core runs: purple.
  - Idea 1 (compatible-set routing): blue; Idea 2 (hierarchical shared experts): green.
- **Numbers:**
  - Every number is measured; keep them exactly as written and don't add values.
  - "Learned" = the robot's return is above half of the best return observed for it.
  - "Kept" = still learned at 100k updates.
- **Robots:**
  - Quadrupeds: A1, Go1, Go2, ANYmal B/C, Barkour v0/vB, Badger, Bittle.
  - Bipeds: H1, G1, Talos, OP3, NAO, Cassie.
  - One hexapod.

---

# Part 1: The setting and EG

## Slide 1. Title

**Sharing what helps, separating what conflicts**
Two expert-routing designs for cross-embodiment offline RL

Subtitle: EG analysis → insights from 48 finished runs → compatible-set routing and hierarchical shared experts

---

## Slide 2. Problem setting

**Headline:** One policy must learn to walk on 16 robots from data that is mostly suboptimal.

- **Robots:** 16 in MuJoCo (9 quadrupeds, 6 bipeds, 1 hexapod), all commanded to walk forward.
- **Data:** 1M transitions per robot, the benchmark's "70% Suboptimal Replay" set.
  - 700k come from early, suboptimal PPO training and 300k from late, expert training.
  - We verified the split for every robot: returns are 0–33 in the first 700k and 33–109 in the last 300k.
- **Policy:** URMA, trained with IQL (an advantage-weighted regression actor).
  - Per-joint encoders feed a whole-body **core**, then a per-joint action **decoder**.
- **Metric:** mean return over the last 10 evaluations at 100k updates.
  - Each evaluation runs one deterministic episode per robot, every 100 updates.

**Visual:**
- Diagram of the URMA policy: joint and foot encoders → core → decoder (one output per joint).
- Highlight "core" and "decoder", the two blocks we later turn into experts.
- Add small robot icons in three families.

---

## Slide 3. What EG found

**Headline:** EG (Abe et al., ICLR 2026) showed that training robots together helps some of them and hurts others, and traced this to conflicting gradients.

- **Transfer on 70% suboptimal data** (IQL, single-robot training → joint training):
  - **Positive:** A1 14.6 → 27.4, Go1 14.5 → 40.1, Go2 13.8 → 52.4, Badger 16.0 → 40.5.
  - **Negative:** H1 54.5 → 6.0, G1 78.9 → 0.9.
- **Conflict grows with suboptimal data:** the share of robot pairs whose actor gradients point in opposite directions (negative cosine) rises:
  - 0.16 with expert data;
  - 0.27 with 30% suboptimal;
  - 0.32 with 70% suboptimal.
- **Conflict grows with diversity:**
  - A1, Go1 and Go2: 0.15;
  - all 9 quadrupeds: 0.29;
  - all 16 robots: 0.32.
- **Correlations:**
  - a robot's transfer gain tracks how aligned its gradients are with the others' (r = 0.82);
  - morphological similarity tracks gradient alignment (r = 0.63).

**Visual:**
- Paired bars, single-robot vs joint training: A1, Go1, Go2 and Badger rise; H1 and G1 drop.
- Inset line: share of conflicting pairs vs share of suboptimal data (0.16, 0.27, 0.32).

*Source: Abe et al., Table 2 and Fig. 2.*

---

## Slide 4. What EG does

**Headline:** EG clusters robots by morphology and gives each cluster its own actor step, all on one shared network.

- **Groups:** robots are clustered once, before training, by graph (FGW) distance between their embodiments, into M = 4 groups.
- **Each update:**
  - one critic update on all robots;
  - then one actor step per group, in random order, using only that group's samples.
- **Results** (70% suboptimal, forward walking, 5 seeds):

  | Method | Return |
  |---|---|
  | IQL | 36.6 |
  | + PCGrad | 39.6 |
  | + SEL | 44.6 |
  | **+ EG** | **51.2** |

  - Compute-matched IQL scores 44.2 against EG's 52.0.
- **The grouping matters:**
  - random groups: 38.7 (+3%);
  - an intuitive biped / quadruped / hexapod split: 34.5 (−8%);
  - EG groups: 52.0 (+38%).
- **EG groups on our benchmark (M = 4):**
  - {A1, Go1, Go2, Barkour v0, Barkour vB, Badger, Bittle, hexapod}
  - {ANYmal B, ANYmal C, Cassie}
  - {H1, Talos}
  - {G1, OP3, NAO}

**Visual:**
- Algorithm strip: critic update (all robots) → actor step for group 1 → group 2 → group 3 → group 4.
- All four arrows write into the same network θ; emphasize "one network".
- Beside it, the four groups as colored clusters.

*Source: Abe et al., Algorithm 1, Tables 3–5.*

---

## Slide 5. EG reproduced

**Headline:** We reproduce EG's numbers. EG is the only one of our 48 finished runs that keeps A1, Go1, H1 and G1 together.

- **URMA**, with 4 actor steps per update to match EG's compute: 42.7 over 3 seeds (42.4 / 35.5 / 50.3). The paper's compute-matched IQL scores 44.2.
- **EG** (seed 0): 50.8, against the paper's 51.2–52.0. URMA on the same seed: 42.4.
- **What EG learns:** 13 of 16 robots, including A1 (53), Go1 (53), H1 (54), G1 (77) and Talos (111).
- **What EG loses or misses:**
  - NAO, learned and then lost: 89 at 48k → 19 at 100k;
  - Cassie: peaks at 30, ends at 2;
  - the hexapod: never learned.

**Visual:** one table row per robot, return at 100k, URMA seed 0 vs EG seed 0. Highlight A1, Go1, H1 and G1 (gains) and NAO (loss).

| | A1 | Go1 | Go2 | ANYmal B | ANYmal C | Barkour v0 | Barkour vB | Badger | Bittle | H1 | G1 | Talos | OP3 | NAO | Cassie | hexapod | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| URMA s0 | 14 | 15 | 52 | 48 | 44 | 47 | 55 | 53 | 45 | 2 | 2 | 110 | 100 | 89 | 2 | 0 | 42.4 |
| EG s0 | 53 | 53 | 52 | 48 | 44 | 46 | 55 | 53 | 45 | 54 | 77 | 111 | 102 | 19 | 2 | 0 | 50.8 |

*Note: EG has one seed so far; seeds 1 and 2 are planned.*

*Supporting runs: URMA-IQL-A4 s0, T1-URMA-A4 s1/s2, EG-IQL-M4 s0.*

---

## Slide 6. Why EG is not enough

**Headline:** EG changes *when* robots update the network, not *which parameters* they share.

1. **Scheduling, not separation:** every group still writes into every parameter.
2. **Wrong granularity:** conflicts sit below the robot level, between joints.
3. **Frozen grouping:** who conflicts with whom changes within a few updates, but EG's groups are fixed before training.
4. **Blind to data quality:** each robot's own data mixes suboptimal and expert behavior.

**Visual:** four quadrants with icons, one per limitation.

---

## Slide 7. Limitation 1: scheduling, not separation

**Headline:** Group steps still overwrite each other, because every group shares every parameter.

- **The steps chain on the same weights:** θ₁ = θ − η∇L_G1(θ), then θ₂ = θ₁ − η∇L_G2(θ₁). Group 2's step changes the function group 1 just fitted.
- **No parameter belongs to a family or a robot.** EG keeps URMA's single network (0.43M parameters).
- **EG still loses robots it has learned:** NAO 89 → 19 (48k → 100k), Cassie 30 → 2.
- **Fixing gradients inside shared parameters helps even less:**
  - robot-level PCGrad gains +3.0 in the paper (39.6 vs 36.6);
  - our joint-level PCGrad on URMA's shared decoder scores 37.0 vs 42.4, and loses Talos (18 vs 110).

**Visual:**
- Left: two update arrows (group 1, group 2) hitting the same parameter box.
- Right ("our goal"): updates landing in different expert boxes, with a shared box in the middle.

*Supporting runs: EG-IQL-M4 s0, URMA-IQL-A4 s0, T3-PCGrad s0 (URMA + PCGrad); Abe et al. Table 3.*

---

## Slide 8. Limitation 2: wrong granularity

**Headline:** Harmful and helpful interactions are organized by joints, not by robots, and an EG split barely separates them.

- **Test 1** (URMA, seeds 1 and 2): we measured how each joint's gradient affects every other joint's, repeatedly over training.
- **Protocol:** each two-way split of the joints is fitted on half of the measurements and scored on the other half.
- **Share of interaction kept inside groups** (lower harmful and higher helpful is better):

  | Split | Harmful kept | Helpful kept |
  |---|---|---|
  | Learned from gradients | 16–18% | 75–79% |
  | EG groups | ~55% | ~60% |
  | Morphology | 61% | 70% |
  | Random | 50% | 50% |

- **The joints' morphology descriptors could not predict the gradient-learned split.**

**Visual:** grouped bars (harmful kept, helpful kept) for the four splits. The ideal corner is low harmful, high helpful.

*Supporting runs: T1-URMA-A4 s1/s2 (measurements from 1k to 50k updates), T1t-early s1/s2, T1t-mid s1.*

---

## Slide 9. Limitation 3: frozen grouping

**Headline:** Who conflicts with whom changes within about 10 updates, so any grouping fixed in advance goes stale.

- **Fast turnover:** about 70% of the reproducible joint-pair interaction signal changes within 10 updates.
- **Old measurements don't predict:**
  - interactions measured 3k or more updates earlier predict about none of the later transfer;
  - measured at the same checkpoint, they predict 0.44–0.83 of it.
- **Fixed tables built from measured conflict do no better than random ones** (decoder experts, seed 1, at 100k):

  | Decoder routing | Return |
  |---|---|
  | One shared decoder | 35.2 |
  | Split by robot | 38.7 |
  | Split by measured conflict | 39.2 |
  | Random split | 40.1 |

- **Slow averaging doesn't help either:** a compatibility loss with slowly averaged per-joint-type estimates scores the same as a plain MoE (47.5 vs 47.3, 3 seeds, 100k).
- **EG's own authors name static grouping as their main limitation.**
- **Takeaway:** separation itself helps (+4–5 over a shared decoder); what fails is choosing the partition once.

**Visual:** bars for the four decoder routings above, plus a small sparkline showing the signal decaying within 10 updates.

*Supporting runs:*
- *Test 1 measurements (as on slide 8).*
- *T2b shared / camps / random / robot (s1); T2c persistent-camps and persistent-random (s1).*
- *MORSE-IQL-v2-noconf-A4 s0–s2 vs T3-compat s0–s2.*

---

## Slide 10. Limitation 4: blind to data quality

**Headline:** Each robot's own data mixes two behaviors, and EG only separates robots.

- **Every robot has 700k suboptimal and 300k expert transitions** (verified for all 16).
- **The two behaviors differ sharply:** suboptimal bipeds fall within tens of steps, while expert episodes walk the full 1,000 steps.
- **EG's own analysis:** conflict rises with the share of suboptimal data (0.16 → 0.27 → 0.32), but EG's remedy acts on groups of robots.
- **IQL's advantage weight** decides how much a sample counts, not which parameters it updates.

**Visual:** one robot's data as a bar, 70% light (suboptimal) and 30% dark (expert), with both parts pointing into the same parameters.

*Note: we have not yet measured whether this mix causes harmful interference. The Talos experiment tests it.*

*Supporting data: our dataset check (first 700k vs last 300k transitions, all 16 robots); Abe et al. Fig. 2a.*

---

# Part 2: What our runs show

## Slide 11. Key insight: sharing vs separation depends on the robot

**Headline:** A shared core gives A1 and Go1 their transfer; a separated (MoE) core gives the bipeds theirs. No architecture we ran gets both.

**Table:** robots kept at 100k, over all finished runs.

| Robot | Shared core (12 runs) | MoE core (36 runs) |
|---|---|---|
| A1 | 8 / 12 | 0 / 36 |
| Go1 | 7 / 12 | 0 / 36 |
| H1 | 3 / 12 | 34 / 36 |
| G1 | 8 / 12 | 30 / 36 |
| Talos | 4 / 12 | 31 / 36 |

- **A1 under a shared core:** learned at some point in 12 of 12 runs.
- **A1 under an MoE core:** learned in only 1 of 36 runs, and only at 2k updates.
- **Go1 under an MoE core:** learned briefly in 23 of 36 runs (between 23k and 70k) and lost every time.
- **Where MoE cores leave A1 and Go1:** about 14.4, which is EG's single-robot level (14.6 / 14.5). Separating the core removes exactly the positive transfer EG reported.
- **The bipeds go the other way,** matching EG's negative transfer for H1 and G1.
- **Only EG (seed 0) keeps A1, Go1, H1 and G1 together.**

**Visual:** diverging bar chart. For each robot, % of runs that kept it with a shared core (orange) vs an MoE core (purple).

*Supporting runs, shared core (12):*
- *URMA-IQL-A4 s0; T1-URMA-A4 s1/s2.*
- *URMA + per-joint calibration s0–s2.*
- *EG s0.*
- *URMA + PCGrad s0.*
- *Decoder-only MoE with a shared core: T2b shared / camps / random / robot (s1).*

*Supporting runs, MoE core (36):*
- *CAP 4×64, 8×64, 8×64 top-2, 8×64 top-4 (s0).*
- *MoE v2 (s0); MoE v2 without the conflict loss (s0–s2).*
- *v3 (s0–s2, MI 0.0005, no MI); v4 (s0–s2); v5 (s0–s2).*
- *v6 ½ and ¾ width (s0–s4 each); v6 robot-level MI (s0–s1).*
- *T3 MI and MI-joint-equal (s0); T3 compatibility (s0–s2).*

*Speaker notes:*
- *The shared-core runs cover seeds 0–2. A1 is kept in 6/6 seed-1 runs, 2/4 seed-0 runs and 0/2 seed-2 runs.*
- *The MoE cores also have more parameters. It is still open whether A1 and Go1 need full sharing or the pressure of a small network. The dense control with 4.4M parameters will separate the two.*

---

## Slide 12. Insight: routed cores rediscover EG's groups

**Headline:** Without any group label, routed cores group robots by EG's morphology groups. But sharing a few routed experts still doesn't give A1 and Go1 their transfer.

- **Data:** 10 cores trained with only a load-balancing loss (v6, ½ and ¾ width, seeds 0–4), at the end of training.
- **Measure:** overlap between two robots' expert sets (1 = the same 4 experts; random sets give 0.5).
- **Same EG group:** 0.74; **different groups:** 0.43. Overlap is higher within groups in 10 of 10 runs.
- **Top pairs:** Go1–Go2 0.94, H1–Talos 0.93, A1–Go2 0.93, Go2–Barkour v0 0.93, OP3–NAO 0.89. The 15 most-overlapping pairs are all within one EG group.
- **Nearest robot:** for 13 of 16 robots, the robot with the most similar expert set is in its own EG group. The exceptions are Bittle→OP3, Cassie→Talos and G1→H1.
- **The catch:** A1 and Go1 share 0.92–0.94 of their core experts with Go2, yet they are never learned.
- **Takeaway:** the task already wants EG's groups. What the routed core lacks is a shared path for every robot.

**Visual:** two bars (0.74 within groups vs 0.43 across, dashed line at 0.5) and a short list of the top pairs.

*Supporting runs: MORSE-v6-w50 s0–s4, MORSE-v6-w75 s0–s4 (routing logs).*

---

## Slide 13. Insight: robot-level MI removes sharing

**Headline:** Mutual-information (MI) routing on robot identity gives every robot its own expert, so MI should be group-level, weak and bounded.

- **MI** here rewards routing that reveals the label (the robot or its group), i.e. robots specializing on different experts.
- **Robot-level MI on the core** (weight 0.005) puts each robot on one expert:
  - H(E|robot) is 0.00–0.01 and MI is 1.95–2.01;
  - the top expert's weight is 1.00.
- **With load balancing only:**
  - H(E|robot) is 1.47–1.72;
  - each robot uses about 3.4 experts.
- **It did not hurt returns:** 50.7 / 45.1 vs 50.6 / 38.4 for the balance-only core (seeds 0 / 1, at 100k). Separation is fine; what's missing is a sharing path.
- **Strong MI hurts:** joint-type MI at 0.005 scores 41.3, vs 48.8 at 0.0005 and 48.0 with no MI (v3, seed 0).
- **Even weak MI sharpens routing:**
  - decoder MI at 0.001 leaves 1.3–1.5 effective experts per joint, vs 2.7 with balance only;
  - this happens although its router gradient was only 0.04–0.09× the task gradient.

**Visual:** routing weights over 4 experts for one example robot (Go2): balance only 0.37 / 0.27 / 0.26 / 0.11, robot-level MI 0.999 / 0.001 / 0 / 0.

*Supporting runs:*
- *MORSE-v6-w50-robotMI s0–s1 vs MORSE-v6-w50 s0–s1.*
- *MORSE-v3 s0, v3-mi0005 s0, v3-noMI s0.*
- *CAP-8x64-top4 s0 vs MORSE-v6 runs (decoder statistics).*

---

## Slide 14. Insight: model size is not the lever

**Headline:** Wider experts did not reliably help over five paired seeds. The question is how parameters are shared, not how many there are.

- **v6, ½ vs ¾ expert width** (1.4M vs 2.2M parameters, same seeds):
  - at 100k, 47.6 vs 48.5;
  - paired difference +0.9 (per seed: −7.9, +17.5, +3.2, −2.3, −6.0; standard deviation 10.2);
  - averaged over 50k–100k: +2.3 (standard deviation 6.7).
- **The earlier 3-seed lead of ¾ width came from seed 1 alone.**
- **What did hold:** separated capacity beats one shared network early. MoE vs URMA at 50k: +6.3 / +14.2 / +7.1 (seeds 0 / 1 / 2).
- **Takeaway:** hold the parameter count fixed (16 experts in every variant) and change the sharing structure. Keep a dense control with the same total parameters.

**Visual:** paired dots per seed (½ and ¾ width) joined by lines, five seeds.

*Supporting runs: MORSE-v6-w50 s0–s4, MORSE-v6-w75 s0–s4; URMA (3 seeds) vs MORSE-IQL-v2-noconf-A4 s0–s2.*

---

## Slide 15. Insight: the router hears little from the task

**Headline:** With top-k routing, an expert the router doesn't pick gets no learning signal from the task, so routing freezes early.

- **Exact top-k:** experts outside the top-k receive zero task gradient. The router only learns which of its current experts is best, never whether an unpicked one would be better.
- **Routing freezes early:** the median robot's core expert set stops changing between 2.75k and 39.75k updates, across 8 v6 runs.
- **Early on, auxiliary losses can take over:**
  - in a short pilot run, the router's task gradient fell from 0.0055 to 0.0001 within 50 updates;
  - an auxiliary routing loss can then dominate.
- **Our fix:**
  - both methods add learned noise to the routing during training (Mod-Squad style, minimum scale 0.01);
  - Idea 1 also scores every expert, including the ones not picked.

**Visual:** 16 experts with the top 4 lit; the 12 unlit ones labeled "no gradient".

*Supporting runs: MORSE-v6-w50/w75 s0–s2 and MORSE-v6-w50-robotMI s0–s1 (routing logs); dry-v3-s0.*

---

## Slide 16. Insight: how to read results

**Headline:** Single robots flip between zero and full return, so we compare paired seeds and per-robot outcomes.

- **Six robots decide the mean:** Go2, G1, Talos, NAO, Cassie and the hexapod. Each moves the 16-robot mean by 3–7 points.
- **Seed-to-seed spread:** the 100k mean varies by 4–12 points between seeds.
- **A common plateau:**
  - seven seed-0 runs of different designs all end at 54.0–54.2 on the 15 robots without the hexapod;
  - our best single run (56.4) is that plateau plus the hexapod.
- **Paired seeds:** the same seed gives the same minibatches and the same critic (the IQL critic never sees the actor).
  - Actor loss over the first 500 updates is within 1.489–1.502 across 12 seed-0 runs.
  - The new runs log a minibatch checksum and the critic norms. These were identical across all 7 seed-0 variants in our pilot runs.

**Visual:** a per-robot outcome grid (kept / learned then lost / never learned) for 3–4 example runs.

*Supporting runs: CAP-* family (seed 0), v4, v5, v6, URMA, MoE (seed-paired tables); V7 pilot runs (seed 0).*

---

# Part 3: Two proposed methods

## Slide 17. From insights to design

**Headline:** Keep EG's insight that morphology groups share, but change its mechanism: share through parameters and decide online.

| What we learned | Design consequence |
|---|---|
| A shared core gives A1/Go1 their transfer; separation gives the bipeds theirs | Give every robot both a shared path and experts it can separate into |
| Routed cores rediscover EG's groups | Use EG's groups as a built-in structure, not an update schedule |
| Conflicts are finer than robots and change fast | Decide sharing per transition and per joint, from the current update |
| Robot-identity MI gives every robot its own expert; strong MI hurts | Group-level MI, weight 0.001, its gradient capped at 0.5× the task gradient |
| Model size is not the lever | Same 16 experts and same parameter count in every variant |
| The top-k router hears little from the task | Routing noise during training; score all experts |

- **Idea 1, compatible-set routing:** decide online which experts an update should share.
- **Idea 2, hierarchical shared experts:** build sharing across all robots and within each EG group into the parameters.

---

## Slide 18. Common architecture

**Headline:** Both ideas use the same sparse mixture-of-experts policy and the same training protocol.

- **Core:** 16 experts, each 400 → 384 → 192 → 128 (LayerNorm, ELU). One routing decision per transition (whole body).
- **Decoder:** 16 experts, each 260 → 64 → 1. Routed per joint, top-4.
- **Router:**
  - input: the robot's morphology descriptor plus its state (log-scaled);
  - output heads start at zero;
  - training adds learned Gaussian noise to the logits (σ = softplus(·) + 0.01); evaluation is noise-free.
- **Load balancing:** KL(p̄ ‖ uniform), weight 0.01 per block, as in our best historical run.
- **Training:**
  - IQL with 4 actor steps per update (matching EG's 4 group steps);
  - 1,024 transitions per robot per update, 100k updates;
  - about 4.4M policy parameters.

**Visual:** encoders → core (16 experts, router) → decoder (16 experts, routed per joint) → joint actions.

---

## Slide 19. Idea 1: compatible-set routing (concept)

**Headline:** Route each update toward the experts it agrees with, and keep several of them in play.

- **The question:** which experts should this transition (core) or this joint (decoder) update?
- **How we answer it:**
  - from the current minibatch, compare the unit's gradient on each expert with that expert's current update from other transitions;
  - avoid the experts it conflicts with;
  - spread over all non-conflicting experts, so it doesn't collapse onto one.
- **Why it fits our findings:**
  - it works where conflict lives (per state and per joint);
  - it adapts every update;
  - it also acts within one robot's mixed-quality data.

**Visual:**
- One unit u and five experts, with arrows labeled by gradient cosine: 0.7, 0.2, 0.02, 0, −0.6.
- The −0.6 expert is crossed out.
- Target bars below: ¼, ¼, ¼, ¼, 0.

---

## Slide 20. Idea 1: method

**Headline:** A fixed uniform target over the experts an update reliably doesn't conflict with.

1. **Score:** s_ue = cos(g_ue, G_e,−t(u)).
   - This compares unit u's gradient on expert e with e's actual update from other transitions, using the real routing and advantage weights. u's own transition is left out.
   - All 16 experts are scored, including those not picked. This costs one extra backward pass on the first actor step, with 64 (core) and 32 (decoder) transitions per robot.
2. **Label** using two independent references (even vs odd transitions), with τ = 0.05:
   - non-conflicting if both scores are ≥ −τ;
   - conflicting if both are < −τ;
   - otherwise unknown, and left out.
3. **Target:** q_u is uniform over the non-conflicting experts.
4. **Loss:** L_compat = mean over units of KL(q_u ‖ p̃_u).
   - p̃ is the router's noisy probabilities, before top-k, renormalized over the experts that have a label.
   - The loss splits into two effects:
     - −log p̃(A) moves probability into the compatible set;
     - KL(U_A ‖ p̃(·|A)) keeps several compatible experts in play.
5. **Objective:** L = L_task + 0.01·L_balance,core + 0.01·L_balance,decoder + λ·(L_compat,core + L_compat,decoder).
   - λ = 0.01, reached after a 5k warm-up and a 5k ramp. It is checked in a pilot run, then frozen.

**Equations (LaTeX):**

$$s_{ue}=\cos\big(g_{ue},\,G_{e,-t(u)}\big),\qquad q_{ue}=\frac{\mathbf 1[e\in\mathcal A_u]}{|\mathcal A_u|},\qquad \mathcal L_{\text{compat}}=\mathbb E_u\,\mathrm{KL}\big(q_u\,\|\,\tilde p_u\big)$$

$$\mathrm{KL}(U_{\mathcal A}\,\|\,\tilde p)=-\log\tilde p(\mathcal A)+\mathrm{KL}\big(U_{\mathcal A}\,\|\,\tilde p(\cdot\mid\mathcal A)\big)$$

**Example:** scores [0.7, 0.2, 0.02, 0, −0.6] give the target [¼, ¼, ¼, ¼, 0]. Neutral experts stay available.

*Related work: comparing a token's gradient with its expert's mean gradient follows STGC (ICLR 2025). The compatible-set target and its use for embodiment and data-quality differences are ours.*

---

## Slide 21. Idea 2: hierarchical shared experts (concept)

**Headline:** Build sharing into the parameters: one expert for all robots, one per EG group, and routed experts for everything else.

- **The same 16 core experts, reassigned:** 1 global, 4 group-shared (one per EG group) and 11 routed.
- **Every transition uses 4 experts:** the global expert, its group's expert, and the top-2 routed experts.
- **What each level is for:**
  - **Global expert:** transfer across groups, which is what A1 and Go1 get from a shared core.
  - **Group expert:** what one EG group shares, which routed cores already rediscover.
  - **Routed experts:** what must be kept apart, which is what the bipeds get from MoE cores.
- **Origin:** inspired by DeepSeekMoE's always-active shared experts. The group-shared level and the fixed mixing weights are ours.

**Visual:**
- Three tiers:
  - a global expert (gray) used by all 16 robots;
  - four group experts, each in its EG group's color with its robot icons;
  - 11 routed experts, of which the router picks 2.
- Output = 0.25·global + 0.25·group + 0.5·routed mix.

---

## Slide 22. Idea 2: method

**Headline:** Supervise only the routed experts, and keep MI weak and bounded.

- **Core output:** h(x, r) = 0.25·S_global(x) + 0.25·S_g(r)(x) + 0.50·Σ_{e ∈ top-2} p̃_e(x, r)·F_e(x).
  - S_global is the global expert and S_g(r) the expert of robot r's EG group.
  - F_e are the routed experts; p̃ is renormalized over the top-2.
- **EG groups (M = 4):** {A1, Go1, Go2, Barkour v0, Barkour vB, Badger, Bittle, hexapod}, {ANYmal B, ANYmal C, Cassie}, {H1, Talos}, {G1, OP3, NAO}.
- **Router:** covers only the 11 routed experts, and any robot may pick any of them.
- **Losses on the routed experts only:**
  - MI: −0.001·I(G_EG; E_routed), on the noisy probabilities before top-k;
  - balance: 0.01·KL(p̄_routed ‖ U₁₁).
  - The shared experts get task gradients only.
- **Decoder unchanged:** 16 experts, top-4 per joint, MI on the (robot, joint) label, balance 0.01.
- **MI gradient cap, per router:** g = g_task + g_balance + c·g_MI, with c = min(1, 0.5‖g_task‖ / ‖g_MI‖). No gradient flows through c.

**Equations (LaTeX):**

$$h(x,r)=0.25\,S_{\text{global}}(x)+0.25\,S_{g(r)}(x)+0.50\!\!\sum_{e\in\mathrm{Top2}(p(x,r))}\!\!\tilde p_e(x,r)\,F_e(x)$$

$$\mathcal L_{\text{MI,core}}=-0.001\,I(G_{\text{EG}};E_{\text{routed}}),\qquad c=\mathrm{sg}\Big[\min\Big(1,\tfrac{0.5\,\|g_T\|}{\|g_S\|+\epsilon}\Big)\Big]$$

**Architecture variants** (all have 16 core experts and use 4 per transition):

| Variant | Experts | Used per transition | Output mixing |
|---|---|---|---|
| Original (A, B, C) | 16 routed | top-4 | routed 1.0 |
| Global only | 1 + 15 routed | 1 + top-3 | 0.25 / 0.75 |
| Group only | 4 + 12 routed | 1 + top-3 | 0.25 / 0.75 |
| **Hierarchical (A-H, C-H)** | 1 + 4 + 11 routed | 1 + 1 + top-2 | 0.25 / 0.25 / 0.5 |

---

## Slide 23. How each idea answers EG's limitations

| | EG | Idea 1: compatible-set routing | Idea 2: hierarchical shared experts |
|---|---|---|---|
| What is separated | actor steps | parameters (experts) | parameters (experts) |
| Unit of decision | robot | transition (core), joint (decoder) | EG group (shared experts) + transition (routed experts) |
| When it is decided | once, before training | every update, from current gradients | groups fixed, routing learned |
| Sharing across EG groups | everything shared | where updates agree | global expert, always |
| Mixed data quality within a robot | not addressed | scored per transition | not targeted |
| Extra cost | M sequential actor steps | one scoring pass (first actor step) | none beyond the MoE |

---

## Slide 24. Experiments: the first 37 runs

**Headline:** 37 runs answer the two main questions; everything else waits for a positive result.

| Variant | Seeds | Question |
|---|---|---|
| A: original MoE, balance only | 3 | matched baseline |
| B: compatible-set routing | 3 | does Idea 1 beat A? |
| C: original MoE + MI | 3 | MI without shared experts |
| A-H: hierarchical, no MI | 3 | does the architecture help on its own? |
| **C-H: hierarchical + MI** | 3 | the full Idea 2 |
| EG (seeds 1–2) | 2 | completes EG's 3 seeds |
| Best historical configuration (seeds 1–2) | 2 | completes the 56.4 run's 3 seeds |
| URMA | 3 | one-shared-network reference |
| Dense URMA, same 4.4M parameters | 3 | is it just parameters, or full sharing? |
| Talos only: A/B × expert-only (300k) or mixed (1M) data | 12 | does Idea 1 help within one robot? |

- **Idea 2's design is 2×2:** {original, hierarchical} × {no MI, MI}, i.e. A, C, A-H and C-H.
- **Priority:** A-H and C-H run first.
- **Deferred until a result justifies them:**
  - global-only and group-only variants;
  - repeating the data-quality test on Go2;
  - the robot-level MI and robot-private controls;
  - B's controls: shuffled labels, conflict-only loss, uniform target.

**Visual:** a 2×2 grid for Idea 2, and a separate box for Idea 1 (16 robots + Talos).

---

## Slide 25. Predictions and success criteria

**Headline:** Each idea makes a specific, checkable prediction.

- **Idea 2 (C-H):**
  - it keeps A1 and Go1 (through the shared paths) and H1, G1 and Talos (through the routed experts) in the same run;
  - it beats both C and A-H;
  - if the dense 4.4M control also keeps A1 and Go1, full sharing explains them, not the hierarchy.
- **Idea 1 (B):**
  - it beats A on the 16 robots, or its gain over A is larger on Talos's mixed data than on expert-only data;
  - shuffled labels should remove the gain.
- **Screening rule:**
  - a better 3-seed mean, and better on at least 2 of 3 paired seeds;
  - always report per-robot outcomes and robots that were learned and then lost.
- **Paired seeds** share minibatches and critics, so differences between variants are not critic noise.

---

## Slide 26. Summary

- **EG** shows that morphology groups matter, but it only reorders updates on one shared network, with fixed robot-level groups.
- **Our runs show:**
  - sharing and separation help different robots (A1/Go1 vs the bipeds);
  - routed cores rediscover EG's groups;
  - conflicts are finer than robots and change fast;
  - model size is not the lever.
- **Idea 1** decides sharing per update from gradient compatibility.
- **Idea 2** builds sharing across all robots and within each EG group into the parameters, with routed experts for the rest.
- **Next:** implement the hierarchical experts, then launch the 37 runs, starting with A-H and C-H.

---

# Appendix

## A. Run glossary

| Name | What it is |
|---|---|
| URMA | one shared network (no experts); 4 actor steps per update |
| URMA + calibration | URMA plus a per-joint output scale and bias |
| EG | URMA network with Embodiment Grouping updates (M = 4) |
| URMA + PCGrad | URMA with joint-level gradient projection on the decoder |
| MoE v2 / no conflict loss | 4 dense full-width core experts + 4 decoder experts |
| CAP family (seed 0) | same core as MoE v2; decoder 4×64 or 8×64 dense, or 8×64 top-2 / top-4. **56.4 run** = 8×64 top-4 |
| T2 runs | shared core + decoder experts with fixed routing tables; branched from URMA seed 1 at 2k updates |
| T3 runs (seed 0) | routing-loss tests: MI, compatibility, PCGrad |
| v3 / v4 / v5 | joint-routed cores: narrow / ½ width / ¾ width with top-2 |
| v6 | body-routed core, 8 experts top-4, ½ or ¾ width, balance-only core; decoder joint MI 0.001 |
| v6 robot-MI | v6 ½ with robot-level MI 0.005 on the core |

## B. Definitions

- **Return:** mean of the last 10 evaluations (one deterministic episode per robot every 100 updates).
- **Learned:** return ≥ half of the best return observed for that robot. The thresholds were frozen before these comparisons.
- **Kept:** still learned at 100k updates.
- **Learned then lost:** the 20-evaluation average crossed the threshold at some point, but the final return is below it.
- **Expert-set overlap:** Σ_e min(f_i(e), f_j(e)) / 4, where f(e) is the share of a robot's states whose top-4 includes expert e. Random sets of 4 out of 8 experts give 0.5.
