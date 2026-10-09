# MORSE editorial review - 2026-10-04

Reviewed the complete manuscript fragment in `morse_cvpr_revised.tex`: abstract, introduction, related work,
preliminaries, method, equations, captions, and algorithm. It does not yet contain an experiments or conclusion
section. The version before this review is preserved in `morse_cvpr_before_editorial_review.tex`; the original
`morse_cvpr_draft.tex` is also preserved.

## Removed or consolidated

| Material | Change and reason |
| --- | --- |
| Repeated explanations of sharing between joints | Consolidated the motivation around one hip/ankle example and one method overview. Removed repetitions from related work, expert descriptions, and routing. The abstract remains self-contained. |
| Suboptimal-data ratio as a motivating explanation for conflict | Removed this sentence from the introduction. EG remains cited for its study of transfer and gradient conflicts. MORSE's motivation no longer depends on data quality causing conflict. |
| Three-part contribution recap | Removed because it repeated the preceding method paragraphs and presented the elementary gradient calculation as a separate contribution. The architecture and the adaptation of MI remain explicit in the introduction. |
| Robot-gradient decomposition and first-order loss-change equations | Removed from Preliminaries. They introduced notation and a local gradient-descent argument without establishing improved control under the actual optimizer. |
| Standalone sparse-gradient derivation | Replaced with a short explanation immediately after the top-k equation. The relevant fact remains: a joint contributes directly to the parameters of its selected experts. Shared modules and possible conflict on selected experts are still acknowledged. |
| Actor-step algorithm | Removed because it repeated the architecture equations and training paragraph. Router-only MI gradients, loss composition, and update frequency remain in the method. |
| Generic caveat about multi-task optimizer baselines | Removed the tangential sentence and its `xin2022mto` / `kurin2022unitary` citations from the narrative. Bibliography entries, where present, were not deleted. |
| STGC comparison | Removed the token-level vision-language comparison; closer robot-control and multi-task MoE references are more relevant here. Its existing bibliography entry was not deleted. |
| Statements describing comparisons and ablations as completed | Replaced the detailed introductory list with an explicit placeholder for verified full-method comparisons and findings. No new experimental result was added. |
| Low-level router implementation details | Moved activation, transformation, initialization, and tie-handling details out of the main narrative; preserved below for the implementation section or supplement. |
| Long author-only comments | Moved their substantive implementation and evidence notes into this review record. |

## Corrected or added

- The abstract describes MI as encouraging consistent routing preferences and expert diversity. It does not imply
  that the objective guarantees different choices for different joints within a robot.
- The method now states explicitly that equal robot weights define the MI prior; they do not equalize gradient
  magnitudes. It also states that the objective can assign the same experts to joints of the same robot.
- Added M2oE as a relevant prior method and added `liu2026m2oe` to `morse_new_refs.bib`. Its authors' project page
  describes a shared expert pool and attention-based selection for individual robot modules. This makes a broad
  claim to be the first joint/module-level expert-sharing method inappropriate.
- The M2oE description and bibliography were checked against https://ryuuchou17.github.io/m2oe/ . The PDF could
  not be fetched through the browser tool because of its size. No claim is made here about its exact sparsity,
  regularizer, or computational cost. A detailed novelty comparison still requires reading that full paper.
- Renamed the per-robot minibatch size from `b` to `n`, keeping `b` for the core/decoder pool index.
- Generalized the dataset wording to allow trajectories of varying quality, including expert-only datasets.
- Preserved the actual implementation scope: experts produce Gaussian means; the log-standard-deviation network
  is shared; MI is evaluated on dense probabilities with detached router inputs; all decoder experts are computed.

## Implementation details to retain in the paper's implementation section or supplement

These details were checked against `algorithms/morse/policy.py`, `moe.py`, and `updater.py` during this review.

- Each router has morphology and state branches. Both are two-layer MLPs with an ELU hidden activation.
- The state branch applies `symlog(v) = sign(v) log(1 + |v|)` elementwise before its MLP.
- Top-k selection resolves tied scores by expert index. Small random router-output initialization breaks the
  initially equal scores. Use the final experiment configuration when reporting the initialization scale.
- Core expert outputs are computed once per observation, then combined separately for each joint. The decoder
  combines per-joint expert outputs. The Gaussian mean and standard deviation are clipped according to the base
  policy configuration; report these bounds with the final hyperparameters.
- The local joint encoding is detached in the decoder input and decoder-router state input, as retained in the
  equations. All router inputs are detached for the auxiliary MI computation, which updates routers only.
- MI uses all transitions in the current actor minibatch and is applied at every actor optimization step,
  including additional actor steps within an IQL update.
- The handoff's proposed configuration uses eight experts, top-4 selection in both pools, expert hidden width 64,
  core output width 128, router hidden width 64, and router output initialization standard deviation 0.01.
  MI weights 0, 0.0005, and 0.005 are under comparison. These are recorded settings, not a verified final selection.

## Evidence and submission items that remain necessary

1. Replace the abstract and introduction result placeholders with completed results for the revised full method,
   including datasets, seeds, checkpoints, and baseline comparisons. Legacy decoder-only MoE results cannot be
   presented as results for joint-conditioned core-and-decoder MORSE.
2. The main claim needs comparisons that distinguish the effects of expert capacity, sparse selection,
   joint-conditioned representation routing, and MI. Report training budgets and parameter counts for these
   comparisons. The present editorial pass does not establish which ablations have finished.
3. Routing capability is not evidence of learned within-robot specialization. The handoff's 200-update subset
   dry run at MI weight 0.005 showed common selections across the joints of each robot. Total I(R,J;E) can be high
   even when I(J;E|R) is small. Preserve this distinction when reporting routing analyses; do not interpret high
   total MI as proof of joint-specific differentiation.
4. Keep the distinction between gradient support and optimization outcomes. Sparsity restricts direct contributions
   to expert gradients; it does not isolate the entire policy, guarantee aligned remaining gradients, or guarantee
   better rollout return. No claim of reduced inference computation is supported by the current implementation.
5. Replace both figure placeholders with actual diagrams. This fragment also needs the completed experiments,
   limitations, and conclusion sections before it is a submission-ready paper.
6. `morse_new_refs.bib` is an addition to the parent bibliography, not a complete bibliography for this fragment.
   Retain the parent entries and include the newly added M2oE entry.

No LaTeX compilation or new experiment was performed for this editorial review.

Static checks passed for balanced braces and environments, internal references, duplicate labels, and the new
bibliography key. The manuscript text was reduced by approximately 36% under a whitespace-based count that excludes
comments but includes mathematical notation. This is a content-length estimate, not a compiled page count.
