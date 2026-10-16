# Research protocol: superposition vs. recursive generalization in single neurons

Status: pilot protocol, v0.1. Fill in the dates and the model revision before you start
data collection. Any change to the decision rules after hypotheses are frozen must be
reported as a deviation.

## 1. Question

When one MLP neuron in a language model fires strongly in contexts that look
semantically unrelated, which explanation fits better?

- **H1 (superposition / polysemanticity):** the neuron responds to two or more
  relatively independent features A, B, …, and the apparent unrelatedness is real.
- **H2 (recursive generalization):** the neuron responds to one coherent abstraction
  C. A, B, … are instances of C, so the apparent unrelatedness reflects the
  interpreter's coarse categories, not the neuron's.

Terminology: strictly, "superposition" means a layer represents more features than it has
dimensions. A neuron responding to several features is *polysemanticity*, for which
superposition is the usual explanation. We use H1 for the neuron-level claim only. This
pilot does not test layer-level superposition.

**The two hypotheses are not mutually exclusive.** A neuron can carry an abstraction C
whose instances are also separately useful downstream. It can also respond to C plus an
unrelated feature D. And either description may be only one useful compression among
several. This protocol therefore asks which hypothesis *predicts better on the items
where they disagree*. It does not ask which one is "true". "Both fit and cannot be
separated" and "neither fits" are first-class outcomes.

## 2. Definitions

- **Neuron.** For Qwen3 (SwiGLU) the MLP computes `down_proj(SiLU(gate_proj(x)) * up_proj(x))`,
  where `x` is the post-attention-RMSNorm residual stream. Neuron *i* of layer *l* is
  `h_i = SiLU(w_gate_i · x) * (w_up_i · x)`, the *i*-th element of the input to
  `down_proj`. Its write to the residual stream is exactly `h_i · W_down[:, i]`.
  `gate_proj` and `up_proj` outputs are *not* treated as neurons.
- **Polarity.** `h_i` is signed: `w_up_i · x` can take either sign, and SiLU is slightly
  negative for negative inputs. Positive and negative values write in opposite directions
  along `W_down[:, i]`. Each candidate is a `(layer, neuron, polarity)` triple, and all
  activations are reported multiplied by the polarity's sign.
- **Active.** An activation is active if it exceeds `τ = 0.25 × median(top-40 discovery
  activations)` for that candidate. τ is computed from discovery data only, before any
  evaluation data exists. Continuous metrics (AUC) are reported alongside so that
  conclusions do not hinge on τ.

## 3. Procedure and leakage controls

| Step | Data the researcher can see | Control |
|---|---|---|
| 1. Discovery | Discovery split (75% of documents, stratified by genre) | Split is by **document**, seeded |
| 2. Candidate selection | none (automatic) | Pre-specified rule: selectivity ≥ threshold, ranked by √(genre entropy × token entropy). Plus 1 random selective neuron per layer as a **baseline** |
| 3. Phase 3 subset | none (automatic) | Seeded layer-stratified random sample, fixed **before** hypotheses are written |
| 4. Hypotheses | Discovery example cards only | Annotator is blind to selection type and metrics. Both H1 and H2 must be written, each with explicit "predicts active" **and** "predicts inactive" statements, plus prior credences |
| 5. Freeze | — | SHA-256 of each record. Evaluation refuses edited records |
| 6. Authored test sets | Frozen hypotheses only (no model access) | Every item carries a definite H1 and H2 prediction. Category consistency is validated |
| 7. Blind held-out sheets | Held-out contexts, **activations hidden** | Created only after the test set is locked (its SHA is recorded). Mixes top-activating, token-matched low-activating, and random tokens |
| 8. Scoring | Everything | Pre-registered rule (§4) |
| 9. Interventions | Everything | Matched random-neuron controls |

**Required item types per neuron (authored set), minimum 30 items:**

- ≥ 5 `shared_positive` (both predict active). These check that both hypotheses fit
  the basics.
- ≥ 5 `h2_only`: *novel* instances of the abstraction C that match none of H1's features.
  If you cannot write these, H2 is not distinguishable from "A or B", and you should
  record that.
- ≥ 5 `h1_only`: instances of feature A or B that *lack* C's defining property
  (surface-feature matches without the abstraction).
- ≥ 5 `negative`: both predict inactive.
- ≥ 5 `counterexample`: deliberately hard cases that are surface-similar to the strongest
  examples (same tokens, same genre, same syntax) but predicted inactive by at least one
  hypothesis.

Write the items *before* running the model on any of them. Exploratory probing with the
model is allowed only on sentences that are then **excluded** from scoring.

**Guarding against gerrymandered abstractions.** Any set of examples can be "unified" by
a disjunction ("things that are A or B") or by an abstraction vague enough to cover
anything. H2 earns credit only through `h2_only` and `h1_only` items, where it must
forbid or permit something that H1 does not. An H2 that never disagrees with H1 cannot
be supported, by construction (see `test_unified_story_alone_cannot_win`).

## 4. Pre-registered decision rule (per neuron)

Computed on the combined authored + blind items. Items labelled `unsure` count against
coverage and are excluded from accuracy. Parameters live in the config (`evaluation:`).

1. If both hypotheses have balanced accuracy < 0.65 → **neither**.
2. Else if there are fewer than 8 discriminating items (H1 and H2 make opposite
   definite predictions) → **inconclusive**.
3. Else if one hypothesis wins significantly more discriminating items (exact two-sided
   sign test, α = 0.05) and also has balanced accuracy ≥ 0.75 → **supports H1** or
   **supports H2**.
4. Else if both have balanced accuracy ≥ 0.75 → **inconclusive (both fit)**. This is
   consistent with non-exclusivity.
5. Otherwise → **inconclusive**.

Also reported, without changing the label: bootstrap 95% CIs for balanced accuracy,
AUC, coverage, the false-positive rate on counterexamples, and performance on blind
items split by source (top / token-matched / random).

### What would support or challenge each hypothesis

| Observation | Bearing |
|---|---|
| Novel instances of C, matching none of A/B (`h2_only`), activate the neuron | **For H2**, against H1 |
| Surface instances of A that lack C (`h1_only`) activate | **For H1**, against H2 |
| Neither `h2_only` nor `h1_only` items activate, but shared positives do | Both too broad. A narrower account is needed (often **neither**) |
| Token-matched held-out controls activate as strongly as top contexts | Neuron is closer to a token detector. Both semantic accounts are undermined |
| Counterexamples activate | Against whichever hypothesis predicted inactive |
| MLP inputs at top tokens form well-separated clusters aligned with A/B | Weakly compatible with H1 (descriptive only). An abstraction can be computed from distinct inputs |
| Ablation effects are similar across A- and B-contexts | Weakly compatible with a shared downstream role. Not evidence for H2's *semantic* content |
| Ablation has large, control-exceeding effects | The neuron matters causally here. **Silent on H1 vs H2** |

## 5. Causal evidence (Phase 4) and how it is kept separate

Interventions edit `h_i` immediately before `down_proj`: zero-ablation, mean-ablation
(discovery mean), and scaling. We report KL(baseline ‖ edited) at and after the edited
position, the change in log-probability of the true next token, and the percentile of
the target's KL against ≥ 50 random neurons in the same layer. Control neurons are
matched on |activation| at the edited position, within ±25%, widening if needed.

Causal results go in their own column of the summary table and **never** change the
predictive verdict. A neuron can be causally important and have no good semantic
description, or have a good description and no measurable effect. Single-neuron ablation
also underestimates importance whenever other units carry redundant information.

## 6. Reporting rules

- Report **every** neuron in the Phase 3 subset, including random baselines and
  neurons that were never fully evaluated (listed as "not evaluated" with the reason).
- Report verdict counts separately for heterogeneity-selected and random-baseline
  neurons. If interpreters "find" coherent H2 stories as often for random neurons,
  unified descriptions are cheap. That is a result in its own right.
- Report prior credences next to outcomes. Report how often the higher-prior
  hypothesis lost.
- Deviations from this protocol (changed thresholds, dropped items, re-written
  hypotheses as new versions) go in a deviations log with the reason.
- Do **not** extrapolate to "the fraction of interpretable neurons in Qwen3". The sample
  is ~24 candidates chosen by a heterogeneity rule from three layers.

## 7. Limitations (state these in any write-up)

- **Small sample, small corpus.** About 80k tokens across 8 genres (a few thousand in the
  bundled example corpus). Rare features will not show up in the top-k, and "heterogeneous" is relative
  to coarse genre labels and token identity.
- **Corpus selection.** The genres and sources (Wikipedia, TinyStories, HumanEval, AG
  News, Dolly, GSM8K, SciQ, IMDB) shape which neurons look heterogeneous. Wikitext has
  residual tokenization artifacts.
- **Subjective interpretation.** The hypotheses, test items, and blind labels all come
  from the researcher. Blinding and pre-registration reduce bias but do not remove it.
  Ideally a second person, who did not write the hypotheses, labels the blind sheets.
  Inter-rater agreement should then be reported.
- **Truncated view.** Top-k examples show only the extremes. Mid-range activations may
  carry different structure.
- **Item difficulty is chosen by the author.** A motivated author can write easy items
  for a favoured hypothesis. Counterexamples and the blind held-out sheets partly offset this.
- **Basis choice.** Neurons are a privileged basis only to the extent that SiLU gating
  makes them so. Features may be better captured by directions (e.g. SAE latents).
- **Model scale.** Qwen3-4B/8B results need not transfer to other models or scales.
- **Thresholding.** The binary active/inactive call depends on τ. AUC is reported to
  check robustness.
