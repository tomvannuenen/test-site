# Neural generalization vs. superposition in Qwen3

A small, reproducible pilot for one mechanistic-interpretability question, prompted by
Nick Cammarata's essay [*Neural Networks Are Already Clean*](https://already-clean.goodfire.com/):

> When a single MLP neuron fires in apparently unrelated contexts, does it carry several
> independent features (**H1, superposition**), or one coherent higher-order abstraction
> that the contexts share (**H2, recursive generalization**)?

The goal is **not** to show that neurons are interpretable. It is a test that *could*
come out either way: a neuron can be described in one unified sentence and still lose
to H1. The design rule is that an appealing explanation never counts as evidence.
Only predictions on data the hypothesis-writer has not seen count. See
[`PROTOCOL.md`](PROTOCOL.md) for the pre-registered decision rules, and read it before
collecting data.

## What counts as a "neuron" here

Qwen3's MLP is SwiGLU: `down_proj( SiLU(gate_proj(x)) * up_proj(x) )`, with `x` the
post-attention-RMSNorm residual stream. **Neuron *i* of layer *l* is**

```
h_i = SiLU(w_gate_i · x) * (w_up_i · x)      # element i of the INPUT to down_proj
```

Its entire effect on the residual stream is `h_i · W_down[:, i]`. We read `h_i` with a
forward *pre*-hook on `mlp.down_proj`, and interventions edit that same tensor. That is
"immediately before the down projection". `h_i` is signed (`up_proj` can be negative), and
the two signs write in opposite directions, so each candidate is a
`(layer, neuron, polarity)` triple, e.g. `L18N1234+`. `gate_proj`/`up_proj` outputs alone
are not treated as neurons. The unit tests check this definition directly against
`SiLU(gate(x)) * up(x)`.

## Hardware and memory

`python scripts/check_env.py` reports the device and estimated memory *before* anything
is downloaded:

| model | params | bf16 weights | est. peak (batch 4 × 256) | runs on |
|---|---|---|---|---|
| Qwen3-0.6B | 0.60 B | 1.2 GB | ~2 GB | anything (smoke tests) |
| **Qwen3-4B** (default) | 4.02 B | 8.0 GB | ~9.6 GB | 16 GB Apple-Silicon Mac (MPS), CUDA ≥ 12 GB, CPU with ≥ 16 GB RAM (slow) |
| Qwen3-8B | 8.19 B | 16.4 GB | ~19 GB | ≥ 24 GB unified / GPU memory |

`load_model` downloads `config.json` first, estimates peak memory, and refuses to load
weights that won't fit. Device order is CUDA, then MPS, then CPU. Discovery never computes
logits: it runs only the base model and stops after the last needed layer. It never
holds the full `[tokens × d_mlp]` activation matrix. Exact top-k/bottom-k and running
moments are streamed instead.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,notebook]"          # torch, transformers, numpy, pyyaml, matplotlib, pandas, pyarrow
pytest -q                                 # 28 tests, tiny random Qwen3, no downloads
python scripts/check_env.py               # hardware + memory table
python scripts/smoke_test.py --tiny       # whole read/ablate path, no downloads
python scripts/smoke_test.py --model Qwen/Qwen3-0.6B   # same on real weights (~1.2 GB)
```

## Procedure

```bash
CFG=configs/qwen3_4b.yaml     # or configs/qwen3_8b.yaml, configs/smoke_qwen3_0.6b.yaml

# Phase 1 — discovery (discovery split only)
python scripts/run_discovery.py --config $CFG
#   -> outputs/<run>/discovery/candidates.md   (strongest examples, readable)
#   -> outputs/<run>/discovery/phase3_subset.json (random subset, fixed now)

# Phase 2 — write H1 and H2 for each candidate (blind to how it was selected)
jupyter lab notebooks/01_inspect_and_hypothesize.ipynb   # or: python scripts/annotate.py --config $CFG --subset
python scripts/freeze_hypotheses.py                       # validates + hashes; no later edits

# Phase 3 — falsifiable tests, written before running the model on them
python scripts/prepare_evaluation.py --config $CFG --stage tests   # templates in research/test_sets/
#   ... write >= 30 items per neuron (see PROTOCOL.md §3) ...
python scripts/prepare_evaluation.py --config $CFG --stage blind   # locks test sets; blind held-out sheets
#   ... label research/blind_sheets/*.yaml without running the model ...
python scripts/evaluate.py --config $CFG

# Phase 4 — causal interventions with matched random-neuron controls
python scripts/run_interventions.py --config $CFG

# Phase 5 — analysis
jupyter lab notebooks/02_test_and_intervene.ipynb
```

### Phase 1: discovery
- **Corpus** (`neurogen/corpus.py`): about 80k tokens, 10k from each of 8 genres:
  encyclopedia (WikiText-103), fiction (TinyStories), code (HumanEval), news (AG News),
  instruction/QA (Dolly-15k), math word problems (GSM8K), science (SciQ support passages),
  and reviews (IMDB). Documents are sampled with a fixed seed and truncated to 256 tokens.
  The resolved dataset commit SHAs go into a manifest next to the cached corpus. The split
  is **by document**, stratified by genre: 75% discovery, 25% held-out. An offline
  40-document example corpus (`data/example/`, original text) is used by the smoke config.
- **Layers**: 15%, 50% and 85% of depth, i.e. layers 5, 18 and 30 of Qwen3-4B's 36.
- **Candidates**: a fixed rule, not hand-picked. The neuron must be selective (mean of
  top-40 ≥ 6 SD), and is ranked by √(genre entropy × token entropy) of its strongest
  examples, with one example per document. That rule excludes token detectors (same token
  everywhere) and single-genre neurons. Eight are taken per layer, plus **one random
  selective neuron per layer as a baseline**. Both polarities are scored.
- **Outputs**: per-candidate example cards with per-token context activations,
  full activation traces on both splits (candidates only), and MLP inputs at
  top-activating tokens for an optional geometry check. Held-out data goes to
  `heldout_SEALED/`.

### Phase 2: hypotheses
For each neuron the researcher records H1 (≥ 2 features, each tied to example ids),
H2 (one abstraction, described *without* listing H1's features), and what each predicts
**active and inactive**. They also record example ids neither explains, a "neither" note,
and prior credences. Records that forbid nothing cannot be frozen. Freezing stores a
SHA-256, and every later stage checks it.

### Phase 3: predictive tests
- **Authored test sets** (YAML): each item names a target substring and a category
  (`shared_positive`, `h1_only`, `h2_only`, `negative`, `counterexample`), plus a definite
  prediction from each hypothesis. The activation measured is the max over the target's tokens.
- **Blind held-out sheets**: held-out contexts with activations hidden. They mix the
  strongest held-out activations, *token-matched* low-activation contexts, and random tokens.
- **Scoring**: accuracy, balanced accuracy with bootstrap CIs, AUC, coverage,
  counterexample false-positive rate, and an exact sign test on the items where H1 and H2
  disagree. The verdict (`supports H1` / `supports H2` / `neither` / `inconclusive`)
  follows the pre-registered rule in `PROTOCOL.md` §4. H2 cannot win without
  discriminating items, however good its story (there is a test for this).

### Phase 4: interventions
Zero-ablation, mean-ablation and scaling of `h_i` at chosen token positions. Reported:
KL(baseline‖edited) from the edit onward, with earlier positions checked to be unchanged;
the change in log-prob of the true next token; and the target's percentile against ≥ 50
random neurons in the same layer whose |activation| at that position is matched.
Also reported: dose-response, the similarity of full-vocabulary effects across contexts
H1 assigns to different features, and an approximate direct-logit readout (descriptive).
**Causal results are reported alongside the predictive verdict, never folded into it.**

### Phase 5: analysis
`notebooks/02_test_and_intervene.ipynb` shows predictions against observations, held-out
genre distributions, intervention plots, and a final table with predictive and causal
evidence in separate columns. That table lists every subset neuron (including "not
evaluated"), with verdicts cross-tabulated by selection type.

## Repository layout

```
neurogen/            package
  hardware.py        device detection, parameter count, memory estimate
  model.py           loading, SwiGLU neuron definition, tiny test model
  hooks.py           NeuronRecorder, NeuronEdit, Intervention (pre-hooks on down_proj)
  corpus.py          genre-labelled corpus, document-level split, manifest
  discovery.py       streaming top-k, candidate rule, traces, example cards
  hypotheses.py      hypothesis records, freezing, tamper checks
  predictions.py     test sets, target measurement, blind sheets
  analysis.py        metrics, sign test, pre-registered verdict, geometry
  interventions.py   ablation/scaling, KL, matched controls, dose-response
  viz.py             matplotlib + HTML token highlighting
  pipeline.py        orchestration and output layout
configs/             qwen3_4b.yaml (default), qwen3_8b.yaml, smoke_qwen3_0.6b.yaml
scripts/             check_env, smoke_test, run_discovery, annotate, freeze_hypotheses,
                     prepare_evaluation, evaluate, run_interventions
notebooks/           01_inspect_and_hypothesize, 02_test_and_intervene
data/example/        small offline corpus (original text)
research/            hypotheses/, test_sets/, blind_sheets/ — the researcher's records (commit these)
tests/               unit + end-to-end tests on a tiny random Qwen3
PROTOCOL.md          research protocol and decision rules
```

`outputs/` and `cache/` are git-ignored. `research/` is meant to be committed, so the
hypothesis → test → result chain can be audited.

## Reproducibility

- One seed (`seed:` in the config) drives corpus sampling, the split, baseline selection,
  the Phase 3 subset, control-neuron sampling and the bootstraps.
- Run directories are keyed by a hash of the model, corpus and discovery settings, and
  pass-1 statistics are cached. Pin `model.revision` to a commit SHA for a publishable
  run. The corpus manifest records the dataset SHAs.
- MPS/CUDA kernels are not bit-for-bit deterministic. Expect tiny differences in
  activations across devices. Candidate selection is robust to this except near ties.

## Status and limitations

- **Verified:** 28 tests on a randomly initialised miniature Qwen3, using the real
  `transformers` Qwen3 classes. They check that the hook reads exactly
  `SiLU(gate(x))·up(x)`; that ablation changes only the intended
  `(position, neuron)` element and shifts `down_proj`'s output by exactly
  `−h_i·W_down[:, i]`; that edits are causal in position; that streaming top-k matches
  brute force; and the freeze/tamper checks, verdict logic and the full pipeline. All
  scripts and both notebooks have been executed end to end on the tiny model.
- **Not yet verified:** a run on real Qwen3 weights. The development environment could
  not reach the Hugging Face Hub. Run `scripts/smoke_test.py --model Qwen/Qwen3-0.6B`
  first, then the 4B config.
- The research limitations (small sample, corpus choice, subjective interpretation,
  author-chosen item difficulty, the neuron basis, model scale) are in `PROTOCOL.md` §7.
  This pilot says nothing about what fraction of Qwen3's neurons are interpretable.
