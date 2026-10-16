"""End-to-end orchestration shared by scripts/ and notebooks/.

Output layout (``outputs/<run_id>/``)::

    discovery/                 <- everything the hypothesis-writer may look at
        stats_L{layer}.npz       pass-1 top/bottom-k + moments, all neurons
        candidates.json          selected candidates + selection metrics
        phase3_subset.json       seeded random subset for Phase 3 (fixed *before* hypotheses)
        cards/{cid}.json         strongest discovery examples per candidate
        candidates.md            the same, as one readable Markdown file
        traces.npz               per-token traces of candidates on the discovery split
        geometry.npz             MLP inputs at top-activating tokens (optional analysis)
    heldout_SEALED/            <- do not open until hypotheses are frozen
        traces.npz, blind_keys/{cid}.json
    evaluation/{cid}.json      Phase 3 scores and verdicts
    interventions/{cid}.json   Phase 4 results
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from . import discovery as D
from .config import ExperimentConfig, set_seed
from .corpus import Document, build_corpus
from .hypotheses import sha256_file
from .model import LoadedModel, resolve_layers


def run_dir(cfg: ExperimentConfig) -> Path:
    name = cfg.model.name.split("/")[-1]
    return cfg.path("outputs") / f"{name}_{cfg.fingerprint()}"


def split_corpora(docs: list[Document], tokenizer) -> tuple[D.TokenizedCorpus, D.TokenizedCorpus]:
    disc = [d for d in docs if d.split == "discovery"]
    held = [d for d in docs if d.split == "heldout"]
    return D.TokenizedCorpus.build(disc, tokenizer), D.TokenizedCorpus.build(held, tokenizer)


def load_corpora(cfg: ExperimentConfig, tokenizer):
    docs = build_corpus(cfg.corpus, tokenizer, cfg.seed, Path(cfg.root))
    return split_corpora(docs, tokenizer)


def choose_phase3_subset(candidates: list[dict], n: int, seed: int) -> list[str]:
    """Seeded, layer-stratified random subset. Chosen at discovery time, so it cannot
    depend on which hypotheses later turn out to look good."""
    rng = random.Random(seed + 7)
    layers = sorted({c["layer"] for c in candidates})
    by_layer = {l: [c["cid"] for c in candidates if c["layer"] == l] for l in layers}
    for l in layers:
        rng.shuffle(by_layer[l])
    out: list[str] = []
    while len(out) < min(n, len(candidates)):
        for l in layers:
            if by_layer[l] and len(out) < n:
                out.append(by_layer[l].pop())
    return out


def run_discovery(cfg: ExperimentConfig, lm: LoadedModel, progress: bool = True) -> Path:
    set_seed(cfg.seed)
    out = run_dir(cfg)
    ddir, hdir = out / "discovery", out / "heldout_SEALED"
    ddir.mkdir(parents=True, exist_ok=True)
    hdir.mkdir(parents=True, exist_ok=True)
    tok, model, dc = lm.tokenizer, lm.model, cfg.discovery
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    disc, held = load_corpora(cfg, tok)
    layers = resolve_layers(cfg.model, lm.n_layers)
    (out / "run.json").write_text(json.dumps(dict(config=cfg.to_dict(), layers=layers,
                                                  n_tokens_discovery=disc.n_tokens,
                                                  n_tokens_heldout=held.n_tokens,
                                                  device=lm.device), indent=2))
    if progress:
        print(f"layers {layers}; discovery {disc.n_tokens} tokens / {len(disc.docs)} docs; "
              f"held-out {held.n_tokens} tokens / {len(held.docs)} docs")

    # Pass 1 (cached)
    stats = {}
    if all((ddir / f"stats_L{l}.npz").exists() for l in layers):
        stats = {l: dict(np.load(ddir / f"stats_L{l}.npz")) for l in layers}
    else:
        stats = D.stream_layer_stats(model, disc, layers, dc.top_k, dc.batch_size, pad, lm.device, progress)
        for l, s in stats.items():
            np.savez_compressed(ddir / f"stats_L{l}.npz", **s)

    cands = D.select_candidates(stats, disc, dc.n_candidates, dc.min_selectivity, cfg.seed,
                                n_random_baseline=dc.n_random_baseline, dedupe_docs=dc.dedupe_docs)
    if not cands:
        raise RuntimeError("No candidates passed the selectivity filter; lower min_selectivity")
    for c in cands:
        s = stats[c["layer"]]
        topk = s["top_v"][:, c["neuron"]] if c["polarity"] == "pos" else -s["bot_v"][:, c["neuron"]]
        c["discovery_topk_signed"] = [float(x) for x in topk]
        c["active_threshold"] = float(cfg.evaluation.active_threshold_frac * np.median(topk))
        c["layer_mean"] = float(s["mean"][c["neuron"]])

    # Pass 2: traces on both splits; MLP inputs above the k-th largest discovery value.
    kth = {c["cid"]: c["discovery_topk_signed"][-1] for c in cands}
    tr_d = D.trace_candidates(model, disc, cands, dc.batch_size, pad, lm.device, keep_input_above=kth)
    tr_h = D.trace_candidates(model, held, cands, dc.batch_size, pad, lm.device)
    np.savez_compressed(ddir / "traces.npz", **tr_d["traces"])
    np.savez_compressed(hdir / "traces.npz", **tr_h["traces"])
    geo = {}
    for cid, pairs in tr_d["inputs"].items():
        if pairs:
            geo[f"{cid}__gid"] = np.array([g for g, _ in pairs])
            geo[f"{cid}__x"] = np.stack([x for _, x in pairs]).astype(np.float32)
    np.savez_compressed(ddir / "geometry.npz", **geo)

    md = ["# Discovery examples (discovery split only)\n",
          "Focus token in [[double brackets]]; ⏎ = newline. Generated automatically.\n"]
    # Iterate in cid order: the selection order would reveal which neurons are random baselines.
    for c in sorted(cands, key=lambda c: (c["layer"], c["neuron"])):
        cards = D.example_cards(tr_d["traces"][c["cid"]], disc, tok, c["polarity"], dc.top_k,
                                dc.context_window, dc.dedupe_docs)
        D.save_json(cards, ddir / "cards" / f"{c['cid']}.json")
        c["discovery_cards_sha"] = sha256_file(ddir / "cards" / f"{c['cid']}.json")
        md.append(D.cards_to_markdown(c, cards[:20]))
    (ddir / "candidates.md").write_text("\n".join(md))
    D.save_json(cands, ddir / "candidates.json")
    D.save_json(dict(cids=choose_phase3_subset(cands, cfg.evaluation.n_test_neurons, cfg.seed),
                     rule="seeded layer-stratified random sample of all candidates"),
                ddir / "phase3_subset.json")
    D.save_json(dict(discovery=[dict(doc_id=d.doc_id, genre=d.genre) for d in disc.docs],
                     heldout=[dict(doc_id=d.doc_id, genre=d.genre) for d in held.docs]),
                out / "corpus_index.json")
    if progress:
        print(f"{len(cands)} candidates -> {ddir / 'candidates.md'}")
    return out


def genre_array(corpus: D.TokenizedCorpus) -> np.ndarray:
    g = np.empty(corpus.n_tokens, dtype=object)
    for i, d in enumerate(corpus.docs):
        g[corpus.offsets[i]:corpus.offsets[i + 1]] = d.genre
    return g


def load_candidates(out: Path) -> list[dict]:
    return json.loads((out / "discovery" / "candidates.json").read_text())


def load_cards(out: Path, cid: str) -> list[dict]:
    return json.loads((out / "discovery" / "cards" / f"{cid}.json").read_text())
