"""Phase 1: activation discovery.

Pass 1 streams over the *discovery* split and keeps, for every neuron in the
selected layers, the exact top-k and bottom-k activations (value + token id) and
running moments -- never the full activation matrix.

Candidate selection is an automatic, pre-specified rule (no hand-picking):
a (layer, neuron, polarity) is a candidate if it is *selective* (top-k mean is
many SDs from zero) and its top-k examples are *heterogeneous*: spread across
genres AND across distinct activating tokens. A neuron that fires on one token in
every genre (a token detector) or on many tokens within one genre is not
heterogeneous by this rule. A few selective neurons are also sampled uniformly at
random as a baseline, so we can check whether interpreters "find" unified
abstractions in neurons that were not chosen for looking heterogeneous.

Pass 2 records complete activation traces for candidates only, on both splits.
Held-out results are written to a separate directory that the hypothesis
notebook never reads.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .corpus import Document
from .hooks import NeuronRecorder, record_activations, stop_after_layer


# --------------------------------------------------------------------------- tokens
@dataclass
class TokenizedCorpus:
    docs: list[Document]
    ids: list[list[int]]                 # per-doc token ids
    offsets: np.ndarray                  # [n_docs+1] global token offsets

    @classmethod
    def build(cls, docs: list[Document], tokenizer) -> "TokenizedCorpus":
        ids = [tokenizer(d.text, add_special_tokens=False)["input_ids"] for d in docs]
        offsets = np.concatenate([[0], np.cumsum([len(x) for x in ids])]).astype(np.int64)
        return cls(docs, ids, offsets)

    @property
    def n_tokens(self) -> int:
        return int(self.offsets[-1])

    def locate(self, gid: int) -> tuple[int, int]:
        """global token id -> (doc index, position)"""
        d = int(np.searchsorted(self.offsets, gid, side="right") - 1)
        return d, int(gid - self.offsets[d])

    def batches(self, batch_size: int, pad_id: int) -> Iterator[tuple[torch.Tensor, torch.Tensor, list[int]]]:
        for s in range(0, len(self.ids), batch_size):
            idx = list(range(s, min(s + batch_size, len(self.ids))))
            L = max(len(self.ids[i]) for i in idx)
            inp = torch.full((len(idx), L), pad_id, dtype=torch.long)
            mask = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                n = len(self.ids[i])
                inp[r, :n] = torch.tensor(self.ids[i])
                mask[r, :n] = 1
            yield inp, mask, idx


# --------------------------------------------------------------------------- pass 1
class StreamingStats:
    """Exact running top-k / bottom-k and moments for every neuron of one layer."""

    def __init__(self, n_neurons: int, k: int):
        self.k = k
        self.top_v = torch.full((0, n_neurons), -math.inf)
        self.top_i = torch.zeros((0, n_neurons), dtype=torch.long)
        self.bot_v = torch.full((0, n_neurons), math.inf)
        self.bot_i = torch.zeros((0, n_neurons), dtype=torch.long)
        self.s1 = torch.zeros(n_neurons, dtype=torch.float64)
        self.s2 = torch.zeros(n_neurons, dtype=torch.float64)
        self.npos = torch.zeros(n_neurons, dtype=torch.float64)
        self.n = 0

    def update(self, acts: torch.Tensor, gids: torch.Tensor) -> None:
        """acts [M, N] for M valid tokens, gids [M] their global token ids."""
        a64 = acts.double()
        self.s1 += a64.sum(0)
        self.s2 += (a64 * a64).sum(0)
        self.npos += (acts > 0).sum(0).double()
        self.n += acts.shape[0]
        g = gids[:, None].expand_as(acts)
        v = torch.cat([self.top_v, acts]); i = torch.cat([self.top_i, g])
        kk = min(self.k, v.shape[0])
        tv, ti = v.topk(kk, dim=0)
        self.top_v, self.top_i = tv, i.gather(0, ti)
        v = torch.cat([self.bot_v, acts]); i = torch.cat([self.bot_i, g])
        bv, bi = v.topk(kk, dim=0, largest=False)
        self.bot_v, self.bot_i = bv, i.gather(0, bi)

    def summary(self) -> dict[str, np.ndarray]:
        mean = self.s1 / max(self.n, 1)
        var = (self.s2 / max(self.n, 1) - mean ** 2).clamp_min(0)
        return dict(top_v=self.top_v.numpy(), top_i=self.top_i.numpy(),
                    bot_v=self.bot_v.numpy(), bot_i=self.bot_i.numpy(),
                    mean=mean.float().numpy(), std=var.sqrt().float().numpy(),
                    frac_pos=(self.npos / max(self.n, 1)).float().numpy(), n=np.array(self.n))


@torch.no_grad()
def stream_layer_stats(model, corpus: TokenizedCorpus, layers: list[int], k: int,
                       batch_size: int, pad_id: int, device: str,
                       progress: bool = True) -> dict[int, dict[str, np.ndarray]]:
    d_mlp = model.model.layers[layers[0]].mlp.down_proj.in_features
    stats = {l: StreamingStats(d_mlp, k) for l in layers}
    state: dict = {}

    def cb(layer, h):  # h: [B, T, N] on CPU
        valid = state["mask"].bool()
        stats[layer].update(h[valid], state["gids"][valid])

    base = model.model
    n_batches = math.ceil(len(corpus.ids) / batch_size)
    for b, (inp, mask, idx) in enumerate(corpus.batches(batch_size, pad_id)):
        gids = torch.zeros_like(inp)
        for r, i in enumerate(idx):
            n = len(corpus.ids[i])
            gids[r, :n] = torch.arange(corpus.offsets[i], corpus.offsets[i] + n)
        state.update(mask=mask, gids=gids)
        with NeuronRecorder(model, layers, callback=cb), stop_after_layer(model, max(layers)):
            base(input_ids=inp.to(device), attention_mask=mask.to(device), use_cache=False)
        if progress and (b % 10 == 0 or b == n_batches - 1):
            print(f"  pass1 batch {b + 1}/{n_batches}", flush=True)
    return {l: s.summary() for l, s in stats.items()}


# --------------------------------------------------------------------------- selection
def _entropy_norm(labels: list, n_categories: int) -> float:
    if len(labels) <= 1 or n_categories <= 1:
        return 0.0
    c = np.array(list(Counter(labels).values()), dtype=float)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / math.log(min(n_categories, len(labels))))


def heterogeneity(gids: np.ndarray, corpus: TokenizedCorpus, n_genres: int,
                  dedupe_docs: bool = True) -> dict:
    seen, genres, toks = set(), [], []
    for gid in gids:
        d, pos = corpus.locate(int(gid))
        if dedupe_docs and d in seen:
            continue
        seen.add(d)
        genres.append(corpus.docs[d].genre)
        toks.append(corpus.ids[d][pos])
    hg = _entropy_norm(genres, n_genres)
    ht = _entropy_norm(toks, len(toks))
    return dict(genre_entropy=hg, token_entropy=ht, score=math.sqrt(hg * ht),
                n_unique_docs=len(seen), genre_counts=dict(Counter(genres)))


def select_candidates(stats: dict[int, dict], corpus: TokenizedCorpus, n_candidates: int,
                      min_selectivity: float, seed: int, n_random_baseline: int = 1,
                      dedupe_docs: bool = True, prefilter: int = 400) -> list[dict]:
    """Deterministic, pre-specified candidate rule (see module docstring)."""
    n_genres = len({d.genre for d in corpus.docs})
    rng = random.Random(seed)
    per_layer = max(1, n_candidates // len(stats))
    out: list[dict] = []
    for layer in sorted(stats):
        s = stats[layer]
        std = np.maximum(s["std"], 1e-6)
        rows = []
        for pol, vals, ids in (("pos", s["top_v"], s["top_i"]), ("neg", -s["bot_v"], s["bot_i"])):
            sel = vals.mean(0) / std
            for n in np.argsort(-sel)[:prefilter]:   # only score the most selective
                if sel[n] < min_selectivity:
                    break
                h = heterogeneity(ids[:, n], corpus, n_genres, dedupe_docs)
                rows.append(dict(layer=int(layer), neuron=int(n), polarity=pol,
                                 selectivity=float(sel[n]), **h))
        rows.sort(key=lambda r: (-r["score"], -r["selectivity"]))
        chosen, used = [], set()
        for r in rows:
            if r["neuron"] in used:
                continue          # one polarity per neuron keeps the set diverse
            chosen.append({**r, "selection": "heterogeneity"}); used.add(r["neuron"])
            if len(chosen) >= per_layer:
                break
        pool = [r for r in rows if r["neuron"] not in used]
        for r in rng.sample(pool, min(n_random_baseline, len(pool))):
            chosen.append({**r, "selection": "random_baseline"})
        out += chosen
    for i, c in enumerate(out):
        c["cid"] = f"L{c['layer']}N{c['neuron']}{'+' if c['polarity'] == 'pos' else '-'}"
    return out


# --------------------------------------------------------------------------- pass 2
@torch.no_grad()
def trace_candidates(model, corpus: TokenizedCorpus, candidates: list[dict], batch_size: int,
                     pad_id: int, device: str, keep_input_above: dict[str, float] | None = None,
                     max_inputs: int = 64) -> dict:
    """Full per-token activation traces for candidate neurons on one corpus split.

    Returns {cid: float32[n_tokens]} plus, if ``keep_input_above`` is given, MLP-input
    vectors at tokens where the (signed) activation exceeds the threshold
    (used for the optional context-geometry analysis).
    """
    layers = sorted({c["layer"] for c in candidates})
    neurons = {l: sorted({c["neuron"] for c in candidates if c["layer"] == l}) for l in layers}
    col = {(l, n): j for l in layers for j, n in enumerate(neurons[l])}
    traces = {c["cid"]: np.zeros(corpus.n_tokens, dtype=np.float32) for c in candidates}
    inputs: dict[str, list] = {c["cid"]: [] for c in candidates}
    for inp, mask, idx in corpus.batches(batch_size, pad_id):
        rec = record_activations(model, inp.to(device), layers, attention_mask=mask.to(device),
                                 neurons=neurons, keep_mlp_input=keep_input_above is not None)
        for c in candidates:
            a = rec.acts[c["layer"]][..., col[(c["layer"], c["neuron"])]]
            sign = 1.0 if c["polarity"] == "pos" else -1.0
            for r, i in enumerate(idx):
                n = len(corpus.ids[i])
                o = corpus.offsets[i]
                traces[c["cid"]][o:o + n] = a[r, :n].numpy()
                room = max_inputs - len(inputs[c["cid"]])
                if keep_input_above is not None and room > 0:
                    hit = torch.nonzero(sign * a[r, :n] >= keep_input_above[c["cid"]]).flatten()
                    for p in hit.tolist()[:room]:
                        inputs[c["cid"]].append((int(o + p), rec.mlp_inputs[c["layer"]][r, p].numpy()))
    return {"traces": traces, "inputs": inputs}


# --------------------------------------------------------------------------- cards
def _tok_strs(tokenizer, ids: list[int]) -> list[str]:
    return [tokenizer.decode([t]) for t in ids]


def example_cards(trace: np.ndarray, corpus: TokenizedCorpus, tokenizer, polarity: str,
                  k: int, window: int, dedupe_docs: bool = True) -> list[dict]:
    """Strongest examples (in the candidate's polarity) with per-token context activations."""
    signed = trace if polarity == "pos" else -trace
    order = np.argsort(-signed)
    cards, seen = [], set()
    for gid in order:
        d, pos = corpus.locate(int(gid))
        if dedupe_docs and d in seen:
            continue
        seen.add(d)
        o = corpus.offsets[d]
        lo, hi = max(0, pos - window), min(len(corpus.ids[d]), pos + window + 1)
        cards.append(dict(
            doc_id=corpus.docs[d].doc_id, genre=corpus.docs[d].genre, position=pos,
            global_token=int(gid), activation=float(trace[gid]),
            token=tokenizer.decode([corpus.ids[d][pos]]),
            context_tokens=_tok_strs(tokenizer, corpus.ids[d][lo:hi]),
            context_acts=[float(x) for x in trace[o + lo:o + hi]], focus_index=pos - lo))
        if len(cards) >= k:
            break
    return cards


def cards_to_markdown(cand: dict, cards: list[dict], title_suffix: str = "") -> str:
    lines = [f"## {cand['cid']}{title_suffix}",
             # Selection type (heterogeneity vs random baseline) and selection metrics are
             # deliberately omitted: the annotator should be blind to them.
             f"layer {cand['layer']}, neuron {cand['neuron']}, polarity {cand['polarity']}",
             "", "| # | act | genre | token | context |", "|---|---|---|---|---|"]
    for j, c in enumerate(cards):
        ctx = "".join(f"[[{t}]]" if i == c["focus_index"] else t
                      for i, t in enumerate(c["context_tokens"]))
        ctx = ctx.replace("\n", "⏎").replace("|", "\\|")
        lines.append(f"| {j} | {c['activation']:+.2f} | {c['genre']} | `{c['token'].strip() or repr(c['token'])}` | {ctx} |")
    return "\n".join(lines) + "\n"


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))
