"""Phase 3: falsifiable predictions, measured against actual activations.

Two kinds of evaluation data, both created *after* hypotheses are frozen:

1. **Authored test sets** (YAML, one per neuron). Each item is a sentence, a target
   substring whose tokens are measured, a category, and an explicit
   ``active``/``inactive``/``unsure`` prediction from *each* hypothesis. Categories:

   - ``shared_positive``  both hypotheses predict active
   - ``h1_only``          H1 predicts active, H2 inactive (e.g. a surface instance of
                          one H1 feature that lacks H2's abstract property)
   - ``h2_only``          H2 predicts active, H1 inactive (a new instance of the
                          abstraction that matches none of H1's features)
   - ``negative``         both predict inactive
   - ``counterexample``   deliberately hard: surface-similar to positives but
                          predicted inactive by the hypothesis it targets

2. **Blind held-out sheets.** Contexts from the held-out split -- its strongest
   activations, uniformly random tokens, and *token-matched* controls (the same
   token ids as strong examples, but in low-activation contexts) -- shuffled, with
   activations hidden. The researcher labels what each hypothesis predicts; the
   key is stored separately and only joined at scoring time.
"""
from __future__ import annotations

import datetime as _dt
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from .hooks import record_activations

CATEGORIES = ("shared_positive", "h1_only", "h2_only", "negative", "counterexample")
LABELS = ("active", "inactive", "unsure")


@dataclass
class TestItem:
    __test__ = False  # not a pytest class
    id: str
    text: str
    target: str
    category: str
    h1: str
    h2: str
    occurrence: int = 1
    rationale: str = ""

    def validate(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"{self.id}: category {self.category!r} not in {CATEGORIES}")
        for h in (self.h1, self.h2):
            if h not in LABELS:
                raise ValueError(f"{self.id}: prediction {h!r} not in {LABELS}")
        if self.text.count(self.target) < self.occurrence:
            raise ValueError(f"{self.id}: target {self.target!r} occurs < {self.occurrence} times")
        expected = {"shared_positive": ("active", "active"), "h1_only": ("active", "inactive"),
                    "h2_only": ("inactive", "active"), "negative": ("inactive", "inactive")}
        if self.category in expected and (self.h1, self.h2) != expected[self.category]:
            raise ValueError(f"{self.id}: category {self.category} implies (h1,h2)="
                             f"{expected[self.category]}, got {(self.h1, self.h2)}")


@dataclass
class TestSet:
    __test__ = False
    cid: str
    hypothesis_sha: str
    author: str
    created_at: str
    items: list[TestItem] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def load(cls, path: Path) -> "TestSet":
        raw = yaml.safe_load(Path(path).read_text())
        items = [TestItem(**it) for it in raw.pop("items")]
        ts = cls(items=items, **raw)
        ids = [i.id for i in items]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate item ids")
        for it in items:
            it.validate()
        return ts


def target_token_indices(tokenizer, text: str, target: str, occurrence: int = 1) -> tuple[list[int], list[int]]:
    """Token ids for ``text`` and the indices of tokens overlapping the target span."""
    start = -1
    for _ in range(occurrence):
        start = text.index(target, start + 1)
    end = start + len(target)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    idx = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < end and b > start and b > a]
    if not idx:
        raise ValueError(f"target {target!r} maps to no tokens")
    return enc["input_ids"], idx


@torch.no_grad()
def measure_items(model, tokenizer, layer: int, neuron: int, polarity: str,
                  items: list[TestItem], device: str) -> list[dict]:
    """Signed activation (in the candidate's polarity) at each item's target tokens."""
    sign = 1.0 if polarity == "pos" else -1.0
    out = []
    for it in items:
        ids, idx = target_token_indices(tokenizer, it.text, it.target, it.occurrence)
        rec = record_activations(model, torch.tensor([ids], device=device), [layer],
                                 neurons={layer: [neuron]})
        a = sign * rec.acts[layer][0, :, 0].numpy()
        out.append(dict(id=it.id, category=it.category, h1=it.h1, h2=it.h2,
                        target_act=float(a[idx].max()), max_act_anywhere=float(a.max()),
                        argmax_token=tokenizer.decode([ids[int(a.argmax())]]),
                        target_tokens=[tokenizer.decode([ids[i]]) for i in idx],
                        token_acts=[float(x) for x in a]))
    return out


def active_threshold(discovery_top_signed: np.ndarray, frac: float) -> float:
    """Pre-specified "active" threshold from *discovery* data only."""
    return float(frac * np.median(discovery_top_signed))


def new_test_set_template(cid: str, hypothesis_sha: str, author: str = "") -> str:
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    ex = [dict(id=f"{c}_1", text="REPLACE with a sentence containing the target.",
               target="target", category=c,
               h1={"h1_only": "active", "shared_positive": "active"}.get(c, "inactive"),
               h2={"h2_only": "active", "shared_positive": "active"}.get(c, "inactive"),
               rationale="why each hypothesis predicts this") for c in CATEGORIES]
    ex[-1]["h1"] = ex[-1]["h2"] = "inactive"
    return yaml.safe_dump(dict(cid=cid, hypothesis_sha=hypothesis_sha, author=author,
                               created_at=now, notes="", items=ex), sort_keys=False, width=100)


# --------------------------------------------------------------------------- blind sheets
def make_blind_sheet(trace: np.ndarray, corpus, tokenizer, polarity: str, threshold: float,
                     n_each: int, window: int, seed: int) -> tuple[list[dict], dict]:
    """Held-out items with hidden activations. Returns (sheet_items, key)."""
    rng = random.Random(seed)
    signed = trace if polarity == "pos" else -trace
    order = np.argsort(-signed)
    top = [int(g) for g in order[: n_each * 3] if signed[g] > threshold][:n_each]
    top_tok = {corpus.ids[corpus.locate(g)[0]][corpus.locate(g)[1]] for g in top}
    low = np.nonzero(signed < threshold * 0.5)[0]
    matched = [int(g) for g in low
               if corpus.ids[corpus.locate(int(g))[0]][corpus.locate(int(g))[1]] in top_tok]
    matched = rng.sample(matched, min(n_each, len(matched)))
    rand = rng.sample(range(len(signed)), min(n_each, len(signed)))
    chosen = [(g, "top") for g in top] + [(g, "token_matched") for g in matched] + \
             [(g, "random") for g in rand if g not in set(top) | set(matched)]
    rng.shuffle(chosen)
    sheet, key = [], {}
    for j, (g, src) in enumerate(chosen):
        d, pos = corpus.locate(g)
        ids = corpus.ids[d]
        lo, hi = max(0, pos - window), min(len(ids), pos + window + 1)
        ctx = tokenizer.decode(ids[lo:pos]) + "[[" + tokenizer.decode([ids[pos]]) + "]]" + \
            tokenizer.decode(ids[pos + 1:hi])
        iid = f"b{j:03d}"
        sheet.append(dict(id=iid, context=ctx.replace("\n", " ⏎ "), h1="", h2=""))
        key[iid] = dict(global_token=g, source=src, signed_act=float(signed[g]),
                        genre=corpus.docs[d].genre)
    return sheet, key


def score_blind_sheet(sheet_path: Path, key_path: Path, threshold: float) -> list[dict]:
    sheet = yaml.safe_load(Path(sheet_path).read_text())
    key = json.loads(Path(key_path).read_text())
    rows = []
    for it in sheet["items"]:
        for h in ("h1", "h2"):
            if it[h] not in LABELS:
                raise ValueError(f"{it['id']}: {h} label {it[h]!r} not in {LABELS} (fill every item)")
        k = key[it["id"]]
        rows.append(dict(id=it["id"], h1=it["h1"], h2=it["h2"], source=k["source"],
                         genre=k["genre"], target_act=k["signed_act"],
                         category="blind_" + k["source"]))
    return rows
