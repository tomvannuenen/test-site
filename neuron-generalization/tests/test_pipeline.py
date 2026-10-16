"""End-to-end: discovery -> hypotheses -> freeze -> test set -> blind sheet -> scoring ->
interventions, on the tiny random Qwen3 and the bundled example corpus."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from neurogen import pipeline as P
from neurogen.analysis import VerdictRule, evaluate_neuron
from neurogen.config import load_config
from neurogen.hypotheses import Feature, HypothesisRecord, freeze, save_record, verify_frozen
from neurogen.interventions import intervention_effect
from neurogen.model import LoadedModel
from neurogen.predictions import (TestSet, make_blind_sheet, measure_items, new_test_set_template,
                                  score_blind_sheet)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def cfg(tmp_path):
    c = load_config(REPO / "configs" / "smoke_qwen3_0.6b.yaml", root=str(tmp_path))
    c.corpus.local_path = str(REPO / "data" / "example" / "example_corpus.jsonl")
    c.corpus.max_tokens_per_doc = 128
    c.discovery.top_k = 6
    c.discovery.min_selectivity = 0.5
    c.model.name = "tiny"
    return c


def test_end_to_end(cfg, tiny_model, tok, tmp_path):
    lm = LoadedModel(tiny_model, tok, "cpu", "tiny")
    out = P.run_discovery(cfg, lm, progress=False)
    cands = P.load_candidates(out)
    subset = json.loads((out / "discovery" / "phase3_subset.json").read_text())["cids"]
    assert len(subset) == cfg.evaluation.n_test_neurons
    assert {c["selection"] for c in cands} == {"heterogeneity", "random_baseline"}
    assert (out / "heldout_SEALED" / "traces.npz").exists()
    md = (out / "discovery" / "candidates.md").read_text()
    assert "selection" not in md and "random" not in md
    order = [l.split()[1] for l in md.splitlines() if l.startswith("## ")]
    assert order == [c["cid"] for c in sorted(cands, key=lambda c: (c["layer"], c["neuron"]))]

    # discovery/held-out are disjoint documents
    idx = json.loads((out / "corpus_index.json").read_text())
    assert not {d["doc_id"] for d in idx["discovery"]} & {d["doc_id"] for d in idx["heldout"]}

    # Re-running reuses pass-1 cache and is deterministic.
    out2 = P.run_discovery(cfg, lm, progress=False)
    assert [c["cid"] for c in P.load_candidates(out2)] == [c["cid"] for c in cands]

    cid = subset[0]
    c = next(x for x in cands if x["cid"] == cid)
    hyp = tmp_path / "hyp"
    save_record(HypothesisRecord(
        cid=cid, layer=c["layer"], neuron=c["neuron"], polarity=c["polarity"], researcher="test",
        h1_features=[Feature("A", "a", [0]), Feature("B", "b", [1])],
        h1_predicts_active="A|B", h1_predicts_inactive="other", h2_abstraction="C",
        h2_description="c", h2_predicts_active="C", h2_predicts_inactive="not C",
        prior={"h1": 0.5, "h2": 0.3, "neither": 0.2}, discovery_cards_sha=c["discovery_cards_sha"]), hyp)
    freeze(hyp)
    sha = verify_frozen(hyp, cid)

    ts_path = tmp_path / "ts.yaml"
    raw = yaml.safe_load(new_test_set_template(cid, sha))
    for i, it in enumerate(raw["items"]):
        it["text"] = f"The river flows into the sea, item {i}."
        it["target"] = "sea"
    ts_path.write_text(yaml.safe_dump(raw))
    ts = TestSet.load(ts_path)
    rows = measure_items(tiny_model, tok, c["layer"], c["neuron"], c["polarity"], ts.items, "cpu")
    assert all(np.isfinite(r["target_act"]) for r in rows)

    disc, held = P.load_corpora(cfg, tok)
    trace = np.load(out / "heldout_SEALED" / "traces.npz")[cid]
    sheet, key = make_blind_sheet(trace, held, tok, c["polarity"], c["active_threshold"], 4, 6, 0)
    assert all("signed_act" not in it for it in sheet)   # activations hidden
    sources = {k["source"] for k in key.values()}
    assert "random" in sources
    for it in sheet:
        it["h1"], it["h2"] = "active", "inactive"
    sp, kp = tmp_path / "sheet.yaml", tmp_path / "key.json"
    sp.write_text(yaml.safe_dump(dict(cid=cid, items=sheet)))
    kp.write_text(json.dumps(key))
    blind = score_blind_sheet(sp, kp, c["active_threshold"])
    res = evaluate_neuron(rows + blind, c["active_threshold"], VerdictRule(), n_boot=100)
    assert res["verdict"] in {"supports H1", "supports H2", "neither", "inconclusive"}

    # Intervention at the strongest discovery context
    card = P.load_cards(out, cid)[0]
    d = [i for i, x in enumerate(disc.docs) if x.doc_id == card["doc_id"]][0]
    ids = torch.tensor([disc.ids[d][: card["position"] + 4]])
    eff = intervention_effect(tiny_model, ids, c["layer"], c["neuron"], [card["position"]], "set", 0.0,
                              n_controls=5, seed=0)
    assert eff["kl_before_edit_max"] < 1e-6 and len(eff["control_neurons"]) == 5
