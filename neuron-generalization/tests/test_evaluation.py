import json

import numpy as np
import pytest
import torch

from neurogen.analysis import (VerdictRule, auc, binom_two_sided_p, discrimination,
                               evaluate_neuron, two_cluster_separation)
from neurogen.hooks import record_activations
from neurogen.hypotheses import Feature, HypothesisRecord, freeze, save_record, verify_frozen
from neurogen.predictions import TestItem, TestSet, measure_items, target_token_indices

RULE = VerdictRule(min_discriminating_items=8, support_balanced_acc=0.75, neither_balanced_acc=0.65)


def _row(cat, h1, h2, act):
    return dict(category=cat, h1=h1, h2=h2, target_act=act)


def test_binomial_and_auc():
    assert binom_two_sided_p(0, 10) == pytest.approx(2 / 1024)
    assert binom_two_sided_p(5, 10) == pytest.approx(1.0)
    assert auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert auc(np.array([1.0]), np.array([1.0])) == 0.5


def _rows(h2_right: bool, n_disc=10):
    rows = [_row("shared_positive", "active", "active", 5.0) for _ in range(4)]
    rows += [_row("negative", "inactive", "inactive", 0.0) for _ in range(4)]
    for _ in range(n_disc // 2):
        rows.append(_row("h2_only", "inactive", "active", 5.0 if h2_right else 0.0))
        rows.append(_row("h1_only", "active", "inactive", 0.0 if h2_right else 5.0))
    return rows


def test_verdict_supports_h2_only_when_discriminating_items_favour_it():
    r = evaluate_neuron(_rows(h2_right=True), threshold=1.0, rule=RULE, n_boot=200)
    assert r["verdict"] == "supports H2"
    r = evaluate_neuron(_rows(h2_right=False), threshold=1.0, rule=RULE, n_boot=200)
    assert r["verdict"] == "supports H1"


def test_verdict_inconclusive_with_few_discriminating_items():
    r = evaluate_neuron(_rows(h2_right=True, n_disc=4), threshold=1.0, rule=RULE, n_boot=200)
    assert r["verdict"] == "inconclusive"


def test_verdict_neither_when_both_fail():
    rows = [_row("shared_positive", "active", "active", 0.0) for _ in range(6)]
    rows += [_row("negative", "inactive", "inactive", 5.0) for _ in range(6)]
    r = evaluate_neuron(rows, threshold=1.0, rule=RULE, n_boot=200)
    assert r["verdict"] == "neither"


def test_unified_story_alone_cannot_win():
    """If H2 only ever agrees with H1 (no discriminating items), H2 is never 'supported',
    however well both predict."""
    rows = [_row("shared_positive", "active", "active", 5.0) for _ in range(20)]
    rows += [_row("negative", "inactive", "inactive", 0.0) for _ in range(20)]
    r = evaluate_neuron(rows, threshold=1.0, rule=RULE, n_boot=200)
    assert r["verdict"] == "inconclusive" and r["discrimination"]["n_discriminating"] == 0


def test_unsure_items_excluded_and_discrimination_counts():
    rows = [_row("counterexample", "unsure", "inactive", 5.0), _row("h2_only", "inactive", "active", 5.0)]
    d = discrimination(rows, 1.0)
    assert d == dict(n_discriminating=1, h1_wins=0, h2_wins=1, sign_test_p=1.0)


def _record(cid="L1N7+"):
    return HypothesisRecord(
        cid=cid, layer=1, neuron=7, polarity="pos", researcher="t",
        h1_features=[Feature("A", "a"), Feature("B", "b")],
        h1_predicts_active="A or B", h1_predicts_inactive="neither A nor B",
        h2_abstraction="C", h2_description="c", h2_predicts_active="C", h2_predicts_inactive="not C",
        prior={"h1": 0.4, "h2": 0.4, "neither": 0.2}, discovery_cards_sha="abc")


def test_freeze_blocks_edits_and_requires_falsifiability(tmp_path):
    bad = _record("L1N8+")
    bad.h2_predicts_inactive = ""
    save_record(bad, tmp_path)
    with pytest.raises(ValueError, match="predicts active AND inactive"):
        freeze(tmp_path)
    (tmp_path / "L1N8+.json").unlink()

    save_record(_record(), tmp_path)
    with pytest.raises(PermissionError):
        verify_frozen(tmp_path, "L1N7+")
    freeze(tmp_path)
    sha = verify_frozen(tmp_path, "L1N7+")
    assert len(sha) == 64
    with pytest.raises(PermissionError):
        save_record(_record(), tmp_path)
    p = tmp_path / "L1N7+.json"
    d = json.loads(p.read_text()); d["h2_abstraction"] = "edited after the fact"
    p.write_text(json.dumps(d))
    with pytest.raises(PermissionError, match="modified"):
        verify_frozen(tmp_path, "L1N7+")


def test_test_item_validation(tmp_path):
    with pytest.raises(ValueError):
        TestItem("x", "hello world", "world", "h2_only", h1="active", h2="active").validate()
    with pytest.raises(ValueError):
        TestItem("x", "hello", "world", "negative", h1="inactive", h2="inactive").validate()
    p = tmp_path / "t.yaml"
    p.write_text("cid: L1N7+\nhypothesis_sha: s\nauthor: a\ncreated_at: now\nitems:\n"
                 "- {id: a, text: 'the cat sat', target: cat, category: negative, h1: inactive, h2: inactive}\n")
    assert TestSet.load(p).items[0].target == "cat"


def test_target_measurement_matches_direct_read(tiny_model, tok):
    text = "a cat and a cat"
    ids, idx = target_token_indices(tok, text, "cat", occurrence=2)
    assert idx == [12, 13, 14]
    item = TestItem("i", text, "cat", "negative", "inactive", "inactive", occurrence=2)
    m = measure_items(tiny_model, tok, 1, 7, "neg", [item], "cpu")[0]
    direct = record_activations(tiny_model, torch.tensor([ids]), [1]).acts[1][0, :, 7]
    assert m["target_act"] == pytest.approx(float((-direct[idx]).max()), abs=1e-6)


def test_two_cluster_separation():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(20, 16)) * 0.05 + np.eye(16)[0]
    b = rng.normal(size=(20, 16)) * 0.05 + np.eye(16)[1]
    sep = two_cluster_separation(np.vstack([a, b]))
    assert sep["silhouette"] > 0.8 and sorted(sep["sizes"]) == [20, 20]
