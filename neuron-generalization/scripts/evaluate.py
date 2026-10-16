"""Phase 3: measure authored test sets + blind held-out sheets, score, apply verdict rule.

    python scripts/evaluate.py --config configs/qwen3_4b.yaml

Every neuron in the Phase 3 subset appears in the summary, including those not yet
evaluated -- nothing is silently dropped.
"""
import argparse
import json

import pandas as pd
import yaml
from _common import ROOT, setup

from neurogen.analysis import VerdictRule, evaluate_neuron
from neurogen.hypotheses import sha256_file, verify_frozen
from neurogen.pipeline import load_candidates, run_dir
from neurogen.predictions import TestSet, measure_items, score_blind_sheet

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/qwen3_4b.yaml")
ap.add_argument("--hyp-dir", default=str(ROOT / "research" / "hypotheses"))
args = ap.parse_args()
cfg, lm = setup(args.config)
out = run_dir(cfg)
ev = cfg.evaluation
rule = VerdictRule(ev.min_discriminating_items, ev.support_balanced_acc, ev.neither_balanced_acc)
cands = {c["cid"]: c for c in load_candidates(out)}
subset = json.loads((out / "discovery" / "phase3_subset.json").read_text())["cids"]
(out / "evaluation").mkdir(parents=True, exist_ok=True)
summary = []
for cid in subset:
    c = cands[cid]
    row = dict(cid=cid, layer=c["layer"], neuron=c["neuron"], polarity=c["polarity"],
               selection=c["selection"], verdict="not evaluated", reason="")
    tp = ROOT / "research" / "test_sets" / f"{cid}.yaml"
    try:
        sha = verify_frozen(args.hyp_dir, cid)
        if not tp.exists() or "REPLACE" in tp.read_text():
            raise FileNotFoundError(f"test set {tp.name} missing or incomplete")
        ts = TestSet.load(tp)
        if ts.hypothesis_sha != sha:
            raise PermissionError("test set was written against a different hypothesis version")
    except (PermissionError, FileNotFoundError) as e:
        row["reason"] = str(e)
        summary.append(row)
        continue
    rows = measure_items(lm.model, lm.tokenizer, c["layer"], c["neuron"], c["polarity"], ts.items, lm.device)
    for r in rows:
        r["source"] = "authored"
    sp = ROOT / "research" / "blind_sheets" / f"{cid}.yaml"
    kp = out / "heldout_SEALED" / "blind_keys" / f"{cid}.json"
    blind = []
    if sp.exists() and kp.exists():
        header = yaml.safe_load(sp.read_text())
        if header.get("test_set_sha") != sha256_file(tp):
            raise SystemExit(f"{cid}: test set was modified after the blind sheet was created")
        try:
            blind = score_blind_sheet(sp, kp, c["active_threshold"])
        except ValueError as e:
            print(f"{cid}: blind sheet incomplete ({e}); scoring authored items only")
    t = c["active_threshold"]
    res = dict(cid=cid, threshold=t, hypothesis_sha=sha,
               authored=evaluate_neuron(rows, t, rule, ev.n_bootstrap, cfg.seed),
               blind=evaluate_neuron(blind, t, rule, ev.n_bootstrap, cfg.seed) if blind else None,
               combined=evaluate_neuron(rows + blind, t, rule, ev.n_bootstrap, cfg.seed),
               items=rows, blind_items=blind)
    (out / "evaluation" / f"{cid}.json").write_text(json.dumps(res, indent=1, default=float))
    comb = res["combined"]
    row.update(verdict=comb["verdict"], reason=comb["reason"], n_items=len(rows) + len(blind),
               n_discriminating=comb["discrimination"]["n_discriminating"],
               h1_bal_acc=comb["h1"]["balanced_accuracy"], h2_bal_acc=comb["h2"]["balanced_accuracy"],
               h1_wins=comb["discrimination"]["h1_wins"], h2_wins=comb["discrimination"]["h2_wins"])
    summary.append(row)
df = pd.DataFrame(summary)
df.to_csv(out / "evaluation" / "summary.csv", index=False)
print(df.to_string(index=False))
