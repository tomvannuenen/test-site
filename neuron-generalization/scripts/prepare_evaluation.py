"""After freezing hypotheses, prepare Phase 3 evaluation data in two ordered stages.

    python scripts/prepare_evaluation.py --config configs/qwen3_4b.yaml --stage tests
        -> research/test_sets/<cid>.yaml templates. Write your items (do NOT run the model).
    python scripts/prepare_evaluation.py --config configs/qwen3_4b.yaml --stage blind
        -> locks each completed test set (records its SHA-256), then writes
           research/blind_sheets/<cid>.yaml from the HELD-OUT split with activations hidden.
           Answer keys go to outputs/.../heldout_SEALED/blind_keys/ -- don't open them.

The order matters: blind sheets show held-out contexts, so they are created only after
the authored test set is locked, and evaluate.py refuses a test set changed afterwards.
"""
import argparse
import json

import numpy as np
import yaml
from _common import ROOT, setup

from neurogen.hypotheses import sha256_file, verify_frozen
from neurogen.pipeline import load_candidates, load_corpora, run_dir
from neurogen.predictions import TestSet, make_blind_sheet, new_test_set_template

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/qwen3_4b.yaml")
ap.add_argument("--stage", choices=["tests", "blind"], required=True)
ap.add_argument("--n-each", type=int, default=10, help="items per source (top / token-matched / random)")
ap.add_argument("--hyp-dir", default=str(ROOT / "research" / "hypotheses"))
args = ap.parse_args()
cfg, _ = setup(args.config, need_model=False)
out = run_dir(cfg)
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(cfg.model.name, revision=cfg.model.revision)
_, held = load_corpora(cfg, tok)
cands = {c["cid"]: c for c in load_candidates(out)}
subset = json.loads((out / "discovery" / "phase3_subset.json").read_text())["cids"]
traces = np.load(out / "heldout_SEALED" / "traces.npz")
(ROOT / "research" / "blind_sheets").mkdir(parents=True, exist_ok=True)
(ROOT / "research" / "test_sets").mkdir(parents=True, exist_ok=True)
for cid in subset:
    try:
        sha = verify_frozen(args.hyp_dir, cid)
    except PermissionError as e:
        print(f"skip {cid}: {e}")
        continue
    tp = ROOT / "research" / "test_sets" / f"{cid}.yaml"
    if args.stage == "tests":
        if not tp.exists():
            tp.write_text(new_test_set_template(cid, sha))
        print(f"{cid}: edit {tp.relative_to(ROOT)}")
        continue
    if not tp.exists() or "REPLACE" in tp.read_text():
        print(f"skip {cid}: complete {tp.name} first (remove template placeholders)")
        continue
    ts = TestSet.load(tp)  # validates
    if ts.hypothesis_sha != sha:
        raise SystemExit(f"{cid}: test set hypothesis_sha does not match the frozen record")
    c = cands[cid]
    sheet, key = make_blind_sheet(traces[cid], held, tok, c["polarity"], c["active_threshold"],
                                  args.n_each, cfg.discovery.context_window, cfg.seed)
    kdir = out / "heldout_SEALED" / "blind_keys"
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / f"{cid}.json").write_text(json.dumps(key, indent=1))
    sp = ROOT / "research" / "blind_sheets" / f"{cid}.yaml"
    if sp.exists():
        print(f"{cid}: {sp.name} already exists; not overwriting")
        continue
    sp.write_text("# For each item, without running the model, write what each FROZEN hypothesis\n"
                  "# predicts at the [[bracketed]] token: active | inactive | unsure\n" +
                  yaml.safe_dump(dict(cid=cid, hypothesis_sha=sha, test_set_sha=sha256_file(tp),
                                      items=sheet), sort_keys=False, allow_unicode=True, width=120))
    print(f"{cid}: locked {tp.name}; blind sheet {sp.name} ({len(sheet)} items)")
