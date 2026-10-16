"""Phase 2 terminal interface: show discovery examples, record H1 / H2 / neither.

    python scripts/annotate.py --config configs/qwen3_4b.yaml [--cid L18N1234+] [--subset]

Only the discovery split is shown. Records go to research/hypotheses/<cid>.json.
"""
import argparse
import json

from _common import ROOT, setup

from neurogen.discovery import cards_to_markdown
from neurogen.hypotheses import Feature, HypothesisRecord, save_record
from neurogen.pipeline import load_candidates, load_cards, run_dir


def ask(prompt, default=""):
    v = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
    return v or default


def ids(s):
    return [int(x) for x in s.replace(",", " ").split()] if s else []


ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/qwen3_4b.yaml")
ap.add_argument("--cid", default=None)
ap.add_argument("--subset", action="store_true", help="only the Phase 3 subset")
ap.add_argument("--hyp-dir", default=str(ROOT / "research" / "hypotheses"))
args = ap.parse_args()
cfg, _ = setup(args.config, need_model=False)
out = run_dir(cfg)
cands = load_candidates(out)
cands.sort(key=lambda c: c["cid"])  # stable order that does not reveal selection type
if args.subset:
    keep = set(json.loads((out / "discovery" / "phase3_subset.json").read_text())["cids"])
    cands = [c for c in cands if c["cid"] in keep]
if args.cid:
    cands = [c for c in cands if c["cid"] == args.cid]
researcher = ask("Your name/initials")
for c in cands:
    cards = load_cards(out, c["cid"])
    print("\n" + "=" * 100 + "\n" + cards_to_markdown(c, cards))
    print("Write BOTH hypotheses even if one seems obviously wrong; 'neither' is a legitimate outcome.\n")
    if ask("Annotate this neuron? (y/n/q)", "y") != "y":
        continue
    feats = []
    print("H1 (superposition): enter >=2 independent features; blank name to stop.")
    while True:
        name = ask(f"  feature {len(feats) + 1} name")
        if not name:
            break
        feats.append(Feature(name, ask("    description"), ids(ask("    example ids it covers"))))
    rec = HypothesisRecord(
        cid=c["cid"], layer=c["layer"], neuron=c["neuron"], polarity=c["polarity"],
        researcher=researcher, h1_features=feats,
        h1_predicts_active=ask("H1 predicts ACTIVE on"), h1_predicts_inactive=ask("H1 predicts INACTIVE on"),
        h2_abstraction=ask("H2 (single abstraction), short name"),
        h2_description=ask("H2 description"), h2_example_ids=ids(ask("H2 covers example ids")),
        h2_predicts_active=ask("H2 predicts ACTIVE on"), h2_predicts_inactive=ask("H2 predicts INACTIVE on"),
        unexplained_example_ids=ids(ask("example ids NEITHER explains")),
        neither_note=ask("'Neither' note (why neither may fit; optional)"),
        prior={k: float(ask(f"prior credence {k}", d)) for k, d in (("h1", "0.4"), ("h2", "0.4"), ("neither", "0.2"))},
        discovery_cards_sha=c["discovery_cards_sha"])
    p = save_record(rec, args.hyp_dir)
    print(f"saved {p}; problems: {rec.problems() or 'none'}")
    if ask("continue? (y/n)", "y") != "y":
        break
