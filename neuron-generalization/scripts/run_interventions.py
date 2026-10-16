"""Phase 4: causal interventions for the Phase 3 subset.

For each neuron, on its strongest discovery contexts:
  * zero- and mean-ablation at the focus token, vs matched random-neuron controls
  * dose-response (scaling) on the strongest context
  * effect on the log-prob of the true next token
  * similarity of full-vocab effects between contexts assigned to different H1 features

    python scripts/run_interventions.py --config configs/qwen3_4b.yaml [--n-contexts 8]
"""
import argparse
import json

import numpy as np
import torch
from _common import ROOT, setup

from neurogen.interventions import direct_logit_effect, dose_response, effect_similarity, intervention_effect
from neurogen.pipeline import load_candidates, load_cards, load_corpora, run_dir

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/qwen3_4b.yaml")
ap.add_argument("--n-contexts", type=int, default=8)
ap.add_argument("--after", type=int, default=8, help="tokens kept after the focus token")
ap.add_argument("--cids", nargs="*", default=None)
args = ap.parse_args()
cfg, lm = setup(args.config)
out = run_dir(cfg)
ic = cfg.intervention
disc, _ = load_corpora(cfg, lm.tokenizer)
doc_index = {d.doc_id: i for i, d in enumerate(disc.docs)}
cands = {c["cid"]: c for c in load_candidates(out)}
subset = args.cids or json.loads((out / "discovery" / "phase3_subset.json").read_text())["cids"]
(out / "interventions").mkdir(parents=True, exist_ok=True)
for cid in subset:
    c = cands[cid]
    layer, neuron = c["layer"], c["neuron"]
    hp = ROOT / "research" / "hypotheses" / f"{cid}.json"
    groups = {}
    if hp.exists():
        for f in json.loads(hp.read_text()).get("h1_features", []):
            for j in f["example_ids"]:
                groups[j] = f["name"]
    cards = load_cards(out, cid)[: args.n_contexts]
    per_ctx, deltas = [], []
    for j, card in enumerate(cards):
        ids = disc.ids[doc_index[card["doc_id"]]]
        pos = card["position"]
        seq = torch.tensor([ids[: pos + 1 + args.after]], device=lm.device)
        nxt = ids[pos + 1] if pos + 1 < len(ids) else None
        track = [nxt] if nxt is not None else []
        zero = intervention_effect(lm.model, seq, layer, neuron, [pos], "set", 0.0, track,
                                   n_controls=ic.n_random_controls, tolerance=ic.control_match_tolerance,
                                   seed=cfg.seed + j, return_delta=True)
        mean = intervention_effect(lm.model, seq, layer, neuron, [pos], "set", c["layer_mean"], track)
        deltas.append(zero.pop("delta_first"))
        per_ctx.append(dict(example_id=j, genre=card["genre"], h1_feature=groups.get(j),
                            activation=card["activation"],
                            true_next_token=lm.tokenizer.decode([nxt]) if nxt is not None else None,
                            zero_ablation=zero, mean_ablation=mean))
        print(f"{cid} ctx{j} [{card['genre']}] KL={zero['kl_mean']:.3g} "
              f"pct_vs_controls={zero['percentile_vs_controls']:.0f}")
    sim = effect_similarity(deltas) if deltas else np.zeros((0, 0))
    first = cards[0]
    ids = disc.ids[doc_index[first["doc_id"]]][: first["position"] + 1 + args.after]
    dose = dose_response(lm.model, torch.tensor([ids], device=lm.device), layer, neuron,
                         [first["position"]], [*ic.scale_factors, 1.0])
    dla = direct_logit_effect(lm.model, layer, neuron)
    dla["promoted_tokens"] = [lm.tokenizer.decode([t]) for t in dla["promoted"]]
    dla["suppressed_tokens"] = [lm.tokenizer.decode([t]) for t in dla["suppressed"]]
    res = dict(cid=cid, contexts=per_ctx, effect_similarity=sim.tolist(),
               dose_response=[dict(factor=d["factor"], kl_mean=d["kl_mean"]) for d in dose],
               direct_logit_effect=dla)
    (out / "interventions" / f"{cid}.json").write_text(json.dumps(res, indent=1, default=float))
print("done")
