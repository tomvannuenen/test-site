import numpy as np
import torch

from neurogen.corpus import Document, split_documents
from neurogen.discovery import (StreamingStats, TokenizedCorpus, example_cards, heterogeneity,
                                select_candidates, stream_layer_stats, trace_candidates)
from neurogen.hooks import record_activations

TEXTS = {
    "news": ["The council voted on Tuesday to fund the bridge.", "Markets fell sharply after the report."],
    "code": ["def add(a, b):\n    return a + b", "for i in range(10):\n    print(i)"],
    "fiction": ["Once upon a time a fox found a key.", "The dragon slept under the hill."],
}


def _docs():
    docs = [Document(f"{g}-{i}", g, "t", t) for g, ts in TEXTS.items() for i, t in enumerate(ts)]
    return split_documents(docs, 0.5, seed=0)


def test_streaming_topk_matches_bruteforce():
    torch.manual_seed(0)
    s = StreamingStats(n_neurons=5, k=4)
    allv, allg = [], []
    for b in range(6):
        a = torch.randn(13, 5)
        g = torch.arange(b * 13, (b + 1) * 13)
        s.update(a, g)
        allv.append(a); allg.append(g)
    A = torch.cat(allv); G = torch.cat(allg)
    out = s.summary()
    tv, ti = A.topk(4, dim=0)
    bv, bi = A.topk(4, dim=0, largest=False)
    assert np.allclose(out["top_v"], tv.numpy()) and np.array_equal(out["top_i"], G[ti].numpy())
    assert np.allclose(out["bot_v"], bv.numpy()) and np.array_equal(out["bot_i"], G[bi].numpy())
    assert np.allclose(out["mean"], A.mean(0).numpy(), atol=1e-5)
    assert np.allclose(out["std"], A.std(0, unbiased=False).numpy(), atol=1e-5)


def test_split_is_by_document_and_stratified():
    docs = _docs()
    for g in TEXTS:
        splits = {d.split for d in docs if d.genre == g}
        assert splits == {"discovery", "heldout"}
    assert _docs()[0].split == docs[0].split  # deterministic


def test_stream_stats_agree_with_direct_recording(tiny_model, tok):
    corp = TokenizedCorpus.build(_docs(), tok)
    stats = stream_layer_stats(tiny_model, corp, [0, 2], k=3, batch_size=2, pad_id=0,
                               device="cpu", progress=False)
    # Recompute brute force, one doc at a time (no padding involved).
    full = []
    for ids in corp.ids:
        full.append(record_activations(tiny_model, torch.tensor([ids]), [2]).acts[2][0])
    A = torch.cat(full)
    assert np.allclose(stats[2]["top_v"], A.topk(3, dim=0).values.numpy(), atol=1e-5)
    assert int(stats[2]["n"]) == corp.n_tokens  # padding excluded


def test_heterogeneity_distinguishes_token_detector(tok):
    corp = TokenizedCorpus.build(_docs(), tok)
    # same character ('e') across genres => low token entropy
    same_tok = [corp.offsets[i] + corp.ids[i].index(tok.to_id["e"]) for i in range(len(corp.ids))
                if tok.to_id["e"] in corp.ids[i]]
    h = heterogeneity(np.array(same_tok), corp, n_genres=3)
    assert h["token_entropy"] == 0.0 and h["score"] == 0.0 and h["genre_entropy"] > 0.5


def test_selection_traces_and_cards(tiny_model, tok):
    corp = TokenizedCorpus.build(_docs(), tok)
    stats = stream_layer_stats(tiny_model, corp, [1], k=4, batch_size=3, pad_id=0,
                               device="cpu", progress=False)
    cands = select_candidates(stats, corp, n_candidates=3, min_selectivity=0.0, seed=0,
                              n_random_baseline=1)
    assert len(cands) == 4 and {c["selection"] for c in cands} == {"heterogeneity", "random_baseline"}
    assert len({c["neuron"] for c in cands}) == 4
    tr = trace_candidates(tiny_model, corp, cands, 2, 0, "cpu",
                          keep_input_above={c["cid"]: -1e9 for c in cands}, max_inputs=5)
    c = cands[0]
    trace = tr["traces"][c["cid"]]
    assert trace.shape == (corp.n_tokens,)
    signed_top = stats[1]["top_v"][0, c["neuron"]] if c["polarity"] == "pos" else stats[1]["bot_v"][0, c["neuron"]]
    assert np.isclose(trace.max() if c["polarity"] == "pos" else trace.min(), signed_top, atol=1e-5)
    cards = example_cards(trace, corp, tok, c["polarity"], k=3, window=4)
    assert len({x["doc_id"] for x in cards}) == len(cards)  # deduped by doc
    assert cards[0]["context_acts"][cards[0]["focus_index"]] == cards[0]["activation"]
    assert len(tr["inputs"][c["cid"]]) == 5 and tr["inputs"][c["cid"]][0][1].shape == (32,)
