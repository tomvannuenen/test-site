"""Corpus parsing and building, with the network fetch replaced by local files."""
import json

import pandas as pd

from neurogen.config import CorpusConfig
from neurogen.corpus import HF_SOURCES, _detok_wikitext, _texts_from_file, build_corpus


def test_wikitext_articles_and_detok(tmp_path):
    lines = [" = Danube = \n", " The river @-@ basin is large , and old . \n", " = = History = = \n",
             " It ( once ) was n't calm . \n", " = Octopus = \n", " Eight arms . \n"]
    p = tmp_path / "w.parquet"
    pd.DataFrame({"text": lines}).to_parquet(p)
    arts = _texts_from_file(str(p), "wikitext")
    assert len(arts) == 2
    assert arts[0].startswith("Danube") and "river-basin is large, and old." in arts[0]
    assert "(once) wasn't calm." in arts[0] and "History" in arts[0]
    assert _detok_wikitext(" 1 @,@ 000 @.@ 5 ") == "1,000.5"


def test_each_source_kind_parses(tmp_path):
    frames = {
        "humaneval": pd.DataFrame({"prompt": ["def f():\n"], "canonical_solution": ["    return 1\n"]}),
        "gsm8k": pd.DataFrame({"question": ["2+2?"], "answer": ["4"]}),
        "col:support": pd.DataFrame({"support": ["Sound is a wave."]}),
    }
    for kind, df in frames.items():
        p = tmp_path / f"{kind.replace(':', '_')}.parquet"
        df.to_parquet(p)
        assert _texts_from_file(str(p), kind)[0]
    j = tmp_path / "d.jsonl"
    j.write_text(json.dumps({"instruction": "Q?", "response": "A.", "context": "", "category": "x"}) + "\n")
    assert _texts_from_file(str(j), "dolly") == ["Q?\n\nA."]
    assert {s["kind"] for s in HF_SOURCES} <= {"wikitext", "humaneval", "dolly", "gsm8k", "col:text", "col:support"}


def test_build_corpus_hf_mixture_with_stub_fetch(tmp_path, tok):
    def fake_fetch(src, cache):
        return [f"{src['genre']} document number {i} " * 5 for i in range(30)], "sha-" + src["genre"]

    cfg = CorpusConfig(source="hf_mixture", tokens_per_genre=1000, max_tokens_per_doc=200,
                       min_tokens_per_doc=10, cache_dir="cache")
    docs = build_corpus(cfg, tok, seed=3, root=tmp_path, fetch=fake_fetch)
    genres = {d.genre for d in docs}
    assert genres == {s["genre"] for s in HF_SOURCES}
    for g in genres:
        n = sum(len(tok(d.text)["input_ids"]) for d in docs if d.genre == g)
        assert 1000 <= n < 1000 + 200
        assert {d.split for d in docs if d.genre == g} == {"discovery", "heldout"}
    manifest = json.loads(next((tmp_path / "cache").glob("manifest_*.json")).read_text())
    assert all(s["revision_sha"].startswith("sha-") for s in manifest["sources"])
    again = build_corpus(cfg, tok, seed=3, root=tmp_path, fetch=None)   # served from cache
    assert [d.doc_id for d in again] == [d.doc_id for d in docs]
