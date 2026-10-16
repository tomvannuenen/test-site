"""Corpus construction: a small genre-labelled English mixture, split by document.

Genre labels are what later let us ask whether a neuron's strongest activations
span *apparently unrelated* contexts. They are coarse and are a property of the
corpus, not of the neuron. All sampling is seeded; the resolved dataset commit
SHAs are written to a manifest next to the cached corpus.

The split is by **document** (stratified by genre): hypotheses are written from the
discovery split only, and the held-out split is reserved for evaluation.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator

from .config import CorpusConfig


@dataclass
class Document:
    doc_id: str
    genre: str
    source: str
    text: str
    split: str = ""          # "discovery" | "heldout"


# --------------------------------------------------------------------------- sources
# (genre, repo_id, filename, row -> text). File paths were checked against the Hub.
def _wiki_articles(df) -> Iterator[str]:
    head = re.compile(r"^ = [^=].* = $")
    cur: list[str] = []
    for line in df["text"]:
        if head.match(line.rstrip("\n")) and cur:
            yield "".join(cur)
            cur = []
        cur.append(line)
    if cur:
        yield "".join(cur)


def _detok_wikitext(t: str) -> str:
    t = t.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    t = re.sub(r" ([,.;:!?)\]'])", r"\1", t)
    t = re.sub(r"([(\[]) ", r"\1", t)
    t = re.sub(r" n't", "n't", t)
    t = re.sub(r"^ = (.*) = $", r"\1", t, flags=re.M)
    t = re.sub(r"^ = = (.*) = = $", r"\1", t, flags=re.M)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


HF_SOURCES: list[dict] = [
    dict(genre="encyclopedia", repo="Salesforce/wikitext",
         file="wikitext-103-raw-v1/test-00000-of-00001.parquet", kind="wikitext"),
    dict(genre="fiction", repo="roneneldan/TinyStories",
         file="data/validation-00000-of-00001-869c898b519ad725.parquet", kind="col:text"),
    dict(genre="code", repo="openai/openai_humaneval",
         file="openai_humaneval/test-00000-of-00001.parquet", kind="humaneval"),
    dict(genre="news", repo="fancyzhx/ag_news", file="data/test-00000-of-00001.parquet",
         kind="col:text"),
    dict(genre="instruction", repo="databricks/databricks-dolly-15k",
         file="databricks-dolly-15k.jsonl", kind="dolly"),
    dict(genre="math", repo="openai/gsm8k", file="main/test-00000-of-00001.parquet",
         kind="gsm8k"),
    dict(genre="science", repo="allenai/sciq", file="data/test-00000-of-00001.parquet",
         kind="col:support"),
    dict(genre="review", repo="stanfordnlp/imdb", file="plain_text/test-00000-of-00001.parquet",
         kind="col:text"),
]


def _texts_from_file(path: str, kind: str) -> list[str]:
    import pandas as pd

    df = pd.read_json(path, lines=True) if path.endswith(".jsonl") else pd.read_parquet(path)
    if kind == "wikitext":
        return [_detok_wikitext(a) for a in _wiki_articles(df)]
    if kind == "humaneval":
        return [p + s for p, s in zip(df["prompt"], df["canonical_solution"])]
    if kind == "dolly":
        return [f"{i}\n\n{r}" for i, r in zip(df["instruction"], df["response"])]
    if kind == "gsm8k":
        return [f"{q}\n{a}" for q, a in zip(df["question"], df["answer"])]
    if kind.startswith("col:"):
        return list(df[kind[4:]])
    raise ValueError(kind)


def _fetch_hf(src: dict, cache_dir: Path, revision: str = "main") -> tuple[list[str], str]:
    from huggingface_hub import HfApi, hf_hub_download

    sha = HfApi().dataset_info(src["repo"], revision=revision).sha
    path = hf_hub_download(src["repo"], src["file"], repo_type="dataset", revision=sha,
                           cache_dir=str(cache_dir / "hf"))
    return _texts_from_file(path, src["kind"]), sha


# --------------------------------------------------------------------------- build
def _truncate(text: str, tokenizer, max_tokens: int) -> tuple[str, int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text, len(ids)
    return tokenizer.decode(ids[:max_tokens]), max_tokens


def sample_genre(texts: list[str], genre: str, source: str, tokenizer, cfg: CorpusConfig,
                 rng: random.Random) -> list[Document]:
    order = list(range(len(texts)))
    rng.shuffle(order)
    docs, total = [], 0
    for i in order:
        t = (texts[i] or "").strip()
        if not t:
            continue
        t, n = _truncate(t, tokenizer, cfg.max_tokens_per_doc)
        if n < cfg.min_tokens_per_doc:
            continue
        docs.append(Document(doc_id=f"{genre}-{i}", genre=genre, source=source, text=t))
        total += n
        if total >= cfg.tokens_per_genre:
            break
    return docs


def split_documents(docs: list[Document], heldout_fraction: float, seed: int) -> list[Document]:
    """Stratified-by-genre document split. Deterministic given ``seed``."""
    rng = random.Random(seed + 1)
    by_genre: dict[str, list[Document]] = {}
    for d in docs:
        by_genre.setdefault(d.genre, []).append(d)
    for genre in sorted(by_genre):
        group = sorted(by_genre[genre], key=lambda d: d.doc_id)
        rng.shuffle(group)
        n_held = max(1, round(heldout_fraction * len(group))) if len(group) > 1 else 0
        for j, d in enumerate(group):
            d.split = "heldout" if j < n_held else "discovery"
    return docs


def build_corpus(cfg: CorpusConfig, tokenizer, seed: int, root: Path,
                 fetch: Callable = _fetch_hf) -> list[Document]:
    """Build (or load from cache) the corpus. Cache key includes tokenizer and settings."""
    cache = root / cfg.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    tok_name = getattr(tokenizer, "name_or_path", "tok").replace("/", "_")
    key = f"{cfg.source}_{tok_name}_{cfg.tokens_per_genre}_{cfg.max_tokens_per_doc}_{seed}"
    out = cache / f"corpus_{key}.jsonl"
    if out.exists():
        return load_jsonl(out)

    rng = random.Random(seed)
    manifest: dict = {"seed": seed, "config": asdict(cfg), "sources": []}
    docs: list[Document] = []
    if cfg.source == "local":
        raw = load_jsonl(root / cfg.local_path, keep_split=False)
        genres = sorted({d.genre for d in raw})
        for g in genres:
            texts = [d.text for d in raw if d.genre == g]
            docs += sample_genre(texts, g, "local", tokenizer, cfg, rng)
        manifest["sources"].append({"local_path": cfg.local_path})
    elif cfg.source == "hf_mixture":
        for src in HF_SOURCES:
            texts, sha = fetch(src, cache)
            docs += sample_genre(texts, src["genre"], src["repo"], tokenizer, cfg, rng)
            manifest["sources"].append({**src, "revision_sha": sha, "n_available": len(texts)})
    else:
        raise ValueError(f"unknown corpus source {cfg.source!r}")

    docs = split_documents(docs, cfg.heldout_fraction, seed)
    save_jsonl(docs, out)
    manifest["n_docs"] = len(docs)
    manifest["genres"] = sorted({d.genre for d in docs})
    (cache / f"manifest_{key}.json").write_text(json.dumps(manifest, indent=2))
    return docs


def save_jsonl(docs: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for d in docs:
            f.write(json.dumps(asdict(d)) + "\n")


def load_jsonl(path: Path, keep_split: bool = True) -> list[Document]:
    docs = []
    with open(path) as f:
        for i, line in enumerate(f):
            if line.strip():
                r = json.loads(line)
                docs.append(Document(doc_id=r.get("doc_id", f"doc-{i}"), genre=r["genre"],
                                     source=r.get("source", "local"), text=r["text"],
                                     split=r.get("split", "") if keep_split else ""))
    return docs
