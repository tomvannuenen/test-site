"""Experiment configuration: YAML files -> plain dataclasses, plus seeding."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3-4B"
    revision: str = "main"
    dtype: str = "bfloat16"          # bfloat16 | float16 | float32
    device: str = "auto"             # auto | mps | cuda | cpu
    # Layers expressed as fractions of depth so the same config works for 4B/8B.
    layer_fractions: list[float] = field(default_factory=lambda: [0.15, 0.5, 0.85])
    layers: list[int] | None = None  # explicit override; takes precedence


@dataclass
class CorpusConfig:
    source: str = "hf_mixture"       # hf_mixture | local
    local_path: str = "data/example/example_corpus.jsonl"
    tokens_per_genre: int = 10_000
    max_tokens_per_doc: int = 256
    min_tokens_per_doc: int = 24
    heldout_fraction: float = 0.25   # documents, not tokens, are split
    cache_dir: str = "cache/corpus"


@dataclass
class DiscoveryConfig:
    batch_size: int = 4
    top_k: int = 40                  # strongest examples kept per neuron per sign
    context_window: int = 24         # tokens shown either side in example cards
    n_candidates: int = 24           # heterogeneity-selected, total across layers
    n_random_baseline: int = 1       # extra random selective neurons per layer (controls)
    min_selectivity: float = 6.0     # (top-k mean) / std; filters noisy neurons
    dedupe_docs: bool = True         # one example per document in diversity metrics
    output_dir: str = "outputs/discovery"


@dataclass
class EvaluationConfig:
    n_test_neurons: int = 8          # Phase 3 subset size, sampled (not hand-picked)
    active_threshold_frac: float = 0.25  # "active" = > frac * median(discovery top-k)
    min_discriminating_items: int = 8
    support_balanced_acc: float = 0.75
    neither_balanced_acc: float = 0.65
    n_bootstrap: int = 2000
    output_dir: str = "outputs/evaluation"


@dataclass
class InterventionConfig:
    n_random_controls: int = 50
    scale_factors: list[float] = field(default_factory=lambda: [0.0, 0.5, 2.0, 4.0])
    control_match_tolerance: float = 0.25   # |act| within ±25% of the target's
    output_dir: str = "outputs/interventions"


@dataclass
class ExperimentConfig:
    seed: int = 1234
    model: ModelConfig = field(default_factory=ModelConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    root: str = "."                 # paths below are relative to this

    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else Path(self.root) / p

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        """Short hash of the settings that determine discovery outputs."""
        d = self.to_dict()
        keep = {k: d[k] for k in ("seed", "model", "corpus")}
        keep["discovery"] = {k: d["discovery"][k] for k in ("top_k", "batch_size")}
        return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:12]


def _merge(dc_cls, data: dict | None):
    data = data or {}
    unknown = set(data) - {f.name for f in dataclasses.fields(dc_cls)}
    if unknown:
        raise ValueError(f"Unknown keys for {dc_cls.__name__}: {sorted(unknown)}")
    return dc_cls(**data)


def load_config(path: str | os.PathLike, root: str | None = None) -> ExperimentConfig:
    """Load a YAML config. ``root`` defaults to the repository root (parent of configs/)."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    cfg = ExperimentConfig(
        seed=raw.get("seed", 1234),
        model=_merge(ModelConfig, raw.get("model")),
        corpus=_merge(CorpusConfig, raw.get("corpus")),
        discovery=_merge(DiscoveryConfig, raw.get("discovery")),
        evaluation=_merge(EvaluationConfig, raw.get("evaluation")),
        intervention=_merge(InterventionConfig, raw.get("intervention")),
        root=root or str(path.resolve().parent.parent),
    )
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass
