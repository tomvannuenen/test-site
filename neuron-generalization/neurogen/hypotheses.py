"""Phase 2: recording competing hypotheses, and freezing them before evaluation.

A hypothesis record holds, for one candidate (layer, neuron, polarity):

* **H1 (superposition)** -- two or more relatively independent features, each with
  a description and the discovery examples it is meant to cover.
* **H2 (recursive generalization)** -- one abstraction meant to cover all examples.
* For *each* hypothesis: what it predicts should activate and what it predicts
  should NOT activate. Records without both are rejected at freeze time, because a
  hypothesis that forbids nothing cannot be tested.
* A prior credence over {H1, H2, neither}, and a free-text "neither" note.

``freeze()`` writes ``FROZEN.json`` with a SHA-256 of every record. Evaluation code
refuses to score a test set whose ``hypothesis_sha`` does not match the frozen
record, so hypotheses cannot be silently edited after seeing evaluation data.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Feature:
    name: str
    description: str
    example_ids: list[int] = field(default_factory=list)   # indices into discovery cards


@dataclass
class HypothesisRecord:
    cid: str
    layer: int
    neuron: int
    polarity: str
    researcher: str = ""
    h1_features: list[Feature] = field(default_factory=list)
    h1_predicts_active: str = ""
    h1_predicts_inactive: str = ""
    h2_abstraction: str = ""
    h2_description: str = ""
    h2_example_ids: list[int] = field(default_factory=list)
    h2_predicts_active: str = ""
    h2_predicts_inactive: str = ""
    unexplained_example_ids: list[int] = field(default_factory=list)
    neither_note: str = ""
    prior: dict = field(default_factory=lambda: {"h1": None, "h2": None, "neither": None})
    discovery_cards_sha: str = ""    # which example set the researcher looked at
    created_at: str = ""

    def problems(self) -> list[str]:
        p = []
        if len(self.h1_features) < 2 and not self.neither_note:
            p.append("H1 needs >= 2 features (or explain in neither_note why you can't form one)")
        if not self.h2_abstraction and not self.neither_note:
            p.append("H2 needs an abstraction (or explain in neither_note)")
        for h in ("h1", "h2"):
            if (getattr(self, f"{h}_features" if h == "h1" else "h2_abstraction")
                    and not (getattr(self, f"{h}_predicts_active") and getattr(self, f"{h}_predicts_inactive"))):
                p.append(f"{h.upper()} must state both what it predicts active AND inactive")
        pr = self.prior
        if any(pr.get(k) is None for k in ("h1", "h2", "neither")):
            p.append("prior credences for h1/h2/neither must be recorded")
        elif abs(sum(pr.values()) - 1) > 1e-6:
            p.append("prior credences must sum to 1")
        if not self.discovery_cards_sha:
            p.append("discovery_cards_sha missing (record which examples you saw)")
        return p

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "HypothesisRecord":
        d = dict(d)
        d["h1_features"] = [Feature(**f) for f in d.get("h1_features", [])]
        return cls(**d)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_record(rec: HypothesisRecord, directory: Path) -> Path:
    directory = Path(directory)
    if (directory / "FROZEN.json").exists():
        frozen = json.loads((directory / "FROZEN.json").read_text())
        if rec.cid in frozen["records"]:
            raise PermissionError(f"{rec.cid} is frozen; write a new, versioned record instead")
    directory.mkdir(parents=True, exist_ok=True)
    rec.created_at = rec.created_at or _dt.datetime.now(_dt.timezone.utc).isoformat()
    path = directory / f"{rec.cid}.json"
    path.write_text(rec.to_json())
    return path


def load_record(directory: Path, cid: str) -> HypothesisRecord:
    return HypothesisRecord.from_dict(json.loads((Path(directory) / f"{cid}.json").read_text()))


def freeze(directory: Path, cids: list[str] | None = None) -> dict:
    """Validate and freeze records. Already-frozen records are kept unchanged."""
    directory = Path(directory)
    fpath = directory / "FROZEN.json"
    frozen = json.loads(fpath.read_text()) if fpath.exists() else {"records": {}}
    files = sorted(directory.glob("L*N*.json"))
    targets = [f for f in files if cids is None or f.stem in cids]
    errors = {}
    for f in targets:
        if f.stem in frozen["records"]:
            continue
        rec = HypothesisRecord.from_dict(json.loads(f.read_text()))
        if rec.problems():
            errors[rec.cid] = rec.problems()
    if errors:
        raise ValueError(f"Cannot freeze; fix these records first: {json.dumps(errors, indent=1)}")
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    for f in targets:
        frozen["records"].setdefault(f.stem, {"sha256": sha256_file(f), "frozen_at": now})
    fpath.write_text(json.dumps(frozen, indent=2, sort_keys=True))
    return frozen


def verify_frozen(directory: Path, cid: str) -> str:
    """Return the frozen SHA for ``cid``; raise if missing or modified since freezing."""
    directory = Path(directory)
    fpath = directory / "FROZEN.json"
    if not fpath.exists():
        raise PermissionError("No FROZEN.json: freeze hypotheses before evaluation")
    frozen = json.loads(fpath.read_text())["records"]
    if cid not in frozen:
        raise PermissionError(f"{cid} has not been frozen")
    actual = sha256_file(directory / f"{cid}.json")
    if actual != frozen[cid]["sha256"]:
        raise PermissionError(f"{cid}.json was modified after freezing")
    return actual
