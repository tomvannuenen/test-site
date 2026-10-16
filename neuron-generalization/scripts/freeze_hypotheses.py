"""Freeze hypothesis records (validates falsifiability, writes SHA-256 hashes).

    python scripts/freeze_hypotheses.py [--cids L18N1234+ ...]
"""
import argparse
import json

from _common import ROOT

from neurogen.hypotheses import freeze

ap = argparse.ArgumentParser()
ap.add_argument("--hyp-dir", default=str(ROOT / "research" / "hypotheses"))
ap.add_argument("--cids", nargs="*", default=None)
args = ap.parse_args()
frozen = freeze(args.hyp_dir, args.cids)
print(json.dumps(frozen, indent=1))
