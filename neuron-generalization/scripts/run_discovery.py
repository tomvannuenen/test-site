"""Phase 1: stream the discovery split, select candidates, save inspection files.

    python scripts/run_discovery.py --config configs/qwen3_4b.yaml
"""
import argparse

from _common import setup

from neurogen.pipeline import run_discovery

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/qwen3_4b.yaml")
args = ap.parse_args()
cfg, lm = setup(args.config)
out = run_discovery(cfg, lm)
print(f"\nDone. Inspect {out / 'discovery' / 'candidates.md'} or notebooks/01_inspect_and_hypothesize.ipynb")
print("Do NOT open heldout_SEALED/ before freezing hypotheses.")
