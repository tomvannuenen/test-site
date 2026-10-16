import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurogen.config import load_config, set_seed  # noqa: E402
from neurogen.model import load_model  # noqa: E402


def setup(config_path: str, need_model: bool = True):
    cfg = load_config(config_path, root=str(ROOT))
    set_seed(cfg.seed)
    lm = load_model(cfg.model, batch_size=cfg.discovery.batch_size,
                    seq_len=cfg.corpus.max_tokens_per_doc) if need_model else None
    return cfg, lm
