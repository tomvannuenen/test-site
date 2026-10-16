"""Report hardware and estimated memory for each model BEFORE downloading weights.

    python scripts/check_env.py                 # offline: built-in Qwen3 shapes
    python scripts/check_env.py --online Qwen/Qwen3-4B   # fetch config.json only
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import Qwen3Config  # noqa: E402

from neurogen.hardware import check_fits, detect_hardware, estimate_memory_gb  # noqa: E402

# Shapes copied from the official config.json files on the Hugging Face Hub.
KNOWN = {
    "Qwen/Qwen3-0.6B": dict(hidden_size=1024, intermediate_size=3072, num_hidden_layers=28,
                            num_attention_heads=16, num_key_value_heads=8, head_dim=128,
                            vocab_size=151936, tie_word_embeddings=True),
    "Qwen/Qwen3-4B": dict(hidden_size=2560, intermediate_size=9728, num_hidden_layers=36,
                          num_attention_heads=32, num_key_value_heads=8, head_dim=128,
                          vocab_size=151936, tie_word_embeddings=True),
    "Qwen/Qwen3-8B": dict(hidden_size=4096, intermediate_size=12288, num_hidden_layers=36,
                          num_attention_heads=32, num_key_value_heads=8, head_dim=128,
                          vocab_size=151936, tie_word_embeddings=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", nargs="*", default=None, help="model ids to fetch config.json for")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    args = ap.parse_args()

    import torch
    hw = detect_hardware()
    print(f"torch {torch.__version__}; {hw.summary()}")
    print(f"MPS built={torch.backends.mps.is_built()} available={torch.backends.mps.is_available()}; "
          f"CUDA available={torch.cuda.is_available()}\n")
    if args.online is not None:
        from transformers import AutoConfig
        cfgs = {m: AutoConfig.from_pretrained(m) for m in (args.online or KNOWN)}
    else:
        cfgs = {m: Qwen3Config(**kw) for m, kw in KNOWN.items()}
    for name, cfg in cfgs.items():
        for dtype in ("bfloat16", "float32"):
            est = estimate_memory_gb(cfg, dtype, args.batch_size, args.seq_len)
            ok, msg = check_fits(hw, est)
            print(f"{name:18s} {dtype:9s} params {est['n_params_b']:.2f}B  weights {est['weights_gb']:5.1f} GB  "
                  f"layers {cfg.num_hidden_layers} d_mlp {cfg.intermediate_size}  -> {msg}")
    print("\nNotes: discovery skips the LM head (no logits), so its peak is ~weights + activations.\n"
          "On CPU use bfloat16 or float32 (float16 is auto-promoted). On MPS bfloat16 is recommended.")


if __name__ == "__main__":
    main()
