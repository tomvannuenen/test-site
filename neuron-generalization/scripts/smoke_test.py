"""Smallest end-to-end experiment: one model, one neuron, a few sentences.

1. Record neuron (layer, idx) on a handful of sentences; print per-token activations.
2. Verify the hook reads exactly SiLU(gate(x)) * up(x) on this model.
3. Zero-ablate the neuron at its peak token; verify the edit landed, report KL and
   compare with matched random-neuron controls.

    python scripts/smoke_test.py --tiny                         # random tiny Qwen3, no download
    python scripts/smoke_test.py --model Qwen/Qwen3-0.6B        # real weights (~1.2 GB)
    python scripts/smoke_test.py --model Qwen/Qwen3-4B --layer 18 --neuron 1234
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from neurogen.config import ModelConfig, set_seed  # noqa: E402
from neurogen.hooks import Intervention, NeuronEdit, record_activations  # noqa: E402
from neurogen.interventions import intervention_effect  # noqa: E402
from neurogen.model import get_mlp, load_model, tiny_qwen3  # noqa: E402

SENTENCES = [
    "The river flows into the sea near the old harbour.",
    "def add(a, b):\n    return a + b",
    "She opened the door and found the room empty.",
    "Shares fell nine percent after the forecast was cut.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--neuron", type=int, default=None)
    args = ap.parse_args()
    set_seed(0)

    if args.tiny:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        from conftest import CharTokenizer
        model, tok, device = tiny_qwen3(), CharTokenizer(), "cpu"
    else:
        lm = load_model(ModelConfig(name=args.model, dtype=args.dtype), batch_size=1, seq_len=64)
        model, tok, device = lm.model, lm.tokenizer, lm.device
    n_layers = len(model.model.layers)
    layer = args.layer if args.layer is not None else n_layers // 2
    mlp = get_mlp(model, layer)
    print(f"model has {n_layers} layers, d_mlp={mlp.down_proj.in_features}; using layer {layer}")

    encoded = [tok(s, add_special_tokens=False)["input_ids"] for s in SENTENCES]
    # Pick the neuron with the largest |activation| on the first sentence unless given.
    first = record_activations(model, torch.tensor([encoded[0]], device=device), [layer]).acts[layer][0]
    neuron = args.neuron if args.neuron is not None else int(first.abs().max(0).values.argmax())
    print(f"neuron {neuron}\n")

    # 1. per-token activations
    for s, ids in zip(SENTENCES, encoded):
        a = record_activations(model, torch.tensor([ids], device=device), [layer],
                               neurons={layer: [neuron]}).acts[layer][0, :, 0]
        toks = [tok.decode([t]) for t in ids]
        print(" ".join(f"{t!r}:{v:+.2f}" for t, v in zip(toks, a.tolist())), "\n")

    # 2. hook reads SiLU(gate(x)) * up(x)
    ids = torch.tensor([encoded[0]], device=device)
    rec = record_activations(model, ids, [layer], keep_mlp_input=True)
    x = rec.mlp_inputs[layer].to(device=device, dtype=mlp.gate_proj.weight.dtype)
    manual = (F.silu(mlp.gate_proj(x)) * mlp.up_proj(x)).float().cpu()
    err = (manual - rec.acts[layer]).abs().max().item()
    print(f"[check] max |hook - SiLU(gate x)*up x| = {err:.2e}")
    assert err < 1e-3 * (manual.abs().max().item() + 1e-6), "hook does not match manual SwiGLU"

    # 3. ablation lands where intended
    peak = int(rec.acts[layer][0, :, neuron].abs().argmax())
    with Intervention(model, [NeuronEdit(layer, neuron, "set", 0.0, [peak])]):
        edited = record_activations(model, ids, [layer]).acts[layer][0]
    changed = (edited != rec.acts[layer][0]).nonzero().tolist()
    print(f"[check] ablation at position {peak}: changed entries {changed} (expected [[{peak}, {neuron}]])")
    assert changed == [[peak, neuron]]

    res = intervention_effect(model, ids, layer, neuron, [peak], "set", 0.0, n_controls=20, seed=0)
    print(f"[effect] mean KL over positions >= {peak}: {res['kl_mean']:.4g} nats "
          f"(max {res['kl_max']:.4g}); KL before edit {res['kl_before_edit_max']:.1e} (must be ~0)")
    print(f"[effect] vs {len(res['control_neurons'])} matched controls: "
          f"{res['percentile_vs_controls']:.0f}th percentile, z={res['z_vs_controls']:.2f}")
    print("top increased next tokens:", [tok.decode([t]) for t in res["top_increased"][:8]])
    print("top decreased next tokens:", [tok.decode([t]) for t in res["top_decreased"][:8]])
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
