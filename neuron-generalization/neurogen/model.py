"""Model loading and access to Qwen3 SwiGLU MLP blocks.

Qwen3 MLP (transformers ``Qwen3MLP``)::

    down_proj( act_fn(gate_proj(x)) * up_proj(x) ),   act_fn = SiLU

where ``x`` is the post-attention-RMSNorm residual stream (width ``hidden_size``),
``gate_proj``/``up_proj`` map ``hidden_size -> intermediate_size`` and ``down_proj``
maps back. **A neuron is one coordinate i of the intermediate vector**

    h_i(x) = SiLU(w_gate_i . x) * (w_up_i . x)

i.e. element ``i`` of the *input* to ``down_proj``. Its contribution to the residual
stream is exactly ``h_i * down_proj.weight[:, i]``. We read and intervene on ``h_i``
with a forward *pre*-hook on ``down_proj``. Because ``w_up_i . x`` can take either
sign (and SiLU is slightly negative for negative inputs), ``h_i`` is signed and both
extremes write to the residual stream along +/- the same direction; we therefore
track the positive and negative tails separately.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import ModelConfig
from .hardware import TORCH_DTYPES, check_fits, detect_hardware, estimate_memory_gb


@dataclass
class LoadedModel:
    model: nn.Module
    tokenizer: object
    device: str
    name: str

    @property
    def n_layers(self) -> int:
        return len(decoder_layers(self.model))

    @property
    def d_mlp(self) -> int:
        return get_mlp(self.model, 0).down_proj.in_features


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    base = model.model if hasattr(model, "model") else model
    return base.layers


def get_mlp(model: nn.Module, layer: int) -> nn.Module:
    mlp = decoder_layers(model)[layer].mlp
    for attr in ("gate_proj", "up_proj", "down_proj", "act_fn"):
        if not hasattr(mlp, attr):
            raise TypeError(f"Layer {layer} MLP lacks '{attr}'; not a SwiGLU (Qwen3/Llama-style) MLP")
    return mlp


def resolve_layers(cfg: ModelConfig, n_layers: int) -> list[int]:
    if cfg.layers:
        layers = list(cfg.layers)
    else:
        layers = [min(n_layers - 1, max(0, round(f * (n_layers - 1)))) for f in cfg.layer_fractions]
    if any(l < 0 or l >= n_layers for l in layers):
        raise ValueError(f"layers {layers} out of range for {n_layers}-layer model")
    return sorted(set(layers))


def preflight(cfg: ModelConfig, batch_size: int = 4, seq_len: int = 256, strict: bool = True) -> str:
    """Download only config.json, estimate memory, and refuse to load if it won't fit."""
    from transformers import AutoConfig

    hf_cfg = AutoConfig.from_pretrained(cfg.name, revision=cfg.revision)
    hw = detect_hardware(cfg.device)
    dtype = cfg.dtype
    if hw.device == "cpu" and dtype == "float16":
        dtype = "float32"  # fp16 matmuls are poorly supported on CPU
    est = estimate_memory_gb(hf_cfg, dtype, batch_size, seq_len)
    ok, msg = check_fits(hw, est)
    report = (f"{hw.summary()}\nModel {cfg.name}: {est['n_params_b']:.2f}B params, "
              f"{dtype} weights {est['weights_gb']:.1f} GB\n{msg}")
    if strict and not ok:
        raise MemoryError(report + "\nUse a smaller model, lower dtype, or smaller batch.")
    return report


def load_model(cfg: ModelConfig, check_memory: bool = True, batch_size: int = 4,
               seq_len: int = 256, verbose: bool = True) -> LoadedModel:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if check_memory:
        report = preflight(cfg, batch_size, seq_len, strict=True)
        if verbose:
            print(report)
    hw_device = detect_hardware(cfg.device).device
    dtype = cfg.dtype
    if hw_device == "cpu" and dtype == "float16":
        dtype = "float32"
    tok = AutoTokenizer.from_pretrained(cfg.name, revision=cfg.revision)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.name, revision=cfg.revision, dtype=TORCH_DTYPES[dtype], low_cpu_mem_usage=True)
    model.to(hw_device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return LoadedModel(model=model, tokenizer=tok, device=hw_device, name=cfg.name)


def tiny_qwen3(seed: int = 0, n_layers: int = 3, hidden: int = 32, inter: int = 64,
               vocab: int = 101) -> nn.Module:
    """A randomly initialised miniature Qwen3 for tests (no download)."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=hidden, intermediate_size=inter,
                      num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=8, max_position_embeddings=128, tie_word_embeddings=True)
    m = Qwen3ForCausalLM(cfg).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m
