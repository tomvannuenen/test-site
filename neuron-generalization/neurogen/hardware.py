"""Hardware detection and model-memory estimation, run *before* loading weights."""
from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass

import torch

DTYPE_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2}
TORCH_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass
class HardwareInfo:
    device: str
    system: str
    machine: str
    cpu_count: int
    total_ram_gb: float
    available_ram_gb: float | None
    accelerator_mem_gb: float | None   # CUDA free memory / MPS recommended max

    def summary(self) -> str:
        avail = f"{self.available_ram_gb:.1f}" if self.available_ram_gb is not None else "?"
        acc = f", accelerator budget {self.accelerator_mem_gb:.1f} GB" if self.accelerator_mem_gb else ""
        return (f"device={self.device} ({self.system}/{self.machine}, {self.cpu_count} CPUs), "
                f"RAM total {self.total_ram_gb:.1f} GB, available {avail} GB{acc}")


def _total_ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):  # pragma: no cover (Windows)
        return float("nan")


def _available_ram_gb() -> float | None:
    if os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 / 1e9
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            page = 16384 if platform.machine() == "arm64" else 4096
            for line in out.splitlines():
                if "page size of" in line:
                    page = int(line.split("page size of")[1].split()[0])
            pages = 0
            for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
                for line in out.splitlines():
                    if line.startswith(key):
                        pages += int(line.split(":")[1].strip().rstrip("."))
            return pages * page / 1e9
        except Exception:  # pragma: no cover
            return None
    return None


def pick_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def detect_hardware(preference: str = "auto") -> HardwareInfo:
    device = pick_device(preference)
    acc = None
    if device == "cuda":
        free, _total = torch.cuda.mem_get_info()
        acc = free / 1e9
    elif device == "mps" and hasattr(torch.mps, "recommended_max_memory"):
        try:
            acc = torch.mps.recommended_max_memory() / 1e9
        except Exception:  # pragma: no cover
            acc = None
    return HardwareInfo(
        device=device,
        system=platform.system(),
        machine=platform.machine(),
        cpu_count=os.cpu_count() or 1,
        total_ram_gb=_total_ram_gb(),
        available_ram_gb=_available_ram_gb(),
        accelerator_mem_gb=acc,
    )


def count_params_from_config(cfg) -> int:
    """Parameter count of a Qwen3-style decoder from its HF config (no weights needed)."""
    d, n_layers, inter = cfg.hidden_size, cfg.num_hidden_layers, cfg.intermediate_size
    n_heads, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or d // n_heads
    vocab = cfg.vocab_size
    attn = d * n_heads * head_dim + 2 * d * n_kv * head_dim + n_heads * head_dim * d
    attn += 2 * head_dim  # q_norm / k_norm
    mlp = 3 * d * inter
    per_layer = attn + mlp + 2 * d
    embed = vocab * d
    lm_head = 0 if getattr(cfg, "tie_word_embeddings", False) else vocab * d
    return embed + lm_head + n_layers * per_layer + d


def estimate_memory_gb(cfg, dtype: str, batch_size: int, seq_len: int) -> dict[str, float]:
    """Rough peak-memory estimate for inference (weights + working activations)."""
    n_params = count_params_from_config(cfg)
    b = DTYPE_BYTES[dtype]
    weights = n_params * b / 1e9
    # Working set for one forward pass without KV cache retention: a few residual-
    # width and MLP-width buffers plus fp32 logits for the whole batch.
    act = batch_size * seq_len * (8 * cfg.hidden_size + 3 * cfg.intermediate_size) * b / 1e9
    logits = batch_size * seq_len * cfg.vocab_size * 4 / 1e9
    return {"n_params_b": n_params / 1e9, "weights_gb": weights,
            "activations_gb": act, "logits_gb": logits,
            "total_gb": weights * 1.1 + act + logits}


def check_fits(hw: HardwareInfo, est: dict[str, float], margin_gb: float = 2.0) -> tuple[bool, str]:
    budget = hw.accelerator_mem_gb if hw.device in ("cuda", "mps") and hw.accelerator_mem_gb else None
    if budget is None:
        budget = hw.available_ram_gb if hw.available_ram_gb is not None else hw.total_ram_gb
    need = est["total_gb"] + margin_gb
    ok = need <= budget
    msg = (f"Estimated peak {est['total_gb']:.1f} GB (+{margin_gb:.0f} GB margin) "
           f"vs budget {budget:.1f} GB on {hw.device}: {'OK' if ok else 'DOES NOT FIT'}")
    return ok, msg
