"""Forward hooks for reading and editing SwiGLU neuron activations.

Everything here hooks ``mlp.down_proj`` with a forward **pre**-hook, so the tensor
seen is exactly ``act_fn(gate_proj(x)) * up_proj(x)`` of shape
``[batch, seq, intermediate_size]`` -- the neuron activations defined in
:mod:`neurogen.model`. Editing that tensor in the pre-hook is therefore an
intervention "immediately before the down projection".
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch
from torch import nn

from .model import decoder_layers, get_mlp

ActCallback = Callable[[int, torch.Tensor], None]


class _StopForward(Exception):
    pass


class NeuronRecorder:
    """Record neuron activations (and optionally the MLP input) for selected layers.

    Parameters
    ----------
    neurons: mapping layer -> neuron indices to keep, or ``None`` for all neurons.
        Keeping a subset avoids materialising the full ``[B, T, d_mlp]`` tensor on CPU.
    callback: if given, called as ``callback(layer, acts)`` instead of storing,
        which lets discovery stream over a corpus without retaining activations.
    keep_mlp_input: also store the MLP input ``x`` (post-RMSNorm residual).
    """

    def __init__(self, model: nn.Module, layers: Iterable[int],
                 neurons: dict[int, Sequence[int]] | None = None,
                 callback: ActCallback | None = None, keep_mlp_input: bool = False,
                 to_cpu: bool = True):
        self.model = model
        self.layers = sorted(set(layers))
        self.neurons = {l: torch.as_tensor(list(v), dtype=torch.long)
                        for l, v in (neurons or {}).items()}
        self.callback = callback
        self.keep_mlp_input = keep_mlp_input
        self.to_cpu = to_cpu
        self.acts: dict[int, torch.Tensor] = {}
        self.mlp_inputs: dict[int, torch.Tensor] = {}
        self._handles: list = []

    def _act_hook(self, layer: int):
        def hook(_module, args):
            h = args[0].detach()
            if layer in self.neurons:
                h = h[..., self.neurons[layer].to(h.device)]
            h = h.float()
            if self.to_cpu:
                h = h.cpu()
            if self.callback is not None:
                self.callback(layer, h)
            else:
                self.acts[layer] = h
        return hook

    def _input_hook(self, layer: int):
        def hook(_module, args):
            x = args[0].detach().float()
            self.mlp_inputs[layer] = x.cpu() if self.to_cpu else x
        return hook

    def __enter__(self):
        for l in self.layers:
            mlp = get_mlp(self.model, l)
            self._handles.append(mlp.down_proj.register_forward_pre_hook(self._act_hook(l)))
            if self.keep_mlp_input:
                self._handles.append(mlp.register_forward_pre_hook(self._input_hook(l)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


@contextmanager
def stop_after_layer(model: nn.Module, layer: int):
    """Abort the forward pass after decoder layer ``layer`` (saves compute and logits)."""
    def hook(_m, _a, _o):
        raise _StopForward
    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        yield
    except _StopForward:
        pass
    finally:
        handle.remove()


@torch.no_grad()
def record_activations(model: nn.Module, input_ids: torch.Tensor, layers: Iterable[int],
                       attention_mask: torch.Tensor | None = None,
                       neurons: dict[int, Sequence[int]] | None = None,
                       keep_mlp_input: bool = False, early_stop: bool = True) -> NeuronRecorder:
    """One forward pass; returns the recorder with ``.acts[layer] -> [B, T, n]`` (float32, CPU)."""
    layers = sorted(set(layers))
    rec = NeuronRecorder(model, layers, neurons=neurons, keep_mlp_input=keep_mlp_input)
    base = model.model if hasattr(model, "model") else model
    with rec:
        if early_stop:
            with stop_after_layer(model, layers[-1]):
                base(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        else:
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return rec


@dataclass
class NeuronEdit:
    """Edit neuron ``neuron`` of layer ``layer`` before ``down_proj``.

    mode: ``"set"`` (h <- value; value=0 is zero-ablation, value=mean is mean-ablation),
          ``"scale"`` (h <- value * h) or ``"add"`` (h <- h + value).
    positions: token positions to edit; ``None`` = all positions.
    batch_rows: batch rows to edit; ``None`` = all rows (lets one batched forward pass
        carry a different edit per row, e.g. for random-neuron controls).
    """
    layer: int
    neuron: int
    mode: str = "set"
    value: float = 0.0
    positions: Sequence[int] | None = None
    batch_rows: Sequence[int] | None = None

    def apply(self, h: torch.Tensor) -> torch.Tensor:
        B, T, _ = h.shape
        rows = torch.arange(B) if self.batch_rows is None else torch.as_tensor(list(self.batch_rows))
        pos = torch.arange(T) if self.positions is None else torch.as_tensor(list(self.positions))
        idx = (rows[:, None].to(h.device), pos[None, :].to(h.device), self.neuron)
        cur = h[idx]
        if self.mode == "set":
            new = torch.full_like(cur, self.value)
        elif self.mode == "scale":
            new = cur * self.value
        elif self.mode == "add":
            new = cur + self.value
        else:
            raise ValueError(f"unknown mode {self.mode!r}")
        h[idx] = new
        return h


class Intervention:
    """Context manager applying one or more :class:`NeuronEdit` during forward passes."""

    def __init__(self, model: nn.Module, edits: Sequence[NeuronEdit]):
        self.model = model
        self.edits = list(edits)
        self._handles: list = []

    def __enter__(self):
        by_layer: dict[int, list[NeuronEdit]] = {}
        for e in self.edits:
            by_layer.setdefault(e.layer, []).append(e)
        for layer, edits in by_layer.items():
            def hook(_module, args, _edits=edits):
                h = args[0].clone()
                for e in _edits:
                    h = e.apply(h)
                return (h,) + tuple(args[1:])
            down = get_mlp(self.model, layer).down_proj
            # prepend=True: edits run before any recorder hook on the same module,
            # so a recorder sees the *edited* activation.
            self._handles.append(down.register_forward_pre_hook(hook, prepend=True))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False
