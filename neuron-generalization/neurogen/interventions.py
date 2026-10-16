"""Phase 4: causal interventions on single SwiGLU neurons.

Edits act on ``h = act_fn(gate_proj(x)) * up_proj(x)`` immediately before
``down_proj`` (see :class:`neurogen.hooks.NeuronEdit`). We compare next-token
distributions against an unmodified baseline (KL divergence, log-prob changes of
chosen tokens) and against **matched random-neuron controls**: other neurons in the
same layer whose |activation| at the edited positions is similar to the target's,
edited in the same way. Matching matters: ablating a neuron that is ~0 at a
position trivially does nothing, so an unmatched control would make any active
neuron look important.

Interpretation caveat (also in PROTOCOL.md): a large, specific causal effect shows
the neuron *matters* for the model's output in that context. It does not show what
the neuron "means", and it does not by itself favour H1 or H2.
"""
from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .hooks import Intervention, NeuronEdit, record_activations
from .model import get_mlp


@torch.no_grad()
def logprobs(model, input_ids: torch.Tensor, edits: Sequence[NeuronEdit] = ()) -> torch.Tensor:
    """Next-token log-probabilities [B, T, V] (float32), optionally under edits."""
    with Intervention(model, edits):
        logits = model(input_ids=input_ids, use_cache=False).logits
    return F.log_softmax(logits.float(), dim=-1)


def kl_per_position(base_lp: torch.Tensor, new_lp: torch.Tensor) -> torch.Tensor:
    """KL(base || new) in nats, per position. Inputs [..., T, V] log-probs.
    Computed in float64: single-neuron effects can be ~1e-7 nats, below float32 round-off."""
    b, n = base_lp.double(), new_lp.double()
    return (b.exp() * (b - n)).sum(-1).clamp_min(0.0).float()


def _resolve_value(mode: str, value, neuron: int, layer_means: np.ndarray | None) -> float:
    if value == "mean":
        if layer_means is None:
            raise ValueError("mean-ablation needs layer_means (from discovery stats)")
        return float(layer_means[neuron])
    return float(value)


@torch.no_grad()
def matched_controls(model, input_ids: torch.Tensor, layer: int, neuron: int,
                     positions: Sequence[int], n: int, tolerance: float, seed: int) -> list[int]:
    """Random neurons in ``layer`` with mean |activation| at ``positions`` within
    ±tolerance (relative) of the target's; widens the band if too few qualify."""
    rec = record_activations(model, input_ids[:1], [layer])
    a = rec.acts[layer][0, list(positions)].abs().mean(0).numpy()   # [d_mlp]
    t = a[neuron]
    rng = random.Random(seed)
    tol = tolerance
    while True:
        pool = [int(i) for i in np.nonzero(np.abs(a - t) <= tol * max(t, 1e-6))[0] if i != neuron]
        if len(pool) >= n or tol > 64:
            break
        tol *= 2
    return sorted(rng.sample(pool, min(n, len(pool))))


@torch.no_grad()
def intervention_effect(model, input_ids: torch.Tensor, layer: int, neuron: int,
                        positions: Sequence[int], mode: str = "set", value=0.0,
                        track_token_ids: Sequence[int] = (), layer_means: np.ndarray | None = None,
                        n_controls: int = 0, tolerance: float = 0.25, seed: int = 0,
                        control_batch: int = 8, return_delta: bool = False) -> dict:
    """Effect of editing one neuron at ``positions`` on next-token distributions.

    Measured at every position >= min(positions) (earlier positions cannot be
    affected in a causal LM; we report their max KL as a sanity check).
    """
    input_ids = input_ids[:1]
    T = input_ids.shape[1]
    p0 = min(positions)
    base = logprobs(model, input_ids)[0]
    edit = NeuronEdit(layer, neuron, mode, _resolve_value(mode, value, neuron, layer_means), positions)
    new = logprobs(model, input_ids, [edit])[0]
    kl = kl_per_position(base, new)
    track = list(track_token_ids)
    res = dict(layer=layer, neuron=neuron, positions=list(positions), mode=mode, value=edit.value,
               kl=kl[p0:].tolist(), kl_mean=float(kl[p0:].mean()), kl_max=float(kl[p0:].max()),
               kl_before_edit_max=float(kl[:p0].max()) if p0 > 0 else 0.0,
               delta_logprob={int(t): (new[p0:, t] - base[p0:, t]).tolist() for t in track},
               measured_positions=list(range(p0, T)))
    # Largest changes in log-prob at the first edited position.
    d = (new[p0] - base[p0])
    res["top_increased"] = d.topk(10).indices.tolist()
    res["top_decreased"] = (-d).topk(10).indices.tolist()
    if return_delta:
        res["delta_first"] = d.cpu()   # full-vocab change, for effect_similarity()

    if n_controls:
        ctrl = matched_controls(model, input_ids, layer, neuron, positions, n_controls, tolerance, seed)
        ctrl_kl = []
        for s in range(0, len(ctrl), control_batch):
            chunk = ctrl[s:s + control_batch]
            ids = input_ids.expand(len(chunk), -1)
            edits = [NeuronEdit(layer, c, mode, _resolve_value(mode, value, c, layer_means),
                                positions, batch_rows=[r]) for r, c in enumerate(chunk)]
            lp = logprobs(model, ids, edits)
            ctrl_kl += kl_per_position(base[None].expand_as(lp), lp)[:, p0:].mean(-1).tolist()
        ctrl_kl = np.array(ctrl_kl)
        res.update(control_neurons=ctrl, control_kl_mean=ctrl_kl.tolist(),
                   percentile_vs_controls=float((ctrl_kl < res["kl_mean"]).mean() * 100) if len(ctrl_kl) else np.nan,
                   z_vs_controls=float((res["kl_mean"] - ctrl_kl.mean()) / (ctrl_kl.std() + 1e-12))
                   if len(ctrl_kl) > 1 else np.nan)
    return res


@torch.no_grad()
def dose_response(model, input_ids, layer, neuron, positions, factors: Sequence[float],
                  track_token_ids: Sequence[int] = ()) -> list[dict]:
    """KL and tracked log-prob change as the activation is scaled by each factor."""
    return [intervention_effect(model, input_ids, layer, neuron, positions, "scale", f,
                                track_token_ids) | {"factor": f} for f in factors]


@torch.no_grad()
def direct_logit_effect(model, layer: int, neuron: int, k: int = 15) -> dict:
    """Tokens most promoted / suppressed by the neuron's output direction, read
    straight through the unembedding (scaled by the final norm's gain).

    Approximate and *descriptive*: ignores all downstream layers and the per-token
    RMS normalisation. Useful as a hint, not as evidence for either hypothesis.
    """
    w = get_mlp(model, layer).down_proj.weight[:, neuron].float()
    g = model.model.norm.weight.float()
    U = model.get_output_embeddings().weight.float()
    s = U @ (w * g)
    return dict(promoted=s.topk(k).indices.tolist(), promoted_scores=s.topk(k).values.tolist(),
                suppressed=(-s).topk(k).indices.tolist(), suppressed_scores=(-(-s).topk(k).values).tolist())


def effect_similarity(deltas: list[torch.Tensor]) -> np.ndarray:
    """Cosine similarity between full-vocab log-prob change vectors from different contexts.

    Compare effects in contexts that H1 assigns to *different* features. High
    similarity across them is compatible with a shared downstream role; low
    similarity means the model uses the neuron's (fixed) output direction
    differently by context. Neither pattern alone decides H1 vs H2.
    """
    M = torch.stack([d / (d.norm() + 1e-12) for d in deltas])
    return (M @ M.T).cpu().numpy()
