"""Neuron definition, recording and intervention correctness on a tiny Qwen3."""
import torch
import torch.nn.functional as F

from neurogen.hooks import Intervention, NeuronEdit, NeuronRecorder, record_activations
from neurogen.interventions import intervention_effect, kl_per_position, logprobs
from neurogen.model import get_mlp

IDS = torch.tensor([[5, 17, 33, 2, 64, 9, 41, 12, 77, 3]])
LAYER, NEURON = 1, 7


def test_neuron_is_silu_gate_times_up(tiny_model):
    """The recorded activation equals SiLU(gate(x)) * up(x) for the MLP input x."""
    rec = record_activations(tiny_model, IDS, [LAYER], keep_mlp_input=True, early_stop=False)
    mlp = get_mlp(tiny_model, LAYER)
    x = rec.mlp_inputs[LAYER]
    manual = F.silu(mlp.gate_proj(x)) * mlp.up_proj(x)
    assert torch.allclose(rec.acts[LAYER], manual, atol=1e-6)
    assert rec.acts[LAYER].shape == (1, IDS.shape[1], mlp.down_proj.in_features)


def test_neuron_subset_and_early_stop_match_full(tiny_model):
    full = record_activations(tiny_model, IDS, [0, LAYER], early_stop=False).acts
    sub = record_activations(tiny_model, IDS, [0, LAYER], neurons={LAYER: [NEURON, 3]}).acts
    assert torch.allclose(sub[LAYER], full[LAYER][..., [NEURON, 3]])
    assert torch.allclose(sub[0], full[0])


def test_ablation_modifies_exactly_the_intended_activation(tiny_model):
    """Zero-ablation at chosen positions changes only (positions, NEURON) of the
    down_proj input, and the down_proj output changes by exactly -h * W_down[:, i]."""
    positions = [3, 6]
    mlp = get_mlp(tiny_model, LAYER)
    outs = {}

    def grab(name):
        def hook(_m, _a, out):
            outs[name] = out.detach().clone()
        return hook

    h = mlp.down_proj.register_forward_hook(grab("base"))
    base = record_activations(tiny_model, IDS, [LAYER], early_stop=False).acts[LAYER]
    h.remove()

    h = mlp.down_proj.register_forward_hook(grab("edit"))
    with Intervention(tiny_model, [NeuronEdit(LAYER, NEURON, "set", 0.0, positions)]):
        edited = record_activations(tiny_model, IDS, [LAYER], early_stop=False).acts[LAYER]
    h.remove()

    # The recorder runs after the edit hook, so it sees the edited intermediate tensor.
    assert torch.all(edited[0, positions, NEURON] == 0)
    assert torch.all(base[0, positions, NEURON] != 0)
    mask = torch.ones_like(base, dtype=torch.bool)
    mask[0, positions, NEURON] = False
    assert torch.equal(edited[mask], base[mask]), "other neurons/positions must be untouched"

    expected = outs["base"].clone()
    w = mlp.down_proj.weight[:, NEURON]
    for p in positions:
        expected[0, p] -= base[0, p, NEURON] * w
    assert torch.allclose(outs["edit"], expected, atol=1e-5)


def test_ablation_is_causal_in_position(tiny_model):
    """Editing at position p cannot change predictions at positions < p, and does change them at >= p."""
    base = logprobs(tiny_model, IDS)
    new = logprobs(tiny_model, IDS, [NeuronEdit(LAYER, NEURON, "scale", 10.0, [5])])
    kl = kl_per_position(base[0], new[0])
    assert torch.all(kl[:5].abs() < 1e-6)
    assert kl[5:].max() > 1e-6


def test_scale_one_is_identity_and_hooks_are_removed(tiny_model):
    base = logprobs(tiny_model, IDS)
    same = logprobs(tiny_model, IDS, [NeuronEdit(LAYER, NEURON, "scale", 1.0)])
    assert torch.allclose(base, same, atol=1e-6)
    mlp = get_mlp(tiny_model, LAYER)
    assert len(mlp.down_proj._forward_pre_hooks) == 0


def test_batch_rows_edit_only_their_row(tiny_model):
    ids = IDS.expand(3, -1)
    with Intervention(tiny_model, [NeuronEdit(LAYER, NEURON, "set", 0.0, [2], batch_rows=[1])]):
        acts = record_activations(tiny_model, ids, [LAYER], early_stop=False).acts[LAYER]
    assert acts[1, 2, NEURON] == 0
    assert acts[0, 2, NEURON] != 0 and acts[2, 2, NEURON] != 0
    assert torch.allclose(acts[0], acts[2])


def test_mean_and_add_modes(tiny_model):
    base = record_activations(tiny_model, IDS, [LAYER], early_stop=False).acts[LAYER]
    with Intervention(tiny_model, [NeuronEdit(LAYER, NEURON, "add", 0.5, [1])]):
        a = record_activations(tiny_model, IDS, [LAYER], early_stop=False).acts[LAYER]
    assert torch.isclose(a[0, 1, NEURON], base[0, 1, NEURON] + 0.5, atol=1e-6)


def test_intervention_effect_and_controls(tiny_model):
    res = intervention_effect(tiny_model, IDS, LAYER, NEURON, [4], "set", 0.0,
                              track_token_ids=[5, 9], n_controls=6, tolerance=0.5, seed=0)
    assert res["kl_before_edit_max"] < 1e-6
    assert len(res["kl"]) == IDS.shape[1] - 4
    assert NEURON not in res["control_neurons"] and len(res["control_kl_mean"]) == len(res["control_neurons"])
    assert 0 <= res["percentile_vs_controls"] <= 100
    assert set(res["delta_logprob"]) == {5, 9}


def test_recorder_callback_streams(tiny_model):
    seen = []
    with NeuronRecorder(tiny_model, [0, 2], callback=lambda l, h: seen.append((l, h.shape))):
        tiny_model(input_ids=IDS)
    assert [s[0] for s in seen] == [0, 2]
