"""Simple Matplotlib / HTML views for inspection. No frontend framework."""
from __future__ import annotations

import html

import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993", "#0b0b0b", "#52514e"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
                     "font.size": 9})


def card_html(card: dict, scale: float | None = None) -> str:
    """Tokens shaded by signed activation (blue = positive, orange = negative)."""
    acts = np.array(card["context_acts"])
    s = scale or (np.abs(acts).max() + 1e-9)
    spans = []
    for i, (t, a) in enumerate(zip(card["context_tokens"], acts)):
        alpha = min(1.0, abs(a) / s)
        rgb = "42,120,214" if a >= 0 else "235,104,52"
        border = "border-bottom:2px solid #0b0b0b;" if i == card["focus_index"] else ""
        spans.append(f'<span title="{a:+.3f}" style="background:rgba({rgb},{alpha:.2f});{border}">'
                     f'{html.escape(t).replace(chr(10), "⏎")}</span>')
    head = (f'<b>{card["activation"]:+.2f}</b> <span style="color:#52514e">[{card["genre"]}] '
            f'{card["doc_id"]}@{card["position"]}</span> ')
    return f'<div style="font-family:monospace;font-size:12px;margin:3px 0;white-space:pre-wrap">{head}{"".join(spans)}</div>'


def cards_html(cards: list[dict], ids: bool = True) -> str:
    scale = max(abs(c["activation"]) for c in cards) if cards else 1.0
    return "".join((f"<div><code>#{j}</code>" if ids else "<div>") + card_html(c, scale) + "</div>"
                   for j, c in enumerate(cards))


def plot_genre_distribution(trace: np.ndarray, genres: np.ndarray, polarity: str, threshold: float,
                            ax=None, title: str = ""):
    """Signed activation per genre (all tokens): box plot + top-tail strip."""
    ax = ax or plt.subplots(figsize=(7, 3))[1]
    signed = trace if polarity == "pos" else -trace
    names = sorted(set(genres))
    data = [signed[genres == g] for g in names]
    ax.boxplot(data, showfliers=False, widths=0.5,
               medianprops=dict(color=INK), boxprops=dict(color=MUTED), whiskerprops=dict(color=MUTED),
               capprops=dict(color=MUTED))
    for j, d in enumerate(data):
        top = d[d > threshold]
        ax.scatter(np.full(len(top), j + 1) + np.random.default_rng(j).uniform(-0.15, 0.15, len(top)),
                   top, s=10, color=BLUE, alpha=0.6, linewidths=0)
    ax.axhline(threshold, color=ORANGE, lw=1, ls="--", label=f"active threshold {threshold:.3g}")
    labels = [f"{g}\n{(d > threshold).mean() * 100:.1f}% active" for g, d in zip(names, data)]
    ax.set_xticks(range(1, len(names) + 1), labels, rotation=30, ha="right")
    ax.set_ylabel(f"activation ({polarity} polarity)")
    ax.set_title(title or "Activation by genre (dots: tokens above threshold)", loc="left")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.08), fontsize=7)
    return ax


def plot_predictions(rows: list[dict], threshold: float, ax=None, title: str = ""):
    """Observed target activation per item, grouped by category; marker shows what H1/H2 predicted."""
    ax = ax or plt.subplots(figsize=(7, 3))[1]
    cats = list(dict.fromkeys(r["category"] for r in rows))
    for j, c in enumerate(cats):
        rs = [r for r in rows if r["category"] == c]
        xs = j + np.linspace(-0.25, 0.25, len(rs)) if len(rs) > 1 else np.array([j])
        for x, r in zip(xs, rs):
            ax.scatter(x - 0.04, r["target_act"], marker="^" if r["h1"] == "active" else "v",
                       color=BLUE, s=28, alpha=0.9 if r["h1"] != "unsure" else 0.3, linewidths=0)
            ax.scatter(x + 0.04, r["target_act"], marker="^" if r["h2"] == "active" else "v",
                       color=ORANGE, s=28, alpha=0.9 if r["h2"] != "unsure" else 0.3, linewidths=0)
    ax.axhline(threshold, color=GRAY, lw=1, ls="--")
    ax.set_xticks(range(len(cats)), cats, rotation=20, ha="right")
    ax.set_ylabel("observed activation at target")
    ax.scatter([], [], marker="^", color=BLUE, label="H1 predicts active (▼ inactive)")
    ax.scatter([], [], marker="^", color=ORANGE, label="H2 predicts active (▼ inactive)")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_title(title or "Predictions vs observed (dashed: pre-registered threshold)", loc="left")
    return ax


def plot_control_comparison(res: dict, ax=None, title: str = ""):
    """Histogram of matched random-neuron KL effects with the target's effect marked."""
    ax = ax or plt.subplots(figsize=(5, 2.6))[1]
    c = np.array(res.get("control_kl_mean", []))
    if len(c):
        ax.hist(c, bins=20, color=GRAY, alpha=0.8, label=f"{len(c)} matched controls")
    ax.axvline(res["kl_mean"], color=BLUE, lw=2,
               label=f"target L{res['layer']}N{res['neuron']} ({res.get('percentile_vs_controls', float('nan')):.0f}th pct)")
    ax.set_xlabel("mean KL(baseline || edited), nats")
    ax.set_ylabel("count")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(title or f"{res['mode']}={res['value']:.2g} at positions {res['positions']}", loc="left")
    return ax


def plot_dose_response(results: list[dict], tokenizer=None, ax=None):
    ax = ax or plt.subplots(figsize=(5, 2.6))[1]
    f = [r["factor"] for r in results]
    ax.plot(f, [r["kl_mean"] for r in results], "-o", color=BLUE, lw=2, ms=5, label="mean KL")
    ax.set_xlabel("activation scale factor (1 = unedited)")
    ax.set_ylabel("mean KL, nats")
    ax.axvline(1.0, color=GRAY, lw=1, ls=":")
    ax.set_title("Dose-response", loc="left")
    return ax
