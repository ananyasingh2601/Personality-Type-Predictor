from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .config import GROUP_ACCENTS, GROUP_ORDER, TYPE_GROUPS


FIG_BG = "#141B28"
AX_BG = "#1A2233"
TEXT = "#D7E0EC"
MUTED = "#9AABBE"
GRID = "#445168"
SPINE = "#5D6B82"


def make_figure(width: float = 5.6, height: float = 4.2):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    return fig, ax


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", linestyle="--", color=GRID, alpha=0.45)


def build_radar_chart(rows: list[dict[str, object]], accent_color: str):
    labels = [row["title"] for row in rows]
    values = [float(row["confidence"]) for row in rows]
    values += values[:1]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.8, 4.8), subplot_kw={"polar": True})
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    ax.plot(angles, values, color=accent_color, linewidth=2.05, solid_capstyle="round")
    ax.fill(angles, values, color=accent_color, alpha=0.14)
    ax.set_thetagrids(
        np.degrees(angles[:-1]),
        labels,
        fontsize=9,
        color=TEXT,
        fontweight="normal",
        fontfamily="DejaVu Sans",
    )
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color=MUTED, fontsize=7)
    ax.tick_params(axis="x", pad=8)
    ax.grid(color=GRID, alpha=0.48, linewidth=0.8)
    ax.spines["polar"].set_color(SPINE)
    ax.spines["polar"].set_linewidth(0.9)
    fig.subplots_adjust(top=0.90, bottom=0.16, left=0.12, right=0.88)
    return fig


def build_probability_chart(candidates: list[dict[str, object]], accent_color: str):
    fig, ax = make_figure(5.6, 3.35)
    labels = [candidate["type"] for candidate in candidates]
    probs = [float(candidate["probability"]) for candidate in candidates]
    positions = np.arange(len(labels))
    colors = [accent_color if index == 0 else "#5B6E8A" for index in range(len(labels))]
    ax.barh(positions, probs, color=colors, edgecolor="none")
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(100, max(probs) + 10))
    ax.set_xlabel("Probability (%)", color=MUTED)
    for pos, prob in zip(positions, probs):
        ax.text(prob + 1.2, pos, f"{prob:.1f}%", va="center", color=TEXT, fontsize=9, fontweight="bold")
    style_axes(ax)
    fig.subplots_adjust(left=0.18, right=0.96, top=0.92, bottom=0.18)
    return fig


def build_group_donut_chart(group_probabilities: dict[str, float]):
    fig, ax = plt.subplots(figsize=(4.5, 3.35))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    labels = [group for group in GROUP_ORDER if group in group_probabilities]
    values = [group_probabilities[group] for group in labels]
    if not values or sum(values) <= 0:
        ax.text(0.5, 0.5, "No group data", ha="center", va="center", color=MUTED)
        ax.set_axis_off()
        return fig
    colors = [GROUP_ACCENTS[group] for group in labels]
    wedges, _ = ax.pie(values, colors=colors, startangle=90, wedgeprops={"width": 0.36, "edgecolor": AX_BG})
    legend = ax.legend(
        wedges,
        [f"{label} {value:.1f}%" for label, value in zip(labels, values)],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
    )
    for text in legend.get_texts():
        text.set_color(TEXT)
    ax.set_aspect("equal")
    fig.subplots_adjust(left=0.06, right=0.78, top=0.95, bottom=0.08)
    return fig


def build_model_comparison_chart(metrics: dict[str, object]):
    fig, ax = make_figure(6.0, 3.4)
    model_names = list(metrics.keys())
    accuracy = [metrics[name]["accuracy"] * 100 for name in model_names]
    f1_scores = [metrics[name]["weighted_avg_f1"] * 100 for name in model_names]
    x = np.arange(len(model_names))
    width = 0.34
    ax.bar(x - width / 2, accuracy, width, label="Accuracy", color="#326BFF")
    ax.bar(x + width / 2, f1_scores, width, label="Weighted F1", color="#FF9F1C")
    ax.set_xticks(x, model_names)
    ax.set_ylabel("Score (%)", color=MUTED)
    ax.legend(frameon=False)
    for idx, score in enumerate(accuracy):
        ax.text(idx - width / 2, score + 1, f"{score:.1f}", ha="center", fontsize=9, color=TEXT)
    for idx, score in enumerate(f1_scores):
        ax.text(idx + width / 2, score + 1, f"{score:.1f}", ha="center", fontsize=9, color=TEXT)
    style_axes(ax)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.18)
    return fig


def build_type_distribution_chart(type_counts: dict[str, int]):
    fig, ax = make_figure(6.8, 4.8)
    labels = [label for label, count in type_counts.items() if count > 0]
    values = [type_counts[label] for label in labels]
    if not values:
        ax.text(0.5, 0.5, "No dataset distribution", ha="center", va="center", color=MUTED)
        ax.set_axis_off()
        return fig
    colors = [GROUP_ACCENTS[TYPE_GROUPS[label]] for label in labels]
    ax.barh(labels, values, color=colors, edgecolor="none")
    ax.invert_yaxis()
    ax.set_xlabel("Profiles in dataset", color=MUTED)
    style_axes(ax)
    return fig


def build_dimension_balance_chart(rows: list[dict[str, object]], accent_color: str):
    fig, ax = make_figure(5.6, 3.35)
    labels = [row["title"] for row in rows]
    signed_values = []
    annotations = []
    for row in rows:
        balance = float(row["right_pct"]) - float(row["left_pct"])
        signed_values.append(balance)
        annotations.append(f"{row['winner']} leaning")
    y = np.arange(len(labels))
    ax.axvline(0, color=SPINE, linewidth=1.0)
    colors = [accent_color if value >= 0 else "#5B6E8A" for value in signed_values]
    ax.barh(y, signed_values, color=colors, edgecolor="none")
    ax.set_yticks(y, labels)
    ax.set_xlim(-100, 100)
    ax.set_xlabel("Left/right preference balance", color=MUTED)
    for pos, value, note in zip(y, signed_values, annotations):
        anchor = value + 3 if value >= 0 else value - 3
        align = "left" if value >= 0 else "right"
        ax.text(anchor, pos, note, va="center", ha=align, fontsize=9, color=TEXT)
    style_axes(ax)
    fig.subplots_adjust(left=0.22, right=0.96, top=0.90, bottom=0.18)
    return fig


def build_confusion_heatmap(confusion_matrix_data: list[list[int]], labels: list[str], title: str):
    matrix = np.array(confusion_matrix_data)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    image = ax.imshow(matrix, cmap="YlGnBu")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.tick_params(colors=MUTED)
    ax.set_title(title, color=TEXT, pad=14)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = int(matrix[i, j])
            if value == 0:
                continue
            color = "#DDE5F0" if value < matrix.max() * 0.55 else "#FFFFFF"
            ax.text(j, i, value, ha="center", va="center", color=color, fontsize=7)
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=MUTED)
    cbar.outline.set_edgecolor(SPINE)
    fig.subplots_adjust(left=0.09, right=0.96, top=0.92, bottom=0.14)
    return fig
