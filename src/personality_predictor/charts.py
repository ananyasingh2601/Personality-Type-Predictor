from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def build_radar_chart(rows: list[dict[str, object]], accent_color: str):
    labels = [row["title"] for row in rows]
    values = [float(row["confidence"]) for row in rows]
    values += values[:1]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.4, 5.4), subplot_kw={"polar": True})
    fig.patch.set_facecolor("#FFF9F1")
    ax.set_facecolor("#FFF9F1")
    ax.plot(angles, values, color=accent_color, linewidth=2.5)
    ax.fill(angles, values, color=accent_color, alpha=0.20)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10, color="#152238")
    ax.set_ylim(0, 100)
    ax.set_rlabel_position(10)
    ax.tick_params(colors="#5F6B7A")
    ax.grid(color="#D8CFBE", alpha=0.7)
    return fig

