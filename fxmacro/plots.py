from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .fair_value import FairValueEstimate  # noqa: E402


def fair_value_chart(estimate: FairValueEstimate, path: str | Path, days: int = 750) -> Path:
    frame = estimate.frame.iloc[-days:]
    actual = np.exp(frame["fv"] + frame["resid"])
    fair = np.exp(frame["fv"])
    band = frame["resid_sd"]
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    top.plot(frame.index, actual, label="price", linewidth=1.2)
    top.plot(frame.index, fair, label="fair value", linewidth=1.2)
    top.fill_between(
        frame.index,
        np.exp(frame["fv"] - 2 * band),
        np.exp(frame["fv"] + 2 * band),
        alpha=0.15,
        label="+/- 2 sd",
    )
    top.set_title(f"{estimate.instrument} - price against model fair value")
    top.legend(loc="upper left")
    bottom.plot(frame.index, frame["resid_z"], linewidth=1.0)
    bottom.axhline(0.0, color="grey", linewidth=0.8)
    bottom.set_ylabel("gap z-score")
    figure.tight_layout()
    target = Path(path)
    figure.savefig(target, dpi=130)
    plt.close(figure)
    return target


def score_chart(table: pd.DataFrame, path: str | Path, title: str) -> Path:
    ordered = table["score"].sort_values()
    figure, axis = plt.subplots(figsize=(8, max(3, 0.35 * len(ordered))))
    colours = ["#2e7d32" if value >= 0 else "#c62828" for value in ordered]
    axis.barh(ordered.index, ordered.to_numpy(), color=colours)
    axis.axvline(0.0, color="grey", linewidth=0.8)
    axis.set_xlim(-1, 1)
    axis.set_title(title)
    figure.tight_layout()
    target = Path(path)
    figure.savefig(target, dpi=130)
    plt.close(figure)
    return target
