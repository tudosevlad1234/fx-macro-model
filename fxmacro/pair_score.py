from __future__ import annotations

import numpy as np
import pandas as pd

from .config import INSTRUMENTS
from .currency_score import currency_components
from .data import MarketData
from .fair_value import FairValueEstimate
from .macro import surprise_index
from .scoring import ScoreParams, score_target, squash, standardise
from .volatility import ewma_variance, log_returns

PAIR_PRIOR = {
    "rate_momentum": 0.25,
    "fv_momentum": 0.15,
    "fv_gap": 0.15,
    "carry": 0.10,
    "macro": 0.15,
    "surprise": 0.15,
    "skew": 0.10,
}

ORTHOGONAL = (("fv_momentum", "rate_momentum"),)


def rate_series(estimate: FairValueEstimate) -> pd.Series | None:
    design = estimate.full_design
    if design is None or "rate_diff" not in design:
        return None
    return design["rate_diff"]


def macro_difference(
    components: dict[str, pd.DataFrame], base: str, quote: str, index: pd.DatetimeIndex
) -> pd.Series:
    parts = []
    for name in ("infl", "unemp", "trade"):
        table = components.get(name)
        if table is None or base not in table or quote not in table:
            continue
        parts.append((table[base] - table[quote]).reindex(index))
    if not parts:
        return pd.Series(np.nan, index=index)
    return pd.concat(parts, axis=1).mean(axis=1)


def pair_components(
    name: str,
    estimate: FairValueEstimate,
    data: MarketData,
    currency_table: dict[str, pd.DataFrame],
    skew: float | None = None,
) -> pd.DataFrame:
    instrument = INSTRUMENTS[name]
    close = data.prices[name].dropna()
    index = close.index
    frame = pd.DataFrame(index=index)
    rate = rate_series(estimate)
    if rate is not None:
        rate = rate.reindex(index).ffill(limit=5)
        frame["rate_momentum"] = squash(standardise(rate - rate.shift(60), 750, 250))
        level = rate - rate.rolling(750, min_periods=250).mean()
        frame["carry"] = squash(standardise(level, 750, 250))
    fair = estimate.frame
    frame["fv_gap"] = squash((-fair["resid_z"]).reindex(index))
    push = fair["push20"].reindex(index)
    frame["fv_momentum"] = squash(standardise(push, 750, 250))
    if not instrument.metal:
        frame["macro"] = squash(
            macro_difference(currency_table, instrument.base, instrument.quote, index)
        )
        own = surprise_index(data.events, instrument.base, index)
        other = surprise_index(data.events, instrument.quote, index)
        if own.notna().any() and other.notna().any():
            frame["surprise"] = squash(own - other)
    if skew is not None and np.isfinite(skew):
        column = pd.Series(np.nan, index=index)
        column.iloc[-1] = skew
        frame["skew"] = column
    return frame


def score_pair(
    name: str,
    estimate: FairValueEstimate,
    data: MarketData,
    currency_table: dict[str, pd.DataFrame],
    params: ScoreParams | None = None,
    skew: float | None = None,
) -> dict:
    params = params or ScoreParams()
    frame = pair_components(name, estimate, data, currency_table, skew)
    close = data.prices[name].dropna()
    target = np.log(close[close > 0])
    returns = log_returns(close)
    noise = float(np.sqrt(ewma_variance(returns).iloc[-1]))
    outcome = score_target(frame, target, PAIR_PRIOR, params, ORTHOGONAL, noise_daily=noise)
    if outcome:
        outcome["components"] = {
            column: float(frame[column].dropna().iloc[-1])
            for column in frame
            if frame[column].notna().any()
        }
    return outcome


def score_pairs(
    estimates: dict[str, FairValueEstimate],
    data: MarketData,
    params: ScoreParams | None = None,
    skews: dict[str, float] | None = None,
) -> dict[str, dict]:
    index = data.index
    table = currency_components(data, index)
    skews = skews or {}
    results = {}
    for name, estimate in estimates.items():
        outcome = score_pair(name, estimate, data, table, params, skews.get(name))
        if outcome:
            results[name] = outcome
    return results
