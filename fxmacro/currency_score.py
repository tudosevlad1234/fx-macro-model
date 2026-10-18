from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MAJORS, SHORT_TENORS
from .data import MarketData
from .fair_value import carry_forward
from .macro import macro_currency_z, surprise_index
from .scoring import ScoreParams, score_target, squash, standardise

CURRENCY_PRIOR = {
    "infl": 1.0,
    "unemp": 1.0,
    "trade": 0.7,
    "surprise": 1.0,
    "rate_mom": 1.4,
    "rate_lvl": 0.7,
}

MACRO_FIELDS = {"CPI": "infl", "UNEMP": "unemp", "TRADE": "trade"}


def usd_legs(prices: dict[str, pd.Series]) -> pd.DataFrame:
    legs = {"USD": None}
    for name, close in prices.items():
        if name.startswith("X") or len(name) != 6:
            continue
        base, quote = name[:3], name[3:]
        log_price = np.log(close[close > 0])
        if quote == "USD" and base in MAJORS:
            legs[base] = log_price
        elif base == "USD" and quote in MAJORS:
            legs[quote] = -log_price
    available = {k: v for k, v in legs.items() if v is not None}
    if not available:
        return pd.DataFrame()
    frame = pd.DataFrame(available).sort_index().ffill(limit=5)
    frame["USD"] = 0.0
    return frame.dropna()


def basket_relative(legs: pd.DataFrame) -> pd.DataFrame:
    total = legs.sum(axis=1)
    count = legs.shape[1]
    return pd.DataFrame({c: legs[c] - (total - legs[c]) / (count - 1) for c in legs.columns})


def relative_to_others(frame: pd.DataFrame) -> pd.DataFrame:
    count = frame.notna().sum(axis=1)
    total = frame.sum(axis=1, min_count=1)
    return pd.DataFrame(
        {
            column: frame[column]
            - (total - frame[column].fillna(0.0)) / (count - 1).where(count > 1)
            for column in frame.columns
        }
    ).where(frame.notna())


def short_rate(data: MarketData, currency: str, index: pd.DatetimeIndex) -> pd.Series | None:
    for tenor in SHORT_TENORS:
        series = data.yields.get(currency, {}).get(tenor)
        if series is not None and len(series.dropna()) > 300:
            return carry_forward(series, index, 5)
    return None


def raw_components(data: MarketData, index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    raw: dict[str, dict[str, pd.Series]] = {name: {} for name in CURRENCY_PRIOR}
    for currency in MAJORS:
        z = macro_currency_z(data.macro.get(currency, {}), index)
        for field, name in MACRO_FIELDS.items():
            if field in z and z[field].notna().any():
                raw[name][currency] = z[field]
        surprise = surprise_index(data.events, currency, index)
        if surprise.notna().any():
            raw["surprise"][currency] = surprise
        rate = short_rate(data, currency, index)
        if rate is not None:
            momentum = rate - rate.shift(60)
            raw["rate_mom"][currency] = standardise(momentum, 750, 250)
            level = rate - rate.rolling(750, min_periods=250).mean()
            raw["rate_lvl"][currency] = standardise(level, 750, 250)
    return {name: pd.DataFrame(columns_) for name, columns_ in raw.items() if columns_}


def currency_components(data: MarketData, index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    relative = {}
    for name, frame in raw_components(data, index).items():
        if frame.shape[1] < 3:
            continue
        centred = relative_to_others(frame)
        relative[name] = pd.DataFrame(
            squash(centred.to_numpy()), index=frame.index, columns=frame.columns
        )
    return relative


def score_currencies(data: MarketData, params: ScoreParams | None = None) -> dict[str, dict]:
    params = params or ScoreParams()
    legs = usd_legs(data.prices)
    if legs.empty:
        return {}
    index = legs.index
    target = basket_relative(legs)
    components = currency_components(data, index)
    results = {}
    for currency in legs.columns:
        frame = pd.DataFrame(
            {name: table[currency] for name, table in components.items() if currency in table}
        ).reindex(index)
        if frame.empty:
            continue
        outcome = score_target(frame, target[currency], CURRENCY_PRIOR, params)
        if outcome:
            outcome["components"] = {
                name: float(frame[name].iloc[-1])
                for name in frame
                if np.isfinite(frame[name].iloc[-1])
            }
            results[currency] = outcome
    return results
