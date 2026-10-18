from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .config import PUBLICATION_LAG_DAYS
from .scoring import standardise


def infer_frequency(series: pd.Series) -> str:
    spacing = series.dropna().index.to_series().diff().dt.days.median()
    if spacing <= 5:
        return "daily"
    if spacing <= 45:
        return "monthly"
    return "quarterly"


def lagged_daily(
    series: pd.Series, index: pd.DatetimeIndex, lag_days: int, limit: int
) -> pd.Series:
    shifted = series.copy()
    shifted.index = shifted.index + pd.Timedelta(days=lag_days)
    shifted = shifted[~shifted.index.duplicated(keep="last")]
    return shifted.reindex(shifted.index.union(index)).ffill(limit=limit).reindex(index)


def macro_currency_z(macro: dict[str, pd.Series], index: pd.DatetimeIndex) -> pd.DataFrame:
    output = {}
    for field, series in macro.items():
        if series is None or len(series.dropna()) < 30:
            continue
        frequency = infer_frequency(series)
        if frequency == "daily":
            series = series.resample("ME").last().dropna()
            frequency = "monthly"
        monthly = frequency == "monthly"
        short, long, per_year = (3, 12, 12) if monthly else (1, 4, 4)
        window, minimum = (120, 36) if monthly else (40, 12)
        if field == "CPI":
            yearly = 100 * (np.log(series) - np.log(series.shift(per_year)))
            statistic, sign = yearly - yearly.shift(short), 1.0
        elif field == "UNEMP":
            statistic, sign = series - series.shift(short), -1.0
        elif field == "TRADE":
            statistic = series.rolling(short).mean() - series.shift(short).rolling(long).mean()
            sign = 1.0
        else:
            continue
        z = sign * standardise(statistic.dropna(), window, minimum)
        output[field] = lagged_daily(
            z.dropna(),
            index,
            PUBLICATION_LAG_DAYS.get(field, 30),
            130 if monthly else 200,
        )
    return pd.DataFrame(output, index=index)


def _surprise_scale(events: pd.DataFrame) -> pd.Series:
    if "sd" in events and events["sd"].notna().any():
        return events["sd"].astype(float)
    surprise = events["actual"] - events["forecast"]
    expanding = surprise.groupby(events["ccy"]).transform(
        lambda s: s.expanding(min_periods=10).std().shift(1)
    )
    return expanding


def surprise_index(
    events: pd.DataFrame | None, currency: str, index: pd.DatetimeIndex, half_life: float = 10.0
) -> pd.Series:
    empty = pd.Series(np.nan, index=index, name=currency)
    if events is None or events.empty:
        return empty
    own = events[events["ccy"] == currency].copy()
    if own.empty:
        return empty
    scale = _surprise_scale(events).loc[own.index]
    z = ((own["actual"] - own["forecast"]) / scale.where(scale > 0)).clip(-4, 4)
    weight = own["impact"].astype(float) if "impact" in own else 1.0
    shocks = (z * weight).dropna()
    daily = np.zeros(len(index))
    dates = pd.to_datetime(own.loc[shocks.index, "date"])
    positions = index.searchsorted(dates)
    for position, value in zip(positions, shocks.to_numpy(), strict=False):
        if position < len(index):
            daily[position] += value
    decay = 0.5 ** (1.0 / half_life)
    smoothed = lfilter([1.0], [1.0, -decay], daily)
    first = positions[positions < len(index)].min() if (positions < len(index)).any() else None
    series = pd.Series(smoothed, index=index, name=currency)
    if first is not None:
        series.iloc[:first] = np.nan
    return series
