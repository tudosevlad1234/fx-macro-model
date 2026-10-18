from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close[close > 0]).diff().dropna()


def realised_vol(returns: pd.Series, window: int = 20) -> float:
    return float(returns.tail(window).std(ddof=1) * math.sqrt(TRADING_DAYS))


def ewma_variance(returns: pd.Series, decay: float = 0.94) -> pd.Series:
    return (returns**2).ewm(alpha=1.0 - decay, adjust=False).mean()


def ewma_vol(returns: pd.Series, decay: float = 0.94) -> float:
    return float(math.sqrt(ewma_variance(returns, decay).iloc[-1] * TRADING_DAYS))


def expected_daily_move(
    close: pd.Series, pip: float, implied_vol: float | None = None, decay: float = 0.94
) -> dict[str, float | str]:
    returns = log_returns(close)
    annual = implied_vol if implied_vol and implied_vol > 0 else ewma_vol(returns, decay)
    sigma = annual / math.sqrt(TRADING_DAYS)
    price = float(close.iloc[-1])
    sigma_pips = sigma * price / pip
    return {
        "source": "implied" if implied_vol and implied_vol > 0 else "ewma",
        "annual_vol_pct": annual * 100.0,
        "sigma_pct": sigma * 100.0,
        "sigma_pips": sigma_pips,
        "mean_abs_pips": sigma_pips * math.sqrt(2.0 / math.pi),
        "range_68_pips": sigma_pips,
        "range_95_pips": 1.96 * sigma_pips,
        "realised_vol_pct": realised_vol(returns) * 100.0,
    }


def vol_premium(implied_vol: float | None, realised: float) -> float:
    if implied_vol is None or not np.isfinite(implied_vol):
        return math.nan
    return float(implied_vol - realised)
