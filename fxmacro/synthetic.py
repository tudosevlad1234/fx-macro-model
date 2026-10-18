from __future__ import annotations

import numpy as np
import pandas as pd

from .config import INSTRUMENTS, MAJORS
from .data import MarketData
from .smile import black_price

RATE_LEVEL = {
    "USD": 2.0,
    "EUR": 0.8,
    "GBP": 1.6,
    "JPY": 0.2,
    "CHF": 0.4,
    "AUD": 2.2,
    "CAD": 1.8,
    "NZD": 2.5,
}
RATE_BETA = 0.035
DRIVER_BETA = {
    "JPY": {"SPX": -0.12},
    "CHF": {"SPX": -0.08},
    "AUD": {"COPPER": 0.20, "VIX": -0.05},
    "NZD": {"COPPER": 0.15, "VIX": -0.04},
    "CAD": {"WTI": 0.15},
}


def ou(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    out = np.zeros(n)
    shocks = rng.standard_normal(n) * sigma
    for t in range(1, n):
        out[t] = phi * out[t - 1] + shocks[t]
    return out


def make_market(rng: np.random.Generator, index: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(index)
    spx = np.cumsum(0.0003 + 0.009 * rng.standard_normal(n))
    vix = 2.9 + ou(rng, n, 0.97, 0.05) - 0.4 * np.diff(spx, prepend=spx[0]) * 20
    copper = np.cumsum(0.0001 + 0.011 * rng.standard_normal(n))
    wti = np.cumsum(0.00005 + 0.018 * rng.standard_normal(n))
    return pd.DataFrame(
        {
            "SPX": 3300 * np.exp(spx),
            "VIX": np.exp(vix),
            "COPPER": 3.2 * np.exp(copper),
            "WTI": 70 * np.exp(wti),
        },
        index=index,
    )


def make_rates(
    rng: np.random.Generator, index: pd.DatetimeIndex
) -> dict[str, dict[str, pd.Series]]:
    n = len(index)
    yields: dict[str, dict[str, pd.Series]] = {}
    for currency in MAJORS:
        level = RATE_LEVEL[currency]
        path = level + ou(rng, n, 0.9995, 0.03)
        if currency == "NZD":
            path = yields["AUD"]["Y2"].to_numpy() + 0.3 + ou(rng, n, 0.98, 0.02)
        yields[currency] = {
            "Y2": pd.Series(path, index=index),
            "Y10": pd.Series(path + 0.8 + ou(rng, n, 0.99, 0.02), index=index),
        }
    return yields


def make_macro(
    rng: np.random.Generator, index: pd.DatetimeIndex
) -> dict[str, dict[str, pd.Series]]:
    months = pd.date_range(index[0], index[-1], freq="ME")
    macro: dict[str, dict[str, pd.Series]] = {}
    for currency in MAJORS:
        inflation = 2.0 + ou(rng, len(months), 0.95, 0.15)
        price_level = 100 * np.exp(np.cumsum(inflation / 1200.0))
        unemployment = 6.0 + np.cumsum(0.05 * rng.standard_normal(len(months)))
        trade = 5.0 + ou(rng, len(months), 0.9, 0.8)
        macro[currency] = {
            "CPI": pd.Series(price_level, index=months),
            "UNEMP": pd.Series(np.clip(unemployment, 2.0, 15.0), index=months),
            "TRADE": pd.Series(trade, index=months),
        }
    return macro


def make_events(rng: np.random.Generator, index: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    dates = pd.date_range(index[0], index[-1], freq="2W-FRI")
    for currency in MAJORS:
        for date in dates:
            forecast = 2.0 + 0.2 * rng.standard_normal()
            rows.append(
                {
                    "date": date,
                    "ccy": currency,
                    "impact": float(rng.choice([1.0, 2.0, 3.0])),
                    "forecast": forecast,
                    "actual": forecast + 0.3 * rng.standard_normal(),
                }
            )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def make_legs(
    rng: np.random.Generator,
    index: pd.DatetimeIndex,
    market: pd.DataFrame,
    yields: dict[str, dict[str, pd.Series]],
) -> pd.DataFrame:
    n = len(index)
    logs = {name: np.log(market[name].to_numpy()) for name in market}
    legs = {}
    for currency in MAJORS:
        if currency == "USD":
            continue
        spread = yields[currency]["Y2"].to_numpy() - yields["USD"]["Y2"].to_numpy()
        fair = RATE_BETA * spread
        for driver, beta in DRIVER_BETA.get(currency, {}).items():
            fair = fair + beta * (logs[driver] - logs[driver].mean())
        legs[currency] = fair + ou(rng, n, 0.975, 0.0035)
    return pd.DataFrame(legs, index=index)


def make_prices(legs: pd.DataFrame) -> dict[str, pd.Series]:
    anchors = {
        "EUR": 1.15,
        "GBP": 1.32,
        "AUD": 0.75,
        "NZD": 0.69,
        "JPY": 110.0,
        "CHF": 0.95,
        "CAD": 1.30,
    }
    level = {}
    for currency in legs:
        value = legs[currency]
        if currency in ("EUR", "GBP", "AUD", "NZD"):
            level[currency] = anchors[currency] * np.exp(value)
        else:
            level[currency] = anchors[currency] * np.exp(-value)
    usd_pairs = {
        "EURUSD": level["EUR"],
        "GBPUSD": level["GBP"],
        "AUDUSD": level["AUD"],
        "NZDUSD": level["NZD"],
        "USDJPY": level["JPY"],
        "USDCHF": level["CHF"],
        "USDCAD": level["CAD"],
    }
    dollar = {
        "EUR": usd_pairs["EURUSD"],
        "GBP": usd_pairs["GBPUSD"],
        "AUD": usd_pairs["AUDUSD"],
        "NZD": usd_pairs["NZDUSD"],
        "JPY": 1 / usd_pairs["USDJPY"],
        "CHF": 1 / usd_pairs["USDCHF"],
        "CAD": 1 / usd_pairs["USDCAD"],
    }
    prices = dict(usd_pairs)
    for name, instrument in INSTRUMENTS.items():
        if instrument.metal or name in prices:
            continue
        prices[name] = dollar[instrument.base] / dollar[instrument.quote]
    return {name: series.rename(name) for name, series in prices.items()}


def make_chain(
    spot: float,
    asof: pd.Timestamp,
    days: int,
    atm: float,
    skew: float,
    curvature: float,
    rate: float = 0.0,
    strikes: int = 31,
) -> pd.DataFrame:
    years = days / 365.0
    forward = spot * np.exp(rate * years)
    moneyness = np.linspace(-0.12, 0.12, strikes)
    rows = []
    for x in moneyness:
        strike = forward * float(np.exp(x))
        vol = atm + skew * x + curvature * x**2
        rows.append(
            {
                "expiry": asof + pd.Timedelta(days=days),
                "strike": strike,
                "call": black_price(forward, strike, years, rate, vol, True),
                "put": black_price(forward, strike, years, rate, vol, False),
            }
        )
    return pd.DataFrame(rows)


def make_options(prices: dict[str, pd.Series]) -> dict[str, pd.DataFrame]:
    settings = {
        "EURUSD": (0.075, -0.10, 0.6),
        "GBPUSD": (0.085, -0.12, 0.7),
        "USDJPY": (0.09, 0.15, 0.8),
    }
    options = {}
    for name, (atm, skew, curvature) in settings.items():
        if name not in prices:
            continue
        series = prices[name].dropna()
        spot = float(series.iloc[-1])
        underlying = 1.0 / spot if name == "USDJPY" else spot
        options[name] = make_chain(underlying, series.index[-1], 30, atm, skew, curvature)
    return options


def generate(years: float = 14.0, seed: int = 7, start: str = "2012-01-02") -> MarketData:
    rng = np.random.default_rng(seed)
    n = int(years * 261)
    index = pd.bdate_range(start, periods=n)
    market = make_market(rng, index)
    yields = make_rates(rng, index)
    macro = make_macro(rng, index)
    legs = make_legs(rng, index, market, yields)
    prices = make_prices(legs)
    return MarketData(
        prices=prices,
        yields=yields,
        macro=macro,
        market=market,
        events=make_events(rng, index),
        options=make_options(prices),
    )
