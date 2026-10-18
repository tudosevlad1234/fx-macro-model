from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


@dataclass(frozen=True)
class Smile:
    forward: float
    atm_vol: float
    rr25: float
    bf25: float
    points: int


def black_price(
    forward: float, strike: float, expiry: float, rate: float, vol: float, is_call: bool
) -> float:
    discount = math.exp(-rate * expiry)
    spread = vol * math.sqrt(expiry)
    d1 = (math.log(forward / strike) + 0.5 * spread**2) / spread
    d2 = d1 - spread
    if is_call:
        return discount * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    return discount * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))


def implied_vol(
    price: float, forward: float, strike: float, expiry: float, rate: float, is_call: bool
) -> float | None:
    if not np.isfinite(price) or price <= 0:
        return None
    low, high = 1e-4, 5.0
    try:
        if (
            black_price(forward, strike, expiry, rate, low, is_call) > price
            or black_price(forward, strike, expiry, rate, high, is_call) < price
        ):
            return None
        return float(
            brentq(
                lambda v: black_price(forward, strike, expiry, rate, v, is_call) - price,
                low,
                high,
                xtol=1e-10,
            )
        )
    except ValueError:
        return None


def call_delta(forward: float, strike: float, expiry: float, vol: float) -> float:
    spread = vol * math.sqrt(expiry)
    d1 = (math.log(forward / strike) + 0.5 * spread**2) / spread
    return float(norm.cdf(d1))


def implied_forward(chain: pd.DataFrame, spot: float, expiry: float, rate: float) -> float:
    both = chain.dropna(subset=["call", "put"])
    if both.empty:
        return spot * math.exp(rate * expiry)
    nearest = both.iloc[(both["strike"] - spot).abs().argsort()[:5]]
    forwards = nearest["strike"] + math.exp(rate * expiry) * (nearest["call"] - nearest["put"])
    return float(forwards.median())


def smile_points(
    chain: pd.DataFrame, forward: float, expiry: float, rate: float
) -> list[tuple[float, float]]:
    points = []
    discount = math.exp(-rate * expiry)
    for row in chain.itertuples():
        strike = float(row.strike)
        use_put = strike < forward
        price = row.put if use_put else row.call
        if not np.isfinite(price):
            other = row.call if use_put else row.put
            if not np.isfinite(other):
                continue
            parity = discount * (forward - strike)
            price = other - parity if use_put else other + parity
        vol = implied_vol(float(price), forward, strike, expiry, rate, not use_put)
        if vol is None:
            continue
        points.append((call_delta(forward, strike, expiry, vol), vol))
    return sorted(points)


def build_smile(
    chain: pd.DataFrame, spot: float, expiry: float, rate: float, min_points: int = 5
) -> Smile | None:
    forward = implied_forward(chain, spot, expiry, rate)
    points = smile_points(chain, forward, expiry, rate)
    if len(points) < min_points:
        return None
    usable = [(d, v) for d, v in points if 0.05 <= d <= 0.95]
    if len(usable) < min_points:
        return None
    deltas = np.array([p[0] for p in usable])
    vols = np.array([p[1] for p in usable])
    if deltas.min() > 0.27 or deltas.max() < 0.73:
        return None
    coefficients = np.polyfit(norm.ppf(deltas), vols, 2)

    def at(delta: float) -> float:
        return float(np.polyval(coefficients, norm.ppf(delta)))

    atm, call_25, put_25 = at(0.5), at(0.25), at(0.75)
    return Smile(
        forward=forward,
        atm_vol=atm,
        rr25=call_25 - put_25,
        bf25=0.5 * (call_25 + put_25) - atm,
        points=len(usable),
    )
