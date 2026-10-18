from __future__ import annotations

import math

import pandas as pd

from .scoring import squash
from .smile import Smile, build_smile

INVERTED_UNDERLYING = frozenset({"USDJPY", "USDCHF", "USDCAD"})


def choose_expiry(
    chain: pd.DataFrame, asof: pd.Timestamp, target_days: int = 30, minimum_days: int = 7
) -> pd.Timestamp | None:
    expiries = pd.to_datetime(chain["expiry"]).drop_duplicates()
    days = (expiries - asof).dt.days
    valid = expiries[days >= minimum_days]
    if valid.empty:
        return None
    return valid.iloc[((valid - asof).dt.days - target_days).abs().argmin()]


def smile_for_pair(
    name: str, chain: pd.DataFrame, spot: float, asof: pd.Timestamp, rate: float = 0.0
) -> Smile | None:
    expiry = choose_expiry(chain, asof)
    if expiry is None:
        return None
    selected = chain[pd.to_datetime(chain["expiry"]) == expiry]
    years = (expiry - asof).days / 365.0
    underlying_spot = 1.0 / spot if name in INVERTED_UNDERLYING else spot
    smile = build_smile(selected[["strike", "call", "put"]], underlying_spot, years, rate)
    if smile is None:
        return None
    if name in INVERTED_UNDERLYING:
        return Smile(
            forward=1.0 / smile.forward,
            atm_vol=smile.atm_vol,
            rr25=-smile.rr25,
            bf25=smile.bf25,
            points=smile.points,
        )
    return smile


def leg_risk_reversal(smiles: dict[str, Smile], currency: str) -> float | None:
    if currency == "USD":
        return 0.0
    direct, inverse = smiles.get(f"{currency}USD"), smiles.get(f"USD{currency}")
    if direct is not None:
        return direct.rr25
    if inverse is not None:
        return -inverse.rr25
    return None


def cross_risk_reversal(smiles: dict[str, Smile], name: str) -> float | None:
    if name in smiles:
        return smiles[name].rr25
    base, quote = leg_risk_reversal(smiles, name[:3]), leg_risk_reversal(smiles, name[3:])
    if base is None or quote is None:
        return None
    return base - quote


def atm_vol(smiles: dict[str, Smile], name: str) -> float | None:
    if name in smiles:
        return smiles[name].atm_vol
    first, second = (
        smiles.get(f"{name[:3]}USD") or smiles.get(f"USD{name[:3]}"),
        smiles.get(f"{name[3:]}USD") or smiles.get(f"USD{name[3:]}"),
    )
    if first is None or second is None:
        return None
    return math.sqrt(
        max(first.atm_vol**2 + second.atm_vol**2 - 0.5 * first.atm_vol * second.atm_vol, 1e-12)
    )


def skew_signal(rr25: float | None, atm: float | None, scale: float = 0.15) -> float:
    if rr25 is None or atm is None or atm <= 0:
        return math.nan
    return float(squash(rr25 / atm, scale))


def collect_smiles(
    options: dict[str, pd.DataFrame], prices: dict[str, pd.Series], asof: pd.Timestamp
) -> dict[str, Smile]:
    smiles = {}
    for name, chain in options.items():
        if name not in prices:
            continue
        smile = smile_for_pair(name, chain, float(prices[name].dropna().iloc[-1]), asof)
        if smile is not None:
            smiles[name] = smile
    return smiles
