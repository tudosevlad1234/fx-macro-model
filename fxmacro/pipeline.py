from __future__ import annotations

import math

import pandas as pd

from .config import INSTRUMENTS
from .currency_score import score_currencies
from .data import MarketData
from .fair_value import FairValueParams, estimate_all
from .options import atm_vol, collect_smiles, cross_risk_reversal, skew_signal
from .pair_score import score_pairs
from .regime import classify
from .scoring import ScoreParams
from .statarb import ArbParams, backtest, current_signal, scan
from .volatility import expected_daily_move, log_returns


def ppp_reading(estimate) -> tuple[float, float]:
    ppp = estimate.ppp
    if ppp is None or ppp.empty:
        return math.nan, math.nan
    latest = ppp.iloc[-1]
    return float(latest["ppp"]), float(latest["deviation_pct"])


def fair_value_table(estimates: dict) -> pd.DataFrame:
    rows = []
    for name, estimate in estimates.items():
        latest = estimate.latest
        ppp_level, ppp_gap = ppp_reading(estimate)
        rows.append(
            {
                "pair": name,
                "fair_value": estimate.fair_value,
                "gap_pct": estimate.gap_pct,
                "ppp": ppp_level,
                "ppp_gap_pct": ppp_gap,
                "z": float(latest["resid_z"]),
                "r2": float(latest["r2"]),
                "half_life_days": float(latest["half_life"]),
                "drivers": ",".join(estimate.kept),
                "cointegrated": bool(estimate.cointegration.get("cointegrated", False)),
                "reliable": estimate.reliable,
                "source": estimate.source,
            }
        )
    return pd.DataFrame(rows).set_index("pair")


def regime_table(data: MarketData, names: list[str], window: int = 1500) -> dict[str, dict]:
    table = {}
    for name in names:
        returns = log_returns(data.prices[name]).iloc[-window:].to_numpy() * 100.0
        if len(returns) < 500:
            continue
        summary = classify(returns, volatility_only=True, starts=4)
        summary.pop("smoothed_high")
        table[name] = summary
    return table


def run(
    data: MarketData,
    fair_value_params: FairValueParams | None = None,
    score_params: ScoreParams | None = None,
    arb_params: ArbParams | None = None,
    instruments: list[str] | None = None,
    include_stat_arb: bool = True,
) -> dict:
    names = instruments or [n for n in INSTRUMENTS if n in data.prices]
    estimates = estimate_all([INSTRUMENTS[n] for n in names], data, fair_value_params)
    asof = data.index[-1]
    smiles = collect_smiles(data.options, data.prices, asof)
    skews = {}
    implied = {}
    for name in estimates:
        rr = cross_risk_reversal(smiles, name)
        atm = atm_vol(smiles, name)
        skews[name] = skew_signal(rr, atm)
        implied[name] = atm
    result = {
        "asof": asof,
        "estimates": estimates,
        "fair_value": fair_value_table(estimates),
        "currency_scores": score_currencies(data, score_params),
        "pair_scores": score_pairs(estimates, data, score_params, skews),
        "regimes": regime_table(data, list(estimates)),
        "smiles": smiles,
        "expected_move": {
            name: expected_daily_move(data.prices[name], INSTRUMENTS[name].pip, implied.get(name))
            for name in estimates
        },
    }
    if include_stat_arb:
        prices = pd.DataFrame(
            {n: data.prices[n] for n in names if not INSTRUMENTS[n].metal}
        ).dropna()
        accepted, rejected = scan(prices, params=arb_params)
        result["stat_arb"] = {
            "accepted": accepted,
            "rejected": len(rejected),
            "backtests": [
                {
                    k: v
                    for k, v in backtest(prices, c["first"], c["second"], arb_params).items()
                    if k != "pnl"
                }
                for c in accepted
            ],
            "signals": [current_signal(prices, c, arb_params) for c in accepted],
        }
    return result


def currency_table(scores: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for currency, outcome in scores.items():
        row = {"currency": currency, "score": outcome["score"]}
        for horizon, detail in outcome["horizons"].items():
            row[f"{horizon}_verdict"] = detail["verdict"]
            row[f"{horizon}_p_up_pct"] = detail["p_up_pct"]
            row[f"{horizon}_calibrated"] = detail["calibrated"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("currency").sort_values("score", ascending=False)


def pair_table(scores: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, outcome in scores.items():
        row = {"pair": name, "score": outcome["score"]}
        for horizon, detail in outcome["horizons"].items():
            row[f"{horizon}_verdict"] = detail["verdict"]
            row[f"{horizon}_expected_pct"] = detail["expected_pct"]
            row[f"{horizon}_p_up_pct"] = detail["p_up_pct"]
            row[f"{horizon}_calibrated"] = detail["calibrated"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("pair").sort_values("score", ascending=False)
