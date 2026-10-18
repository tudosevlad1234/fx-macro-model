from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from . import stats
from .config import INSTRUMENTS


@dataclass(frozen=True)
class ArbParams:
    train_fraction: float = 0.65
    min_observations: int = 800
    half_life_min: float = 3.0
    half_life_max: float = 60.0
    edge_cost_ratio: float = 2.0
    entry_z: float = 2.0
    exit_z: float = 0.25
    stop_z: float = 3.5
    z_window: int = 120
    beta_window: int = 250
    mechanical_tolerance: float = 0.30
    min_abs_beta: float = 0.10
    test_p_value: float = 0.05
    null_draws: int = 50
    null_candidates: int = 12
    null_p_value: float = 0.05
    seed: int = 0


def shares_currency(first: str, second: str) -> bool:
    return bool({first[:3], first[3:]} & {second[:3], second[3:]})


def is_mechanical(first: str, second: str, beta: float, tolerance: float) -> bool:
    return shares_currency(first, second) and abs(abs(beta) - 1.0) <= tolerance


def phase_randomise(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    centred = series - series.mean()
    spectrum = np.fft.rfft(centred)
    phases = rng.uniform(0, 2 * np.pi, len(spectrum))
    phases[0] = 0.0
    if len(series) % 2 == 0:
        phases[-1] = 0.0
    shuffled = np.abs(spectrum) * np.exp(1j * phases)
    return np.fft.irfft(shuffled, n=len(series)) + series.mean()


def cost_in_log_units(first: str, second: str, prices: pd.DataFrame) -> float:
    total = 0.0
    for name in (first, second):
        instrument = INSTRUMENTS[name]
        total += instrument.cost_pips * instrument.pip / float(prices[name].iloc[-1])
    return total


def fit_pair(log_prices: pd.DataFrame, first: str, second: str, params: ArbParams) -> dict | None:
    pair = log_prices[[first, second]].dropna()
    if len(pair) < params.min_observations:
        return None
    split = int(len(pair) * params.train_fraction)
    train, test = pair.iloc[:split], pair.iloc[split:]
    x = stats.add_constant([train[second].to_numpy()])
    coefficients, resid = stats.ols(x, train[first].to_numpy())
    intercept, beta = float(coefficients[0]), float(coefficients[1])
    train_test = stats.engle_granger(train[first].to_numpy(), train[[second]].to_numpy())
    train_ok = bool(train_test["statistic"] < train_test["critical_1pct"])
    test_spread = (test[first] - beta * test[second] - intercept).to_numpy()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            out_of_sample_p = float(adfuller(test_spread, maxlag=8, autolag="AIC")[1])
    except ValueError:
        out_of_sample_p = 1.0
    life = stats.half_life(resid)
    return {
        "first": first,
        "second": second,
        "beta": beta,
        "intercept": intercept,
        "train_statistic": float(train_test["statistic"]),
        "train_critical_1pct": float(train_test["critical_1pct"]),
        "train_cointegrated": train_ok,
        "test_p_value": out_of_sample_p,
        "half_life": float(life),
        "spread_sd": float(resid.std(ddof=1)),
        "train_observations": len(train),
        "test_observations": len(test),
        "mechanical": is_mechanical(first, second, beta, params.mechanical_tolerance),
    }


def screen(candidate: dict, cost: float, params: ArbParams) -> list[str]:
    reasons = []
    if candidate["mechanical"]:
        reasons.append("mechanical pair")
    if abs(candidate["beta"]) < params.min_abs_beta:
        reasons.append("degenerate hedge ratio")
    if not candidate["train_cointegrated"]:
        reasons.append("not cointegrated in training window")
    if not candidate["test_p_value"] < params.test_p_value:
        reasons.append("spread not stationary out of sample")
    if not params.half_life_min <= candidate["half_life"] <= params.half_life_max:
        reasons.append("half-life outside range")
    edge = (params.entry_z - params.exit_z) * candidate["spread_sd"]
    candidate["edge_cost_ratio"] = edge / cost if cost > 0 else math.inf
    if candidate["edge_cost_ratio"] < params.edge_cost_ratio:
        reasons.append("edge does not cover costs")
    return reasons


def null_p_value(
    log_prices: pd.DataFrame, first: str, second: str, params: ArbParams, rng: np.random.Generator
) -> float:
    pair = log_prices[[first, second]].dropna()
    split = int(len(pair) * params.train_fraction)
    train = pair.iloc[:split]
    observed = stats.engle_granger(train[first].to_numpy(), train[[second]].to_numpy())["statistic"]
    draws = []
    for _ in range(params.null_draws):
        a = phase_randomise(train[first].to_numpy(), rng)
        b = phase_randomise(train[second].to_numpy(), rng)
        draws.append(stats.engle_granger(a, b[:, None])["statistic"])
    draws = np.asarray(draws)
    return float((1 + np.sum(draws <= observed)) / (1 + len(draws)))


def scan(prices: pd.DataFrame, names: list[str] | None = None, params: ArbParams | None = None):
    params = params or ArbParams()
    names = names or [n for n in prices.columns if n in INSTRUMENTS and not INSTRUMENTS[n].metal]
    log_prices = np.log(prices[names])
    accepted, rejected = [], []
    for first, second in itertools.combinations(names, 2):
        candidate = fit_pair(log_prices, first, second, params)
        if candidate is None:
            continue
        cost = cost_in_log_units(first, second, prices)
        reasons = screen(candidate, cost, params)
        candidate["reasons"] = reasons
        (rejected if reasons else accepted).append(candidate)
    accepted.sort(key=lambda item: item["train_statistic"])
    rng = np.random.default_rng(params.seed)
    survivors = []
    for rank, candidate in enumerate(accepted):
        if rank >= params.null_candidates:
            candidate["reasons"] = ["outside null-test shortlist"]
            rejected.append(candidate)
            continue
        candidate["null_p_value"] = null_p_value(
            log_prices, candidate["first"], candidate["second"], params, rng
        )
        if candidate["null_p_value"] > params.null_p_value:
            candidate["reasons"] = ["fails phase-randomised null"]
            rejected.append(candidate)
        else:
            survivors.append(candidate)
    return survivors, rejected


def trailing_spread(pair: pd.DataFrame, first: str, second: str, window: int) -> pd.DataFrame:
    a, b = pair[first], pair[second]
    variance = b.rolling(window).var()
    beta = a.rolling(window).cov(b) / variance.where(variance > 0)
    intercept = a.rolling(window).mean() - beta * b.rolling(window).mean()
    beta, intercept = beta.shift(1), intercept.shift(1)
    return pd.DataFrame({"beta": beta, "spread": a - beta * b - intercept})


def simulate(
    z: np.ndarray, spread_returns: np.ndarray, cost: float, params: ArbParams
) -> tuple[np.ndarray, list[float], int]:
    position, entries, blocked = 0, 0, False
    open_pnl = 0.0
    pnl = np.zeros(len(z))
    closed: list[float] = []
    for i, value in enumerate(z):
        daily = 0.0
        if position != 0 and np.isfinite(spread_returns[i]):
            daily = position * spread_returns[i]
            open_pnl += daily
        target = position
        if not np.isfinite(value):
            target = 0
        elif position == 0:
            if blocked and abs(value) < params.entry_z:
                blocked = False
            if not blocked and abs(value) >= params.entry_z:
                target = -1 if value > 0 else 1
        else:
            reverted = position * value >= -params.exit_z
            stopped = position * value <= -params.stop_z
            if reverted or stopped:
                target = 0
                blocked = stopped
        if target != position:
            daily -= cost
            if position == 0:
                entries += 1
                open_pnl = -cost
            else:
                closed.append(open_pnl - cost)
                open_pnl = 0.0
            position = target
        pnl[i] = daily
    return pnl, closed, entries


def backtest(
    prices: pd.DataFrame, first: str, second: str, params: ArbParams | None = None
) -> dict:
    params = params or ArbParams()
    log_pair = np.log(prices[[first, second]].dropna())
    split = int(len(log_pair) * params.train_fraction)
    built = trailing_spread(log_pair, first, second, params.beta_window)
    spread = built["spread"]
    mean = spread.rolling(params.z_window).mean().shift(1)
    deviation = spread.rolling(params.z_window).std().shift(1)
    z = ((spread - mean) / deviation.where(deviation > 0)).iloc[split:]
    returns = log_pair.diff().iloc[split:]
    spread_returns = np.array(
        returns[first] - built["beta"].iloc[split:] * returns[second], dtype=float
    )
    spread_returns[0] = np.nan
    cost = cost_in_log_units(first, second, prices)
    pnl, closed, entries = simulate(z.to_numpy(), spread_returns, cost, params)
    series = pd.Series(pnl, index=z.index)
    daily_sd = series.std(ddof=1)
    sharpe = float(series.mean() / daily_sd * math.sqrt(252)) if daily_sd > 0 else math.nan
    wins = [r for r in closed if r > 0]
    cumulative = series.cumsum()
    return {
        "pair": f"{first}/{second}",
        "trades": entries,
        "closed": len(closed),
        "hit_rate": len(wins) / len(closed) if closed else math.nan,
        "total_return_pct": float(series.sum() * 100.0),
        "sharpe": sharpe,
        "max_drawdown_pct": float((cumulative - cumulative.cummax()).min() * 100.0),
        "pnl": series,
    }


def current_signal(prices: pd.DataFrame, candidate: dict, params: ArbParams | None = None) -> dict:
    params = params or ArbParams()
    first, second = candidate["first"], candidate["second"]
    log_pair = np.log(prices[[first, second]].dropna())
    spread = log_pair[first] - candidate["beta"] * log_pair[second] - candidate["intercept"]
    window = spread.iloc[-params.z_window :]
    z = float((spread.iloc[-1] - window.mean()) / window.std(ddof=1))
    action = "flat"
    if z >= params.entry_z:
        action = f"short {first} / long {second}"
    elif z <= -params.entry_z:
        action = f"long {first} / short {second}"
    return {"pair": f"{first}/{second}", "z": z, "action": action, "beta": candidate["beta"]}
