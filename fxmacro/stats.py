from __future__ import annotations

import math

import numpy as np
from statsmodels.tsa.stattools import coint


def add_constant(columns) -> np.ndarray:
    columns = [np.asarray(column, dtype=float) for column in columns]
    return np.column_stack([np.ones(len(columns[0]))] + columns)


def ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta, y - x @ beta


def weighted_least_squares(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, ridge: float = 0.0
) -> np.ndarray:
    root = np.sqrt(weights)[:, None]
    xw, yw = x * root, y * root[:, 0]
    if ridge > 0 and x.shape[1] > 1:
        scale = np.sqrt(np.mean(xw**2, axis=0))
        scale[0] = 0.0
        penalty = np.diag(np.sqrt(ridge) * scale * len(x))
        xw = np.vstack([xw, penalty])
        yw = np.concatenate([yw, np.zeros(x.shape[1])])
    beta, *_ = np.linalg.lstsq(xw, yw, rcond=None)
    return beta


def newey_west_se(x: np.ndarray, resid: np.ndarray, lags: int | None = None) -> np.ndarray:
    n = x.shape[0]
    lags = int(4 * (n / 100) ** (2 / 9)) if lags is None else lags
    bread = np.linalg.pinv(x.T @ x)
    scores = x * resid[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1)
        cross = scores[lag:].T @ scores[:-lag]
        meat += weight * (cross + cross.T)
    covariance = bread @ meat @ bread
    return np.sqrt(np.maximum(np.diag(covariance), 1e-300))


def r_squared(y: np.ndarray, resid: np.ndarray) -> float:
    total = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(resid @ resid) / total if total > 0 else float("nan")


def half_life(series: np.ndarray) -> float:
    values = np.asarray(series, dtype=float)
    lagged, current = values[:-1], values[1:]
    denominator = float(lagged @ lagged)
    if denominator <= 0:
        return math.inf
    phi = float(lagged @ current) / denominator
    if phi >= 1.0:
        return math.inf
    if phi <= 0.0:
        return 0.0
    return math.log(0.5) / math.log(phi)


def variance_ratio(series: np.ndarray, horizon: int = 20) -> tuple[float, float]:
    values = np.asarray(series, dtype=float)
    steps = np.diff(values)
    n = len(steps)
    if n < 5 * horizon:
        return math.nan, math.nan
    mean = steps.mean()
    centered = steps - mean
    denominator = float(np.sum(centered**2))
    var_one = denominator / (n - 1)
    spans = values[horizon:] - values[:-horizon] - horizon * mean
    normaliser = horizon * (n - horizon + 1) * (1 - horizon / n)
    var_horizon = float(np.sum(spans**2)) / normaliser
    ratio = var_horizon / var_one if var_one > 0 else math.nan
    asymptotic = 0.0
    for lag in range(1, horizon):
        numerator = float(np.sum((centered[lag:] ** 2) * (centered[:-lag] ** 2)))
        asymptotic += (2 * (horizon - lag) / horizon) ** 2 * numerator / denominator**2
    z = (ratio - 1.0) / math.sqrt(asymptotic) if asymptotic > 0 else math.nan
    return float(ratio), float(z)


def engle_granger(y: np.ndarray, x: np.ndarray, max_lag: int = 8) -> dict[str, float]:
    regressors = np.asarray(x, dtype=float)
    if regressors.ndim == 1:
        regressors = regressors[:, None]
    statistic, p_value, critical = coint(
        np.asarray(y, dtype=float), regressors, trend="c", maxlag=max_lag, autolag="bic"
    )
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "critical_1pct": float(critical[0]),
        "critical_5pct": float(critical[1]),
        "critical_10pct": float(critical[2]),
    }
