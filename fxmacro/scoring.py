from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScoreParams:
    horizons: tuple[tuple[str, int], ...] = (("20d", 20), ("60d", 60))
    ic_horizon: int = 60
    ic_window: int = 750
    ic_min_obs: int = 250
    ic_shrink: float = 0.70
    calibration_window: int = 1000
    calibration_min_obs: int = 250
    probability_buckets: int = 8
    max_calibration_gap_pct: float = 5.0
    min_calibration_obs: int = 500
    strong_ratio: float = 0.35
    weak_ratio: float = 0.15


def squash(values, scale: float = 1.5):
    return np.tanh(np.asarray(values, dtype=float) / scale)


def standardise(values: pd.Series, window: int, min_periods: int) -> pd.Series:
    deviation = values.rolling(window, min_periods=min_periods).std()
    return (values / deviation.where(deviation > 1e-9)).clip(-4, 4)


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def _rolling_moments(x: pd.Series, f: pd.Series, window: int):
    present = x.notna() & f.notna()
    xs, fs = x.where(present, 0.0), f.where(present, 0.0)
    options = dict(window=window, min_periods=1)
    count = present.astype(float).rolling(**options).sum()
    positive = count.where(count > 0)
    mean_x = xs.rolling(**options).sum() / positive
    mean_f = fs.rolling(**options).sum() / positive
    covariance = (xs * fs).rolling(**options).sum() / positive - mean_x * mean_f
    var_x = (xs * xs).rolling(**options).sum() / positive - mean_x**2
    var_f = (fs * fs).rolling(**options).sum() / positive - mean_f**2
    return covariance, var_x, var_f, count


def _lagged_correlation(
    x: pd.Series, forward: pd.Series, window: int, min_obs: int, horizon: int
) -> pd.Series:
    covariance, var_x, var_f, count = _rolling_moments(x, forward, window)
    scale = np.sqrt(var_x.clip(lower=0) * var_f.clip(lower=0))
    correlation = (covariance / scale.where(scale > 1e-12)).clip(-1, 1).where(count >= min_obs)
    return correlation.shift(horizon)


def forward_return(target: pd.Series, horizon: int) -> pd.Series:
    return target.shift(-horizon) - target


def rolling_ic(components: pd.DataFrame, target: pd.Series, params: ScoreParams) -> pd.DataFrame:
    forward = forward_return(target, params.ic_horizon)
    return pd.DataFrame(
        {
            column: _lagged_correlation(
                components[column], forward, params.ic_window, params.ic_min_obs, params.ic_horizon
            )
            for column in components.columns
        },
        index=components.index,
    )


def orthogonalise(
    components: pd.DataFrame, pairs: tuple[tuple[str, str], ...], window: int
) -> pd.DataFrame:
    result = components.copy()
    for target, onto in pairs:
        if target not in result or onto not in components:
            continue
        a, b = result[target], components[onto]
        covariance = (a * b).rolling(window, min_periods=100).mean() - a.rolling(
            window, min_periods=100
        ).mean() * b.rolling(window, min_periods=100).mean()
        variance = b.rolling(window, min_periods=100).var(ddof=0)
        beta = (covariance / variance.where(variance > 1e-12)).clip(-3, 3).fillna(0.0)
        result[target] = a - beta * b
    return result


def ic_weights(ic: pd.DataFrame, prior: dict[str, float], shrink: float) -> pd.DataFrame:
    columns = [column for column in ic.columns if column in prior]
    if not columns:
        return pd.DataFrame(index=ic.index)
    base = pd.Series({column: float(prior[column]) for column in columns})
    base = base / base.sum()
    positive = ic[columns].clip(lower=0.0)
    total = positive.sum(axis=1)
    measured = positive.div(total.where(total > 1e-9), axis=0)
    blended = shrink * base + (1 - shrink) * measured.fillna(base)
    return blended.div(blended.sum(axis=1), axis=0)


def composite(components: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in components.columns if column in weights.columns]
    if not columns:
        return pd.DataFrame({"score": np.nan, "n": 0}, index=components.index)
    values = components[columns]
    active = weights[columns].where(values.notna(), 0.0)
    total = active.sum(axis=1)
    score = (values.fillna(0.0) * active).sum(axis=1) / total.where(total > 1e-9)
    return pd.DataFrame({"score": score, "n": values.notna().sum(axis=1)})


def calibrate(
    score: pd.Series, target: pd.Series, horizon: int, window: int, min_obs: int
) -> pd.DataFrame:
    forward = forward_return(target, horizon)
    covariance, var_score, var_forward, count = _rolling_moments(score, forward, window)
    beta = (covariance / var_score.where(var_score > 1e-12)).where(count >= min_obs)
    sd_forward = np.sqrt(var_forward.clip(lower=0))
    denominator = (np.sqrt(var_score.clip(lower=0)) * sd_forward).where(lambda v: v > 1e-12)
    correlation = (covariance / denominator).clip(-1, 1)
    residual_sd = (sd_forward * np.sqrt((1 - correlation**2).clip(lower=0.0))).where(
        count >= min_obs
    )
    agree = (
        (np.sign(score) == np.sign(forward)).where(score.notna() & forward.notna()).astype(float)
    )
    hit = (agree.rolling(window, min_periods=1).sum() / count.where(count > 0)).where(
        count >= min_obs
    )
    frame = pd.DataFrame({"beta": beta, "resid_sd": residual_sd, "hit": hit, "n": count})
    return frame.shift(horizon)


def probability_reliability(
    score: pd.Series, target: pd.Series, horizon: int, calibration: pd.DataFrame, buckets: int
) -> dict:
    forward = forward_return(target, horizon)
    expected = calibration["beta"].clip(lower=0.0) * score
    deviation = calibration["resid_sd"]
    probability = (expected / deviation.where(deviation > 0)).map(
        lambda v: normal_cdf(v) if np.isfinite(v) else np.nan
    )
    joined = pd.concat([probability.rename("p"), forward.rename("f")], axis=1).dropna()
    if len(joined) < 200:
        return {"n": len(joined), "mean_abs_gap_pct": math.nan, "buckets": []}
    groups = pd.qcut(joined["p"], min(buckets, joined["p"].nunique()), duplicates="drop")
    table = joined.groupby(groups, observed=True).agg(
        predicted=("p", "mean"), realised=("f", lambda v: float((v > 0).mean())), n=("f", "size")
    )
    gap = float((table["predicted"] - table["realised"]).abs().mean())
    return {
        "n": int(len(joined)),
        "mean_abs_gap_pct": gap * 100.0,
        "buckets": [
            {
                "predicted_pct": float(row.predicted * 100),
                "realised_pct": float(row.realised * 100),
                "n": int(row.n),
            }
            for row in table.itertuples()
        ],
    }


def verdict(ratio: float, params: ScoreParams) -> str:
    if ratio is None or not np.isfinite(ratio):
        return "n/a"
    side = "bullish" if ratio > 0 else "bearish"
    magnitude = abs(ratio)
    if magnitude >= params.strong_ratio:
        return f"{side.upper()} (strong)"
    if magnitude >= params.weak_ratio:
        return side.capitalize()
    return "Neutral"


def _last(series: pd.Series) -> float:
    return float(series.iloc[-1]) if len(series) and np.isfinite(series.iloc[-1]) else math.nan


def score_target(
    components: pd.DataFrame,
    target: pd.Series,
    prior: dict[str, float],
    params: ScoreParams,
    orthogonal: tuple[tuple[str, str], ...] = (),
    noise_daily: float | None = None,
) -> dict:
    components = components.dropna(how="all", axis=1)
    if components.empty or target is None or not len(target):
        return {}
    target = target.reindex(components.index).astype(float)
    components = orthogonalise(components, orthogonal, params.ic_window)
    ic = rolling_ic(components, target, params)
    weights = ic_weights(ic, prior, params.ic_shrink)
    if not len(weights.columns):
        return {}
    combined = composite(components, weights)
    score = _last(combined["score"])
    result = {
        "score": score,
        "n_components": int(combined["n"].iloc[-1]),
        "n_weighted": int(len(weights.columns)),
        "weights": {k: float(v) for k, v in weights.iloc[-1].dropna().items()},
        "ic": {k: float(v) for k, v in ic.iloc[-1].dropna().items()},
        "horizons": {},
    }
    for name, horizon in params.horizons:
        calibration = calibrate(
            combined["score"],
            target,
            horizon,
            params.calibration_window,
            params.calibration_min_obs,
        )
        beta, deviation = _last(calibration["beta"]), _last(calibration["resid_sd"])
        skill = bool(np.isfinite(beta) and beta > 0)
        expected = max(beta, 0.0) * score if np.isfinite(beta) and np.isfinite(score) else math.nan
        probability = (
            normal_cdf(expected / deviation)
            if np.isfinite(expected) and np.isfinite(deviation) and deviation > 0
            else math.nan
        )
        reliability = probability_reliability(
            combined["score"], target, horizon, calibration, params.probability_buckets
        )
        gap = reliability["mean_abs_gap_pct"]
        observations = int(_last(calibration["n"])) if np.isfinite(_last(calibration["n"])) else 0
        calibrated = bool(
            skill
            and np.isfinite(gap)
            and gap <= params.max_calibration_gap_pct
            and observations >= params.min_calibration_obs
        )
        noise = noise_daily * math.sqrt(horizon) if noise_daily else deviation
        ratio = expected / noise if np.isfinite(expected) and noise and noise > 0 else math.nan
        result["horizons"][name] = {
            "days": horizon,
            "skill": skill,
            "expected_pct": expected * 100 if np.isfinite(expected) else math.nan,
            "dispersion_pct": deviation * 100 if np.isfinite(deviation) else math.nan,
            "p_up_pct": probability * 100 if np.isfinite(probability) else math.nan,
            "hit_pct": _last(calibration["hit"]) * 100,
            "n": observations,
            "calibrated": calibrated,
            "calibration_gap_pct": gap,
            "ratio": ratio,
            "verdict": verdict(ratio, params),
            "reliability": reliability,
        }
    return result
