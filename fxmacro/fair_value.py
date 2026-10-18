from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import stats
from .config import LONG_TENORS, SHORT_TENORS, Instrument
from .data import MarketData


@dataclass(frozen=True)
class FairValueParams:
    window: int = 750
    decay_half_life: float = 120.0
    shrink: float = 0.35
    min_driver_t: float = 1.5
    rate_floor_fraction: float = 0.5
    rate_beta_floor: float = 0.01
    ridge: float = 1e-6
    coint_window: int = 1000
    min_r2: float = 0.30
    max_half_life: float = 60.0
    rate_max_age_days: int = 30
    min_history: int = 300
    hold_limit: int = 5


@dataclass
class LongRunFit:
    betas: dict[str, float]
    constant: float
    t_stats: dict[str, float]
    r2: float
    kept: list[str]
    dropped: list[str]
    observations: int


@dataclass
class FairValueEstimate:
    instrument: str
    design: pd.DataFrame
    frame: pd.DataFrame
    kept: list[str]
    dropped: list[str]
    rate_tenor: str | None
    long_run: LongRunFit | None
    cointegration: dict
    reliable: bool
    issues: list[str]
    source: str = "direct"
    ppp: pd.DataFrame | None = None
    full_design: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def latest(self) -> pd.Series:
        return self.frame.iloc[-1]

    @property
    def fair_value(self) -> float:
        return float(np.exp(self.latest["fv"]))

    @property
    def gap_pct(self) -> float:
        return float((np.exp(self.latest["resid"]) - 1.0) * 100.0)


def carry_forward(series: pd.Series, index: pd.DatetimeIndex, limit: int = 5) -> pd.Series:
    clean = series.dropna()
    clean = clean[~clean.index.duplicated(keep="last")]
    union = clean.index.union(index)
    return clean.reindex(union).ffill(limit=limit).reindex(index)


def rate_spread(
    yields: dict[str, dict[str, pd.Series]],
    base: str,
    quote: str,
    tenors: tuple[str, ...],
    asof: pd.Timestamp,
    params: FairValueParams,
) -> tuple[pd.Series | None, str | None]:
    candidates = []
    for tenor in tenors:
        first = yields.get(base, {}).get(tenor)
        second = yields.get(quote, {}).get(tenor)
        if first is None or second is None:
            continue
        index = first.index.union(second.index)
        spread = (
            carry_forward(first, index, params.hold_limit)
            - carry_forward(second, index, params.hold_limit)
        ).dropna()
        if len(spread) >= params.min_history:
            candidates.append((tenor, spread))
    if not candidates:
        return None, None
    for tenor, spread in candidates:
        if (asof - spread.index[-1]).days <= params.rate_max_age_days:
            return spread, tenor
    tenor, spread = max(candidates, key=lambda item: item[1].index[-1])
    return spread, f"{tenor} (stale)"


def build_design(
    instrument: Instrument, data: MarketData, params: FairValueParams
) -> tuple[pd.DataFrame, str | None]:
    close = data.prices[instrument.name].dropna()
    close = close[close > 0]
    index = close.index
    frame = pd.DataFrame({"y": np.log(close)})
    asof = index[-1]
    tenor = None

    if instrument.metal:
        real = data.yields.get("USD", {}).get("REAL10")
        if real is not None:
            frame["rate_diff"] = carry_forward(-real, index, params.hold_limit)
            tenor = "-US10y real"
    else:
        short, tenor = rate_spread(
            data.yields, instrument.base, instrument.quote, SHORT_TENORS, asof, params
        )
        if short is not None:
            frame["rate_diff"] = carry_forward(short, index, params.hold_limit)
        long, _ = rate_spread(
            data.yields, instrument.base, instrument.quote, LONG_TENORS, asof, params
        )
        if long is not None:
            long_daily = carry_forward(long, index, params.hold_limit)
            if "rate_diff" not in frame or long_daily.corr(frame["rate_diff"]) < 0.97:
                frame["rate_diff_long"] = long_daily

    for driver in instrument.drivers:
        if driver in data.market:
            level = data.market[driver].where(data.market[driver] > 0)
            frame[f"log_{driver}"] = np.log(carry_forward(level, index, params.hold_limit))

    return frame.dropna(), tenor


def long_run_fit(design: pd.DataFrame, params: FairValueParams) -> LongRunFit:
    window = design.iloc[-params.coint_window :]
    drivers = [column for column in window.columns if column != "y"]
    x = stats.add_constant([window[column] for column in drivers])
    y = window["y"].to_numpy()
    beta, resid = stats.ols(x, y)
    t_values = beta / stats.newey_west_se(x, resid)
    kept = [
        column
        for column, t in zip(drivers, t_values[1:], strict=False)
        if abs(t) >= params.min_driver_t
    ]
    if not kept:
        preferred = [c for c in ("rate_diff", "rate_diff_long") if c in drivers]
        kept = preferred[:1] or drivers[:1]
    return LongRunFit(
        betas=dict(zip(drivers, map(float, beta[1:]), strict=False)),
        constant=float(beta[0]),
        t_stats=dict(zip(drivers, map(float, t_values[1:]), strict=False)),
        r2=stats.r_squared(y, resid),
        kept=kept,
        dropped=[column for column in drivers if column not in kept],
        observations=len(window),
    )


def long_run_path(design: pd.DataFrame, params: FairValueParams, step: int = 63) -> pd.DataFrame:
    drivers = [column for column in design.columns if column != "y"]
    rows, stamps = [], []
    for t in range(params.window - 1, len(design), step):
        sample = design.iloc[max(0, t + 1 - params.coint_window) : t + 1]
        x = stats.add_constant([sample[column] for column in drivers])
        beta, _ = stats.ols(x, sample["y"].to_numpy())
        rows.append(beta[1:])
        stamps.append(design.index[t])
    path = pd.DataFrame(rows, index=pd.DatetimeIndex(stamps), columns=drivers)
    return path.reindex(design.index).ffill()


def rolling_fair_value(
    design: pd.DataFrame, prior: pd.DataFrame | None, params: FairValueParams
) -> pd.DataFrame:
    drivers = [column for column in design.columns if column != "y"]
    y = design["y"].to_numpy()
    x = stats.add_constant([design[column] for column in drivers])
    n, window = len(design), params.window
    weights = 0.5 ** (np.arange(window - 1, -1, -1) / params.decay_half_life)
    weights = weights / weights.sum() * window
    prior_matrix = prior[drivers].to_numpy(dtype=float) if prior is not None else None
    prior_vector = np.full(len(drivers), np.nan)
    rate_position = next(
        (drivers.index(c) for c in ("rate_diff", "rate_diff_long") if c in drivers), None
    )

    def rate_floor() -> float:
        if rate_position is None:
            return params.rate_beta_floor
        anchor = prior_vector[rate_position]
        if np.isfinite(anchor) and anchor > 0:
            return max(params.rate_beta_floor, params.rate_floor_fraction * anchor)
        return params.rate_beta_floor

    rows, stamps = [], []
    for t in range(window - 1, n):
        span = slice(t - window + 1, t + 1)
        if prior_matrix is not None:
            prior_vector = prior_matrix[t]
        xs, ys = x[span], y[span]
        beta = stats.weighted_least_squares(xs, ys, weights, params.ridge)
        floored = False
        if prior is not None:
            for j in range(len(drivers)):
                if np.isfinite(prior_vector[j]):
                    beta[j + 1] = (1 - params.shrink) * beta[j + 1] + params.shrink * prior_vector[
                        j
                    ]
        if rate_position is not None and beta[rate_position + 1] < rate_floor():
            fixed = rate_floor()
            others = [0] + [j + 1 for j in range(len(drivers)) if j != rate_position]
            refit = stats.weighted_least_squares(
                xs[:, others], ys - fixed * xs[:, rate_position + 1], weights, params.ridge
            )
            beta = np.zeros(len(drivers) + 1)
            beta[rate_position + 1] = fixed
            for slot, column in enumerate(others):
                beta[column] = refit[slot]
            if prior is not None:
                for column in others[1:]:
                    if np.isfinite(prior_vector[column - 1]):
                        beta[column] = (1 - params.shrink) * beta[
                            column
                        ] + params.shrink * prior_vector[column - 1]
            floored = True
        beta[0] = float(np.sum(weights * (ys - xs[:, 1:] @ beta[1:])) / np.sum(weights))
        fitted = xs @ beta
        resid = ys - fitted
        centre = float(np.sum(weights * ys) / np.sum(weights))
        sse = float(np.sum(weights * resid**2))
        sst = float(np.sum(weights * (ys - centre) ** 2))
        deviation = math.sqrt(sse / max(np.sum(weights) - len(beta), 1.0))
        push = float((x[t, 1:] - x[t - 20, 1:]) @ beta[1:]) if t >= 20 else math.nan
        row = {
            "fv": fitted[-1],
            "resid": resid[-1],
            "resid_sd": deviation,
            "resid_z": resid[-1] / deviation if deviation > 0 else math.nan,
            "r2": 1.0 - sse / sst if sst > 0 else math.nan,
            "half_life": stats.half_life(resid),
            "push20": push,
            "rate_floored": floored,
        }
        for j, column in enumerate(drivers, start=1):
            row[f"beta_{column}"] = beta[j]
        rows.append(row)
        stamps.append(design.index[t])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(stamps))


def error_correction(design: pd.DataFrame, window: int) -> dict[str, float]:
    sample = design.iloc[-window:]
    drivers = [column for column in sample.columns if column != "y"]
    x0 = stats.add_constant([sample[column] for column in drivers])
    _, equilibrium_error = stats.ols(x0, sample["y"].to_numpy())
    dy = np.diff(sample["y"].to_numpy())
    dx = [np.diff(sample[column].to_numpy()) for column in drivers]
    x = stats.add_constant([equilibrium_error[:-1]] + dx)
    beta, resid = stats.ols(x, dy)
    dof = max(len(dy) - x.shape[1], 1)
    covariance = np.linalg.pinv(x.T @ x) * float(resid @ resid) / dof
    alpha = float(beta[1])
    t_alpha = alpha / math.sqrt(max(covariance[1, 1], 1e-300))
    half = math.log(0.5) / math.log(1 + alpha) if -1 < alpha < 0 else math.inf
    return {"alpha": alpha, "t_alpha": float(t_alpha), "half_life": half}


def cointegration_report(design: pd.DataFrame, params: FairValueParams) -> dict:
    window = design.iloc[-params.coint_window :]
    if len(window) < 250:
        return {"error": "not enough history", "cointegrated": False}
    drivers = [column for column in window.columns if column != "y"]
    y = window["y"].to_numpy()
    x = window[drivers].to_numpy()
    test = stats.engle_granger(y, x)
    _, resid = stats.ols(stats.add_constant([window[c] for c in drivers]), y)
    ratio, ratio_z = stats.variance_ratio(resid, 20)
    correction = error_correction(window, params.coint_window)
    verdict = test["p_value"] < 0.05 or (np.isfinite(ratio_z) and ratio_z < -2.0)
    return {
        **test,
        "variance_ratio": ratio,
        "variance_ratio_z": ratio_z,
        "half_life": stats.half_life(resid),
        "ecm_alpha": correction["alpha"],
        "ecm_t": correction["t_alpha"],
        "ecm_half_life": correction["half_life"],
        "window": len(window),
        "cointegrated": bool(verdict),
    }


def quality_issues(latest: pd.Series, params: FairValueParams) -> list[str]:
    issues = []
    if not latest["r2"] >= params.min_r2:
        issues.append(f"R2 {latest['r2']:.2f} below {params.min_r2:.2f}")
    if not latest["half_life"] <= params.max_half_life:
        issues.append(
            f"residual half-life {latest['half_life']:.0f}d above {params.max_half_life:.0f}d"
        )
    return issues


def ppp_anchor(
    close: pd.Series,
    cpi_base: pd.Series | None,
    cpi_quote: pd.Series | None,
    min_months: int = 60,
) -> pd.DataFrame | None:
    if cpi_base is None or cpi_quote is None:
        return None
    spot = np.log(close.resample("ME").mean())
    base = np.log(cpi_base.resample("ME").last().ffill(limit=3))
    quote = np.log(cpi_quote.resample("ME").last().ffill(limit=3))
    monthly = pd.concat({"s": spot, "pb": base, "pq": quote}, axis=1).dropna()
    if len(monthly) < min_months:
        return None
    monthly["q"] = monthly["s"] + monthly["pb"] - monthly["pq"]
    mean_q = monthly["q"].mean()
    monthly["ppp_log"] = mean_q - monthly["pb"] + monthly["pq"]
    monthly["ppp"] = np.exp(monthly["ppp_log"])
    monthly["deviation_pct"] = (np.exp(monthly["s"] - monthly["ppp_log"]) - 1.0) * 100.0
    monthly["q_z"] = (monthly["q"] - mean_q) / monthly["q"].std()
    return monthly


def estimate_fair_value(
    instrument: Instrument, data: MarketData, params: FairValueParams | None = None
) -> FairValueEstimate | None:
    params = params or FairValueParams()
    design, tenor = build_design(instrument, data, params)
    drivers = [column for column in design.columns if column != "y"]
    if not drivers or len(design) < params.window + 50:
        return None
    long_run = long_run_fit(design, params)
    pruned = design[["y"] + long_run.kept]
    frame = rolling_fair_value(pruned, long_run_path(pruned, params), params)
    issues = quality_issues(frame.iloc[-1], params)
    ppp = None
    if not instrument.metal:
        ppp = ppp_anchor(
            data.prices[instrument.name],
            data.macro.get(instrument.base, {}).get("CPI"),
            data.macro.get(instrument.quote, {}).get("CPI"),
        )
    return FairValueEstimate(
        instrument=instrument.name,
        design=pruned,
        frame=frame,
        kept=long_run.kept,
        dropped=long_run.dropped,
        rate_tenor=tenor,
        long_run=long_run,
        cointegration=cointegration_report(pruned, params),
        reliable=not issues,
        issues=issues,
        ppp=ppp,
        full_design=design,
    )


def frame_from_log_fair_value(
    log_fair_value: pd.Series, y: pd.Series, window: int = 250
) -> pd.DataFrame:
    index = log_fair_value.dropna().index.intersection(y.dropna().index)
    fair, actual = log_fair_value.reindex(index), y.reindex(index)
    resid = actual - fair
    minimum = window // 2
    deviation = resid.rolling(window, min_periods=minimum).std()
    lagged = resid.shift(1)
    covariance = resid.rolling(window, min_periods=minimum).cov(lagged)
    variance = lagged.rolling(window, min_periods=minimum).var()
    phi = (covariance / variance.where(variance > 0)).clip(-0.999, 0.999)
    half = np.log(0.5) / np.log(phi.where(phi > 0))
    r2 = (
        1.0
        - resid.rolling(window, min_periods=minimum).var()
        / actual.rolling(window, min_periods=minimum).var()
    )
    frame = pd.DataFrame(
        {
            "fv": fair,
            "resid": resid,
            "resid_sd": deviation,
            "resid_z": resid / deviation.where(deviation > 0),
            "r2": r2,
            "half_life": half,
            "push20": fair.diff(20),
            "rate_floored": False,
        },
        index=index,
    )
    return frame.dropna(subset=["fv", "resid_sd"])


def leg_fair_values(estimates: dict[str, FairValueEstimate]) -> dict[str, pd.Series]:
    legs: dict[str, pd.Series] = {}
    for name, estimate in estimates.items():
        base, quote = name[:3], name[3:]
        if quote == "USD" and not name.startswith("X"):
            legs.setdefault(base, estimate.frame["fv"])
        elif base == "USD":
            legs.setdefault(quote, -estimate.frame["fv"])
    return legs


def estimate_all(
    instruments: list[Instrument], data: MarketData, params: FairValueParams | None = None
) -> dict[str, FairValueEstimate]:
    params = params or FairValueParams()
    estimates: dict[str, FairValueEstimate] = {}
    for instrument in instruments:
        if instrument.name not in data.prices:
            continue
        estimate = estimate_fair_value(instrument, data, params)
        if estimate is not None:
            estimates[instrument.name] = estimate

    legs = leg_fair_values({n: e for n, e in estimates.items() if is_usd_pair(n)})
    for name, estimate in list(estimates.items()):
        if is_usd_pair(name) or name.startswith("X"):
            continue
        base, quote = name[:3], name[3:]
        if base not in legs or quote not in legs:
            continue
        y = estimate.design["y"]
        rebuilt = frame_from_log_fair_value(legs[base] - legs[quote], y)
        if rebuilt.empty:
            continue
        if rebuilt["r2"].iloc[-1] > estimate.frame["r2"].iloc[-1]:
            issues = quality_issues(rebuilt.iloc[-1], params)
            estimates[name] = FairValueEstimate(
                instrument=name,
                design=estimate.design,
                frame=rebuilt,
                kept=estimate.kept,
                dropped=estimate.dropped,
                rate_tenor=estimate.rate_tenor,
                long_run=estimate.long_run,
                cointegration=estimate.cointegration,
                reliable=not issues,
                issues=issues,
                source="usd legs",
                ppp=estimate.ppp,
                full_design=estimate.full_design,
            )
    return estimates


def is_usd_pair(name: str) -> bool:
    return "USD" in (name[:3], name[3:]) and not name.startswith("X")
