import numpy as np
import pandas as pd

from fxmacro import fair_value as fv
from fxmacro.config import INSTRUMENTS
from fxmacro.data import MarketData
from fxmacro.fair_value import FairValueParams
from fxmacro.synthetic import RATE_BETA


def test_estimates_cover_requested_pairs(estimates):
    assert {"EURUSD", "USDJPY", "AUDUSD"} <= set(estimates)


def test_rate_beta_is_positive_and_near_truth(estimates):
    long_run = estimates["EURUSD"].long_run
    assert long_run.betas["rate_diff"] > 0
    assert abs(long_run.betas["rate_diff"] - RATE_BETA) < 0.03


def test_fair_value_frame_is_finite_and_consistent(estimates):
    for estimate in estimates.values():
        frame = estimate.frame
        assert np.isfinite(frame[["fv", "resid", "resid_sd"]]).all().all()
        assert np.allclose(frame["resid"] / frame["resid_sd"], frame["resid_z"])
        assert np.isclose(estimate.fair_value, np.exp(frame["fv"].iloc[-1]))


def test_rate_beta_never_below_floor(estimates):
    for estimate in estimates.values():
        column = "beta_rate_diff" if "beta_rate_diff" in estimate.frame else "beta_rate_diff_long"
        if column in estimate.frame:
            assert (estimate.frame[column] >= FairValueParams().rate_beta_floor - 1e-9).all()


def test_rolling_fair_value_does_not_use_future_data(estimates):
    params = FairValueParams()
    design = estimates["EURUSD"].design
    full = fv.rolling_fair_value(design, fv.long_run_path(design, params), params)
    clipped = design.iloc[:-300]
    partial = fv.rolling_fair_value(clipped, fv.long_run_path(clipped, params), params)
    shared = partial.index
    np.testing.assert_allclose(full.loc[shared, "fv"], partial["fv"], atol=1e-10)


def test_short_history_returns_none(market):
    short = MarketData(
        prices={"EURUSD": market.prices["EURUSD"].iloc[:400]},
        yields=market.yields,
        macro=market.macro,
        market=market.market,
    )
    assert fv.estimate_fair_value(INSTRUMENTS["EURUSD"], short) is None


def test_cointegration_report_flags_unrelated_walks():
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2015-01-01", periods=1200)
    design = pd.DataFrame(
        {"y": np.cumsum(rng.standard_normal(1200)), "x": np.cumsum(rng.standard_normal(1200))},
        index=index,
    )
    assert fv.cointegration_report(design, FairValueParams())["cointegrated"] is False


def test_cointegration_report_accepts_linked_series():
    rng = np.random.default_rng(1)
    index = pd.bdate_range("2015-01-01", periods=1200)
    x = np.cumsum(rng.standard_normal(1200))
    design = pd.DataFrame({"y": 0.7 * x + rng.standard_normal(1200), "x": x}, index=index)
    report = fv.cointegration_report(design, FairValueParams())
    assert report["cointegrated"] is True and report["p_value"] < 0.01


def test_carry_forward_respects_limit():
    series = pd.Series([1.0], index=pd.to_datetime(["2020-01-01"]))
    index = pd.bdate_range("2020-01-01", periods=10)
    filled = fv.carry_forward(series, index, limit=3)
    assert filled.notna().sum() == 4


def test_ppp_anchor_deviation_is_zero_at_mean_real_rate():
    months = pd.date_range("2005-01-31", periods=120, freq="ME")
    daily = pd.bdate_range(months[0], months[-1])
    rng = np.random.default_rng(2)
    close = pd.Series(
        1.2 * np.exp(0.05 * rng.standard_normal(len(daily)).cumsum() * 0.05), index=daily
    )
    cpi_a = pd.Series(100 * 1.002 ** np.arange(120), index=months)
    cpi_b = pd.Series(100 * 1.001 ** np.arange(120), index=months)
    anchor = fv.ppp_anchor(close, cpi_a, cpi_b)
    assert abs(anchor["q_z"].mean()) < 1e-9
    assert fv.ppp_anchor(close, None, cpi_b) is None


def test_usd_leg_rebuild_marks_source(market):
    from fxmacro.fair_value import estimate_all

    names = ["EURUSD", "GBPUSD", "EURGBP"]
    estimates = estimate_all([INSTRUMENTS[n] for n in names], market)
    assert estimates["EURGBP"].source in ("direct", "usd legs")
    assert (
        fv.is_usd_pair("EURUSD") and not fv.is_usd_pair("EURGBP") and not fv.is_usd_pair("XAUUSD")
    )
