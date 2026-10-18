import math

import numpy as np
import pandas as pd

from fxmacro import scoring
from fxmacro.scoring import ScoreParams


def frame(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2015-01-01", periods=n)
    target = pd.Series(np.cumsum(0.005 * rng.standard_normal(n)), index=index)
    signal = pd.Series(rng.standard_normal(n), index=index)
    return index, target, signal


def test_squash_is_bounded_and_odd():
    values = np.array([-50.0, -1.0, 0.0, 1.0, 50.0])
    squashed = scoring.squash(values)
    assert np.all(np.abs(squashed) <= 1.0)
    np.testing.assert_allclose(squashed, -squashed[::-1])


def test_standardise_clips_and_scales():
    series = pd.Series(np.random.default_rng(0).standard_normal(500) * 3)
    z = scoring.standardise(series, 100, 50)
    assert z.dropna().abs().max() <= 4.0
    assert abs(z.dropna().std() - 1.0) < 0.3


def test_normal_cdf_reference_values():
    assert scoring.normal_cdf(0.0) == 0.5
    assert math.isclose(scoring.normal_cdf(1.96), 0.975, abs_tol=1e-3)


def test_forward_return_alignment():
    series = pd.Series([1.0, 2.0, 4.0, 8.0])
    result = scoring.forward_return(series, 2)
    assert result.iloc[0] == 3.0 and result.iloc[1] == 6.0 and result.iloc[2:].isna().all()


def test_rolling_ic_has_no_look_ahead():
    index, target, signal = frame()
    params = ScoreParams(ic_window=500, ic_min_obs=100, ic_horizon=20)
    components = pd.DataFrame({"a": signal})
    cut = index[1000]
    full = scoring.rolling_ic(components, target, params)
    truncated = scoring.rolling_ic(components.loc[:cut], target.loc[:cut], params)
    pd.testing.assert_series_equal(full["a"].loc[:cut], truncated["a"], check_names=False)


def test_rolling_ic_detects_predictive_signal():
    index, target, _ = frame(2500, seed=3)
    horizon = 20
    forward = scoring.forward_return(target, horizon)
    rng = np.random.default_rng(1)
    informative = forward / forward.std() + rng.standard_normal(len(index))
    params = ScoreParams(ic_window=750, ic_min_obs=250, ic_horizon=horizon)
    ic = scoring.rolling_ic(pd.DataFrame({"x": informative}, index=index), target, params)
    assert ic["x"].iloc[-1] > 0.3


def test_ic_weights_sum_to_one_and_shrink_to_prior():
    index = pd.bdate_range("2020-01-01", periods=5)
    ic = pd.DataFrame({"a": [0.5] * 5, "b": [-0.2] * 5}, index=index)
    prior = {"a": 1.0, "b": 3.0}
    weights = scoring.ic_weights(ic, prior, 0.7)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    assert math.isclose(weights["b"].iloc[0], 0.7 * 0.75, rel_tol=1e-9)


def test_ic_weights_fall_back_to_prior_without_history():
    ic = pd.DataFrame({"a": [np.nan], "b": [np.nan]})
    weights = scoring.ic_weights(ic, {"a": 1.0, "b": 1.0}, 0.7)
    np.testing.assert_allclose(weights.iloc[0], [0.5, 0.5])


def test_composite_renormalises_over_available_components():
    index = pd.bdate_range("2020-01-01", periods=2)
    components = pd.DataFrame({"a": [1.0, 1.0], "b": [np.nan, -1.0]}, index=index)
    weights = pd.DataFrame({"a": [0.5, 0.5], "b": [0.5, 0.5]}, index=index)
    result = scoring.composite(components, weights)
    assert result["score"].iloc[0] == 1.0 and result["score"].iloc[1] == 0.0


def test_orthogonalise_removes_loading():
    rng = np.random.default_rng(0)
    base = pd.Series(rng.standard_normal(1000))
    mixed = 0.8 * base + 0.3 * pd.Series(rng.standard_normal(1000))
    frame_ = pd.DataFrame({"fast": mixed, "slow": base})
    result = scoring.orthogonalise(frame_, (("fast", "slow"),), 500)
    assert abs(result["fast"].iloc[600:].corr(base.iloc[600:])) < 0.1


def test_calibration_recovers_known_slope():
    index, target, _ = frame(3000, seed=5)
    horizon = 20
    forward = scoring.forward_return(target, horizon)
    rng = np.random.default_rng(2)
    score = (forward / 0.5 + 0.01 * rng.standard_normal(len(index))).rename("score")
    score = score.fillna(0.0)
    table = scoring.calibrate(score, target, horizon, 1000, 250)
    assert math.isclose(table["beta"].iloc[-1], 0.5, rel_tol=0.1)


def test_verdict_thresholds():
    params = ScoreParams()
    assert scoring.verdict(0.5, params) == "BULLISH (strong)"
    assert scoring.verdict(-0.2, params) == "Bearish"
    assert scoring.verdict(0.05, params) == "Neutral"
    assert scoring.verdict(float("nan"), params) == "n/a"


def test_score_target_returns_horizon_blocks():
    index, target, signal = frame(2200, seed=9)
    components = pd.DataFrame({"a": signal, "b": -signal * 0.3}, index=index)
    outcome = scoring.score_target(components, target, {"a": 1.0, "b": 1.0}, ScoreParams())
    assert set(outcome["horizons"]) == {"20d", "60d"}
    assert math.isclose(sum(outcome["weights"].values()), 1.0)
    assert 0.0 <= outcome["horizons"]["20d"]["p_up_pct"] <= 100.0


def test_anti_predictive_score_is_not_flipped():
    index, target, signal = frame(2200, seed=9)
    forward = scoring.forward_return(target, 20).fillna(0.0)
    components = pd.DataFrame({"a": scoring.squash(-forward / 0.05 + 0.2 * signal)}, index=index)
    outcome = scoring.score_target(components, target, {"a": 1.0}, ScoreParams())
    assert outcome["score"] != 0.0
    for block in outcome["horizons"].values():
        assert block["skill"] is False
        assert block["expected_pct"] == 0.0
        assert block["p_up_pct"] == 50.0
        assert block["verdict"] == "Neutral"
        assert block["calibrated"] is False
