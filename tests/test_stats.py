import math

import numpy as np

from fxmacro import stats


def test_ols_recovers_coefficients():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    y = 1.5 + 2.0 * x + 0.1 * rng.standard_normal(2000)
    beta, resid = stats.ols(stats.add_constant([x]), y)
    np.testing.assert_allclose(beta, [1.5, 2.0], atol=0.02)
    assert abs(resid.mean()) < 1e-9


def test_weighted_least_squares_matches_ols_for_equal_weights():
    rng = np.random.default_rng(1)
    x = stats.add_constant([rng.standard_normal(500), rng.standard_normal(500)])
    y = x @ np.array([0.2, 1.0, -0.5]) + 0.05 * rng.standard_normal(500)
    plain, _ = stats.ols(x, y)
    weighted = stats.weighted_least_squares(x, y, np.ones(500), 0.0)
    np.testing.assert_allclose(plain, weighted, atol=1e-8)


def test_weighted_least_squares_downweights_outliers():
    rng = np.random.default_rng(2)
    x = stats.add_constant([rng.standard_normal(400)])
    y = x @ np.array([0.0, 1.0]) + 0.01 * rng.standard_normal(400)
    y[:20] += 5.0
    weights = np.ones(400)
    weights[:20] = 1e-6
    beta = stats.weighted_least_squares(x, y, weights, 0.0)
    assert abs(beta[0]) < 0.01


def test_newey_west_exceeds_naive_error_under_autocorrelation():
    rng = np.random.default_rng(3)
    n = 3000
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.9 * noise[t - 1] + rng.standard_normal()
    x = stats.add_constant([np.cumsum(rng.standard_normal(n)) * 0.1])
    beta, resid = stats.ols(x, noise)
    naive = np.sqrt(np.diag(np.linalg.inv(x.T @ x)) * resid.var())
    assert (stats.newey_west_se(x, resid) > naive).all()


def test_half_life_of_ar1():
    rng = np.random.default_rng(4)
    series = np.zeros(20000)
    for t in range(1, len(series)):
        series[t] = 0.9 * series[t - 1] + rng.standard_normal()
    assert math.isclose(stats.half_life(series), math.log(0.5) / math.log(0.9), rel_tol=0.1)


def test_half_life_infinite_for_random_walk_like_series():
    walk = np.cumsum(np.random.default_rng(5).standard_normal(5000))
    assert stats.half_life(walk) > 200


def test_variance_ratio_separates_random_walk_from_mean_reversion():
    rng = np.random.default_rng(6)
    walk = np.cumsum(rng.standard_normal(4000))
    reverting = np.zeros(4000)
    for t in range(1, 4000):
        reverting[t] = 0.8 * reverting[t - 1] + rng.standard_normal()
    _, z_walk = stats.variance_ratio(walk, 20)
    ratio, z_reverting = stats.variance_ratio(reverting, 20)
    assert abs(z_walk) < 3.0
    assert ratio < 0.5 and z_reverting < -3.0


def test_engle_granger_verdicts():
    rng = np.random.default_rng(7)
    trend = np.cumsum(rng.standard_normal(1500))
    linked = 0.8 * trend + rng.standard_normal(1500)
    unrelated = np.cumsum(rng.standard_normal(1500))
    related = stats.engle_granger(linked, trend[:, None])
    spurious = stats.engle_granger(unrelated, trend[:, None])
    assert related["p_value"] < 0.01
    assert related["statistic"] < related["critical_1pct"]
    assert spurious["p_value"] > related["p_value"]
