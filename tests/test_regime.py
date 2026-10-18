import numpy as np
import statsmodels.api as sm

from fxmacro.regime import classify, describe_regimes, fit_markov


def simulate(n=2500, seed=1, calm=0.5, wild=1.5, stay=(0.98, 0.95)):
    rng = np.random.default_rng(seed)
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        keep = rng.random() < stay[state[t - 1]]
        state[t] = state[t - 1] if keep else 1 - state[t - 1]
    return rng.standard_normal(n) * np.where(state == 0, calm, wild), state


def test_recovers_volatility_states():
    x, state = simulate()
    fit = fit_markov(x, volatility_only=True)
    np.testing.assert_allclose(fit.sigmas, [0.5, 1.5], rtol=0.1)
    assert np.mean((fit.smoothed[:, 1] > 0.5) == (state == 1)) > 0.9


def test_matches_statsmodels_likelihood():
    x, _ = simulate(seed=3)
    ours = fit_markov(x, volatility_only=True)
    reference = sm.tsa.MarkovRegression(x, k_regimes=2, trend="n", switching_variance=True).fit()
    assert abs(ours.log_likelihood - reference.llf) < 0.5


def test_mean_switching_orders_states_by_mean():
    rng = np.random.default_rng(4)
    x = np.concatenate([rng.normal(-1, 0.5, 600), rng.normal(1, 0.5, 600)] * 2)
    fit = fit_markov(x)
    assert fit.means[0] < 0 < fit.means[1]


def test_probabilities_are_valid():
    x, _ = simulate(n=800)
    fit = fit_markov(x, volatility_only=True)
    np.testing.assert_allclose(fit.filtered.sum(axis=1), 1.0)
    np.testing.assert_allclose(fit.smoothed.sum(axis=1), 1.0)
    np.testing.assert_allclose(fit.transition.sum(axis=1), 1.0)


def test_single_regime_noise_is_not_reported_as_confident():
    x = np.random.default_rng(8).standard_normal(1500)
    summary = describe_regimes(fit_markov(x, volatility_only=True))
    assert not summary["confident"] or summary["vol_ratio"] >= 1.3


def test_classify_reports_current_state():
    x, state = simulate(n=1500, seed=5)
    summary = classify(x)
    assert summary["state"] in ("calm", "turbulent")
    assert summary["persistence"][0] > 0.9
    assert len(summary["smoothed_high"]) == len(x)
