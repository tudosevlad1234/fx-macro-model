import numpy as np
import pandas as pd

from fxmacro import statarb
from fxmacro.statarb import ArbParams


def test_shared_currency_detection():
    assert statarb.shares_currency("EURUSD", "EURJPY")
    assert not statarb.shares_currency("EURUSD", "GBPJPY")


def test_mechanical_pair_requires_shared_currency_and_unit_beta():
    assert statarb.is_mechanical("EURUSD", "EURJPY", 0.95, 0.3)
    assert not statarb.is_mechanical("EURUSD", "GBPJPY", 0.95, 0.3)
    assert not statarb.is_mechanical("EURUSD", "EURJPY", 0.3, 0.3)


def test_phase_randomisation_preserves_spectrum_not_path():
    rng = np.random.default_rng(0)
    series = np.cumsum(rng.standard_normal(1024))
    shuffled = statarb.phase_randomise(series, rng)
    assert not np.allclose(series, shuffled)
    np.testing.assert_allclose(
        np.abs(np.fft.rfft(series - series.mean()))[1:-1],
        np.abs(np.fft.rfft(shuffled - shuffled.mean()))[1:-1],
        rtol=1e-6,
    )


def build_prices(seed=0, linked=True, n=2400):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2012-01-02", periods=n)
    trend = np.cumsum(0.004 * rng.standard_normal(n))
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.93 * noise[t - 1] + 0.004 * rng.standard_normal()
    second = trend + 0.002 * np.cumsum(rng.standard_normal(n)) * (0 if linked else 8)
    first = 0.6 * trend + noise if linked else np.cumsum(0.004 * rng.standard_normal(n))
    return pd.DataFrame(
        {"EURUSD": 1.1 * np.exp(first), "NZDJPY": 80.0 * np.exp(second)}, index=index
    )


def test_scan_accepts_planted_cointegrated_pair():
    accepted, _ = statarb.scan(build_prices(linked=True))
    assert [(c["first"], c["second"]) for c in accepted] == [("EURUSD", "NZDJPY")]
    assert abs(accepted[0]["beta"] - 0.6) < 0.1
    assert accepted[0]["null_p_value"] <= 0.05


def test_scan_rejects_independent_walks():
    accepted, rejected = statarb.scan(build_prices(seed=3, linked=False))
    assert accepted == [] and rejected


def test_backtest_trades_the_planted_pair_profitably():
    prices = build_prices(linked=True)
    result = statarb.backtest(prices, "EURUSD", "NZDJPY")
    assert result["trades"] > 5
    assert result["total_return_pct"] > 0


def test_backtest_costs_reduce_pnl():
    prices = build_prices(linked=True)
    cheap = statarb.backtest(prices, "EURUSD", "NZDJPY")
    original = statarb.INSTRUMENTS["EURUSD"]
    expensive = type(original)(
        original.name,
        original.base,
        original.quote,
        original.pip,
        original.cost_pips * 50,
        original.drivers,
        original.metal,
    )
    statarb.INSTRUMENTS["EURUSD"] = expensive
    try:
        costly = statarb.backtest(prices, "EURUSD", "NZDJPY")
    finally:
        statarb.INSTRUMENTS["EURUSD"] = original
    assert costly["total_return_pct"] < cheap["total_return_pct"]


def test_simulate_charges_costs_on_entry_and_exit():
    params = ArbParams(entry_z=2.0, exit_z=0.25, stop_z=3.5)
    z = np.array([0.0, 2.5, 1.0, 0.1, 0.0])
    returns = np.array([np.nan, 0.0, -0.01, -0.02, 0.05])
    pnl, closed, entries = statarb.simulate(z, returns, 0.001, params)
    assert entries == 1
    np.testing.assert_allclose(pnl, [0.0, -0.001, 0.01, 0.02 - 0.001, 0.0])
    np.testing.assert_allclose(closed, [0.01 + 0.02 - 0.002])


def test_simulate_exits_when_spread_overshoots_the_mean():
    params = ArbParams(entry_z=2.0, exit_z=0.25, stop_z=3.5)
    z = np.array([2.5, -2.4, -2.4])
    _, closed, entries = statarb.simulate(z, np.zeros(3), 0.0, params)
    assert entries == 2 and len(closed) == 1


def test_simulate_waits_after_a_stop():
    params = ArbParams(entry_z=2.0, exit_z=0.25, stop_z=3.5)
    z = np.array([2.5, 3.8, 3.0, 1.0, 2.2])
    _, closed, entries = statarb.simulate(z, np.zeros(5), 0.0, params)
    assert entries == 2 and len(closed) == 1


def test_current_signal_fires_beyond_entry_threshold():
    prices = build_prices(linked=True)
    candidate = {"first": "EURUSD", "second": "NZDJPY", "beta": 0.6, "intercept": 0.0}
    params = ArbParams(z_window=120)
    log_second = np.log(prices["NZDJPY"])
    prices.iloc[-1, 0] = float(np.exp(0.6 * log_second.iloc[-1] + 0.3))
    signal = statarb.current_signal(prices, candidate, params)
    assert signal["z"] > params.entry_z and signal["action"].startswith("short EURUSD")
