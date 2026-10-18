import numpy as np
import pandas as pd

from fxmacro import data_io
from fxmacro.currency_score import (
    CURRENCY_PRIOR,
    basket_relative,
    relative_to_others,
    score_currencies,
    usd_legs,
)
from fxmacro.pair_score import PAIR_PRIOR, macro_difference, score_pairs
from fxmacro.pipeline import currency_table, pair_table, run


def test_usd_legs_orientation(market):
    legs = usd_legs(market.prices)
    assert (legs["USD"] == 0).all()
    assert np.allclose(legs["EUR"], np.log(market.prices["EURUSD"]).reindex(legs.index))
    assert np.allclose(legs["JPY"], -np.log(market.prices["USDJPY"]).reindex(legs.index))


def test_basket_relative_sums_to_zero(market):
    relative = basket_relative(usd_legs(market.prices))
    assert np.allclose(relative.sum(axis=1), 0.0, atol=1e-9)


def test_relative_to_others_is_zero_sum():
    frame = pd.DataFrame(np.random.default_rng(0).standard_normal((50, 5)))
    assert np.allclose(relative_to_others(frame).sum(axis=1), 0.0, atol=1e-9)


def test_currency_scores_structure(market):
    scores = score_currencies(market)
    assert set(scores) == {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"}
    for outcome in scores.values():
        assert -1.0 <= outcome["score"] <= 1.0
        assert abs(sum(outcome["weights"].values()) - 1.0) < 1e-9
        assert set(outcome["weights"]) <= set(CURRENCY_PRIOR)


def test_currency_table_is_sorted(market):
    table = currency_table(score_currencies(market))
    assert table["score"].is_monotonic_decreasing


def test_pair_scores(market, estimates):
    scores = score_pairs(estimates, market)
    assert scores
    for outcome in scores.values():
        assert set(outcome["weights"]) <= set(PAIR_PRIOR)
        for detail in outcome["horizons"].values():
            assert 0.0 <= detail["p_up_pct"] <= 100.0 or np.isnan(detail["p_up_pct"])
    assert "score" in pair_table(scores).columns


def test_skew_enters_pair_score_when_supplied(market, estimates):
    base = score_pairs({"EURUSD": estimates["EURUSD"]}, market)
    tilted = score_pairs({"EURUSD": estimates["EURUSD"]}, market, skews={"EURUSD": 0.9})
    assert "skew" in tilted["EURUSD"]["components"] and "skew" not in base["EURUSD"]["components"]
    assert tilted["EURUSD"]["score"] > base["EURUSD"]["score"]


def test_macro_difference_antisymmetric():
    index = pd.bdate_range("2020-01-01", periods=5)
    table = {"infl": pd.DataFrame({"EUR": 0.5, "USD": 0.1}, index=index)}
    forward = macro_difference(table, "EUR", "USD", index)
    backward = macro_difference(table, "USD", "EUR", index)
    assert np.allclose(forward, -backward)


def test_csv_round_trip(market, tmp_path):
    data_io.save(market, tmp_path)
    loaded = data_io.load(tmp_path)
    assert set(loaded.prices) == set(market.prices)
    pd.testing.assert_series_equal(
        loaded.prices["EURUSD"], market.prices["EURUSD"], check_freq=False, check_names=False
    )
    np.testing.assert_allclose(
        loaded.yields["EUR"]["Y2"].to_numpy(), market.yields["EUR"]["Y2"].to_numpy()
    )
    assert set(loaded.options) == set(market.options)
    assert len(loaded.events) == len(market.events)


def test_load_reports_missing_files(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="prices.csv"):
        data_io.load(tmp_path)


def test_run_end_to_end(market):
    result = run(
        market, instruments=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"], include_stat_arb=False
    )
    assert {"fair_value", "currency_scores", "pair_scores", "regimes", "smiles"} <= set(result)
    assert "EURUSD" in result["smiles"] and "USDJPY" in result["smiles"]
    assert result["fair_value"].loc["EURUSD", "fair_value"] > 0


def test_cli_demo_writes_report(tmp_path):
    from fxmacro.__main__ import main

    assert main(["demo", "--out", str(tmp_path), "--no-stat-arb"]) == 0
    assert (tmp_path / "report.md").read_text().startswith("# FX macro model report")
