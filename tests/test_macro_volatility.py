import math

import numpy as np
import pandas as pd

from fxmacro import volatility
from fxmacro.macro import infer_frequency, lagged_daily, macro_currency_z, surprise_index


def test_infer_frequency():
    daily = pd.Series(1.0, index=pd.bdate_range("2020-01-01", periods=50))
    monthly = pd.Series(1.0, index=pd.date_range("2020-01-01", periods=24, freq="MS"))
    quarterly = pd.Series(1.0, index=pd.date_range("2020-01-01", periods=12, freq="QS"))
    assert [infer_frequency(s) for s in (daily, monthly, quarterly)] == [
        "daily",
        "monthly",
        "quarterly",
    ]


def test_lagged_daily_hides_values_until_published():
    monthly = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-01", "2020-02-01"]))
    index = pd.bdate_range("2020-01-01", "2020-03-31")
    result = lagged_daily(monthly, index, 20, 130)
    assert result.loc[:"2020-01-20"].isna().all()
    assert result.loc["2020-01-21"] == 1.0
    assert result.loc["2020-02-20"] == 1.0
    assert result.loc["2020-02-21"] == 2.0


def test_surprise_index_decays_with_half_life():
    index = pd.bdate_range("2020-01-01", periods=120)
    events = pd.DataFrame(
        {
            "date": [index[60]],
            "ccy": "USD",
            "impact": 1.0,
            "forecast": 0.0,
            "actual": 3.0,
            "sd": 1.0,
        }
    )
    series = surprise_index(events, "USD", index, half_life=10.0)
    assert series.iloc[:60].isna().all()
    assert math.isclose(series.iloc[60], 3.0)
    assert math.isclose(series.iloc[70] / series.iloc[60], 0.5, rel_tol=1e-6)


def test_surprise_index_missing_currency_is_nan():
    index = pd.bdate_range("2020-01-01", periods=20)
    events = pd.DataFrame(
        {
            "date": [index[3]],
            "ccy": ["EUR"],
            "impact": 1.0,
            "forecast": 0.0,
            "actual": 1.0,
            "sd": 1.0,
        }
    )
    assert surprise_index(events, "USD", index).isna().all()
    assert surprise_index(None, "USD", index).isna().all()


def test_macro_z_signs():
    months = pd.date_range("2005-01-01", periods=200, freq="MS")
    rng = np.random.default_rng(0)
    cpi = pd.Series(
        100
        * np.exp(
            np.cumsum(0.002 + 0.0005 * np.arange(200) / 200 + 0.0005 * rng.standard_normal(200))
        ),
        index=months,
    )
    unemp = pd.Series(5 + np.cumsum(0.05 * rng.standard_normal(200)), index=months)
    index = pd.bdate_range("2012-01-02", "2021-01-01")
    z = macro_currency_z({"CPI": cpi, "UNEMP": unemp}, index)
    assert set(z.columns) == {"CPI", "UNEMP"}
    assert z["CPI"].dropna().abs().max() <= 4.0


def test_log_returns_and_vol():
    close = pd.Series(
        100 * np.exp(np.cumsum(np.full(300, 0.01))), index=pd.bdate_range("2020-01-01", periods=300)
    )
    returns = volatility.log_returns(close)
    assert np.allclose(returns, 0.01)
    assert volatility.realised_vol(returns) < 1e-9


def test_expected_daily_move_prefers_implied_vol():
    rng = np.random.default_rng(0)
    close = pd.Series(
        1.1 * np.exp(np.cumsum(0.004 * rng.standard_normal(500))),
        index=pd.bdate_range("2020-01-01", periods=500),
    )
    implied = volatility.expected_daily_move(close, 0.0001, implied_vol=0.10)
    model = volatility.expected_daily_move(close, 0.0001)
    assert implied["source"] == "implied" and model["source"] == "ewma"
    assert math.isclose(implied["sigma_pct"], 10.0 / math.sqrt(252), rel_tol=1e-9)
    assert math.isclose(implied["range_95_pips"], 1.96 * implied["sigma_pips"])
