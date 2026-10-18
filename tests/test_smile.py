import math

import pandas as pd
import pytest
from scipy.optimize import brentq

from fxmacro.options import (
    choose_expiry,
    cross_risk_reversal,
    leg_risk_reversal,
    skew_signal,
    smile_for_pair,
)
from fxmacro.smile import Smile, black_price, build_smile, call_delta, implied_forward, implied_vol
from fxmacro.synthetic import make_chain

ASOF = pd.Timestamp("2025-06-02")


def truth(atm, skew, curvature, days=30, rate=0.0, spot=1.1):
    years = days / 365.0
    forward = spot * math.exp(rate * years)

    def vol_at(x):
        return atm + skew * x + curvature * x**2

    def delta_gap(x, target):
        return call_delta(forward, forward * math.exp(x), years, vol_at(x)) - target

    x25 = brentq(delta_gap, -0.3, 0.3, args=(0.25,))
    x75 = brentq(delta_gap, -0.3, 0.3, args=(0.75,))
    return vol_at(x25) - vol_at(x75)


def test_put_call_parity():
    call = black_price(1.1, 1.05, 0.5, 0.02, 0.1, True)
    put = black_price(1.1, 1.05, 0.5, 0.02, 0.1, False)
    assert math.isclose(call - put, math.exp(-0.01) * (1.1 - 1.05), abs_tol=1e-12)


def test_implied_vol_round_trip():
    price = black_price(1.1, 1.12, 0.25, 0.01, 0.0825, True)
    assert math.isclose(implied_vol(price, 1.1, 1.12, 0.25, 0.01, True), 0.0825, abs_tol=1e-8)


def test_implied_vol_rejects_arbitrage_prices():
    assert implied_vol(-1.0, 1.1, 1.1, 0.25, 0.0, True) is None
    assert implied_vol(5.0, 1.1, 1.1, 0.25, 0.0, True) is None


def test_implied_forward_recovered_from_parity():
    chain = make_chain(1.1, ASOF, 60, 0.08, 0.0, 0.0, rate=0.03)
    assert math.isclose(
        implied_forward(chain, 1.1, 60 / 365, 0.03), 1.1 * math.exp(0.03 * 60 / 365), rel_tol=1e-6
    )


@pytest.mark.parametrize("skew,curvature", [(-0.1, 0.6), (0.15, 0.8), (0.0, 0.4)])
def test_build_smile_recovers_risk_reversal(skew, curvature):
    chain = make_chain(1.1, ASOF, 30, 0.08, skew, curvature)
    smile = build_smile(chain[["strike", "call", "put"]], 1.1, 30 / 365, 0.0)
    assert abs(smile.rr25 - truth(0.08, skew, curvature)) < 0.0015
    assert abs(smile.atm_vol - 0.08) < 0.002


def test_build_smile_requires_wing_coverage():
    chain = make_chain(1.1, ASOF, 30, 0.08, 0.0, 0.5)
    narrow = chain.iloc[14:17]
    assert build_smile(narrow[["strike", "call", "put"]], 1.1, 30 / 365, 0.0, min_points=3) is None


def test_inverted_underlying_flips_risk_reversal_sign():
    chain = make_chain(1 / 120.0, ASOF, 30, 0.09, 0.15, 0.8)
    direct = build_smile(chain[["strike", "call", "put"]], 1 / 120.0, 30 / 365, 0.0)
    flipped = smile_for_pair("USDJPY", chain, 120.0, ASOF)
    assert math.isclose(flipped.rr25, -direct.rr25, abs_tol=1e-12)
    assert math.isclose(flipped.atm_vol, direct.atm_vol, abs_tol=1e-12)


def test_choose_expiry_picks_nearest_valid():
    chain = pd.DataFrame({"expiry": [ASOF + pd.Timedelta(days=d) for d in (3, 28, 90)]})
    assert choose_expiry(chain, ASOF) == ASOF + pd.Timedelta(days=28)


def test_cross_risk_reversal_from_usd_legs():
    smiles = {
        "EURUSD": Smile(1.1, 0.08, -0.01, 0.002, 31),
        "USDJPY": Smile(120.0, 0.09, -0.02, 0.002, 31),
    }
    assert math.isclose(cross_risk_reversal(smiles, "EURUSD"), -0.01)
    assert math.isclose(cross_risk_reversal(smiles, "EURJPY"), -0.01 + (-0.02))
    assert math.isclose(leg_risk_reversal(smiles, "JPY"), 0.02)
    assert math.isclose(leg_risk_reversal(smiles, "EUR"), -0.01)
    assert leg_risk_reversal(smiles, "USD") == 0.0
    assert cross_risk_reversal(smiles, "GBPJPY") is None


def test_skew_signal_sign_and_bounds():
    assert skew_signal(0.01, 0.08) > 0 > skew_signal(-0.01, 0.08)
    assert abs(skew_signal(5.0, 0.08)) <= 1.0
    assert math.isnan(skew_signal(None, 0.08))
