import pytest

from fxmacro.config import INSTRUMENTS
from fxmacro.fair_value import estimate_all
from fxmacro.synthetic import generate

SUBSET = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "EURJPY", "AUDJPY"]


@pytest.fixture(scope="session")
def market():
    return generate(years=12.0, seed=11)


@pytest.fixture(scope="session")
def estimates(market):
    return estimate_all([INSTRUMENTS[name] for name in SUBSET], market)
