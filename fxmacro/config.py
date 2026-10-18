from __future__ import annotations

from dataclasses import dataclass

MAJORS = ("USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD")
SHORT_TENORS = ("Y2", "Y5", "Y1", "ON")
LONG_TENORS = ("Y10", "Y5")
PUBLICATION_LAG_DAYS = {"CPI": 20, "UNEMP": 35, "TRADE": 45}

LEG_DRIVERS = {
    "USD": (),
    "EUR": (),
    "GBP": (),
    "JPY": ("SPX",),
    "CHF": ("SPX",),
    "AUD": ("COPPER",),
    "NZD": ("COPPER",),
    "CAD": ("WTI",),
}


@dataclass(frozen=True)
class Instrument:
    name: str
    base: str
    quote: str
    pip: float
    cost_pips: float
    drivers: tuple[str, ...]
    metal: bool = False

    @property
    def is_usd_pair(self) -> bool:
        return "USD" in (self.base, self.quote)


def _drivers_for(base: str, quote: str) -> tuple[str, ...]:
    ordered = ["VIX"]
    for currency in (base, quote):
        for driver in LEG_DRIVERS.get(currency, ()):
            if driver not in ordered:
                ordered.append(driver)
    return tuple(ordered)


def _currency_pair(name: str, cost_pips: float) -> Instrument:
    base, quote = name[:3], name[3:]
    pip = 0.01 if "JPY" in (base, quote) else 0.0001
    return Instrument(name, base, quote, pip, cost_pips, _drivers_for(base, quote))


_PAIR_COSTS = {
    "EURUSD": 0.6,
    "GBPUSD": 0.9,
    "USDJPY": 0.7,
    "USDCHF": 1.1,
    "AUDUSD": 0.8,
    "NZDUSD": 1.4,
    "USDCAD": 1.2,
    "EURGBP": 1.0,
    "EURJPY": 1.3,
    "GBPJPY": 2.2,
    "EURCHF": 1.4,
    "EURAUD": 2.0,
    "EURCAD": 2.0,
    "EURNZD": 3.0,
    "GBPAUD": 2.6,
    "GBPCAD": 2.8,
    "GBPNZD": 4.0,
    "AUDJPY": 1.5,
    "AUDCAD": 1.8,
    "AUDNZD": 2.2,
    "CADJPY": 1.8,
    "NZDJPY": 2.0,
    "CHFJPY": 2.0,
}

INSTRUMENTS: dict[str, Instrument] = {
    name: _currency_pair(name, cost) for name, cost in _PAIR_COSTS.items()
}
INSTRUMENTS["XAUUSD"] = Instrument("XAUUSD", "XAU", "USD", 0.1, 30.0, ("VIX",), metal=True)
INSTRUMENTS["XAGUSD"] = Instrument("XAGUSD", "XAG", "USD", 0.01, 25.0, ("VIX",), metal=True)

USD_PAIRS = tuple(name for name, item in INSTRUMENTS.items() if item.is_usd_pair and not item.metal)
