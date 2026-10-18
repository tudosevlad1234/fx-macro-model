from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class MarketData:
    prices: dict[str, pd.Series]
    yields: dict[str, dict[str, pd.Series]]
    macro: dict[str, dict[str, pd.Series]]
    market: pd.DataFrame
    events: pd.DataFrame | None = None
    options: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def index(self) -> pd.DatetimeIndex:
        union = None
        for series in self.prices.values():
            union = series.index if union is None else union.union(series.index)
        return union if union is not None else pd.DatetimeIndex([])

    def subset(self, names: list[str]) -> MarketData:
        return MarketData(
            prices={name: self.prices[name] for name in names if name in self.prices},
            yields=self.yields,
            macro=self.macro,
            market=self.market,
            events=self.events,
            options=self.options,
        )
