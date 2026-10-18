from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import MarketData

FILES = ("prices.csv", "yields.csv", "macro.csv", "market.csv")


def _read(path: Path, **options) -> pd.DataFrame:
    return pd.read_csv(path, **options)


def save(data: MarketData, directory: str | Path) -> None:
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data.prices).rename_axis("date").to_csv(folder / "prices.csv")
    yield_rows = [
        {"date": date, "ccy": ccy, "tenor": tenor, "value": value}
        for ccy, tenors in data.yields.items()
        for tenor, series in tenors.items()
        for date, value in series.dropna().items()
    ]
    pd.DataFrame(yield_rows).to_csv(folder / "yields.csv", index=False)
    macro_rows = [
        {"date": date, "ccy": ccy, "field": field, "value": value}
        for ccy, fields in data.macro.items()
        for field, series in fields.items()
        for date, value in series.dropna().items()
    ]
    pd.DataFrame(macro_rows).to_csv(folder / "macro.csv", index=False)
    data.market.rename_axis("date").to_csv(folder / "market.csv")
    if data.events is not None:
        data.events.to_csv(folder / "events.csv", index=False)
    if data.options:
        frames = [chain.assign(pair=name) for name, chain in data.options.items()]
        pd.concat(frames).to_csv(folder / "options.csv", index=False)


def load(directory: str | Path) -> MarketData:
    folder = Path(directory)
    missing = [name for name in FILES if not (folder / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing input files in {folder}: {', '.join(missing)}")
    prices = _read(folder / "prices.csv", index_col="date", parse_dates=True)
    yields: dict[str, dict[str, pd.Series]] = {}
    long_yields = _read(folder / "yields.csv", parse_dates=["date"])
    for (ccy, tenor), group in long_yields.groupby(["ccy", "tenor"]):
        yields.setdefault(ccy, {})[tenor] = group.set_index("date")["value"].sort_index()
    macro: dict[str, dict[str, pd.Series]] = {}
    long_macro = _read(folder / "macro.csv", parse_dates=["date"])
    for (ccy, field), group in long_macro.groupby(["ccy", "field"]):
        macro.setdefault(ccy, {})[field] = group.set_index("date")["value"].sort_index()
    market = _read(folder / "market.csv", index_col="date", parse_dates=True)
    events = None
    if (folder / "events.csv").exists():
        events = _read(folder / "events.csv", parse_dates=["date"])
    options: dict[str, pd.DataFrame] = {}
    if (folder / "options.csv").exists():
        combined = _read(folder / "options.csv", parse_dates=["expiry"])
        for name, group in combined.groupby("pair"):
            options[name] = group.drop(columns="pair").reset_index(drop=True)
    return MarketData(
        prices={name: prices[name].dropna() for name in prices},
        yields=yields,
        macro=macro,
        market=market,
        events=events,
        options=options,
    )
