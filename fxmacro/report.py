from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .pipeline import currency_table, pair_table
from .plots import fair_value_chart, score_chart


def _frame(table: pd.DataFrame, digits: int = 2) -> str:
    return table.round(digits).to_markdown()


def render(result: dict) -> str:
    lines = [f"# FX macro model report - {pd.Timestamp(result['asof']).date()}", ""]
    lines += ["## Currency scores (each currency against the basket of the other majors)", ""]
    lines += ["![currency scores](currency_scores.png)", ""]
    lines += [_frame(currency_table(result["currency_scores"])), ""]
    lines += ["## Fair value", "", _frame(result["fair_value"]), ""]
    lines += ["## Pair scores", "", _frame(pair_table(result["pair_scores"])), ""]
    regimes = pd.DataFrame(
        {
            name: {
                "state": item["state"],
                "probability": item["probability"],
                "vol_ratio": item["vol_ratio"],
                "calm_days": item["expected_duration"][0],
                "turbulent_days": item["expected_duration"][1],
            }
            for name, item in result["regimes"].items()
        }
    ).T
    lines += ["## Volatility regimes", "", _frame(regimes), ""]
    if result["smiles"]:
        smiles = pd.DataFrame(
            {
                n: {"atm_vol": s.atm_vol, "rr25": s.rr25, "bf25": s.bf25}
                for n, s in result["smiles"].items()
            }
        ).T
        lines += ["## Options smiles", "", _frame(smiles, 4), ""]
    arb = result.get("stat_arb")
    if arb is not None:
        lines += ["## Statistical arbitrage", ""]
        if arb["accepted"]:
            rows = []
            for candidate, backtest, signal in zip(
                arb["accepted"], arb["backtests"], arb["signals"], strict=False
            ):
                rows.append(
                    {
                        "pair": signal["pair"],
                        "beta": candidate["beta"],
                        "half_life": candidate["half_life"],
                        "null_p": candidate.get("null_p_value"),
                        "z_now": signal["z"],
                        "action": signal["action"],
                        "oos_trades": backtest["trades"],
                        "oos_return_pct": backtest["total_return_pct"],
                        "oos_sharpe": backtest["sharpe"],
                    }
                )
            lines += [_frame(pd.DataFrame(rows).set_index("pair")), ""]
        else:
            lines += ["No pair passed every screen.", ""]
        lines += [f"{arb['rejected']} candidate pairs were rejected.", ""]
    return "\n".join(lines)


def write(result: dict, directory: str | Path) -> Path:
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    currency_table(result["currency_scores"]).to_csv(folder / "currency_scores.csv")
    pair_table(result["pair_scores"]).to_csv(folder / "pair_scores.csv")
    result["fair_value"].to_csv(folder / "fair_value.csv")
    summary = {
        "asof": str(pd.Timestamp(result["asof"]).date()),
        "regimes": result["regimes"],
        "expected_move": result["expected_move"],
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    score_chart(
        currency_table(result["currency_scores"]), folder / "currency_scores.png", "Currency scores"
    )
    charts = folder / "charts"
    charts.mkdir(exist_ok=True)
    for name, estimate in result["estimates"].items():
        fair_value_chart(estimate, charts / f"{name}.png")
    report = folder / "report.md"
    report.write_text(render(result))
    return report
