from fxmacro import run
from fxmacro.pipeline import currency_table, pair_table
from fxmacro.synthetic import generate

market = generate(seed=7)
result = run(market, instruments=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD"])

print(result["fair_value"][["fair_value", "gap_pct", "z", "r2", "reliable"]].round(3))
print(currency_table(result["currency_scores"]).round(2))
print(pair_table(result["pair_scores"]).round(2))
