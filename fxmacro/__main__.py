from __future__ import annotations

import argparse
import sys

from . import data_io, report
from .pipeline import run
from .synthetic import generate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fxmacro")
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="run the model on synthetic data")
    demo.add_argument("--out", default="output")
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--no-stat-arb", action="store_true")

    template = commands.add_parser(
        "export-template", help="write synthetic CSVs showing the input format"
    )
    template.add_argument("--out", default="data")
    template.add_argument("--seed", type=int, default=7)

    live = commands.add_parser("run", help="run the model on your own CSV inputs")
    live.add_argument("--data", required=True)
    live.add_argument("--out", default="output")
    live.add_argument("--no-stat-arb", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "export-template":
        data_io.save(generate(seed=args.seed), args.out)
        print(f"wrote input template to {args.out}")
        return 0
    try:
        market = generate(seed=args.seed) if args.command == "demo" else data_io.load(args.data)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 2
    result = run(market, include_stat_arb=not args.no_stat_arb)
    path = report.write(result, args.out)
    print(f"report written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
