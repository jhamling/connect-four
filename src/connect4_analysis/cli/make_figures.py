
# src/connect4_analysis/cli/make_figures.py
from __future__ import annotations

import argparse
from pathlib import Path

from ..io.load_results import LoadSpec, load_latest_from_dir, load_results
from ..plots.portfolio import PortfolioPlotConfig, make_portfolio_figures


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="connect4_analysis figures",
        description="Generate portfolio-ready figures/tables from league_results_*.csv",
    )

    # Input selection
    ap.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to a specific results CSV. If omitted, uses latest CSV in --results-dir matching --pattern.",
    )
    ap.add_argument(
        "--results-dir",
        type=str,
        default="data/results",
        help="Directory to search for latest results CSV when --csv is not provided.",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="league_results_*.csv",
        help="Glob pattern to find results CSVs in --results-dir.",
    )

    # Output location
    ap.add_argument(
        "--figures-dir",
        type=str,
        default="data/figures",
        help="Base output directory (subfolders overview/tradeoffs/diagnostics/family/tables will be created).",
    )

    # Filtering
    ap.add_argument(
        "--min-games",
        type=int,
        default=0,
        help="Filter out agents with fewer than this many games (0 disables).",
    )

    # Plot knobs
    ap.add_argument(
        "--top",
        type=int,
        default=20,
        help="Top-N for any leaderboard-style plots.",
    )
    ap.add_argument(
        "--label-best-per-family",
        type=int,
        default=1,
        help="Label best-per-family points (1 labels each family's best; 0 disables all labels).",
    )
    ap.add_argument(
        "--no-legend",
        action="store_true",
        help="Disable legends for a cleaner look.",
    )
    ap.add_argument(
        "--min-games-line",
        type=int,
        default=50,
        help="Vertical reference line on the games-vs-PPG plot. Use -1 to disable.",
    )

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)

    # Resolve CSV path
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = load_latest_from_dir(Path(args.results_dir), pattern=args.pattern)

    df = load_results(LoadSpec(csv_path=csv_path))

    cfg = PortfolioPlotConfig(
        top_n=args.top,
        min_games=args.min_games,
        label_best_per_family=args.label_best_per_family,
        show_legend=not args.no_legend,
        min_games_line=(None if args.min_games_line < 0 else args.min_games_line),
    )

    figures_dir = Path(args.figures_dir)
    created = make_portfolio_figures(df, figures_dir, cfg)

    print(f"Loaded: {csv_path}")
    print(f"Wrote {len(created)} outputs under: {figures_dir.resolve()}")
    for k, p in created.items():
        print(f"- {k}: {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

