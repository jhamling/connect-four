from __future__ import annotations

import argparse
from pathlib import Path

from ..io.load_results import LoadSpec, load_latest_from_dir, load_results
from ..metrics.summarize import SummaryConfig, numeric_summary, top_correlations, top_table
from ..plots.chart import plot_histograms, plot_scatter, plot_top_bar, plot_trend_over_rank


DEFAULT_NUMERIC_PLOTS = [
    "ppg",
    "strength_wilson_lcb",
    "efficiency_score",
    "avg_ms_per_move",
    "wins",
    "points",
    "games",
    "avg_depth",
    "nodes",
]


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Analyze Connect-4 league CSV results.")
    ap.add_argument("--csv", type=str, default=None, help="Path to a results CSV. If omitted, uses latest in --results-dir.")
    ap.add_argument("--results-dir", type=str, default="data/results", help="Directory containing league_results_*.csv")
    ap.add_argument("--pattern", type=str, default="league_results_*.csv", help="Glob pattern for selecting latest file")

    ap.add_argument("--outdir", type=str, default="figures", help="Directory for saving plots")
    ap.add_argument("--show", action="store_true", help="Show plots instead of saving")

    ap.add_argument("--top", type=int, default=20, help="Top N for tables/bar charts")
    ap.add_argument("--metric", type=str, default="efficiency_score", help="Ranking metric (e.g. efficiency_score, strength_wilson_lcb, ppg)")
    ap.add_argument("--min-games", type=int, default=0, help="Filter out agents with fewer than this many games")
    ap.add_argument("--max-ms", type=float, default=None, help="Filter out agents slower than this avg_ms_per_move")

    ap.add_argument("--no-hists", action="store_true", help="Disable histogram generation")
    ap.add_argument("--no-scatter", action="store_true", help="Disable scatter plot generation")
    ap.add_argument("--no-bar", action="store_true", help="Disable top-N bar chart generation")
    ap.add_argument("--no-trend", action="store_true", help="Disable trend plot generation")

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)

    # Choose CSV
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = load_latest_from_dir(Path(args.results_dir), pattern=args.pattern)

    df = load_results(LoadSpec(csv_path=csv_path))

    print(f"\nLoaded: {csv_path}")
    print(f"Rows: {len(df):,}  Cols: {len(df.columns)}")
    print("Columns:", ", ".join(df.columns))

    cfg = SummaryConfig(
        metric=args.metric,  # type: ignore[arg-type]
        top_n=args.top,
        min_games=args.min_games,
        max_avg_ms_per_move=args.max_ms,
    )

    # Table output
    table = top_table(df, cfg)
    print("\n=== Top table ===")
    print(table.to_string(index=False))

    # Numeric summary
    desc = numeric_summary(df)
    if not desc.empty:
        print("\n=== Numeric summary ===")
        print(desc.to_string())

    # Correlations
    corrs = top_correlations(df, top_k=15)
    if not corrs.empty:
        print("\n=== Top correlations (abs) ===")
        print(corrs[["a", "b", "corr"]].to_string(index=False))

    # Plots
    filtered = df.copy()
    if cfg.min_games > 0 and "games" in filtered.columns:
        filtered = filtered[filtered["games"].fillna(0) >= cfg.min_games].copy()
    if cfg.max_avg_ms_per_move is not None and "avg_ms_per_move" in filtered.columns:
        filtered = filtered[filtered["avg_ms_per_move"].fillna(float("inf")) <= cfg.max_avg_ms_per_move].copy()

    if not args.no_hists:
        plot_histograms(filtered, outdir, DEFAULT_NUMERIC_PLOTS, show=args.show)

    if not args.no_scatter:
        # Classic: quality vs speed
        if "avg_ms_per_move" in filtered.columns and args.metric in filtered.columns:
            plot_scatter(filtered, outdir, x="avg_ms_per_move", y=args.metric, show=args.show)

    if not args.no_bar:
        plot_top_bar(filtered, outdir, metric=args.metric, top_n=args.top, show=args.show)

    if not args.no_trend:
        plot_trend_over_rank(filtered, outdir, metric=args.metric, show=args.show)

    if not args.show:
        print(f"\nSaved figures to: {outdir.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
