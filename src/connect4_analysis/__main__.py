from __future__ import annotations

import sys

from .cli.analyze_csv import main as analyze_main
from .cli.make_figures import main as figures_main


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Default behavior: run analysis if no subcommand
    if not argv:
        return analyze_main([])

    # Minimal subcommand router
    cmd = argv[0].lower()
    rest = argv[1:]

    if cmd in {"analyze", "analysis"}:
        return analyze_main(rest)

    if cmd in {"figures", "plots", "make-figures", "make_figures"}:
        return figures_main(rest)

    # Backward compatible: if user passes flags, treat as analyze
    if cmd.startswith("-"):
        return analyze_main(argv)

    print("Usage:")
    print("  python -m connect4_analysis analyze [--csv ...] [--metric ...]")
    print("  python -m connect4_analysis figures [--csv ...] [--figures-dir figures]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

