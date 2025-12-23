from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_EXPECTED_COLS = [
    "name",
    "games", "wins", "draws", "losses",
    "points", "ppg",
    "strength_wilson_lcb",
    "avg_ms_per_move",
    "efficiency_score",
    "moves", "time_ms", "nodes", "avg_depth",
]


@dataclass(frozen=True)
class LoadSpec:
    csv_path: Path
    expected_cols: tuple[str, ...] = tuple(DEFAULT_EXPECTED_COLS)


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def load_results(spec: LoadSpec) -> pd.DataFrame:
    if not spec.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {spec.csv_path}")

    df = pd.read_csv(spec.csv_path)

    # Trim whitespace in column names just in case
    df.columns = [c.strip() for c in df.columns]

    # Ensure "name" exists
    if "name" not in df.columns:
        raise ValueError(f"CSV missing required column 'name'. Columns: {list(df.columns)}")

    # Coerce known numeric columns if present
    numeric_like = [
        "games", "wins", "draws", "losses",
        "points", "ppg",
        "strength_wilson_lcb",
        "avg_ms_per_move",
        "efficiency_score",
        "moves", "time_ms", "nodes", "avg_depth",
    ]
    df = _coerce_numeric(df, numeric_like)

    # Drop rows with no name
    df["name"] = df["name"].astype(str)
    df = df[df["name"].str.len() > 0].copy()

    return df


def load_latest_from_dir(results_dir: Path, pattern: str = "league_results_*.csv") -> Path:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    files = sorted(results_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {results_dir}")

    # Filenames include timestamp, lexicographic sort works
    return files[-1]
