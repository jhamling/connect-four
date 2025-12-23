from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


MetricKey = Literal[
    "ppg",
    "strength_wilson_lcb",
    "efficiency_score",
    "avg_ms_per_move",
    "wins",
    "points",
    "games",
]


@dataclass(frozen=True)
class SummaryConfig:
    metric: MetricKey = "efficiency_score"
    top_n: int = 20
    min_games: int = 0
    # If True, filter out agents that are too slow (ms per move above threshold)
    max_avg_ms_per_move: float | None = None


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Present: {list(df.columns)}")


def filter_rows(df: pd.DataFrame, cfg: SummaryConfig) -> pd.DataFrame:
    out = df.copy()

    if cfg.min_games > 0:
        _require_cols(out, ["games"])
        out = out[out["games"].fillna(0) >= cfg.min_games].copy()

    if cfg.max_avg_ms_per_move is not None:
        _require_cols(out, ["avg_ms_per_move"])
        out = out[out["avg_ms_per_move"].fillna(float("inf")) <= cfg.max_avg_ms_per_move].copy()

    return out


def top_table(df: pd.DataFrame, cfg: SummaryConfig) -> pd.DataFrame:
    _require_cols(df, ["name", cfg.metric])

    out = filter_rows(df, cfg)

    # Sort direction: lower is better only for avg_ms_per_move
    ascending = (cfg.metric == "avg_ms_per_move")
    out = out.sort_values(cfg.metric, ascending=ascending)

    cols = [
        "name",
        "games", "wins", "draws", "losses",
        "ppg",
        "strength_wilson_lcb",
        "avg_ms_per_move",
        "efficiency_score",
        "points",
        "moves", "time_ms", "nodes", "avg_depth",
    ]
    keep = [c for c in cols if c in out.columns]

    out = out[keep].head(cfg.top_n).reset_index(drop=True)
    out.insert(0, "rk", range(1, len(out) + 1))
    return out


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number")
    if num.empty:
        return pd.DataFrame()
    desc = num.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    return desc




def top_correlations(df: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return pd.DataFrame(columns=["a", "b", "corr", "abs"])

    corr = num.corr(numeric_only=True)
    pairs = (
        corr.stack()
        .reset_index()
        .rename(columns={"level_0": "a", "level_1": "b", 0: "corr"})
    )

    # remove self-correlations and duplicates
    pairs = pairs[pairs["a"] < pairs["b"]].copy()
    pairs["abs"] = pairs["corr"].abs()
    pairs = pairs.sort_values("abs", ascending=False).head(top_k)

    return pairs.reset_index(drop=True)

