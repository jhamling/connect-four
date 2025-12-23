from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_histograms(df: pd.DataFrame, outdir: Path, cols: Iterable[str], *, show: bool) -> None:
    num_cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return

    if not show:
        _ensure_dir(outdir)

    for c in num_cols:
        fig = plt.figure()
        plt.hist(df[c].dropna(), bins=30)
        plt.title(f"Histogram: {c}")
        plt.xlabel(c)
        plt.ylabel("count")

        if show:
            plt.show()
        else:
            fig.savefig(outdir / f"hist_{c}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)


def plot_scatter(df: pd.DataFrame, outdir: Path, x: str, y: str, *, show: bool) -> None:
    if x not in df.columns or y not in df.columns:
        return
    if not (pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y])):
        return

    if not show:
        _ensure_dir(outdir)

    fig = plt.figure()
    plt.scatter(df[x], df[y], alpha=0.6)
    plt.title(f"{y} vs {x}")
    plt.xlabel(x)
    plt.ylabel(y)

    if show:
        plt.show()
    else:
        fig.savefig(outdir / f"scatter_{y}_vs_{x}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_top_bar(df: pd.DataFrame, outdir: Path, metric: str, top_n: int, *, show: bool) -> None:
    if "name" not in df.columns or metric not in df.columns:
        return
    if not pd.api.types.is_numeric_dtype(df[metric]):
        return

    if not show:
        _ensure_dir(outdir)

    top = df[["name", metric]].dropna().sort_values(metric, ascending=False).head(top_n)
    fig = plt.figure(figsize=(10, 5))
    plt.bar(top["name"].astype(str), top[metric].astype(float))
    plt.title(f"Top {min(top_n, len(top))}: {metric}")
    plt.xlabel("agent")
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha="right")

    if show:
        plt.show()
    else:
        fig.savefig(outdir / f"top_{top_n}_{metric}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_trend_over_rank(df: pd.DataFrame, outdir: Path, metric: str, *, show: bool) -> None:
    """
    You don't have 'stage' in your exported CSV. This draws a simple trend of metric vs rank.
    """
    if metric not in df.columns or "name" not in df.columns:
        return
    if not pd.api.types.is_numeric_dtype(df[metric]):
        return

    if not show:
        _ensure_dir(outdir)

    ranked = df[["name", metric]].dropna().sort_values(metric, ascending=False).reset_index(drop=True)
    ranked["rk"] = ranked.index + 1

    fig = plt.figure()
    plt.plot(ranked["rk"], ranked[metric])
    plt.title(f"{metric} vs rank")
    plt.xlabel("rank (1 = best)")
    plt.ylabel(metric)

    if show:
        plt.show()
    else:
        fig.savefig(outdir / f"trend_{metric}_over_rank.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
