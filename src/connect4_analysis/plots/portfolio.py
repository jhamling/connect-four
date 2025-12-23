from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Config + styling
# -----------------------------
@dataclass(frozen=True)
class PortfolioPlotConfig:
    top_n: int = 20
    min_games: int = 0

    # Labeling: ONLY best-per-family labels
    label_best_per_family: int = 1  # 1 label per family; 0 disables all labels
    label_baselines: bool = False   # keep False if you want only best-per-family names

    # Plot aesthetics
    point_size: int = 42
    best_size: int = 120
    background_alpha: float = 0.18
    best_alpha: float = 0.95

    title_size: int = 18
    label_size: int = 12

    # Thresholds / callouts
    min_games_line: int | None = 50  # games vs ppg reference line; set None to disable

    # Legend
    show_legend: bool = True
    legend_cols: int = 2


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _require(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Present: {list(df.columns)}")


def _filter(df: pd.DataFrame, cfg: PortfolioPlotConfig) -> pd.DataFrame:
    out = df.copy()
    if cfg.min_games > 0 and "games" in out.columns:
        out = out[out["games"].fillna(0) >= cfg.min_games].copy()
    return out


def _apply_mpl_style(cfg: PortfolioPlotConfig) -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (9, 6),
            "axes.titlesize": cfg.title_size,
            "axes.labelsize": cfg.label_size,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


# -----------------------------
# Name shortening (expanded)
# -----------------------------
_BEAM_RE = re.compile(r"^Beam\s+d(?P<d>\d+)\s+w(?P<w>\d+)\s+t(?P<t>\d+)ms\s+temp(?P<temp>\d+)$", re.IGNORECASE)
_BEAM2_RE = re.compile(r"^Beam2\s+d(?P<d>\d+)\s+w(?P<w>\d+)\s+t(?P<t>\d+)ms\s+temp(?P<temp>\d+)$", re.IGNORECASE)
_MM_RE = re.compile(r"^(MM|Minimax)\s+d(?P<d>\d+)\s+t(?P<t>\d+)ms\s+temp(?P<temp>\d+)$", re.IGNORECASE)
_EXPMM_RE = re.compile(r"^(ExpectiMinimax|Expectiminimax)\s+d(?P<d>\d+)\s+t(?P<t>\d+)ms\s+temp(?P<temp>\d+)$", re.IGNORECASE)


def _truncate(s: str, n: int = 26) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    return s if len(s) <= n else s[: n - 1] + "…"


def short_name(name: str) -> str:
    s = str(name).strip()

    m = _BEAM2_RE.match(s)
    if m:
        return f"Beam2(d{m.group('d')},w{m.group('w')},{m.group('t')}ms,T{m.group('temp')})"

    m = _BEAM_RE.match(s)
    if m:
        return f"Beam(d{m.group('d')},w{m.group('w')},{m.group('t')}ms,T{m.group('temp')})"

    m = _EXPMM_RE.match(s)
    if m:
        return f"ExpMM(d{m.group('d')},{m.group('t')}ms,T{m.group('temp')})"

    m = _MM_RE.match(s)
    if m:
        return f"MM(d{m.group('d')},{m.group('t')}ms,T{m.group('temp')})"

    if s.lower().startswith("mcts"):
        return "MCTS(" + _truncate(s[4:], 18).strip() + ")"

    if s.lower().startswith("heuristic"):
        return "Heur(" + _truncate(s[len("heuristic"):], 18).strip() + ")"

    if s.lower().startswith("greedy"):
        return "Greedy(" + _truncate(s[len("greedy"):], 18).strip() + ")"

    if s.lower().startswith("tacticalgreedy") or s.lower().startswith("tactical_greedy"):
        temp = re.search(r"temp(\d+)", s, re.IGNORECASE)
        tms = re.search(r"t(\d+)ms", s, re.IGNORECASE)
        inner = ",".join([x for x in [f"T{temp.group(1)}" if temp else "", f"{tms.group(1)}ms" if tms else ""] if x])
        return f"TGreedy({inner})" if inner else "TGreedy"

    if s.lower().startswith("tactical"):
        return "Tact(" + _truncate(s[len("tactical"):], 18).strip() + ")"

    if s.lower().startswith("weightedrandom") or s.lower().startswith("weighted_random"):
        seed = re.search(r"seed(\d+)", s, re.IGNORECASE)
        return f"WRand(s{seed.group(1)})" if seed else "WRand"

    if s.lower() == "random":
        return "Random"

    return _truncate(s, 28)


def add_short_name_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["name_short"] = out["name"].astype(str).map(short_name)
    return out


# -----------------------------
# Family classification (based on your agent modules)
# -----------------------------
_FAMILY_RULES: list[tuple[str, re.Pattern]] = [
    ("Beam2", re.compile(r"^\s*beam2\b", re.IGNORECASE)),
    ("Beam", re.compile(r"^\s*beam\b", re.IGNORECASE)),
    ("Expectiminimax", re.compile(r"^\s*expectiminimax\b", re.IGNORECASE)),
    ("Minimax", re.compile(r"^\s*(mm|minimax)\b", re.IGNORECASE)),
    ("MCTS", re.compile(r"^\s*mcts\b", re.IGNORECASE)),
    ("Tactical Greedy", re.compile(r"^\s*tactical[_\s-]*greedy\b|^\s*tacticalgreedy\b", re.IGNORECASE)),
    ("Tactical", re.compile(r"^\s*tactical\b", re.IGNORECASE)),
    ("Heuristic", re.compile(r"^\s*heuristic\b", re.IGNORECASE)),
    ("Greedy", re.compile(r"^\s*greedy\b", re.IGNORECASE)),
    ("Weighted Random", re.compile(r"^\s*weighted[_\s-]*random\b|^\s*weightedrandom\b", re.IGNORECASE)),
    ("Random", re.compile(r"^\s*random\b", re.IGNORECASE)),
]


def agent_family(name: str) -> str:
    s = str(name).strip()
    for fam, pat in _FAMILY_RULES:
        if pat.search(s):
            return fam
    return "Other"


def add_family_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["family"] = out["name"].astype(str).map(agent_family)
    return out


# -----------------------------
# Group analysis
# -----------------------------
def summarize_by_family(df: pd.DataFrame) -> pd.DataFrame:
    _require(df, ["family", "ppg", "strength_wilson_lcb", "efficiency_score", "avg_ms_per_move", "games"])
    d = df.copy()
    grp = (
        d.groupby("family", dropna=False)
        .agg(
            agents=("name", "count"),
            total_games=("games", "sum"),
            mean_ppg=("ppg", "mean"),
            mean_strength=("strength_wilson_lcb", "mean"),
            mean_efficiency=("efficiency_score", "mean"),
            mean_ms=("avg_ms_per_move", "mean"),
            median_ms=("avg_ms_per_move", "median"),
        )
        .sort_values("mean_efficiency", ascending=False)
        .reset_index()
    )
    return grp


def best_agent_per_family(df: pd.DataFrame, metric: str = "efficiency_score") -> pd.DataFrame:
    _require(df, ["family", "name", "name_short", metric])
    d = df.dropna(subset=[metric]).copy()
    idx = d.groupby("family")[metric].idxmax()
    best = d.loc[idx].copy()
    cols = [
        "family",
        "name",
        "name_short",
        "games",
        "ppg",
        "strength_wilson_lcb",
        "avg_ms_per_move",
        "efficiency_score",
        "avg_depth",
        "nodes",
    ]
    present = [c for c in cols if c in best.columns]
    return best[present].sort_values(metric, ascending=False).reset_index(drop=True)


# -----------------------------
# Plot helpers
# -----------------------------
def _annotate_only_best(ax, best: pd.DataFrame, xcol: str, ycol: str) -> None:
    """Annotate only best-per-family points (very small set)."""
    offsets = [(8, 8), (10, -8), (-10, 10), (-10, -10), (12, 0), (0, 12)]
    for i, (_, r) in enumerate(best.iterrows()):
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(
            str(r["name_short"]),
            (float(r[xcol]), float(r[ycol])),
            fontsize=9,
            xytext=(dx, dy),
            textcoords="offset points",
        )


def _scatter_by_family(ax, d: pd.DataFrame, xcol: str, ycol: str, cfg: PortfolioPlotConfig) -> None:
    """Background scatter grouped by family (no labels)."""
    for fam, sub in d.groupby("family", dropna=False):
        ax.scatter(
            sub[xcol],
            sub[ycol],
            alpha=cfg.background_alpha,
            s=cfg.point_size,
            label=str(fam),
        )


# -----------------------------
# Plots (NO labels except best-per-family)
# -----------------------------
def plot_strength_vs_speed(df: pd.DataFrame, outpath: Path, cfg: PortfolioPlotConfig) -> None:
    _apply_mpl_style(cfg)
    _require(df, ["name", "avg_ms_per_move", "strength_wilson_lcb", "efficiency_score"])

    d = add_family_column(add_short_name_column(_filter(df, cfg))).dropna(
        subset=["avg_ms_per_move", "strength_wilson_lcb"]
    ).copy()

    best = best_agent_per_family(d, metric="efficiency_score")

    fig, ax = plt.subplots()

    _scatter_by_family(ax, d, "avg_ms_per_move", "strength_wilson_lcb", cfg)

    # highlight best points
    ax.scatter(
        best["avg_ms_per_move"],
        best["strength_wilson_lcb"],
        alpha=cfg.best_alpha,
        s=cfg.best_size,
        zorder=3,
    )

    ax.set_title("Playing Strength vs Move-Time Cost (Family View)")
    ax.set_xlabel("avg_ms_per_move (lower is faster)")
    ax.set_ylabel("strength_wilson_lcb (higher is stronger)")

    if cfg.label_best_per_family > 0:
        _annotate_only_best(ax, best, "avg_ms_per_move", "strength_wilson_lcb")

    if cfg.show_legend:
        ax.legend(ncol=cfg.legend_cols, fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_strength_vs_efficiency(df: pd.DataFrame, outpath: Path, cfg: PortfolioPlotConfig) -> None:
    _apply_mpl_style(cfg)
    _require(df, ["strength_wilson_lcb", "efficiency_score", "name"])

    d = add_family_column(add_short_name_column(_filter(df, cfg))).dropna(
        subset=["strength_wilson_lcb", "efficiency_score"]
    ).copy()
    best = best_agent_per_family(d, metric="efficiency_score")

    fig, ax = plt.subplots()
    _scatter_by_family(ax, d, "strength_wilson_lcb", "efficiency_score", cfg)

    ax.scatter(
        best["strength_wilson_lcb"],
        best["efficiency_score"],
        alpha=cfg.best_alpha,
        s=cfg.best_size,
        zorder=3,
    )

    ax.set_title("Strength vs Efficiency (Family View)")
    ax.set_xlabel("strength_wilson_lcb (higher is stronger)")
    ax.set_ylabel("efficiency_score (higher is better)")

    if cfg.label_best_per_family > 0:
        _annotate_only_best(ax, best, "strength_wilson_lcb", "efficiency_score")

    if cfg.show_legend:
        ax.legend(ncol=cfg.legend_cols, fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_games_vs_ppg(df: pd.DataFrame, outpath: Path, cfg: PortfolioPlotConfig) -> None:
    _apply_mpl_style(cfg)
    _require(df, ["games", "ppg", "name"])

    d = add_family_column(add_short_name_column(_filter(df, cfg))).dropna(subset=["games", "ppg"]).copy()
    best = best_agent_per_family(d, metric="efficiency_score")
    best = best.dropna(subset=["games", "ppg"])

    fig, ax = plt.subplots()
    _scatter_by_family(ax, d, "games", "ppg", cfg)

    ax.scatter(best["games"], best["ppg"], alpha=cfg.best_alpha, s=cfg.best_size, zorder=3)

    ax.set_title("Win Quality (PPG) vs Sample Size (Family View)")
    ax.set_xlabel("games")
    ax.set_ylabel("ppg (points per game)")

    if cfg.min_games_line is not None:
        ax.axvline(cfg.min_games_line, linestyle="--", alpha=0.5)

    if cfg.label_best_per_family > 0:
        _annotate_only_best(ax, best, "games", "ppg")

    if cfg.show_legend:
        ax.legend(ncol=cfg.legend_cols, fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_depth_vs_strength(df: pd.DataFrame, outpath: Path, cfg: PortfolioPlotConfig) -> None:
    _apply_mpl_style(cfg)
    _require(df, ["avg_depth", "strength_wilson_lcb", "name"])

    d = add_family_column(add_short_name_column(_filter(df, cfg))).dropna(subset=["avg_depth", "strength_wilson_lcb"]).copy()
    best = best_agent_per_family(d, metric="efficiency_score").dropna(subset=["avg_depth", "strength_wilson_lcb"])

    fig, ax = plt.subplots()
    _scatter_by_family(ax, d, "avg_depth", "strength_wilson_lcb", cfg)

    ax.scatter(best["avg_depth"], best["strength_wilson_lcb"], alpha=cfg.best_alpha, s=cfg.best_size, zorder=3)

    ax.set_title("Average Search Depth vs Strength (Family View)")
    ax.set_xlabel("avg_depth")
    ax.set_ylabel("strength_wilson_lcb")

    if cfg.label_best_per_family > 0:
        _annotate_only_best(ax, best, "avg_depth", "strength_wilson_lcb")

    if cfg.show_legend:
        ax.legend(ncol=cfg.legend_cols, fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_nodes_vs_ms(df: pd.DataFrame, outpath: Path, cfg: PortfolioPlotConfig) -> None:
    _apply_mpl_style(cfg)
    _require(df, ["nodes", "avg_ms_per_move", "name"])

    d = add_family_column(add_short_name_column(_filter(df, cfg))).dropna(subset=["nodes", "avg_ms_per_move"]).copy()
    best = best_agent_per_family(d, metric="efficiency_score").dropna(subset=["nodes", "avg_ms_per_move"])

    fig, ax = plt.subplots()
    _scatter_by_family(ax, d, "nodes", "avg_ms_per_move", cfg)

    ax.set_xscale("log")
    ax.scatter(best["nodes"], best["avg_ms_per_move"], alpha=cfg.best_alpha, s=cfg.best_size, zorder=3)

    ax.set_title("Search Volume vs Move-Time Cost (Family View)")
    ax.set_xlabel("nodes evaluated (log scale)")
    ax.set_ylabel("avg_ms_per_move")

    if cfg.label_best_per_family > 0:
        _annotate_only_best(ax, best, "nodes", "avg_ms_per_move")

    if cfg.show_legend:
        ax.legend(ncol=cfg.legend_cols, fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# NEW: Best-per-family graphs (what you asked for)
# -----------------------------
def plot_best_per_family_efficiency(df: pd.DataFrame, outpath: Path, cfg: PortfolioPlotConfig) -> None:
    """
    Bar chart: best efficiency score within each family.
    This is the most portfolio-friendly "who won each family" view.
    """
    _apply_mpl_style(cfg)
    _require(df, ["name", "family", "efficiency_score", "name_short"])

    d = add_family_column(add_short_name_column(_filter(df, cfg))).dropna(subset=["efficiency_score"]).copy()
    best = best_agent_per_family(d, metric="efficiency_score")

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Sort so best overall at top
    best_sorted = best.sort_values("efficiency_score", ascending=True)

    labels = [f"{fam}: {nm}" for fam, nm in zip(best_sorted["family"], best_sorted["name_short"])]
    ax.barh(labels, best_sorted["efficiency_score"].astype(float))

    ax.set_title("Best Agent in Each Family (by Efficiency)")
    ax.set_xlabel("efficiency_score")
    ax.set_ylabel("family → best agent")

    for i, v in enumerate(best_sorted["efficiency_score"].astype(float).tolist()):
        ax.text(v, i, f"  {v:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_best_per_family_pareto(df: pd.DataFrame, outpath: Path, cfg: PortfolioPlotConfig) -> None:
    """
    Scatter: only best-per-family points on the strength vs speed plane.
    """
    _apply_mpl_style(cfg)
    _require(df, ["avg_ms_per_move", "strength_wilson_lcb", "family", "name_short", "efficiency_score"])

    d = add_family_column(add_short_name_column(_filter(df, cfg))).dropna(
        subset=["avg_ms_per_move", "strength_wilson_lcb", "efficiency_score"]
    ).copy()
    best = best_agent_per_family(d, metric="efficiency_score")

    fig, ax = plt.subplots()
    ax.scatter(best["avg_ms_per_move"], best["strength_wilson_lcb"], alpha=cfg.best_alpha, s=cfg.best_size)

    ax.set_title("Best per Family: Strength vs Speed")
    ax.set_xlabel("avg_ms_per_move (lower is faster)")
    ax.set_ylabel("strength_wilson_lcb (higher is stronger)")

    if cfg.label_best_per_family > 0:
        _annotate_only_best(ax, best, "avg_ms_per_move", "strength_wilson_lcb")

    fig.tight_layout()
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Orchestration + exports
# -----------------------------
def make_portfolio_figures(df: pd.DataFrame, figures_dir: Path, cfg: PortfolioPlotConfig) -> dict[str, Path]:
    out: dict[str, Path] = {}

    overview = figures_dir / "overview"
    tradeoffs = figures_dir / "tradeoffs"
    diagnostics = figures_dir / "diagnostics"
    tables = figures_dir / "tables"
    family = figures_dir / "family"

    _ensure_dir(overview)
    _ensure_dir(tradeoffs)
    _ensure_dir(diagnostics)
    _ensure_dir(tables)
    _ensure_dir(family)

    base = add_family_column(add_short_name_column(df))

    # Family-view plots (only best-per-family labeled)
    p1 = overview / "strength_vs_speed_family.png"
    plot_strength_vs_speed(base, p1, cfg)
    out["strength_vs_speed_family"] = p1

    p2 = tradeoffs / "strength_vs_efficiency_family.png"
    plot_strength_vs_efficiency(base, p2, cfg)
    out["strength_vs_efficiency_family"] = p2

    p3 = diagnostics / "games_vs_ppg_family.png"
    plot_games_vs_ppg(base, p3, cfg)
    out["games_vs_ppg_family"] = p3

    p4 = tradeoffs / "avg_depth_vs_strength_family.png"
    plot_depth_vs_strength(base, p4, cfg)
    out["avg_depth_vs_strength_family"] = p4

    p5 = diagnostics / "nodes_vs_ms_family.png"
    plot_nodes_vs_ms(base, p5, cfg)
    out["nodes_vs_ms_family"] = p5

    # NEW: best-per-family graphs
    p6 = family / "best_per_family_efficiency.png"
    plot_best_per_family_efficiency(base, p6, cfg)
    out["best_per_family_efficiency"] = p6

    p7 = family / "best_per_family_pareto.png"
    plot_best_per_family_pareto(base, p7, cfg)
    out["best_per_family_pareto"] = p7

    # Exports
    fam_summary = summarize_by_family(_filter(base, cfg))
    fam_path = tables / "family_summary.csv"
    fam_summary.to_csv(fam_path, index=False)
    out["family_summary_csv"] = fam_path

    best = best_agent_per_family(_filter(base, cfg), metric="efficiency_score")
    best_path = tables / "best_per_family.csv"
    best.to_csv(best_path, index=False)
    out["best_per_family_csv"] = best_path

    return out
