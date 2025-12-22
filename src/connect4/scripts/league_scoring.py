from __future__ import annotations

import math
from typing import List, Tuple

from .league_types import Agg


def ppg(a: Agg) -> float:
    return (a.points / a.games) if a.games else 0.0


def avg_ms_per_move(a: Agg) -> float:
    return (a.time_ms / a.moves) if a.moves else 0.0


def wilson_lcb(p: float, n: int, z: float) -> float:
    if n <= 0:
        return 0.0
    p = max(0.0, min(1.0, p))
    z2 = z * z
    denom = 1.0 + (z2 / n)
    center = p + (z2 / (2.0 * n))
    rad = z * math.sqrt(max(0.0, (p * (1.0 - p) + (z2 / (4.0 * n))) / n))
    return max(0.0, (center - rad) / denom)


def strength_score(a: Agg, z: float) -> float:
    return wilson_lcb(ppg(a), a.games, z)


def compute_factor(ms_per_move: float, ms_target: float, alpha: float, min_factor: float) -> float:
    ms = max(1e-9, float(ms_per_move))
    tgt = max(1e-9, float(ms_target))
    factor = 1.0 / (1.0 + float(alpha) * math.sqrt(ms / tgt))
    return max(float(min_factor), factor)


def efficiency_score(a: Agg, z: float, ms_target: float, alpha: float, min_factor: float) -> float:
    s = strength_score(a, z)
    ms = avg_ms_per_move(a)
    f = compute_factor(ms, ms_target=ms_target, alpha=alpha, min_factor=min_factor)
    return s * f


def pareto_frontier(rows: List[tuple[str, Agg]], z: float) -> List[tuple[str, Agg]]:
    scored: List[Tuple[str, Agg, float, float]] = []
    for name, a in rows:
        scored.append((name, a, strength_score(a, z), avg_ms_per_move(a)))

    scored.sort(key=lambda t: (-t[2], t[3]))

    frontier: List[tuple[str, Agg]] = []
    best_ms = float("inf")
    for name, a, s, ms in scored:
        if ms < best_ms - 1e-9:
            frontier.append((name, a))
            best_ms = ms
    return frontier
