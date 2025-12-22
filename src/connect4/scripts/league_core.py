
from __future__ import annotations

import csv
import math
import os
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .league_format import A
from .league_play import add_result, chunked, run_pairings_batch
from .league_scoring import (
    avg_ms_per_move,
    efficiency_score,
    pareto_frontier,
    ppg,
    strength_score,
)
from .league_types import Agg, Team


# -----------------------------
# Stable terminal formatting
# -----------------------------
def term_width(default: int = 120) -> int:
    try:
        return shutil.get_terminal_size(fallback=(default, 24)).columns
    except Exception:
        return default


def hr(char: str = "─", width: int | None = None) -> str:
    w = width or term_width()
    return char * max(10, w)


def clamp(s: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(s) <= width:
        return s
    if width <= 1:
        return s[:width]
    return s[: width - 1] + "…"


@dataclass(frozen=True)
class Col:
    title: str
    width: int
    align: str = "left"  # "left" | "right"


def _fmt_row(values: Sequence[str], cols: Sequence[Col]) -> str:
    out: List[str] = []
    for v, c in zip(values, cols):
        s = clamp(str(v), c.width)
        out.append(s.rjust(c.width) if c.align == "right" else s.ljust(c.width))
    return "  ".join(out)


def print_table(title: str, cols: Sequence[Col], rows: Iterable[Sequence[str]], *, width: int) -> None:
    print(A.cyan(A.bold(title)))
    print(A.dim(_fmt_row([c.title for c in cols], cols)))
    print(A.dim(hr("─", width)))
    for r in rows:
        print(_fmt_row(r, cols))
    print(A.dim(hr("─", width)))


def league_auto_prune(
    teams: List[Team],
    games_per_pair: int = 2,
    stage_pairings_per_team: int = 10,
    min_games_before_prune: int = 20,
    keep_fraction: float = 0.50,
    final_keep: int = 30,
    seed: int = 1234,
    max_workers: int | None = None,
    batch_pairings: int = 12,
    max_stages: int | None = None,
    prune_z: float = 1.28,
    ms_target: float = 50.0,
    speed_alpha: float = 0.35,
    speed_min_factor: float = 0.75,
    export_csv: bool = True,
) -> None:
    rng = random.Random(seed)
    agg: Dict[str, Agg] = {t.name: Agg() for t in teams}

    def print_tables(title: str, roster: List[Team], top_n: int = 20) -> None:
        w = term_width(120)
        rows = [(t.name, agg[t.name]) for t in roster]

        def fmt_wdl(a: Agg) -> str:
            return f"{a.wins}-{a.draws}-{a.losses}"

        by_strength = sorted(rows, key=lambda r: strength_score(r[1], prune_z), reverse=True)
        by_eff = sorted(
            rows,
            key=lambda r: efficiency_score(r[1], prune_z, ms_target, speed_alpha, speed_min_factor),
            reverse=True,
        )
        frontier = pareto_frontier(rows, prune_z)

        # Allocate most remaining width to agent column.
        # Columns: rk(3) + spaces + strength(10) + eff(10) + ppg(5) + g(3) + ms(7) + wdl(7)
        fixed = 3 + 2 + 10 + 2 + 10 + 2 + 5 + 2 + 3 + 2 + 7 + 2 + 7
        agent_w = max(18, min(60, w - fixed))

        cols = [
            Col("rk", 3, "right"),
            Col("agent", agent_w, "left"),
            Col("strength", 10, "right"),
            Col("eff", 10, "right"),
            Col("ppg", 5, "right"),
            Col("g", 3, "right"),
            Col("ms/mv", 7, "right"),
            Col("W-D-L", 7, "right"),
        ]

        def make_rows(ranking: List[tuple[str, Agg]]) -> List[List[str]]:
            out: List[List[str]] = []
            for i, (name, a) in enumerate(ranking[:top_n], start=1):
                s = strength_score(a, prune_z)
                e = efficiency_score(a, prune_z, ms_target, speed_alpha, speed_min_factor)
                p = ppg(a)
                ms = avg_ms_per_move(a)

                # Color only the PPG text; keep widths stable (color codes don't change padding due to clamp first).
                p_txt = f"{p:0.3f}"
                if p >= 0.75:
                    p_txt = A.green(p_txt)
                elif p >= 0.5:
                    p_txt = A.yellow(p_txt)
                else:
                    p_txt = A.red(p_txt)

                out.append([
                    str(i),
                    name,
                    f"{s:0.6f}",
                    f"{e:0.6f}",
                    p_txt,
                    str(a.games),
                    f"{ms:0.1f}",
                    fmt_wdl(a),
                ])
            return out

        print("\n" + A.bold(f"=== {title} ==="))
        print(A.dim(hr("═", w)))

        print_table("Top by Strength (speed ignored)", cols, make_rows(by_strength), width=w)
        print_table("Top by Efficiency (capped speed factor)", cols, make_rows(by_eff), width=w)

        # Pareto frontier: fewer columns, same stable formatting
        f_cols = [
            Col("agent", agent_w, "left"),
            Col("strength", 10, "right"),
            Col("ms/mv", 7, "right"),
            Col("ppg", 5, "right"),
            Col("g", 3, "right"),
            Col("W-D-L", 7, "right"),
        ]

        f_rows: List[List[str]] = []
        for (name, a) in frontier[:top_n]:
            s = strength_score(a, prune_z)
            ms = avg_ms_per_move(a)
            p = ppg(a)
            f_rows.append([name, f"{s:0.6f}", f"{ms:0.1f}", f"{p:0.3f}", str(a.games), fmt_wdl(a)])

        print_table("Pareto Frontier (no agent is both stronger and faster)", f_cols, f_rows, width=w)

    def schedule_stage(roster: List[Team]) -> List[tuple[Team, Team]]:
        if len(roster) < 2:
            return []

        need = {t.name: stage_pairings_per_team for t in roster}
        name_to_team = {t.name: t for t in roster}

        pool: List[str] = []
        for name, k in need.items():
            pool.extend([name] * k)

        rng.shuffle(pool)

        used_pairs = set()
        matches: list[tuple[Team, Team]] = []

        i = 0
        while i + 1 < len(pool):
            a = pool[i]
            b = pool[i + 1]
            i += 2
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            if key in used_pairs:
                continue
            used_pairs.add(key)
            matches.append((name_to_team[a], name_to_team[b]))

        target_matches = (len(roster) * stage_pairings_per_team) // 2
        tries = 0
        while len(matches) < target_matches and tries < target_matches * 20:
            tries += 1
            A_, B_ = rng.sample(roster, 2)
            key = (A_.name, B_.name) if A_.name < B_.name else (B_.name, A_.name)
            if key in used_pairs:
                continue
            used_pairs.add(key)
            matches.append((A_, B_))

        return matches

    def apply_game_result(A_name: str, B_name: str, a_is_x: bool, outcome: str, stats):
        add_result(agg[A_name], agg[B_name], outcome, a_is_x=a_is_x)

        if a_is_x:
            agg[A_name].moves += stats["X"]["moves"]
            agg[A_name].time_ms += stats["X"]["time_ms"]
            agg[A_name].nodes += stats["X"]["nodes"]
            agg[A_name].depth_sum += stats["X"]["depth"]

            agg[B_name].moves += stats["O"]["moves"]
            agg[B_name].time_ms += stats["O"]["time_ms"]
            agg[B_name].nodes += stats["O"]["nodes"]
            agg[B_name].depth_sum += stats["O"]["depth"]
        else:
            agg[B_name].moves += stats["X"]["moves"]
            agg[B_name].time_ms += stats["X"]["time_ms"]
            agg[B_name].nodes += stats["X"]["nodes"]
            agg[B_name].depth_sum += stats["X"]["depth"]

            agg[A_name].moves += stats["O"]["moves"]
            agg[A_name].time_ms += stats["O"]["time_ms"]
            agg[A_name].nodes += stats["O"]["nodes"]
            agg[A_name].depth_sum += stats["O"]["depth"]

    if max_workers is None:
        max_workers = min(os.cpu_count() or 2, 6)

    roster = teams[:]
    stage = 1

    w = term_width(120)
    print(A.bold(f"Initial roster: {len(roster)} teams"))
    print(A.dim(hr("═", w)))
    print(
        f"Stage settings: pairings/team={stage_pairings_per_team}, games/pair={games_per_pair}, "
        f"keep_fraction={keep_fraction}, final_keep={final_keep}, workers={max_workers}, batch_pairings={batch_pairings}"
    )
    print(
        f"Scoring:\n"
        f"  Strength = WilsonLCB(PPG), z={prune_z}\n"
        f"  Efficiency = Strength * max({speed_min_factor}, 1/(1+alpha*sqrt(ms/ms_target)))  "
        f"(alpha={speed_alpha}, ms_target={ms_target})"
    )
    print(A.dim(hr("═", w)))

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        while len(roster) > final_keep:
            if max_stages is not None and stage > max_stages:
                print(A.yellow(f"Reached max stages ({max_stages}) — stopping early."))
                break

            print("\n" + A.bold(f"--- STAGE {stage} (roster={len(roster)}) ---"))
            matches = schedule_stage(roster)
            print(f"Stage {stage}: scheduled {len(matches)} pairings")

            batch_items = []
            for idx, (A_team, B_team) in enumerate(matches, start=1):
                base_seed = seed + stage * 1_000_000 + idx * 10_000
                batch_items.append((A_team.name, B_team.name, A_team.make, B_team.make, base_seed))

            futures = []
            for chunk in chunked(batch_items, batch_pairings):
                futures.append(ex.submit(run_pairings_batch, (chunk, games_per_pair)))

            for fut in as_completed(futures):
                results = fut.result()
                for (A_name, B_name, a_is_x, outcome, stats) in results:
                    apply_game_result(A_name, B_name, a_is_x, outcome, stats)

            print_tables(f"After Stage {stage}", roster, top_n=20)

            if all(agg[t.name].games >= min_games_before_prune for t in roster):
                keep_n = max(final_keep, int(math.ceil(len(roster) * keep_fraction)))
                keep_n = min(keep_n, len(roster))

                by_eff = sorted(
                    roster,
                    key=lambda t: efficiency_score(agg[t.name], prune_z, ms_target, speed_alpha, speed_min_factor),
                    reverse=True,
                )
                new_roster = by_eff[:keep_n]

                if len(new_roster) == len(roster):
                    print(A.yellow("No pruning occurred — stopping early."))
                    roster = new_roster
                    break

                roster = new_roster
                print(A.bold(f"Pruned to {len(roster)} teams (kept top {keep_n} by Efficiency)"))
            else:
                min_g = min(agg[t.name].games for t in roster)
                print(A.yellow(f"Not pruning yet: minimum games/team = {min_g}, need >= {min_games_before_prune}"))

            stage += 1
            if len(roster) < 2:
                break

        print("\n" + A.bold(f"=== FINAL ROUND-ROBIN (roster={len(roster)}) ==="))
        print(A.dim(hr("═", term_width(120))))

        pair_items = []
        n = len(roster)
        for i in range(n):
            for j in range(i + 1, n):
                A_team = roster[i]
                B_team = roster[j]
                base_seed = seed + 9_000_000 + i * 10_000 + j * 100
                pair_items.append((A_team.name, B_team.name, A_team.make, B_team.make, base_seed))

        final_futures = []
        for chunk in chunked(pair_items, batch_pairings):
            final_futures.append(ex.submit(run_pairings_batch, (chunk, games_per_pair)))

        for fut in as_completed(final_futures):
            results = fut.result()
            for (A_name, B_name, a_is_x, outcome, stats) in results:
                apply_game_result(A_name, B_name, a_is_x, outcome, stats)

        winner_strength = max(roster, key=lambda t: strength_score(agg[t.name], prune_z))
        winner_eff = max(
            roster,
            key=lambda t: efficiency_score(agg[t.name], prune_z, ms_target, speed_alpha, speed_min_factor),
        )

        def print_winner(title: str, team: Team) -> None:
            a = agg[team.name]
            w = term_width(120)
            print("\n" + A.bold(title))
            print(A.dim(hr("═", w)))
            print(team.name)
            print(
                f"  strength={strength_score(a, prune_z):.6f}  "
                f"eff={efficiency_score(a, prune_z, ms_target, speed_alpha, speed_min_factor):.6f}\n"
                f"  ppg={ppg(a):.3f}  W-D-L={a.wins}-{a.draws}-{a.losses}  games={a.games}\n"
                f"  avg_ms/move={avg_ms_per_move(a):.1f}  moves={a.moves}  nodes={a.nodes}"
            )

        print_winner("WINNER (Strength)", winner_strength)
        print_winner("WINNER (Efficiency)", winner_eff)

        print_tables("Final (post round-robin)", roster, top_n=30)

        if export_csv:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = f"league_results_{ts}.csv"
            all_rows = [(t.name, agg[t.name]) for t in teams]

            with open(out_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "name",
                    "games", "wins", "draws", "losses",
                    "points", "ppg",
                    "strength_wilson_lcb",
                    "avg_ms_per_move",
                    "efficiency_score",
                    "moves", "time_ms", "nodes", "avg_depth",
                ])
                for name, a in all_rows:
                    p = ppg(a)
                    s = strength_score(a, prune_z)
                    ms = avg_ms_per_move(a)
                    e = efficiency_score(a, prune_z, ms_target, speed_alpha, speed_min_factor)
                    avg_depth = (a.depth_sum / a.moves) if a.moves else 0.0
                    w.writerow([
                        name,
                        a.games, a.wins, a.draws, a.losses,
                        a.points, round(p, 6),
                        round(s, 6),
                        round(ms, 3),
                        round(e, 6),
                        a.moves, a.time_ms, a.nodes, round(avg_depth, 3),
                    ])

            print("\n" + A.bold("Export"))
            print(A.dim(hr("═", term_width(120))))
            print(f"Wrote CSV: {out_path}")
