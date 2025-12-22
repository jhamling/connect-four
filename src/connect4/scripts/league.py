
from __future__ import annotations

import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Tuple

from connect4.ai.tt import TranspositionTable
from connect4.core.board import Board
from connect4.core.rules import check_winner, is_draw
from connect4.game.state import GameState


def other(p: str) -> str:
    return "O" if p == "X" else "X"


@dataclass(frozen=True)
class Team:
    name: str
    make: Callable[[], object]  # must be picklable (use functools.partial, not lambda)


@dataclass
class Agg:
    games: int = 0
    points: float = 0.0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    moves: int = 0
    time_ms: int = 0
    nodes: int = 0
    depth_sum: int = 0


def _seed(agent, seed: int) -> None:
    if hasattr(agent, "rng"):
        agent.rng.seed(seed)


def play_headless(agent_x, agent_o, seed_base: int = 0) -> Tuple[str, Dict[str, Dict[str, int]]]:
    """
    Headless game loop.
    Returns outcome: "X", "O", or "D"
    Also returns per-side move stats: {"X": {...}, "O": {...}}
    """
    state = GameState(board=Board(), current="X", last_status="")
    stats = {
        "X": {"moves": 0, "time_ms": 0, "nodes": 0, "depth": 0},
        "O": {"moves": 0, "time_ms": 0, "nodes": 0, "depth": 0},
    }

    # deterministic seeds per game for reproducibility
    _seed(agent_x, seed_base + 101)
    _seed(agent_o, seed_base + 202)

    # reset TT once per game (only if agent uses it)
    if hasattr(agent_x, "tt"):
        agent_x.tt = TranspositionTable()
    if hasattr(agent_o, "tt"):
        agent_o.tt = TranspositionTable()

    # --- opening randomization (2 plies) ---
    rng = random.Random(seed_base)
    for _ in range(2):
        moves = state.board.valid_moves()
        if not moves:
            break
        move = rng.choice(moves)
        state.board.drop(move, state.current)
        state.current = other(state.current)
    # --------------------------------------

    while True:
        w = check_winner(state.board)
        if w is not None:
            return w, stats
        if is_draw(state.board):
            return "D", stats

        agent = agent_x if state.current == "X" else agent_o
        move = agent.choose_move(state)

        info = getattr(agent, "last_info", None) or {}
        side_stats = stats[state.current]
        side_stats["moves"] += 1

        ms = int(info.get("time_ms", 0))
        side_stats["time_ms"] += max(1, ms)

        side_stats["nodes"] += int(info.get("nodes", 0))
        side_stats["depth"] += int(info.get("depth", 0))

        state.board.drop(move, state.current)
        state.current = other(state.current)


def add_result(agg_a: Agg, agg_b: Agg, outcome: str, a_is_x: bool) -> None:
    """
    outcome is "X", "O", or "D" for the game.
    a_is_x says whether team A played as X.
    """
    agg_a.games += 1
    agg_b.games += 1

    if outcome == "D":
        agg_a.draws += 1
        agg_b.draws += 1
        agg_a.points += 0.5
        agg_b.points += 0.5
        return

    a_won = (outcome == "X" and a_is_x) or (outcome == "O" and not a_is_x)
    if a_won:
        agg_a.wins += 1
        agg_b.losses += 1
        agg_a.points += 1.0
    else:
        agg_b.wins += 1
        agg_a.losses += 1
        agg_b.points += 1.0


# -----------------------------
# Multiprocessing: batched worker
# -----------------------------
def _run_pairings_batch(args):
    """
    Execute a batch of pairings in one process to reduce IPC/submit overhead.

    args = (
      batch_items,
      games_per_pair
    )

    batch_items is list of tuples:
      (A_name, B_name, A_make, B_make, base_seed)

    Returns:
      list of tuples:
        (A_name, B_name, a_is_x, outcome, stats)
    """
    (batch_items, games_per_pair) = args
    out = []

    for (A_name, B_name, A_make, B_make, base_seed) in batch_items:
        for g in range(games_per_pair):
            if g % 2 == 0:
                ax = A_make()
                bo = B_make()
                outcome, stats = play_headless(ax, bo, seed_base=(base_seed + g))
                out.append((A_name, B_name, True, outcome, stats))   # A is X
            else:
                bx = B_make()
                ao = A_make()
                outcome, stats = play_headless(bx, ao, seed_base=(base_seed + g))
                out.append((A_name, B_name, False, outcome, stats))  # A is O

    return out


def _chunked(lst, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def league_auto_prune(
    teams: List[Team],
    games_per_pair: int = 2,
    stage_pairings_per_team: int = 10,
    min_games_before_prune: int = 20,
    keep_fraction: float = 0.50,
    final_keep: int = 30,
    seed: int = 1234,
    max_workers: int | None = None,
    batch_pairings: int = 12,         # tune: 8..32
    max_stages: int | None = None,    # None = no cap
    prune_z: float = 1.28,            # stability penalty (LCB)
    ms_target: float = 50.0,          # soft compute penalty scale
) -> None:
    """
    Auto-pruning league with:
      - One persistent ProcessPool
      - Batched pairings per task (lower overhead)
      - Parallel staged execution AND parallel final round-robin

    PRUNING METRIC: "all-around best agent"
      score = ppg_lcb / (1 + avg_ms_per_move / ms_target)

    where:
      ppg_lcb = ppg - Z * sqrt(ppg*(1-ppg)/games)

    This favors:
      - strong win rate (ppg)
      - stability (LCB penalizes small-sample luck)
      - reasonable compute (soft penalty, not dominated by speed)
    """
    rng = random.Random(seed)

    agg: Dict[str, Agg] = {t.name: Agg() for t in teams}

    def points_per_game(a: Agg) -> float:
        return (a.points / a.games) if a.games else 0.0

    def avg_ms_per_move(a: Agg) -> float:
        return (a.time_ms / a.moves) if a.moves else 0.0

    def efficiency(a: Agg) -> float:
        ms = avg_ms_per_move(a)
        return (points_per_game(a) / ms) if ms > 0 else 0.0

    def ppg_lcb(a: Agg) -> float:
        if a.games <= 0:
            return 0.0
        ppg = points_per_game(a)
        se = math.sqrt(max(1e-9, ppg * (1.0 - ppg)) / a.games)
        return max(0.0, ppg - prune_z * se)

    def all_around_score(a: Agg) -> float:
        ms = avg_ms_per_move(a)
        if ms <= 0:
            ms = 1.0
        return ppg_lcb(a) / (1.0 + (ms / ms_target))

    def print_table(title: str, roster: List[Team], top_n: int = 20) -> None:
        print(f"\n=== {title} ===")
        rows = sorted(roster, key=lambda t: all_around_score(agg[t.name]), reverse=True)
        for rank, t in enumerate(rows[:top_n], start=1):
            a = agg[t.name]
            print(
                f"{rank:>2}. {t.name:<32} "
                f"score={all_around_score(a):.6f}  "
                f"ppg={points_per_game(a):.3f}  "
                f"lcb={ppg_lcb(a):.3f}  "
                f"W-D-L={a.wins}-{a.draws}-{a.losses}  "
                f"g={a.games:<4}  "
                f"avg_ms/move={avg_ms_per_move(a):.1f}  "
                f"eff={efficiency(a):.6f}"
            )

    def schedule_stage(roster: List[Team]) -> List[tuple[Team, Team]]:
        if len(roster) < 2:
            return []

        need = {t.name: stage_pairings_per_team for t in roster}
        name_to_team = {t.name: t for t in roster}

        pool = []
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
            A, B = rng.sample(roster, 2)
            key = (A.name, B.name) if A.name < B.name else (B.name, A.name)
            if key in used_pairs:
                continue
            used_pairs.add(key)
            matches.append((A, B))

        return matches

    def _apply_game_result(A_name: str, B_name: str, a_is_x: bool, outcome: str, stats: Dict[str, Dict[str, int]]):
        add_result(agg[A_name], agg[B_name], outcome, a_is_x=a_is_x)

        if a_is_x:
            # A is X, B is O
            agg[A_name].moves += stats["X"]["moves"]
            agg[A_name].time_ms += stats["X"]["time_ms"]
            agg[A_name].nodes += stats["X"]["nodes"]
            agg[A_name].depth_sum += stats["X"]["depth"]

            agg[B_name].moves += stats["O"]["moves"]
            agg[B_name].time_ms += stats["O"]["time_ms"]
            agg[B_name].nodes += stats["O"]["nodes"]
            agg[B_name].depth_sum += stats["O"]["depth"]
        else:
            # B is X, A is O
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

    print(f"Initial roster: {len(roster)} teams")
    print(
        f"Stage settings: pairings/team={stage_pairings_per_team}, "
        f"games/pair={games_per_pair}, keep_fraction={keep_fraction}, "
        f"final_keep={final_keep}, workers={max_workers}, batch_pairings={batch_pairings}"
    )
    print(f"Prune metric: Z={prune_z}, ms_target={ms_target}  (score = LCB(ppg) / (1 + ms/ms_target))")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        # ---------- STAGED PRUNING ----------
        while len(roster) > final_keep:
            if max_stages is not None and stage > max_stages:
                print(f"Reached max stages ({max_stages}) — stopping early.")
                break

            print(f"\n--- STAGE {stage} (roster={len(roster)}) ---")
            matches = schedule_stage(roster)
            print(f"Stage {stage}: scheduled {len(matches)} pairings")

            batch_items = []
            for idx, (A, B) in enumerate(matches, start=1):
                base_seed = seed + stage * 1_000_000 + idx * 10_000
                batch_items.append((A.name, B.name, A.make, B.make, base_seed))

            futures = []
            for chunk in _chunked(batch_items, batch_pairings):
                futures.append(ex.submit(_run_pairings_batch, (chunk, games_per_pair)))

            for fut in as_completed(futures):
                results = fut.result()
                for (A_name, B_name, a_is_x, outcome, stats) in results:
                    _apply_game_result(A_name, B_name, a_is_x, outcome, stats)

            print_table(f"After Stage {stage} (Top 20 by all-around score)", roster, top_n=20)

            # prune if enough data
            if all(agg[t.name].games >= min_games_before_prune for t in roster):
                keep_n = max(final_keep, int(math.ceil(len(roster) * keep_fraction)))
                keep_n = min(keep_n, len(roster))

                by_score = sorted(roster, key=lambda t: all_around_score(agg[t.name]), reverse=True)
                new_roster = by_score[:keep_n]

                if len(new_roster) == len(roster):
                    print("No pruning occurred — stopping early.")
                    roster = new_roster
                    break

                roster = new_roster
                print(f"Pruned to {len(roster)} teams (kept top {keep_n} by all-around score)")
            else:
                min_g = min(agg[t.name].games for t in roster)
                print(f"Not pruning yet: minimum games/team = {min_g}, need >= {min_games_before_prune}")

            stage += 1
            if len(roster) < 2:
                break

        # ---------- FINAL ROUND-ROBIN (PARALLEL) ----------
        print(f"\n=== FINAL ROUND-ROBIN (roster={len(roster)}) ===")

        pair_items = []
        n = len(roster)
        for i in range(n):
            for j in range(i + 1, n):
                A = roster[i]
                B = roster[j]
                base_seed = seed + 9_000_000 + i * 10_000 + j * 100
                pair_items.append((A.name, B.name, A.make, B.make, base_seed))

        final_futures = []
        for chunk in _chunked(pair_items, batch_pairings):
            final_futures.append(ex.submit(_run_pairings_batch, (chunk, games_per_pair)))

        for fut in as_completed(final_futures):
            results = fut.result()
            for (A_name, B_name, a_is_x, outcome, stats) in results:
                _apply_game_result(A_name, B_name, a_is_x, outcome, stats)

        # Winner: single best "all-around" agent among final survivors
        winner = max(roster, key=lambda t: all_around_score(agg[t.name]))
        a = agg[winner.name]

        print("\n=== WINNER (All-around best agent) ===")
        print(
            f"{winner.name}\n"
            f"  score={all_around_score(a):.6f}\n"
            f"  ppg={points_per_game(a):.3f}  lcb={ppg_lcb(a):.3f}\n"
            f"  W-D-L={a.wins}-{a.draws}-{a.losses}  games={a.games}\n"
            f"  avg_ms/move={avg_ms_per_move(a):.1f}"
        )




def build_roster() -> List[Team]:
    # Import all agent classes by their REAL names (must match class definitions)
    from connect4.ai.random_agent import RandomAgent
    from connect4.ai.weighted_random_agent import WeightedRandomAgent
    from connect4.ai.greedy_agent import GreedyAgent
    from connect4.ai.tactical_greedy_agent import TacticalGreedyAgent
    from connect4.ai.heuristic_agent import HeuristicAgent
    from connect4.ai.beam_agent import BeamAgent
    from connect4.ai.beam2_agent import Beam2Agent
    from connect4.ai.mcts_agent import MCTSAgent
    from connect4.ai.expectiminimax_agent import ExpectiMiniMaxAgent
    from connect4.ai.minimax_agent import MinimaxAgent

    teams: List[Team] = []

    # --- Baselines / variety ---
    teams.append(Team("Random", partial(RandomAgent)))
    teams.append(Team("WeightedRandom", partial(WeightedRandomAgent)))
    teams.append(Team("Greedy", partial(GreedyAgent)))
    teams.append(Team("TacticalGreedy", partial(TacticalGreedyAgent)))
    teams.append(Team("Heuristic", partial(HeuristicAgent)))
    teams.append(Team("Beam", partial(BeamAgent)))
    teams.append(Team("Beam2", partial(Beam2Agent)))
    teams.append(Team("MCTS", partial(MCTSAgent)))
    teams.append(Team("ExpectiMiniMax", partial(ExpectiMiniMaxAgent)))

    # --- Minimax variants (your main competitive pool) ---
    depths = [3, 5, 7, 9, 11]
    times = [0.05, 0.10, 0.20, 0.35, 0.75]
    temps = [0, 25, 75]

    for d in depths:
        for t in times:
            for temp in temps:
                name = f"MM d{d} t{int(t * 1000)}ms temp{temp}"
                teams.append(
                    Team(
                        name,
                        partial(
                            MinimaxAgent,
                            name=name,
                            depth=d,
                            time_limit_sec=t,
                            temperature=temp,
                        ),
                    )
                )

    return teams


def main() -> None:
    roster = build_roster()
    print(f"Roster size: {len(roster)} teams")

    gpp = input("Games per pairing (default 1): ").strip()
    games_per_pair = int(gpp) if gpp else 1

    ppt = input("Pairings per team per stage (default 4): ").strip()
    stage_pairings_per_team = int(ppt) if ppt else 4

    mg = input("Min games/team before prune (default 8): ").strip()
    min_games_before_prune = int(mg) if mg else 8

    kf = input("Keep fraction each prune (default 0.60): ").strip()
    keep_fraction = float(kf) if kf else 0.60

    fk = input("Final keep (default 3): ").strip()
    final_keep = int(fk) if fk else 3

    sd = input("Seed (default 1234): ").strip()
    seed = int(sd) if sd else 1234

    mw = input("Max workers (default = cpu cores, capped at 6): ").strip()
    max_workers = int(mw) if mw else None

    bp = input("Batch pairings per worker task (default 12): ").strip()
    batch_pairings = int(bp) if bp else 12

    ms = input("Max stages (blank = no cap, recommended 3 for testing): ").strip()
    max_stages = int(ms) if ms else None

    z = input("Prune Z (default 1.28): ").strip()
    prune_z = float(z) if z else 1.28

    mt = input("ms_target (default 50): ").strip()
    ms_target = float(mt) if mt else 50.0

    league_auto_prune(
        roster,
        games_per_pair=games_per_pair,
        stage_pairings_per_team=stage_pairings_per_team,
        min_games_before_prune=min_games_before_prune,
        keep_fraction=keep_fraction,
        final_keep=final_keep,
        seed=seed,
        max_workers=max_workers,
        batch_pairings=batch_pairings,
        max_stages=max_stages,
        prune_z=prune_z,
        ms_target=ms_target,
    )


if __name__ == "__main__":
    main()
