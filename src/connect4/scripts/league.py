from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from connect4.core.board import Board
from connect4.core.rules import check_winner, is_draw
from connect4.game.state import GameState

from connect4.ai.tt import TranspositionTable

def other(p: str) -> str:
    return "O" if p == "X" else "X"


@dataclass
class Team:
    name: str
    make: Callable[[], object]


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
    import random

    state = GameState(board=Board(), current="X", last_status="")
    stats = {
        "X": {"moves": 0, "time_ms": 0, "nodes": 0, "depth": 0},
        "O": {"moves": 0, "time_ms": 0, "nodes": 0, "depth": 0},
    }

    # seed both agents deterministically per game for variety + reproducibility
    _seed(agent_x, seed_base + 101)
    _seed(agent_o, seed_base + 202)

    # Reset transposition tables once per game
    if hasattr(agent_x, "tt"):
        agent_x.tt = TranspositionTable()
    if hasattr(agent_o, "tt"):
        agent_o.tt = TranspositionTable()

    # ---------- OPENING RANDOMIZATION (2 plies) ----------
    rng = random.Random(seed_base)
    for _ in range(2):
        moves = state.board.valid_moves()
        if not moves:
            break
        move = rng.choice(moves)
        state.board.drop(move, state.current)
        state.current = other(state.current)
    # ----------------------------------------------------

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
    outcome is relative to X/O for the game.
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

    # map X/O winner to A/B winner
    a_won = (outcome == "X" and a_is_x) or (outcome == "O" and not a_is_x)
    if a_won:
        agg_a.wins += 1
        agg_b.losses += 1
        agg_a.points += 1.0
    else:
        agg_b.wins += 1
        agg_a.losses += 1
        agg_b.points += 1.0

import math
import random



def league_auto_prune(
    teams: List[Team],
    games_per_pair: int = 2,
    stage_pairings_per_team: int = 10,
    min_games_before_prune: int = 20,
    keep_fraction: float = 0.50,
    final_keep: int = 30,
    seed: int = 1234,
) -> None:
    """
    Auto-pruning league.

    Staged random pairings with pruning.
    Stops when:
      - roster <= final_keep
      - pruning no longer reduces roster
      - max stages reached
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

    def print_table(title: str, roster: List[Team], top_n: int = 20) -> None:
        print(f"\n=== {title} ===")
        rows = sorted(roster, key=lambda t: points_per_game(agg[t.name]), reverse=True)
        for rank, t in enumerate(rows[:top_n], start=1):
            a = agg[t.name]
            print(
                f"{rank:>2}. {t.name:<32} "
                f"ppg={points_per_game(a):.3f}  "
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

    def play_match(A: Team, B: Team, base_seed: int) -> None:
        for g in range(games_per_pair):
            if g % 2 == 0:
                ax = A.make()
                bo = B.make()
                outcome, stats = play_headless(ax, bo, seed_base=(base_seed + g))
                add_result(agg[A.name], agg[B.name], outcome, a_is_x=True)

                agg[A.name].moves += stats["X"]["moves"]
                agg[A.name].time_ms += stats["X"]["time_ms"]
                agg[A.name].nodes += stats["X"]["nodes"]
                agg[A.name].depth_sum += stats["X"]["depth"]

                agg[B.name].moves += stats["O"]["moves"]
                agg[B.name].time_ms += stats["O"]["time_ms"]
                agg[B.name].nodes += stats["O"]["nodes"]
                agg[B.name].depth_sum += stats["O"]["depth"]
            else:
                bx = B.make()
                ao = A.make()
                outcome, stats = play_headless(bx, ao, seed_base=(base_seed + g))
                add_result(agg[A.name], agg[B.name], outcome, a_is_x=False)

                agg[B.name].moves += stats["X"]["moves"]
                agg[B.name].time_ms += stats["X"]["time_ms"]
                agg[B.name].nodes += stats["X"]["nodes"]
                agg[B.name].depth_sum += stats["X"]["depth"]

                agg[A.name].moves += stats["O"]["moves"]
                agg[A.name].time_ms += stats["O"]["time_ms"]
                agg[A.name].nodes += stats["O"]["nodes"]
                agg[A.name].depth_sum += stats["O"]["depth"]

    roster = teams[:]
    stage = 1
    MAX_STAGES = 3   # 🔒 safety cap (raise later for real runs)

    print(f"Initial roster: {len(roster)} teams")
    print(
        f"Stage settings: pairings/team={stage_pairings_per_team}, "
        f"games/pair={games_per_pair}, keep_fraction={keep_fraction}, "
        f"final_keep={final_keep}"
    )

    while len(roster) > final_keep:
        if stage > MAX_STAGES:
            print(f"Reached max stages ({MAX_STAGES}) — stopping early.")
            break

        print(f"\n--- STAGE {stage} (roster={len(roster)}) ---")
        matches = schedule_stage(roster)
        print(f"Stage {stage}: scheduled {len(matches)} pairings")

        for idx, (A, B) in enumerate(matches, start=1):
            base_seed = seed + stage * 1_000_000 + idx * 10_000
            play_match(A, B, base_seed=base_seed)

        print_table(f"After Stage {stage} (Top 20)", roster, top_n=20)

        if all(agg[t.name].games >= min_games_before_prune for t in roster):
            by_ppg = sorted(roster, key=lambda t: points_per_game(agg[t.name]), reverse=True)

            keep_n = max(final_keep, int(math.ceil(len(roster) * keep_fraction)))
            keep_n = min(keep_n, len(roster))

            new_roster = by_ppg[:keep_n]

            if len(new_roster) == len(roster):
                print("No pruning occurred — stopping early.")
                roster = new_roster
                break

            roster = new_roster
            print(f"Pruned to {len(roster)} teams (kept top {keep_n} by ppg)")
        else:
            min_g = min(agg[t.name].games for t in roster)
            print(
                f"Not pruning yet: minimum games/team = {min_g}, "
                f"need >= {min_games_before_prune}"
            )

        stage += 1

        if len(roster) < 2:
            break

    print(f"\n=== FINAL ROUND-ROBIN (roster={len(roster)}) ===")
    league(roster, games_per_pair=games_per_pair)

    print("\n=== FINAL (Efficiency among survivors) ===")
    by_eff = sorted(roster, key=lambda t: efficiency(agg[t.name]), reverse=True)
    for rank, t in enumerate(by_eff, start=1):
        a = agg[t.name]
        print(
            f"{rank:>2}. {t.name:<32} "
            f"eff={efficiency(a):.6f}  "
            f"ppg={points_per_game(a):.3f}  "
            f"avg_ms/move={avg_ms_per_move(a):.1f}  "
            f"g={a.games}"
        )

def league(teams: List[Team], games_per_pair: int = 6) -> None:
    agg: Dict[str, Agg] = {t.name: Agg() for t in teams}

    n = len(teams)
    for i in range(n):
        for j in range(i + 1, n):
            A = teams[i]
            B = teams[j]

            print(f"\n=== {A.name} vs {B.name} ({games_per_pair} games) ===")

            for g in range(games_per_pair):
                # alternate which team is X each game to reduce first-move bias
                if g % 2 == 0:
                    ax = A.make()
                    bo = B.make()
                    outcome, stats = play_headless(ax, bo, seed_base=(i * 10_000 + j * 100 + g))
                    add_result(agg[A.name], agg[B.name], outcome, a_is_x=True)

                    agg[A.name].moves += stats["X"]["moves"]
                    agg[A.name].time_ms += stats["X"]["time_ms"]
                    agg[A.name].nodes += stats["X"]["nodes"]
                    agg[A.name].depth_sum += stats["X"]["depth"]

                    agg[B.name].moves += stats["O"]["moves"]
                    agg[B.name].time_ms += stats["O"]["time_ms"]
                    agg[B.name].nodes += stats["O"]["nodes"]
                    agg[B.name].depth_sum += stats["O"]["depth"]
                else:
                    bx = B.make()
                    ao = A.make()
                    outcome, stats = play_headless(bx, ao, seed_base=(j * 10_000 + i * 100 + g))
                    add_result(agg[A.name], agg[B.name], outcome, a_is_x=False)

                    agg[B.name].moves += stats["X"]["moves"]
                    agg[B.name].time_ms += stats["X"]["time_ms"]
                    agg[B.name].nodes += stats["X"]["nodes"]
                    agg[B.name].depth_sum += stats["X"]["depth"]

                    agg[A.name].moves += stats["O"]["moves"]
                    agg[A.name].time_ms += stats["O"]["time_ms"]
                    agg[A.name].nodes += stats["O"]["nodes"]
                    agg[A.name].depth_sum += stats["O"]["depth"]

            print(f"Completed: {A.name} vs {B.name}")

    def points_per_game(a: Agg) -> float:
        return (a.points / a.games) if a.games else 0.0

    def avg_ms_per_move(a: Agg) -> float:
        return (a.time_ms / a.moves) if a.moves else 0.0

    def efficiency(a: Agg) -> float:
        # strength per compute
        ms = avg_ms_per_move(a)
        return (points_per_game(a) / ms) if ms > 0 else 0.0

    print("\n=== STANDINGS (Strength: points/game) ===")
    by_strength = sorted(agg.items(), key=lambda kv: points_per_game(kv[1]), reverse=True)
    for rank, (name, a) in enumerate(by_strength, start=1):
        print(
            f"{rank:>2}. {name:<28} "
            f"ppg={points_per_game(a):.3f}  "
            f"W-D-L={a.wins}-{a.draws}-{a.losses}  "
            f"avg_ms/move={avg_ms_per_move(a):.1f}  "
            f"avg_nodes/move={(a.nodes / a.moves) if a.moves else 0.0:.0f}  "
            f"avg_depth={(a.depth_sum / a.moves) if a.moves else 0.0:.2f}"
        )

    print("\n=== STANDINGS (Efficiency: ppg / avg_ms_per_move) ===")
    by_eff = sorted(agg.items(), key=lambda kv: efficiency(kv[1]), reverse=True)
    for rank, (name, a) in enumerate(by_eff, start=1):
        print(
            f"{rank:>2}. {name:<28} "
            f"eff={efficiency(a):.6f}  "
            f"ppg={points_per_game(a):.3f}  "
            f"avg_ms/move={avg_ms_per_move(a):.1f}"
        )


def build_roster() -> List[Team]:
    from connect4.ai.random_agent import RandomAgent
    from connect4.ai.minimax_agent import MinimaxAgent
    from connect4.ai.greedy_agent import GreedyAgent
    from connect4.ai.tactical_greedy_agent import TacticalGreedyAgent
    from connect4.ai.beam_agent import BeamSearchAgent
    from connect4.ai.expectiminimax_agent import ExpectiMiniMaxAgent

    teams: List[Team] = [Team("Random", lambda: RandomAgent())]

    # Big roster generator:
    depths = [3, 5, 7, 9, 11]
    times = [0.05, 0.10, 0.20, 0.35, 0.75]   # seconds (50ms..750ms)
    temps = [0, 25, 75]                      # randomness: 0 deterministic, higher = more variety

    # Limit count if you want (this generates 5*5*3 = 75 minimax variants + Random)
    for d in depths:
        for t in times:
            for temp in temps:
                name = f"MM d{d} t{int(t*1000)}ms temp{temp}"
                teams.append(
                    Team(
                        name,
                        lambda d=d, t=t, temp=temp: MinimaxAgent(
                            name=name,
                            depth=d,
                            time_limit_sec=t,
                            temperature=temp,
                        ),
                    )
                )

    return teams




def main():
    from connect4.ai.minimax_agent import MinimaxAgent
    from connect4.ai.greedy_agent import GreedyAgent

    roster = [
        Team("MM d7 200ms", lambda: MinimaxAgent(depth=7, time_limit_sec=0.2)),
        Team("MM d7 200ms #2", lambda: MinimaxAgent(depth=7, time_limit_sec=0.2)),
        Team("Greedy", lambda: GreedyAgent()),
    ]

    league_auto_prune(
        roster,
        games_per_pair=1,
        stage_pairings_per_team=2,
        min_games_before_prune=2,
        keep_fraction=0.7,
        final_keep=2,
        seed=42,
    )


if __name__ == "__main__":
    main()

