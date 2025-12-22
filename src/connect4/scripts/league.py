from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from connect4.core.board import Board
from connect4.core.rules import check_winner, is_draw
from connect4.game.state import GameState


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
    state = GameState(board=Board(), current="X", last_status="")
    stats = {"X": {"moves": 0, "time_ms": 0, "nodes": 0, "depth": 0},
             "O": {"moves": 0, "time_ms": 0, "nodes": 0, "depth": 0}}

    # seed both agents deterministically per game for variety + reproducibility
    _seed(agent_x, seed_base + 101)
    _seed(agent_o, seed_base + 202)

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
        side_stats["time_ms"] += int(info.get("time_ms", 0))
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


def main() -> None:
    roster = build_roster()
    print(f"Roster size: {len(roster)} teams")

    g = input("Games per pairing (default 4): ").strip()
    games_per_pair = int(g) if g else 4

    league(roster, games_per_pair=games_per_pair)


if __name__ == "__main__":
    main()

