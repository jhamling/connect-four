from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

from connect4.ai.tt import TranspositionTable
from connect4.core.board import Board
from connect4.core.rules import check_winner, is_draw
from connect4.game.state import GameState

from .league_types import Agg


def other(p: str) -> str:
    return "O" if p == "X" else "X"


def seed_agent(agent, seed: int) -> None:
    if hasattr(agent, "rng"):
        try:
            agent.rng.seed(seed)
        except Exception:
            pass


def play_headless(agent_x, agent_o, seed_base: int = 0) -> Tuple[str, Dict[str, Dict[str, int]]]:
    state = GameState(board=Board(), current="X", last_status="")
    stats = {
        "X": {"moves": 0, "time_ms": 0, "nodes": 0, "depth": 0},
        "O": {"moves": 0, "time_ms": 0, "nodes": 0, "depth": 0},
    }

    seed_agent(agent_x, seed_base + 101)
    seed_agent(agent_o, seed_base + 202)

    if hasattr(agent_x, "tt"):
        agent_x.tt = TranspositionTable()
    if hasattr(agent_o, "tt"):
        agent_o.tt = TranspositionTable()

    rng = random.Random(seed_base)
    for _ in range(2):
        moves = state.board.valid_moves()
        if not moves:
            break
        move = rng.choice(moves)
        state.board.drop(move, state.current)
        state.current = other(state.current)

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


def run_pairings_batch(args):
    (batch_items, games_per_pair) = args
    out = []
    for (A_name, B_name, A_make, B_make, base_seed) in batch_items:
        for g in range(games_per_pair):
            if g % 2 == 0:
                ax = A_make()
                bo = B_make()
                outcome, stats = play_headless(ax, bo, seed_base=(base_seed + g))
                out.append((A_name, B_name, True, outcome, stats))
            else:
                bx = B_make()
                ao = A_make()
                outcome, stats = play_headless(bx, ao, seed_base=(base_seed + g))
                out.append((A_name, B_name, False, outcome, stats))
    return out


def chunked(lst, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]

