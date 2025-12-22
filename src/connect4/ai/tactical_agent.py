
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Optional

from connect4.core.rules import check_winner, is_draw
from connect4.game.state import GameState


def other(p: str) -> str:
    return "O" if p == "X" else "X"


@dataclass
class TacticalAgent:
    """
    Fast tactical agent:
      1) Play immediate winning move if available
      2) Block opponent immediate winning move
      3) Prefer center-ish columns
      4) Otherwise random valid move

    This agent is intentionally cheap and can "spoil" weaker minimax settings.
    """
    name: str = "Tactical"
    center_bias: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        import random
        self.rng = random.Random(self.seed)
        self.last_info = {}

    def _winning_move(self, state: GameState, player: str) -> Optional[int]:
        moves = state.board.valid_moves()
        for c in moves:
            b2 = copy.deepcopy(state.board)
            b2.drop(c, player)
            if check_winner(b2) == player:
                return c
        return None

    def choose_move(self, state: GameState) -> int:
        t0 = time.perf_counter()
        nodes = 0

        moves = state.board.valid_moves()
        if not moves:
            self.last_info = {"time_ms": 1, "nodes": 0, "depth": 1}
            return 0

        me = state.current
        opp = other(me)

        # 1) win now
        m = self._winning_move(state, me)
        nodes += len(moves)
        if m is not None:
            self.last_info = {
                "time_ms": max(1, int((time.perf_counter() - t0) * 1000)),
                "nodes": nodes,
                "depth": 1,
            }
            return m

        # 2) block opponent win
        m = self._winning_move(state, opp)
        nodes += len(moves)
        if m is not None:
            self.last_info = {
                "time_ms": max(1, int((time.perf_counter() - t0) * 1000)),
                "nodes": nodes,
                "depth": 1,
            }
            return m

        # 3) center preference (Connect Four is center-favoring)
        if self.center_bias:
            # typical board is 7 columns; center index = 3
            center = 3
            moves_sorted = sorted(moves, key=lambda c: abs(c - center))
            best = moves_sorted[0]
            # small randomness among top few to avoid determinism
            top_k = min(3, len(moves_sorted))
            choice = self.rng.choice(moves_sorted[:top_k])
            self.last_info = {
                "time_ms": max(1, int((time.perf_counter() - t0) * 1000)),
                "nodes": nodes,
                "depth": 1,
            }
            return choice

        # 4) random fallback
        choice = self.rng.choice(moves)
        self.last_info = {
            "time_ms": max(1, int((time.perf_counter() - t0) * 1000)),
            "nodes": nodes,
            "depth": 1,
        }
        return choice
