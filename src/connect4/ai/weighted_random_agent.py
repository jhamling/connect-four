
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Optional

from connect4.core.rules import check_winner
from connect4.game.state import GameState


def other(p: str) -> str:
    return "O" if p == "X" else "X"


@dataclass
class WeightedRandomAgent:
    """
    Random agent with:
      - immediate win
      - immediate block
      - otherwise weighted toward center columns
    """
    name: str = "WeightedRandom"
    seed: int = 0

    def __post_init__(self) -> None:
        import random
        self.rng = random.Random(self.seed)
        self.last_info = {}

    def _winning_move(self, state: GameState, player: str) -> Optional[int]:
        for c in state.board.valid_moves():
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

        # win / block
        m = self._winning_move(state, me)
        nodes += len(moves)
        if m is not None:
            self.last_info = {"time_ms": max(1, int((time.perf_counter() - t0) * 1000)), "nodes": nodes, "depth": 1}
            return m

        m = self._winning_move(state, opp)
        nodes += len(moves)
        if m is not None:
            self.last_info = {"time_ms": max(1, int((time.perf_counter() - t0) * 1000)), "nodes": nodes, "depth": 1}
            return m

        # center weights (assumes 7 columns)
        center = 3
        weights = []
        for c in moves:
            # closer to center => higher weight
            w = 10 - abs(c - center) * 2
            weights.append(max(1, w))

        choice = self.rng.choices(moves, weights=weights, k=1)[0]
        self.last_info = {"time_ms": max(1, int((time.perf_counter() - t0) * 1000)), "nodes": nodes, "depth": 1}
        return choice
