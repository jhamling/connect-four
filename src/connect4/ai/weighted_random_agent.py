
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

from connect4.core.rules import check_winner
from connect4.game.state import GameState


def other(p: str) -> str:
    return "O" if p == "X" else "X"


def _softmax_sample(rng, moves: list[int], scores: list[float], temperature: float) -> int:
    """
    temperature:
      0   -> argmax (deterministic best score)
      >0  -> softmax sampling; higher = more random
    """
    if not moves:
        return 0

    if temperature <= 0:
        best_i = 0
        best_s = scores[0]
        for i, s in enumerate(scores):
            if s > best_s:
                best_s = s
                best_i = i
        return moves[best_i]

    t = float(temperature)
    m = max(scores)
    exps = [math.exp((s - m) / t) for s in scores]
    total = sum(exps) or 1.0
    r = rng.random() * total
    acc = 0.0
    for mv, e in zip(moves, exps):
        acc += e
        if r <= acc:
            return mv
    return moves[-1]


def _safe_drop_copy(board, col: int, player: str):
    """
    Fast-ish copy: uses Board.copy() if available, else fallback.
    """
    if hasattr(board, "copy"):
        b2 = board.copy()
    else:
        # defensive fallback; should not be needed with your Board.copy()
        import copy as _copy
        b2 = _copy.deepcopy(board)
    b2.drop(col, player)
    return b2


@dataclass
class WeightedRandomAgent:
    """
    WeightedRandom family (parameterized):

    1) Always take immediate win if available.
    2) Otherwise block opponent's immediate win (strength controlled by block_weight).
    3) Otherwise score each move using:
          score = center_weight * center_score + noise
       and sample via:
          - temperature = 0 => deterministic best
          - temperature > 0 => softmax sampling

    Parameters to sweep in your roster:
      - temperature: 0, 25, 75 (typical)
      - center_weight: 1.0, 2.0, 3.0
      - win_weight / block_weight: used to decide how strongly to force win/block
      - noise: tiny randomness to break ties reproducibly (optional)
    """
    name: str = "WeightedRandom"
    seed: int = 0

    temperature: float = 0.0
    center_weight: float = 2.0
    win_weight: float = 10.0
    block_weight: float = 5.0
    noise: float = 0.0

    def __post_init__(self) -> None:
        import random
        self.rng = random.Random(self.seed)
        self.last_info = {}

    def _winning_move(self, state: GameState, player: str) -> Optional[int]:
        moves = state.board.valid_moves()
        for c in moves:
            b2 = _safe_drop_copy(state.board, int(c), player)
            if check_winner(b2) == player:
                return int(c)
        return None

    def choose_move(self, state: GameState) -> int:
        t0 = time.perf_counter()
        nodes = 0

        moves_raw = state.board.valid_moves()
        moves = [int(c) for c in moves_raw]
        if not moves:
            self.last_info = {"time_ms": 1, "nodes": 0, "depth": 1}
            return 0

        me = state.current
        opp = other(me)

        # 1) Immediate win (always force)
        win = self._winning_move(state, me)
        nodes += len(moves)
        if win is not None:
            self.last_info = {
                "time_ms": max(1, int((time.perf_counter() - t0) * 1000)),
                "nodes": nodes,
                "depth": 1,
            }
            return win

        # 2) Immediate block (force if block_weight > 0)
        if self.block_weight > 0:
            block = self._winning_move(state, opp)
            nodes += len(moves)
            if block is not None:
                self.last_info = {
                    "time_ms": max(1, int((time.perf_counter() - t0) * 1000)),
                    "nodes": nodes,
                    "depth": 1,
                }
                return block

        # 3) Otherwise: center preference + (optional) noise, sampled via temperature
        center = (state.board.cols - 1) / 2.0 if hasattr(state.board, "cols") else 3.0

        scores: list[float] = []
        for c in moves:
            # center_score: higher when closer to center
            dist = abs(c - center)
            center_score = max(0.0, 1.0 - (dist / max(1.0, center)))  # ~[0,1]
            s = self.center_weight * center_score

            if self.noise > 0:
                s += self.rng.uniform(-self.noise, self.noise)

            scores.append(s)

        choice = _softmax_sample(self.rng, moves, scores, self.temperature)

        self.last_info = {
            "time_ms": max(1, int((time.perf_counter() - t0) * 1000)),
            "nodes": nodes,
            "depth": 1,
        }
        return int(choice)
