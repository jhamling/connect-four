from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from connect4.core.rules import check_winner
from connect4.game.state import GameState


def other(p: str) -> str:
    return "O" if p == "X" else "X"


@dataclass
class HeuristicAgent:
    """
    1-ply heuristic agent with tunable evaluation weights + selection randomness.

    Knobs to generate meaningful variants:
      - center_weight, two_weight, three_weight, opp_three_penalty: eval weights
      - win_value: score for a 4-in-a-row window (kept large; still a knob)
      - temperature: best_score - temperature threshold for candidate set
      - time_limit_sec: optional budget (useful if you later add more expensive eval)
      - seed

    Notes:
      - Uses drop/undo (fast) and only does 1-ply eval.
      - Temperature here acts like your MM temperature: it widens the set of "good enough"
        moves to sample from (more diverse / less deterministic).
    """
    name: str = "Heuristic"
    seed: int = 0

    # evaluation weights
    center_weight: int = 4
    two_weight: int = 2
    three_weight: int = 8
    opp_three_penalty: int = 9
    win_value: int = 10_000

    # selection / runtime knobs
    temperature: int = 0
    time_limit_sec: float = 0.0

    rng: object = field(init=False)
    last_info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        import random

        self.rng = random.Random(self.seed)
        self.last_info = {}

        if self.temperature < 0:
            self.temperature = 0
        if self.time_limit_sec < 0:
            self.time_limit_sec = 0.0
        if self.win_value < 1000:
            # keep wins overwhelmingly dominant
            self.win_value = 1000

    # ----- evaluation -----
    def _count_center(self, grid, player: str) -> int:
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        center_c = cols // 2
        return sum(1 for r in range(rows) if grid[r][center_c] == player)

    def _window_score(self, window: List[object], player: str) -> int:
        opp = other(player)
        p = window.count(player)
        o = window.count(opp)
        e = window.count(None)

        score = 0
        if p == 4:
            score += self.win_value
        elif p == 3 and e == 1:
            score += self.three_weight
        elif p == 2 and e == 2:
            score += self.two_weight

        if o == 3 and e == 1:
            score -= self.opp_three_penalty

        return score

    def _evaluate_board(self, board, me: str) -> int:
        grid = board.grid
        rows = board.rows
        cols = board.cols

        score = 0
        score += self.center_weight * self._count_center(grid, me)

        # horizontal
        for r in range(rows):
            row = grid[r]
            for c in range(cols - 3):
                score += self._window_score([row[c], row[c + 1], row[c + 2], row[c + 3]], me)

        # vertical
        for c in range(cols):
            for r in range(rows - 3):
                score += self._window_score([grid[r][c], grid[r + 1][c], grid[r + 2][c], grid[r + 3][c]], me)

        # diag down-right
        for r in range(rows - 3):
            for c in range(cols - 3):
                score += self._window_score(
                    [grid[r][c], grid[r + 1][c + 1], grid[r + 2][c + 2], grid[r + 3][c + 3]],
                    me,
                )

        # diag up-right
        for r in range(3, rows):
            for c in range(cols - 3):
                score += self._window_score(
                    [grid[r][c], grid[r - 1][c + 1], grid[r - 2][c + 2], grid[r - 3][c + 3]],
                    me,
                )

        return score

    # ----- tactical checks -----
    def _winning_move(self, state: GameState, player: str) -> Optional[int]:
        b = state.board
        for col in b.valid_moves():
            b.drop(col, player)
            w = check_winner(b)
            b.undo(col)
            if w == player:
                return int(col)
        return None

    def choose_move(self, state: GameState) -> int:
        t0 = time.perf_counter()
        nodes = 0

        b = state.board
        moves = b.valid_moves()
        if not moves:
            self.last_info = {"time_ms": 1, "nodes": 0, "depth": 1}
            return 0

        me = state.current
        opp = other(me)

        deadline = None
        if self.time_limit_sec and self.time_limit_sec > 0:
            deadline = t0 + float(self.time_limit_sec)

        # win now
        m = self._winning_move(state, me)
        nodes += len(moves)
        if m is not None:
            self.last_info = {
                "time_ms": max(1, int((time.perf_counter() - t0) * 1000)),
                "nodes": nodes,
                "depth": 1,
                "temperature": self.temperature,
                "note": "immediate_win",
            }
            return m

        # block now
        m = self._winning_move(state, opp)
        nodes += len(moves)
        if m is not None:
            self.last_info = {
                "time_ms": max(1, int((time.perf_counter() - t0) * 1000)),
                "nodes": nodes,
                "depth": 1,
                "temperature": self.temperature,
                "note": "block",
            }
            return m

        # evaluate each candidate (1 ply) using drop/undo
        best_score = -10**18
        scored: List[tuple[int, int]] = []

        for col in moves:
            if deadline is not None and time.perf_counter() >= deadline:
                break

            b.drop(col, me)
            nodes += 1
            s = self._evaluate_board(b, me)
            b.undo(col)

            c_int = int(col)
            scored.append((c_int, s))
            if s > best_score:
                best_score = s

        if not scored:
            # budget too small; fall back to random
            choice = int(self.rng.choice(moves))
            self.last_info = {
                "time_ms": max(1, int((time.perf_counter() - t0) * 1000)),
                "nodes": nodes,
                "depth": 1,
                "temperature": self.temperature,
                "note": "budget_fallback",
            }
            return choice

        # Temperature widening: pick among moves within (best - temperature)
        threshold = best_score - int(self.temperature)
        candidates = [c for (c, s) in scored if s >= threshold]
        if not candidates:
            candidates = [c for (c, s) in scored if s == best_score]

        choice = self.rng.choice(candidates)

        self.last_info = {
            "time_ms": max(1, int((time.perf_counter() - t0) * 1000)),
            "nodes": nodes,
            "depth": 1,
            "temperature": self.temperature,
            "best_score": int(best_score),
            "note": "heuristic_1ply",
        }
        return choice
