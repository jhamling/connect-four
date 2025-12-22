
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from connect4.core.rules import check_winner
from connect4.game.state import GameState


def other(p: str) -> str:
    return "O" if p == "X" else "X"


@dataclass
class HeuristicAgent:
    """
    1-ply agent:
      - immediate win
      - immediate block
      - otherwise choose move maximizing heuristic score

    Uses board.drop + board.undo (fast, no deepcopy).
    """
    name: str = "Heuristic"
    seed: int = 0

    center_weight: int = 4
    two_weight: int = 2
    three_weight: int = 8
    opp_three_penalty: int = 9

    def __post_init__(self) -> None:
        import random
        self.rng = random.Random(self.seed)
        self.last_info = {}

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
            score += 10_000
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
                score += self._window_score([grid[r][c], grid[r + 1][c + 1], grid[r + 2][c + 2], grid[r + 3][c + 3]], me)

        # diag up-right
        for r in range(3, rows):
            for c in range(cols - 3):
                score += self._window_score([grid[r][c], grid[r - 1][c + 1], grid[r - 2][c + 2], grid[r - 3][c + 3]], me)

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

        # win now
        m = self._winning_move(state, me)
        nodes += len(moves)
        if m is not None:
            self.last_info = {"time_ms": max(1, int((time.perf_counter() - t0) * 1000)), "nodes": nodes, "depth": 1}
            return m

        # block now
        m = self._winning_move(state, opp)
        nodes += len(moves)
        if m is not None:
            self.last_info = {"time_ms": max(1, int((time.perf_counter() - t0) * 1000)), "nodes": nodes, "depth": 1}
            return m

        # evaluate each candidate (1 ply) using drop/undo
        best_score = -10**18
        best_moves: List[int] = []

        for col in moves:
            b.drop(col, me)
            nodes += 1
            s = self._evaluate_board(b, me)
            b.undo(col)

            c_int = int(col)
            if s > best_score:
                best_score = s
                best_moves = [c_int]
            elif s == best_score:
                best_moves.append(c_int)

        choice = self.rng.choice(best_moves) if best_moves else int(self.rng.choice(moves))
        self.last_info = {"time_ms": max(1, int((time.perf_counter() - t0) * 1000)), "nodes": nodes, "depth": 1}
        return choice
