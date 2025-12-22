
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Tuple

from connect4.core.rules import check_winner, is_draw
from connect4.game.state import GameState


def other(p: str) -> str:
    return "O" if p == "X" else "X"


@dataclass
class Beam2Agent:
    """
    Beam search with heuristic scoring.
    Uses Board.copy() (fast) instead of deepcopy.

    depth: number of plies
    beam_width: keep top K nodes each ply
    """
    name: str = "Beam2"
    depth: int = 4
    beam_width: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        import random
        self.rng = random.Random(self.seed)
        self.last_info = {}

    def _evaluate(self, board, me: str) -> int:
        grid = board.grid
        rows = board.rows
        cols = board.cols

        center = cols // 2
        score = 0
        for r in range(rows):
            if grid[r][center] == me:
                score += 3

        opp = other(me)

        def window_score(w: List[object]) -> int:
            p = w.count(me)
            o = w.count(opp)
            e = w.count(None)

            if o == 0:
                if p == 3 and e == 1:
                    return 20
                if p == 2 and e == 2:
                    return 6
                if p == 1 and e == 3:
                    return 1
            if p == 0:
                if o == 3 and e == 1:
                    return -24
                if o == 2 and e == 2:
                    return -7
            return 0

        # horizontal
        for r in range(rows):
            row = grid[r]
            for c in range(cols - 3):
                score += window_score([row[c], row[c + 1], row[c + 2], row[c + 3]])

        # vertical
        for c in range(cols):
            for r in range(rows - 3):
                score += window_score([grid[r][c], grid[r + 1][c], grid[r + 2][c], grid[r + 3][c]])

        # diag down-right
        for r in range(rows - 3):
            for c in range(cols - 3):
                score += window_score([grid[r][c], grid[r + 1][c + 1], grid[r + 2][c + 2], grid[r + 3][c + 3]])

        # diag up-right
        for r in range(3, rows):
            for c in range(cols - 3):
                score += window_score([grid[r][c], grid[r - 1][c + 1], grid[r - 2][c + 2], grid[r - 3][c + 3]])

        return score

    def choose_move(self, state: GameState) -> int:
        t0 = time.perf_counter()
        nodes = 0

        moves0 = state.board.valid_moves()
        if not moves0:
            self.last_info = {"time_ms": 1, "nodes": 0, "depth": 1}
            return 0

        me = state.current

        # Beam entries: (board, current_player, first_move, score)
        beam: List[Tuple[object, str, int, int]] = []

        for m in moves0:
            b2 = state.board.copy()
            b2.drop(m, me)
            nodes += 1

            w = check_winner(b2)
            if w == me:
                self.last_info = {"time_ms": max(1, int((time.perf_counter() - t0) * 1000)), "nodes": nodes, "depth": 1}
                return int(m)

            beam.append((b2, other(me), int(m), self._evaluate(b2, me)))

        beam.sort(key=lambda x: x[3], reverse=True)
        beam = beam[: self.beam_width]

        max_depth_seen = 1

        for ply in range(1, self.depth):
            new_beam: List[Tuple[object, str, int, int]] = []

            for (b, cur, first_move, _) in beam:
                w = check_winner(b)
                if w is not None or is_draw(b):
                    new_beam.append((b, cur, first_move, self._evaluate(b, me)))
                    continue

                for m in b.valid_moves():
                    b2 = b.copy()
                    b2.drop(m, cur)
                    nodes += 1

                    w2 = check_winner(b2)
                    sc = self._evaluate(b2, me)
                    if w2 == other(me):
                        sc -= 50_000
                    elif w2 == me:
                        sc += 50_000

                    new_beam.append((b2, other(cur), first_move, sc))

            new_beam.sort(key=lambda x: x[3], reverse=True)
            beam = new_beam[: self.beam_width]
            max_depth_seen = ply + 1

        best_first = beam[0][2] if beam else int(self.rng.choice(moves0))
        self.last_info = {
            "time_ms": max(1, int((time.perf_counter() - t0) * 1000)),
            "nodes": nodes,
            "depth": max_depth_seen,
        }
        return best_first
