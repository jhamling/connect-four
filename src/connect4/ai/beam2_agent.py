from __future__ import annotations

import time
from dataclasses import dataclass
from math import inf
from typing import List, Tuple
import random

from connect4.core.rules import check_winner, is_draw
from connect4.game.state import GameState


def other(p: str) -> str:
    return "O" if p == "X" else "X"


@dataclass
class Beam2Agent:
    """
    Beam search with heuristic scoring.
    Uses Board.copy() (fast) instead of deepcopy.

    Knobs:
      - depth
      - beam_width
      - time_limit_sec (optional budget)
      - temperature (root sampling among near-best first moves)
      - win_bonus / loss_penalty
      - root_order: how to order root moves before evaluation
          "heuristic" | "center" | "random"
    """
    name: str = "Beam2"

    depth: int = 4
    beam_width: int = 8

    time_limit_sec: float = 0.0   # 0 => no time limit
    temperature: int = 0          # allow near-best first moves at root

    win_bonus: int = 50_000
    loss_penalty: int = 50_000

    root_order: str = "heuristic"  # "heuristic" | "center" | "random"
    seed: int = 0

    def __post_init__(self) -> None:
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

    def _order_root_moves(self, state: GameState, moves0: List[int], me: str) -> List[int]:
        """
        Applies root_order to moves0:
          - "center": center-first
          - "random": shuffled
          - "heuristic": cheap 1-ply eval ordering (best-first)
        """
        moves = list(moves0)
        cols = state.board.cols
        center = cols // 2

        ro = (self.root_order or "heuristic").lower().strip()

        if ro == "center":
            moves.sort(key=lambda m: abs(int(m) - center))
            return moves

        if ro == "random":
            self.rng.shuffle(moves)
            return moves

        # default: "heuristic"
        scored: List[Tuple[int, int]] = []
        for m in moves:
            b2 = state.board.copy()
            b2.drop(m, me)
            scored.append((int(m), self._evaluate(b2, me)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return [m for (m, _) in scored]

    def choose_move(self, state: GameState) -> int:
        t0 = time.perf_counter()
        deadline = (t0 + self.time_limit_sec) if self.time_limit_sec and self.time_limit_sec > 0 else None
        nodes = 0

        moves0 = state.board.valid_moves()
        if not moves0:
            self.last_info = {"time_ms": 1, "nodes": 0, "depth": 1}
            return 0

        me = state.current

        # Apply root ordering
        moves0 = self._order_root_moves(state, moves0, me)

        # Beam entries: (board, current_player, first_move, score)
        beam: List[Tuple[object, str, int, int]] = []

        for m in moves0:
            if deadline is not None and time.perf_counter() >= deadline:
                break

            b2 = state.board.copy()
            b2.drop(m, me)
            nodes += 1

            w = check_winner(b2)
            if w == me:
                self.last_info = {
                    "time_ms": max(1, int((time.perf_counter() - t0) * 1000)),
                    "nodes": nodes,
                    "depth": 1,
                    "beam_width": self.beam_width,
                    "temperature": self.temperature,
                    "time_limit_ms": int(self.time_limit_sec * 1000),
                    "root_order": self.root_order,
                }
                return int(m)

            sc0 = self._evaluate(b2, me)
            beam.append((b2, other(me), int(m), sc0))

        if not beam:
            choice = int(self.rng.choice(moves0))
            self.last_info = {
                "time_ms": max(1, int((time.perf_counter() - t0) * 1000)),
                "nodes": nodes,
                "depth": 1,
                "beam_width": self.beam_width,
                "temperature": self.temperature,
                "time_limit_ms": int(self.time_limit_sec * 1000),
                "root_order": self.root_order,
            }
            return choice

        beam.sort(key=lambda x: x[3], reverse=True)
        beam = beam[: self.beam_width]

        max_depth_seen = 1

        for ply in range(1, self.depth):
            if deadline is not None and time.perf_counter() >= deadline:
                break

            new_beam: List[Tuple[object, str, int, int]] = []

            for (b, cur, first_move, _) in beam:
                if deadline is not None and time.perf_counter() >= deadline:
                    break

                w = check_winner(b)
                if w is not None or is_draw(b):
                    new_beam.append((b, cur, first_move, self._evaluate(b, me)))
                    continue

                for m in b.valid_moves():
                    if deadline is not None and time.perf_counter() >= deadline:
                        break

                    b2 = b.copy()
                    b2.drop(m, cur)
                    nodes += 1

                    w2 = check_winner(b2)
                    sc = self._evaluate(b2, me)
                    if w2 == other(me):
                        sc -= self.loss_penalty
                    elif w2 == me:
                        sc += self.win_bonus

                    new_beam.append((b2, other(cur), first_move, sc))

            if not new_beam:
                break

            new_beam.sort(key=lambda x: x[3], reverse=True)
            beam = new_beam[: self.beam_width]
            max_depth_seen = ply + 1

        # pick best first move from final beam
        best_score = -inf
        first_scores: dict[int, int] = {}
        for (_, __, first_move, sc) in beam:
            if first_move not in first_scores or sc > first_scores[first_move]:
                first_scores[first_move] = sc
            if sc > best_score:
                best_score = sc

        if not first_scores:
            best_first = int(self.rng.choice(moves0))
        else:
            threshold = best_score - int(self.temperature)
            candidates = [fm for fm, sc in first_scores.items() if sc >= threshold]
            if not candidates:
                candidates = [fm for fm, sc in first_scores.items() if sc == best_score]
            best_first = int(self.rng.choice(candidates))

        self.last_info = {
            "time_ms": max(1, int((time.perf_counter() - t0) * 1000)),
            "nodes": nodes,
            "depth": max_depth_seen,
            "beam_width": self.beam_width,
            "temperature": self.temperature,
            "time_limit_ms": int(self.time_limit_sec * 1000),
            "root_order": self.root_order,
        }
        return best_first

