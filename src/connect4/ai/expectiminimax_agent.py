from __future__ import annotations

from dataclasses import dataclass
from math import inf
import random
import time

from connect4.core.rules import check_winner, is_draw
from connect4.core.scoring import evaluate
from connect4.game.state import GameState
from connect4.types import Move, Player


def _other(p: Player) -> Player:
    return "O" if p == "X" else "X"


@dataclass(slots=True)
class ExpectiMiniMaxAgent:
    name: str = "ExpectiMinimax"
    depth: int = 6
    time_limit_sec: float = 0.20
    p_best: float = 0.80  # opponent plays best response with this probability
    rng: random.Random = random.Random()

    last_info: dict = None

    def choose_move(self, state: GameState) -> Move:
        board = state.board
        me: Player = state.current
        moves = board.valid_moves()
        if not moves:
            raise ValueError("No valid moves.")

        start = time.perf_counter()
        deadline = start + max(0.01, float(self.time_limit_sec))

        center = board.cols // 2
        moves = sorted(moves, key=lambda m: abs(int(m) - center))

        best_move = moves[0]
        best_score = -inf
        nodes = 0

        for m in moves:
            if time.perf_counter() >= deadline:
                break
            board.drop(m, me)
            v, n = self._opp_expect(board, _other(me), self.depth - 1, me, deadline)
            board.undo(m)
            nodes += n

            if v > best_score:
                best_score = v
                best_move = m

        elapsed = time.perf_counter() - start
        self.last_info = {
            "depth": self.depth,
            "nodes": nodes,
            "eval": int(best_score) if best_score not in (inf, -inf) else best_score,
            "move_col": int(best_move) + 1,
            "time_ms": int(elapsed * 1000),
            "time_limit_ms": int(self.time_limit_sec * 1000),
            "p_best": self.p_best,
        }
        return best_move

    def _terminal(self, board, perspective: Player) -> float | None:
        w = check_winner(board)
        if w == perspective:
            return 1_000_000.0
        if w is not None:
            return -1_000_000.0
        if is_draw(board):
            return 0.0
        return None

    def _max_value(self, board, to_play: Player, depth: int, perspective: Player, deadline: float) -> tuple[float, int]:
        if time.perf_counter() >= deadline:
            return float(evaluate(board, perspective)), 1

        term = self._terminal(board, perspective)
        if term is not None:
            return term, 1
        if depth == 0:
            return float(evaluate(board, perspective)), 1

        v = -inf
        nodes = 1
        for m in board.valid_moves():
            if time.perf_counter() >= deadline:
                break
            board.drop(m, to_play)
            child, cn = self._opp_expect(board, _other(to_play), depth - 1, perspective, deadline)
            board.undo(m)
            nodes += cn
            v = max(v, child)

        return v, nodes

    def _opp_expect(self, board, opp: Player, depth: int, perspective: Player, deadline: float) -> tuple[float, int]:
        """
        Opponent node:
        With prob p_best, choose the move that MINIMIZES perspective.
        Otherwise, average the remaining moves.
        """
        if time.perf_counter() >= deadline:
            return float(evaluate(board, perspective)), 1

        term = self._terminal(board, perspective)
        if term is not None:
            return term, 1
        if depth == 0:
            return float(evaluate(board, perspective)), 1

        moves = board.valid_moves()
        if not moves:
            return float(evaluate(board, perspective)), 1

        # Evaluate all opponent replies
        vals: list[tuple[Move, float]] = []
        nodes = 1
        for m in moves:
            if time.perf_counter() >= deadline:
                break
            board.drop(m, opp)
            child, cn = self._max_value(board, _other(opp), depth - 1, perspective, deadline)
            board.undo(m)
            nodes += cn
            vals.append((m, child))

        if not vals:
            return float(evaluate(board, perspective)), nodes

        # Opponent's "best" move minimizes perspective's outcome
        vals.sort(key=lambda t: t[1])
        best_val = vals[0][1]
        rest = [v for (_, v) in vals[1:]]

        if not rest:
            return best_val, nodes

        avg_rest = sum(rest) / len(rest)
        expected = (self.p_best * best_val) + ((1.0 - self.p_best) * avg_rest)
        return expected, nodes
