from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
import random
import time

from connect4.ai.tt import TranspositionTable
from connect4.config import AI_TIME_LIMIT_SEC
from connect4.core.rules import check_winner, is_draw
from connect4.core.scoring import evaluate
from connect4.core.zobrist import side_key
from connect4.game.state import GameState
from connect4.types import Move, Player


def _other(p: Player) -> Player:
    return "O" if p == "X" else "X"


def _ordered_moves(board, to_play: Player) -> list[Move]:
    """
    Tactical move ordering:
    1) Immediate winning moves
    2) Immediate blocks
    3) Center-first fallback
    """
    moves = board.valid_moves()
    if not moves:
        return moves

    center = board.cols // 2
    opp = _other(to_play)

    winning = []
    blocking = []
    rest = []

    for m in moves:
        # Check winning move
        board.drop(m, to_play)
        if check_winner(board) == to_play:
            board.undo(m)
            return [m]  # forced win → best possible ordering
        board.undo(m)

        # Check blocking move
        board.drop(m, opp)
        blocks = check_winner(board) == opp
        board.undo(m)

        if blocks:
            blocking.append(m)
        else:
            rest.append(m)

    # Center-first for non-tactical moves
    rest.sort(key=lambda m: abs(int(m) - center))
    blocking.sort(key=lambda m: abs(int(m) - center))

    return blocking + rest


@dataclass(slots=True)
class MinimaxAgent:
    name: str = "Minimax AI"
    depth: int = 7
    time_limit_sec: float = AI_TIME_LIMIT_SEC
    temperature: int = 0
    tt: TranspositionTable = field(default_factory=TranspositionTable)
    rng: random.Random = field(default_factory=random.Random)

    # Stats
    last_info: dict = field(default_factory=dict)

    _nodes: int = 0
    _tt_hits: int = 0
    _cutoffs: int = 0

    def choose_move(self, state: GameState) -> Move:
        board = state.board
        me: Player = state.current

        moves = board.valid_moves()
        if not moves:
            raise ValueError("No valid moves.")

        center = board.cols // 2
        root_moves = sorted(moves, key=lambda m: abs(int(m) - center))

        start = time.perf_counter()
        deadline = start + max(0.01, float(self.time_limit_sec))

        best_move = root_moves[0]
        best_score = -inf
        best_depth_reached = 0

        # Reset TT per move (bounded memory for league)
        # self.tt = TranspositionTable()

        for d in range(1, self.depth + 1):
            if time.perf_counter() >= deadline:
                break

            self._nodes = 0
            self._tt_hits = 0
            self._cutoffs = 0

            alpha = -inf
            beta = inf

            current_best_score = -inf
            scored_moves: list[tuple[Move, float]] = []

            for m in root_moves:
                if time.perf_counter() >= deadline:
                    break

                board.drop(m, me)
                score = self._min_value(board, _other(me), d - 1, alpha, beta, me, deadline)
                board.undo(m)

                scored_moves.append((m, score))
                current_best_score = max(current_best_score, score)
                alpha = max(alpha, current_best_score)

            if scored_moves:
                threshold = current_best_score - float(self.temperature)
                candidates = [m for (m, s) in scored_moves if s >= threshold]
                if not candidates:
                    candidates = [m for (m, s) in scored_moves if s == current_best_score]
                best_move = self.rng.choice(candidates)
                best_score = current_best_score
                best_depth_reached = d

        elapsed = time.perf_counter() - start
        self.last_info = {
            "depth": best_depth_reached,
            "nodes": self._nodes,
            "tt_hits": self._tt_hits,
            "cutoffs": self._cutoffs,
            "eval": int(best_score) if best_score not in (inf, -inf) else best_score,
            "move_col": int(best_move) + 1,
            "time_ms": max(1, int(elapsed * 1000)),
            "time_limit_ms": int(self.time_limit_sec * 1000),
            "temperature": self.temperature,
        }

        return best_move

    def _terminal_score(self, board, perspective: Player) -> float | None:
        w = check_winner(board)
        if w == perspective:
            return 1_000_000.0
        if w is not None:
            return -1_000_000.0
        if is_draw(board):
            return 0.0
        return None

    def _max_value(self, board, to_play: Player, depth: int, alpha: float, beta: float, perspective: Player, deadline: float) -> float:
        if time.perf_counter() >= deadline:
            return float(evaluate(board, perspective))

        self._nodes += 1

        term = self._terminal_score(board, perspective)
        if term is not None:
            return term
        if depth == 0:
            return float(evaluate(board, perspective))

        h = board.zhash ^ side_key(to_play)
        cached = self.tt.get(h, to_play, depth)
        if cached is not None:
            self._tt_hits += 1
            return cached

        v = -inf
        for m in _ordered_moves(board, to_play):
            if time.perf_counter() >= deadline:
                break

            board.drop(m, to_play)
            v = max(v, self._min_value(board, _other(to_play), depth - 1, alpha, beta, perspective, deadline))
            board.undo(m)

            if v >= beta:
                self._cutoffs += 1
                break
            alpha = max(alpha, v)

        self.tt.put(h, to_play, depth, v)
        return v

    def _min_value(self, board, to_play: Player, depth: int, alpha: float, beta: float, perspective: Player, deadline: float) -> float:
        if time.perf_counter() >= deadline:
            return float(evaluate(board, perspective))

        self._nodes += 1

        term = self._terminal_score(board, perspective)
        if term is not None:
            return term
        if depth == 0:
            return float(evaluate(board, perspective))

        h = board.zhash ^ side_key(to_play)
        cached = self.tt.get(h, to_play, depth)
        if cached is not None:
            self._tt_hits += 1
            return cached

        v = inf
        for m in _ordered_moves(board, to_play):
            if time.perf_counter() >= deadline:
                break

            board.drop(m, to_play)
            v = min(v, self._max_value(board, _other(to_play), depth - 1, alpha, beta, perspective, deadline))
            board.undo(m)

            if v <= alpha:
                self._cutoffs += 1
                break
            beta = min(beta, v)

        self.tt.put(h, to_play, depth, v)
        return v

