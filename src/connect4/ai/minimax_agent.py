
# src/connect4/ai/minimax_agent.py

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


@dataclass(slots=True)
class MinimaxAgent:
    name: str = "Minimax AI"
    depth: int = 7                 # max depth cap (iterative deepening goes up to this)
    time_limit_sec: float = AI_TIME_LIMIT_SEC
    temperature: int = 0           # 0 = best-only; >0 = choose among near-best within this eval margin
    tt: TranspositionTable = field(default_factory=TranspositionTable)
    rng: random.Random = field(default_factory=random.Random)

    # Stats for UI / benchmarking
    last_info: dict = field(default_factory=dict)

    # Internal counters per iteration
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

        # Reset TT per real move (keeps memory bounded for tournaments)
        self.tt = TranspositionTable()

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
                if score > current_best_score:
                    current_best_score = score

                alpha = max(alpha, current_best_score)

            if scored_moves:
                # Temperature sampling at root:
                # keep moves within (best - temperature), pick randomly among them
                threshold = current_best_score - float(self.temperature)
                candidates = [m for (m, s) in scored_moves if s >= threshold]

                # If something odd happens, fall back to strict best
                if not candidates:
                    best_only = [m for (m, s) in scored_moves if s == current_best_score]
                    candidates = best_only if best_only else [scored_moves[0][0]]

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
            "time_ms": int(elapsed * 1000),
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
        moves = board.valid_moves()
        center = board.cols // 2
        moves = sorted(moves, key=lambda m: abs(int(m) - center))

        for m in moves:
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
        moves = board.valid_moves()
        center = board.cols // 2
        moves = sorted(moves, key=lambda m: abs(int(m) - center))

        for m in moves:
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
