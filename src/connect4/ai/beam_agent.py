
from __future__ import annotations

from dataclasses import dataclass, field
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
class BeamAgent:
    """
    Width-limited minimax (beam search).
    Knobs:
      - depth
      - width
      - time_limit_sec
      - temperature (root sampling): choose among near-best root candidates within (best - temperature)
    """
    name: str = "Beam Search"
    depth: int = 6
    width: int = 3
    time_limit_sec: float = 0.15
    temperature: int = 0
    rng: random.Random = field(default_factory=random.Random)

    last_info: dict = field(default_factory=dict)

    def choose_move(self, state: GameState) -> Move:
        board = state.board
        me: Player = state.current
        moves = board.valid_moves()
        if not moves:
            raise ValueError("No valid moves.")

        start = time.perf_counter()
        deadline = start + max(0.01, float(self.time_limit_sec))

        # Order root moves with a cheap heuristic
        scored: list[tuple[Move, float]] = []
        nodes = 0
        for m in moves:
            if time.perf_counter() >= deadline:
                break
            board.drop(m, me)
            s = float(evaluate(board, me))
            board.undo(m)
            nodes += 1
            scored.append((m, s))

        if not scored:
            scored = [(moves[0], float(evaluate(board, me)))]

        scored.sort(key=lambda t: t[1], reverse=True)
        root_candidates = scored[: max(1, self.width)]

        # Evaluate root candidates with beam-limited alpha-beta
        best_score = -inf
        root_scored_deep: list[tuple[Move, float]] = []
        cutoffs = 0

        alpha = -inf
        beta = inf

        for (m, _) in root_candidates:
            if time.perf_counter() >= deadline:
                break
            board.drop(m, me)
            v, dn, dc = self._min_value(board, _other(me), self.depth - 1, alpha, beta, me, deadline)
            board.undo(m)
            nodes += dn
            cutoffs += dc

            root_scored_deep.append((m, v))
            if v > best_score:
                best_score = v
            alpha = max(alpha, best_score)

        if not root_scored_deep:
            root_scored_deep = [(root_candidates[0][0], root_candidates[0][1])]
            best_score = root_scored_deep[0][1]

        # Temperature sampling at root
        threshold = best_score - float(self.temperature)
        candidates = [m for (m, s) in root_scored_deep if s >= threshold]
        if not candidates:
            candidates = [m for (m, s) in root_scored_deep if s == best_score] or [root_scored_deep[0][0]]

        best_move = self.rng.choice(candidates)

        elapsed = time.perf_counter() - start
        self.last_info = {
            "depth": self.depth,
            "nodes": nodes,
            "tt_hits": 0,
            "cutoffs": cutoffs,
            "eval": int(best_score) if best_score not in (inf, -inf) else best_score,
            "move_col": int(best_move) + 1,
            "time_ms": max(1, int(elapsed * 1000)),
            "time_limit_ms": int(self.time_limit_sec * 1000),
            "beam_width": self.width,
            "temperature": self.temperature,
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

    def _topk_moves(self, board, player: Player, k: int, perspective: Player) -> list[Move]:
        moves = board.valid_moves()
        if len(moves) <= k:
            return moves

        scored: list[tuple[Move, float]] = []
        for m in moves:
            board.drop(m, player)
            s = float(evaluate(board, perspective))
            board.undo(m)
            scored.append((m, s))

        scored.sort(key=lambda t: t[1], reverse=(player == perspective))
        return [m for (m, _) in scored[:k]]

    def _max_value(
        self,
        board,
        to_play: Player,
        depth: int,
        alpha: float,
        beta: float,
        perspective: Player,
        deadline: float,
    ) -> tuple[float, int, int]:
        if time.perf_counter() >= deadline:
            return float(evaluate(board, perspective)), 1, 0

        term = self._terminal(board, perspective)
        if term is not None:
            return term, 1, 0
        if depth == 0:
            return float(evaluate(board, perspective)), 1, 0

        v = -inf
        nodes = 1
        cutoffs = 0

        for m in self._topk_moves(board, to_play, self.width, perspective):
            if time.perf_counter() >= deadline:
                break
            board.drop(m, to_play)
            child, cn, cc = self._min_value(board, _other(to_play), depth - 1, alpha, beta, perspective, deadline)
            board.undo(m)
            nodes += cn
            cutoffs += cc

            if child > v:
                v = child
            if v >= beta:
                cutoffs += 1
                break
            alpha = max(alpha, v)

        return v, nodes, cutoffs

    def _min_value(
        self,
        board,
        to_play: Player,
        depth: int,
        alpha: float,
        beta: float,
        perspective: Player,
        deadline: float,
    ) -> tuple[float, int, int]:
        if time.perf_counter() >= deadline:
            return float(evaluate(board, perspective)), 1, 0

        term = self._terminal(board, perspective)
        if term is not None:
            return term, 1, 0
        if depth == 0:
            return float(evaluate(board, perspective)), 1, 0

        v = inf
        nodes = 1
        cutoffs = 0

        for m in self._topk_moves(board, to_play, self.width, perspective):
            if time.perf_counter() >= deadline:
                break
            board.drop(m, to_play)
            child, cn, cc = self._max_value(board, _other(to_play), depth - 1, alpha, beta, perspective, deadline)
            board.undo(m)
            nodes += cn
            cutoffs += cc

            if child < v:
                v = child
            if v <= alpha:
                cutoffs += 1
                break
            beta = min(beta, v)

        return v, nodes, cutoffs

