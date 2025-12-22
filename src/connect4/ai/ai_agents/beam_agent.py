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

    Purpose: create *many distinct variants* via knobs, similar to MM d/t/temp.

    Knobs that matter (and are safe to sweep):
      - depth: search depth cap
      - width: beam width (top-k branching at each ply)
      - time_limit_sec: per-move budget
      - temperature: near-best selection at the ROOT (>= best - temperature)
      - root_order: ordering used for initial root scoring ("center", "heuristic", "random")
      - beam_policy: how we pick top-k per node ("heuristic", "static_center", "random")
      - seed: deterministic randomness (affects random ordering + tie breaks)

    Notes:
      - Uses your evaluate() for ordering and leaf evaluation.
      - Preserves last_info fields used by your league script.
    """
    name: str = "Beam Search"

    depth: int = 6
    width: int = 3
    time_limit_sec: float = 0.15

    temperature: float = 0.0
    root_order: str = "heuristic"   # "heuristic" | "center" | "random"
    beam_policy: str = "heuristic"  # "heuristic" | "static_center" | "random"

    seed: int = 0
    rng: random.Random = field(init=False)

    last_info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

        if self.depth < 1:
            self.depth = 1
        if self.width < 1:
            self.width = 1
        if self.time_limit_sec < 0:
            self.time_limit_sec = 0.0
        if self.temperature < 0.0:
            self.temperature = 0.0

        if self.root_order not in {"heuristic", "center", "random"}:
            self.root_order = "heuristic"
        if self.beam_policy not in {"heuristic", "static_center", "random"}:
            self.beam_policy = "heuristic"

    # ------------------------
    # Core move selection
    # ------------------------
    def choose_move(self, state: GameState) -> Move:
        board = state.board
        me: Player = state.current
        moves = board.valid_moves()
        if not moves:
            raise ValueError("No valid moves.")

        start = time.perf_counter()
        deadline = start + max(0.01, float(self.time_limit_sec))

        # Root ordering / prefilter
        root_moves = self._order_root_moves(board, moves, me, deadline)
        if not root_moves:
            root_moves = moves[:]

        # Beam at root: take top-k as candidates
        root_candidates = root_moves[: max(1, self.width)]

        # Deep evaluate each root candidate with beam-limited alpha-beta
        best_score = -inf
        root_scored_deep: list[tuple[Move, float]] = []
        nodes = 0
        cutoffs = 0

        alpha = -inf
        beta = inf

        for m in root_candidates:
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
            # fallback to first candidate with shallow heuristic
            m0 = root_candidates[0] if root_candidates else moves[0]
            board.drop(m0, me)
            v0 = float(evaluate(board, me))
            board.undo(m0)
            root_scored_deep = [(m0, v0)]
            best_score = v0

        # Temperature sampling at root among near-best
        best_move = self._pick_root_with_temperature(root_scored_deep, best_score)

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
            "root_order": self.root_order,
            "beam_policy": self.beam_policy,
        }
        return best_move

    # ------------------------
    # Helpers
    # ------------------------
    def _terminal(self, board, perspective: Player) -> float | None:
        w = check_winner(board)
        if w == perspective:
            return 1_000_000.0
        if w is not None:
            return -1_000_000.0
        if is_draw(board):
            return 0.0
        return None

    def _pick_root_with_temperature(self, scored: list[tuple[Move, float]], best_score: float) -> Move:
        if self.temperature <= 0.0:
            best_moves = [m for (m, s) in scored if s == best_score]
            return self.rng.choice(best_moves) if best_moves else scored[0][0]

        threshold = best_score - float(self.temperature)
        candidates = [m for (m, s) in scored if s >= threshold]
        if not candidates:
            candidates = [m for (m, s) in scored if s == best_score] or [scored[0][0]]
        return self.rng.choice(candidates)

    def _order_root_moves(self, board, moves: list[Move], me: Player, deadline: float) -> list[Move]:
        """
        Root move ordering affects strength under time limits.
        """
        ms = moves[:]

        if self.root_order == "random":
            self.rng.shuffle(ms)
            return ms

        if self.root_order == "center":
            center = board.cols // 2
            return sorted(ms, key=lambda m: abs(int(m) - center))

        # heuristic order (default): evaluate after one move
        scored: list[tuple[Move, float]] = []
        for m in ms:
            if time.perf_counter() >= deadline:
                break
            board.drop(m, me)
            s = float(evaluate(board, me))
            board.undo(m)
            scored.append((m, s))

        if not scored:
            return ms

        scored.sort(key=lambda t: t[1], reverse=True)
        return [m for (m, _) in scored]

    def _topk_moves(self, board, player: Player, k: int, perspective: Player, deadline: float) -> list[Move]:
        """
        Beam policy determines how we choose the next layer's top-k moves.
        """
        moves = board.valid_moves()
        if len(moves) <= k:
            return moves

        if self.beam_policy == "random":
            ms = moves[:]
            self.rng.shuffle(ms)
            return ms[:k]

        if self.beam_policy == "static_center":
            center = board.cols // 2
            ms = sorted(moves, key=lambda m: abs(int(m) - center))
            return ms[:k]

        # heuristic beam (default): evaluate leaf score to order moves
        scored: list[tuple[Move, float]] = []
        for m in moves:
            if time.perf_counter() >= deadline:
                break
            board.drop(m, player)
            s = float(evaluate(board, perspective))
            board.undo(m)
            scored.append((m, s))

        if not scored:
            return moves[:k]

        # If it's perspective's turn, larger is better; otherwise smaller is better
        scored.sort(key=lambda t: t[1], reverse=(player == perspective))
        return [m for (m, _) in scored[:k]]

    # ------------------------
    # Beam-limited alpha-beta
    # ------------------------
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

        for m in self._topk_moves(board, to_play, self.width, perspective, deadline):
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

        for m in self._topk_moves(board, to_play, self.width, perspective, deadline):
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

