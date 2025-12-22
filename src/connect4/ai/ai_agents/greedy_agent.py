from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
import random
import time

from connect4.core.scoring import evaluate
from connect4.game.state import GameState
from connect4.types import Move, Player


@dataclass(slots=True)
class GreedyAgent:
    """
    1-ply greedy evaluation with tunable "style" knobs so you can generate many variants
    (analogous to Minimax depth/time/temp).

    Meaningful knobs:
      - temperature: widens candidate set (score >= best - temperature)
      - time_limit_sec: optional scan budget
      - scan_order: affects which moves get evaluated first under tight budgets
          * "center": center-first (strong default)
          * "random": random order (adds diversity)
          * "left": left-to-right
          * "right": right-to-left
      - epsilon: with probability epsilon, pick randomly from valid moves (adds exploration/chaos)
      - eval_jitter: adds small uniform noise to evaluation (breaks ties, adds diversity)
      - seed: deterministic randomness per agent instance

    Notes:
      - This remains 1-ply; "depth" reported in last_info stays 1.
      - Under no time limit, scan_order mostly only affects tie-breaking (still useful).
    """
    name: str = "Greedy (1-ply)"
    seed: int = 0

    temperature: int = 0
    time_limit_sec: float = 0.0

    scan_order: str = "center"   # "center" | "random" | "left" | "right"
    epsilon: float = 0.0         # 0..1
    eval_jitter: float = 0.0     # >=0 (small like 0.0..2.0)

    rng: random.Random = field(init=False)
    last_info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.last_info = {}

        if self.temperature < 0:
            self.temperature = 0
        if self.time_limit_sec < 0:
            self.time_limit_sec = 0.0
        if self.eval_jitter < 0:
            self.eval_jitter = 0.0
        if self.epsilon < 0.0:
            self.epsilon = 0.0
        if self.epsilon > 1.0:
            self.epsilon = 1.0

        if self.scan_order not in {"center", "random", "left", "right"}:
            self.scan_order = "center"

    def _ordered_moves(self, moves: list[Move], cols: int) -> list[Move]:
        if self.scan_order == "random":
            ms = moves[:]
            self.rng.shuffle(ms)
            return ms
        if self.scan_order == "left":
            return sorted(moves, key=lambda m: int(m))
        if self.scan_order == "right":
            return sorted(moves, key=lambda m: -int(m))

        # default: center-first
        center = cols // 2
        return sorted(moves, key=lambda m: abs(int(m) - center))

    def choose_move(self, state: GameState) -> Move:
        board = state.board
        me: Player = state.current

        moves = board.valid_moves()
        if not moves:
            raise ValueError("No valid moves.")

        start = time.perf_counter()
        deadline = start + float(self.time_limit_sec) if self.time_limit_sec and self.time_limit_sec > 0 else None

        # epsilon exploration (cheap diversity + can punish brittle opponents)
        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            choice = self.rng.choice(moves)
            elapsed = time.perf_counter() - start
            self.last_info = {
                "depth": 1,
                "nodes": 0,
                "tt_hits": 0,
                "cutoffs": 0,
                "eval": 0,
                "move_col": int(choice) + 1,
                "time_ms": max(1, int(elapsed * 1000)),
                "time_limit_ms": int(self.time_limit_sec * 1000),
                "temperature": self.temperature,
                "scan_order": self.scan_order,
                "epsilon": self.epsilon,
                "eval_jitter": self.eval_jitter,
                "note": "epsilon_random",
            }
            return choice

        ordered = self._ordered_moves(moves, board.cols)

        best_score = -inf
        scored: list[tuple[Move, float]] = []
        nodes = 0

        for m in ordered:
            if deadline is not None and time.perf_counter() >= deadline:
                break

            board.drop(m, me)
            s = float(evaluate(board, me))
            board.undo(m)

            # optional tiny noise for tie-breaking / diversity
            if self.eval_jitter > 0.0:
                s += self.rng.uniform(-self.eval_jitter, self.eval_jitter)

            nodes += 1
            scored.append((m, s))
            if s > best_score:
                best_score = s

        if not scored:
            # budget too small; fall back to first valid move in current ordering
            choice = ordered[0]
            elapsed = time.perf_counter() - start
            self.last_info = {
                "depth": 1,
                "nodes": nodes,
                "tt_hits": 0,
                "cutoffs": 0,
                "eval": 0,
                "move_col": int(choice) + 1,
                "time_ms": max(1, int(elapsed * 1000)),
                "time_limit_ms": int(self.time_limit_sec * 1000),
                "temperature": self.temperature,
                "scan_order": self.scan_order,
                "epsilon": self.epsilon,
                "eval_jitter": self.eval_jitter,
                "note": "budget_fallback",
            }
            return choice

        threshold = best_score - float(self.temperature)
        candidates = [m for (m, s) in scored if s >= threshold]
        if not candidates:
            candidates = [m for (m, s) in scored if s == best_score] or [scored[0][0]]

        choice = self.rng.choice(candidates)

        elapsed = time.perf_counter() - start
        self.last_info = {
            "depth": 1,
            "nodes": nodes,
            "tt_hits": 0,
            "cutoffs": 0,
            "eval": int(best_score) if best_score not in (inf, -inf) else best_score,
            "move_col": int(choice) + 1,
            "time_ms": max(1, int(elapsed * 1000)),
            "time_limit_ms": int(self.time_limit_sec * 1000),
            "temperature": self.temperature,
            "scan_order": self.scan_order,
            "epsilon": self.epsilon,
            "eval_jitter": self.eval_jitter,
        }
        return choice

