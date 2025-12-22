
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
    1-ply greedy evaluation.
    Knobs:
      - temperature: allow near-best moves (score >= best - temperature), choose randomly among them
      - time_limit_sec: stop scanning moves if time expires (useful for symmetry w/ other agents)
    """
    name: str = "Greedy (1-ply)"
    temperature: int = 0
    time_limit_sec: float = 0.0
    rng: random.Random = field(default_factory=random.Random)

    last_info: dict = field(default_factory=dict)

    def choose_move(self, state: GameState) -> Move:
        board = state.board
        me: Player = state.current

        moves = board.valid_moves()
        if not moves:
            raise ValueError("No valid moves.")

        start = time.perf_counter()
        deadline = start + float(self.time_limit_sec) if self.time_limit_sec and self.time_limit_sec > 0 else None

        best_score = -inf
        scored: list[tuple[Move, float]] = []
        nodes = 0

        # Center-first order gives slightly better play under time limits
        center = board.cols // 2
        moves = sorted(moves, key=lambda m: abs(int(m) - center))

        for m in moves:
            if deadline is not None and time.perf_counter() >= deadline:
                break

            board.drop(m, me)
            s = float(evaluate(board, me))
            board.undo(m)

            nodes += 1
            scored.append((m, s))
            if s > best_score:
                best_score = s

        if not scored:
            # Extremely defensive fallback (should be rare)
            scored = [(moves[0], float(evaluate(board, me)))]

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
        }
        return choice

