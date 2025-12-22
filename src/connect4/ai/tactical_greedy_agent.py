
from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
import random
import time

from connect4.core.rules import check_winner
from connect4.core.scoring import evaluate
from connect4.game.state import GameState
from connect4.types import Move, Player


def _other(p: Player) -> Player:
    return "O" if p == "X" else "X"


@dataclass(slots=True)
class TacticalGreedyAgent:
    """
    Tactical rules + greedy fallback.
    Knobs:
      - temperature: applies to greedy fallback and (when multiple) blocks/wins selection
      - time_limit_sec: optional budget for scanning/evaluating moves
    """
    name: str = "Tactical Greedy"
    temperature: int = 0
    time_limit_sec: float = 0.0
    rng: random.Random = field(default_factory=random.Random)

    last_info: dict = field(default_factory=dict)

    def choose_move(self, state: GameState) -> Move:
        board = state.board
        me: Player = state.current
        opp: Player = _other(me)

        moves = board.valid_moves()
        if not moves:
            raise ValueError("No valid moves.")

        start = time.perf_counter()
        deadline = start + float(self.time_limit_sec) if self.time_limit_sec and self.time_limit_sec > 0 else None

        center = board.cols // 2
        moves = sorted(moves, key=lambda m: abs(int(m) - center))

        # 1) Immediate winning move(s)
        winning: list[Move] = []
        for m in moves:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            board.drop(m, me)
            w = check_winner(board)
            board.undo(m)
            if w == me:
                winning.append(m)

        if winning:
            # If multiple wins exist, pick among them (temperature not needed here, but randomness is fine)
            chosen = self.rng.choice(winning)
            elapsed = time.perf_counter() - start
            self.last_info = {
                "depth": 1,
                "nodes": 0,
                "tt_hits": 0,
                "cutoffs": 0,
                "eval": 1_000_000,
                "move_col": int(chosen) + 1,
                "time_ms": int(elapsed * 1000),
                "time_limit_ms": int(self.time_limit_sec * 1000),
                "temperature": self.temperature,
                "note": "immediate_win",
            }
            return chosen

        # 2) Immediate block(s) (prevent opponent win next move)
        blocks: list[Move] = []
        for m in moves:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            board.drop(m, opp)
            w = check_winner(board)
            board.undo(m)
            if w == opp:
                blocks.append(m)

        if blocks:
            # If multiple blocks, optionally apply heuristic to pick "better" blocks using temperature
            scored_blocks: list[tuple[Move, float]] = []
            best = -inf
            nodes = 0
            for m in blocks:
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                board.drop(m, me)
                s = float(evaluate(board, me))
                board.undo(m)
                nodes += 1
                scored_blocks.append((m, s))
                if s > best:
                    best = s

            if not scored_blocks:
                chosen = self.rng.choice(blocks)
            else:
                threshold = best - float(self.temperature)
                candidates = [m for (m, s) in scored_blocks if s >= threshold] or [m for (m, s) in scored_blocks if s == best]
                chosen = self.rng.choice(candidates)

            elapsed = time.perf_counter() - start
            self.last_info = {
                "depth": 1,
                "nodes": nodes,
                "tt_hits": 0,
                "cutoffs": 0,
                "eval": int(best) if best not in (inf, -inf) else best,
                "move_col": int(chosen) + 1,
                "time_ms": int(elapsed * 1000),
                "time_limit_ms": int(self.time_limit_sec * 1000),
                "temperature": self.temperature,
                "note": "block",
            }
            return chosen

        # 3) Greedy fallback (1-ply)
        best_score = -inf
        scored: list[tuple[Move, float]] = []
        nodes = 0

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
            scored = [(moves[0], float(evaluate(board, me)))]

        threshold = best_score - float(self.temperature)
        candidates = [m for (m, s) in scored if s >= threshold]
        if not candidates:
            candidates = [m for (m, s) in scored if s == best_score] or [scored[0][0]]

        chosen = self.rng.choice(candidates)

        elapsed = time.perf_counter() - start
        self.last_info = {
            "depth": 1,
            "nodes": nodes,
            "tt_hits": 0,
            "cutoffs": 0,
            "eval": int(best_score) if best_score not in (inf, -inf) else best_score,
            "move_col": int(chosen) + 1,
            "time_ms": int(elapsed * 1000),
            "time_limit_ms": int(self.time_limit_sec * 1000),
            "temperature": self.temperature,
        }
        return chosen

