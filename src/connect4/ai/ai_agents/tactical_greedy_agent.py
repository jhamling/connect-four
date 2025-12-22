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

    Tunable knobs (to generate many distinct variants):
      - temperature:
          Selection "slack" for greedy/block scoring. Higher => more random among near-best.
      - time_limit_sec:
          Optional time budget for scanning/evaluating moves.
      - seed:
          Deterministic RNG seed (important for reproducibility + multiprocessing).
      - center_bias:
          How strongly to prefer center columns when ordering moves.
          0.0 = no center bias, higher = stronger center preference.
      - block_policy:
          How to pick among multiple blocking moves:
            "random"     -> pick random among blocking moves
            "center"     -> pick most-center block (ties random)
            "evaluate"   -> evaluate each block and apply temperature threshold
      - greedy_policy:
          How to pick in greedy fallback:
            "evaluate"   -> evaluate each move and apply temperature threshold
            "center"     -> prefer center-most (still uses immediate win/block rules)
    """
    name: str = "TacticalGreedy"
    temperature: int = 0
    time_limit_sec: float = 0.0

    seed: int = 0
    center_bias: float = 1.0
    block_policy: str = "evaluate"   # "random" | "center" | "evaluate"
    greedy_policy: str = "evaluate"  # "evaluate" | "center"

    rng: random.Random = field(init=False)
    last_info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

        # normalize policies defensively
        if self.block_policy not in ("random", "center", "evaluate"):
            self.block_policy = "evaluate"
        if self.greedy_policy not in ("evaluate", "center"):
            self.greedy_policy = "evaluate"

        if self.center_bias < 0:
            self.center_bias = 0.0

    def _deadline(self) -> float | None:
        if self.time_limit_sec and self.time_limit_sec > 0:
            return time.perf_counter() + float(self.time_limit_sec)
        return None

    def _time_left(self, deadline: float | None) -> bool:
        return deadline is None or time.perf_counter() < deadline

    def _sort_moves(self, moves: list[Move], cols: int) -> list[Move]:
        if self.center_bias <= 0:
            return moves[:]  # no ordering preference

        center = cols // 2

        # higher center_bias => stronger “center first” ordering
        # (we keep it simple: scale the distance)
        return sorted(moves, key=lambda m: abs(int(m) - center) * self.center_bias)

    def choose_move(self, state: GameState) -> Move:
        board = state.board
        me: Player = state.current
        opp: Player = _other(me)

        moves = board.valid_moves()
        if not moves:
            raise ValueError("No valid moves.")

        start = time.perf_counter()
        deadline = self._deadline()

        moves = self._sort_moves(moves, board.cols)

        # 1) Immediate winning moves
        winning: list[Move] = []
        for m in moves:
            if not self._time_left(deadline):
                break
            board.drop(m, me)
            w = check_winner(board)
            board.undo(m)
            if w == me:
                winning.append(m)

        if winning:
            chosen = self.rng.choice(winning)
            elapsed = time.perf_counter() - start
            self.last_info = {
                "depth": 1,
                "nodes": 0,
                "eval": 1_000_000,
                "time_ms": max(1, int(elapsed * 1000)),
                "time_limit_ms": int(self.time_limit_sec * 1000),
                "temperature": self.temperature,
                "seed": self.seed,
                "center_bias": self.center_bias,
                "block_policy": self.block_policy,
                "greedy_policy": self.greedy_policy,
                "note": "immediate_win",
            }
            return chosen

        # 2) Immediate blocks (prevent opponent win next move)
        blocks: list[Move] = []
        for m in moves:
            if not self._time_left(deadline):
                break
            board.drop(m, opp)
            w = check_winner(board)
            board.undo(m)
            if w == opp:
                blocks.append(m)

        if blocks:
            chosen: Move
            nodes = 0
            best = -inf

            if self.block_policy == "random":
                chosen = self.rng.choice(blocks)
            elif self.block_policy == "center":
                # blocks already center-sorted; pick among best-center ties
                best_dist = abs(int(blocks[0]) - (board.cols // 2))
                candidates = [m for m in blocks if abs(int(m) - (board.cols // 2)) == best_dist]
                chosen = self.rng.choice(candidates)
            else:
                # "evaluate": score each blocking move using evaluate(board, me)
                scored_blocks: list[tuple[Move, float]] = []
                for m in blocks:
                    if not self._time_left(deadline):
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
                    candidates = [m for (m, s) in scored_blocks if s >= threshold]
                    if not candidates:
                        candidates = [m for (m, s) in scored_blocks if s == best] or [scored_blocks[0][0]]
                    chosen = self.rng.choice(candidates)

            elapsed = time.perf_counter() - start
            self.last_info = {
                "depth": 1,
                "nodes": nodes,
                "eval": int(best) if best not in (inf, -inf) else best,
                "time_ms": max(1, int(elapsed * 1000)),
                "time_limit_ms": int(self.time_limit_sec * 1000),
                "temperature": self.temperature,
                "seed": self.seed,
                "center_bias": self.center_bias,
                "block_policy": self.block_policy,
                "greedy_policy": self.greedy_policy,
                "note": "block",
            }
            return chosen

        # 3) Greedy fallback
        if self.greedy_policy == "center":
            # already center-ordered; deterministic aside from RNG tie-break (not needed)
            chosen = moves[0]
            elapsed = time.perf_counter() - start
            self.last_info = {
                "depth": 1,
                "nodes": 0,
                "eval": 0,
                "time_ms": max(1, int(elapsed * 1000)),
                "time_limit_ms": int(self.time_limit_sec * 1000),
                "temperature": self.temperature,
                "seed": self.seed,
                "center_bias": self.center_bias,
                "block_policy": self.block_policy,
                "greedy_policy": self.greedy_policy,
                "note": "greedy_center",
            }
            return chosen

        best_score = -inf
        scored: list[tuple[Move, float]] = []
        nodes = 0

        for m in moves:
            if not self._time_left(deadline):
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
            best_score = scored[0][1]

        threshold = best_score - float(self.temperature)
        candidates = [m for (m, s) in scored if s >= threshold]
        if not candidates:
            candidates = [m for (m, s) in scored if s == best_score] or [scored[0][0]]

        chosen = self.rng.choice(candidates)

        elapsed = time.perf_counter() - start
        self.last_info = {
            "depth": 1,
            "nodes": nodes,
            "eval": int(best_score) if best_score not in (inf, -inf) else best_score,
            "time_ms": max(1, int(elapsed * 1000)),
            "time_limit_ms": int(self.time_limit_sec * 1000),
            "temperature": self.temperature,
            "seed": self.seed,
            "center_bias": self.center_bias,
            "block_policy": self.block_policy,
            "greedy_policy": self.greedy_policy,
            "note": "greedy_eval",
        }
        return chosen

