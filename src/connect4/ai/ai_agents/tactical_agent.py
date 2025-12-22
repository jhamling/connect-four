from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Optional
import random

from connect4.core.rules import check_winner
from connect4.game.state import GameState


def other(p: str) -> str:
    return "O" if p == "X" else "X"


@dataclass(slots=True)
class TacticalAgent:
    """
    Fast tactical agent:
      1) Play immediate winning move if available
      2) Block opponent immediate winning move
      3) Prefer center-ish columns
      4) Otherwise random valid move

    Variant knobs (to generate multiple distinct Tactical agents like MM d/t/temp):
      - seed: deterministic RNG seed
      - center_bias: float (0.0 disables; higher = stronger center preference)
      - top_k: choose randomly among top_k center-ranked moves
      - win_policy: how to pick among multiple winning moves ("random" or "center")
      - block_policy: how to pick among multiple blocking moves ("random" or "center")
      - scan_limit: limit how many candidate columns we scan for win/block each turn
          * None = scan all valid moves (strongest)
          * small int = faster/weaker, produces meaningfully different variants
    """
    name: str = "Tactical"

    seed: int = 0
    center_bias: float = 1.0
    top_k: int = 3

    win_policy: str = "random"    # "random" | "center"
    block_policy: str = "center"  # "random" | "center"

    scan_limit: Optional[int] = None

    rng: random.Random = field(init=False)
    last_info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        if self.center_bias < 0:
            self.center_bias = 0.0
        if self.top_k < 1:
            self.top_k = 1
        if self.win_policy not in ("random", "center"):
            self.win_policy = "random"
        if self.block_policy not in ("random", "center"):
            self.block_policy = "center"
        if self.scan_limit is not None and self.scan_limit < 1:
            self.scan_limit = 1

    def _center_sorted(self, state: GameState, moves: list[int]) -> list[int]:
        if self.center_bias <= 0:
            return moves[:]
        center = state.board.cols // 2
        return sorted(moves, key=lambda c: abs(c - center) * self.center_bias)

    def _winning_moves(self, state: GameState, player: str, candidates: list[int]) -> list[int]:
        wins: list[int] = []
        for c in candidates:
            b2 = copy.deepcopy(state.board)
            b2.drop(c, player)
            if check_winner(b2) == player:
                wins.append(c)
        return wins

    def _pick_among(self, state: GameState, moves: list[int], policy: str) -> int:
        if len(moves) == 1:
            return moves[0]
        if policy == "center":
            return self._center_sorted(state, moves)[0]
        return self.rng.choice(moves)

    def choose_move(self, state: GameState) -> int:
        t0 = time.perf_counter()
        nodes = 0

        moves = state.board.valid_moves()
        if not moves:
            self.last_info = {"time_ms": 1, "nodes": 0, "depth": 1}
            return 0

        me = state.current
        opp = other(me)

        # order candidates by center preference (if enabled)
        ordered = self._center_sorted(state, moves)

        # optionally scan only first N moves (speed/strength tradeoff variant)
        scan = ordered if self.scan_limit is None else ordered[: min(self.scan_limit, len(ordered))]

        # 1) win now (if multiple, pick by win_policy)
        winning = self._winning_moves(state, me, scan)
        nodes += len(scan)
        if winning:
            choice = self._pick_among(state, winning, self.win_policy)
            self.last_info = {
                "time_ms": max(1, int((time.perf_counter() - t0) * 1000)),
                "nodes": nodes,
                "depth": 1,
                "seed": self.seed,
                "center_bias": self.center_bias,
                "top_k": self.top_k,
                "win_policy": self.win_policy,
                "block_policy": self.block_policy,
                "scan_limit": self.scan_limit,
                "note": "win",
            }
            return choice

        # 2) block opponent win (if multiple, pick by block_policy)
        blocks = self._winning_moves(state, opp, scan)
        nodes += len(scan)
        if blocks:
            choice = self._pick_among(state, blocks, self.block_policy)
            self.last_info = {
                "time_ms": max(1, int((time.perf_counter() - t0) * 1000)),
                "nodes": nodes,
                "depth": 1,
                "seed": self.seed,
                "center_bias": self.center_bias,
                "top_k": self.top_k,
                "win_policy": self.win_policy,
                "block_policy": self.block_policy,
                "scan_limit": self.scan_limit,
                "note": "block",
            }
            return choice

        # 3) center-ish preference with controlled randomness among top_k
        if self.center_bias > 0:
            k = min(self.top_k, len(ordered))
            choice = self.rng.choice(ordered[:k])
            self.last_info = {
                "time_ms": max(1, int((time.perf_counter() - t0) * 1000)),
                "nodes": nodes,
                "depth": 1,
                "seed": self.seed,
                "center_bias": self.center_bias,
                "top_k": self.top_k,
                "win_policy": self.win_policy,
                "block_policy": self.block_policy,
                "scan_limit": self.scan_limit,
                "note": "center",
            }
            return choice

        # 4) random fallback
        choice = self.rng.choice(moves)
        self.last_info = {
            "time_ms": max(1, int((time.perf_counter() - t0) * 1000)),
            "nodes": nodes,
            "depth": 1,
            "seed": self.seed,
            "center_bias": self.center_bias,
            "top_k": self.top_k,
            "win_policy": self.win_policy,
            "block_policy": self.block_policy,
            "scan_limit": self.scan_limit,
            "note": "random",
        }
        return choice
