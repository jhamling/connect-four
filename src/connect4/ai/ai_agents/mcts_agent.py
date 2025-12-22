from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from connect4.core.rules import check_winner, is_draw
from connect4.game.state import GameState


def other(p: str) -> str:
    return "O" if p == "X" else "X"


@dataclass
class _Node:
    state: GameState
    player_to_move: str
    parent: Optional["_Node"] = None
    parent_move: Optional[int] = None
    children: Dict[int, "_Node"] = None
    untried_moves: List[int] = None
    visits: int = 0
    wins_for_root_player: float = 0.0  # from root player's perspective

    def __post_init__(self) -> None:
        self.children = {}
        self.untried_moves = self.state.board.valid_moves()


@dataclass(slots=True)
class MCTSAgent:
    """
    UCT Monte Carlo Tree Search agent (time-budgeted).

    Variant knobs (so you can generate a grid like MM d/t/temp):
      - time_limit_sec: overall budget per move
      - exploration_c: UCT exploration constant
      - rollout_max_plies: cap rollout length
      - rollout_top_k: how random vs center-biased rollouts are
      - rollout_center_bias: strength of center ordering in rollouts
      - rollout_check_win_block: whether rollout policy uses cheap win/block checks
      - expansion_policy: how to pick which untried move to expand ("random" or "center")
      - seed: deterministic RNG seed
    """
    name: str = "MCTS"

    time_limit_sec: float = 0.10
    exploration_c: float = 1.4

    rollout_max_plies: int = 80
    rollout_top_k: int = 3
    rollout_center_bias: float = 1.0
    rollout_check_win_block: bool = True

    expansion_policy: str = "random"  # "random" | "center"
    seed: int = 0

    rng: object = field(init=False)
    last_info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        import random

        self.rng = random.Random(self.seed)
        self.last_info = {}

        if self.rollout_top_k < 1:
            self.rollout_top_k = 1
        if self.rollout_center_bias < 0:
            self.rollout_center_bias = 0.0
        if self.exploration_c < 0:
            self.exploration_c = 0.0
        if self.rollout_max_plies < 1:
            self.rollout_max_plies = 1
        if self.expansion_policy not in ("random", "center"):
            self.expansion_policy = "random"

    # --------- Core helpers ---------
    def _terminal_value(self, state: GameState, root_player: str) -> Optional[float]:
        w = check_winner(state.board)
        if w is not None:
            return 1.0 if w == root_player else 0.0
        if is_draw(state.board):
            return 0.5
        return None

    def _select_child_uct(self, node: _Node) -> _Node:
        # UCT: Q/N + c * sqrt(ln(N)/n)
        best_score = -1e18
        best_child: Optional[_Node] = None

        for _move, child in node.children.items():
            if child.visits == 0:
                return child

            exploit = child.wins_for_root_player / child.visits
            explore = self.exploration_c * math.sqrt(math.log(max(1, node.visits)) / child.visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child

        return best_child if best_child is not None else next(iter(node.children.values()))

    def _center_sorted(self, state: GameState, moves: List[int]) -> List[int]:
        if self.rollout_center_bias <= 0:
            return moves[:]
        center = state.board.cols // 2
        return sorted(moves, key=lambda c: abs(c - center) * self.rollout_center_bias)

    def _pick_untried_move(self, node: _Node) -> int:
        if self.expansion_policy == "center":
            ordered = self._center_sorted(node.state, node.untried_moves)
            # pick best-by-center deterministically (still yields variety across seeds via root randomization, etc.)
            move = ordered[0]
            node.untried_moves.remove(move)
            return move

        # random expansion
        idx = self.rng.randrange(len(node.untried_moves))
        return node.untried_moves.pop(idx)

    def _expand(self, node: _Node) -> _Node:
        move = self._pick_untried_move(node)

        s2 = copy.deepcopy(node.state)
        s2.board.drop(move, s2.current)
        s2.current = other(s2.current)

        child = _Node(state=s2, player_to_move=s2.current, parent=node, parent_move=move)
        node.children[move] = child
        return child

    # --------- Rollout policy ---------
    def _rollout_policy(self, s: GameState, moves: List[int]) -> int:
        # optional win/block checks (stronger but slower rollouts)
        if self.rollout_check_win_block:
            me = s.current
            opp = other(me)

            for c in moves:
                b2 = copy.deepcopy(s.board)
                b2.drop(c, me)
                if check_winner(b2) == me:
                    return c

            for c in moves:
                b2 = copy.deepcopy(s.board)
                b2.drop(c, opp)
                if check_winner(b2) == opp:
                    return c

        # center-ish randomness
        ordered = self._center_sorted(s, moves)
        k = min(self.rollout_top_k, len(ordered))
        return self.rng.choice(ordered[:k])

    def _rollout(self, state: GameState, root_player: str) -> float:
        s = copy.deepcopy(state)

        for _ in range(self.rollout_max_plies):
            val = self._terminal_value(s, root_player)
            if val is not None:
                return val

            moves = s.board.valid_moves()
            if not moves:
                return 0.5

            move = self._rollout_policy(s, moves)
            s.board.drop(move, s.current)
            s.current = other(s.current)

        return 0.5

    def _backprop(self, node: _Node, result: float) -> None:
        cur: Optional[_Node] = node
        while cur is not None:
            cur.visits += 1
            cur.wins_for_root_player += result
            cur = cur.parent

    # --------- Public API ---------
    def choose_move(self, state: GameState) -> int:
        t0 = time.perf_counter()

        moves = state.board.valid_moves()
        if not moves:
            self.last_info = {"time_ms": 1, "nodes": 0, "depth": 1}
            return 0

        root_player = state.current
        root = _Node(state=copy.deepcopy(state), player_to_move=root_player)

        iters = 0
        max_depth_seen = 1
        deadline = t0 + max(0.001, float(self.time_limit_sec))

        while time.perf_counter() < deadline:
            node = root
            depth = 1

            # 1) Selection
            while node.untried_moves == [] and node.children:
                node = self._select_child_uct(node)
                depth += 1

            # 2) Expansion
            if node.untried_moves:
                node = self._expand(node)
                depth += 1

            # 3) Simulation
            result = self._rollout(node.state, root_player)

            # 4) Backprop
            self._backprop(node, result)

            iters += 1
            if depth > max_depth_seen:
                max_depth_seen = depth

        # pick move by most visits (standard)
        best_move: Optional[int] = None
        best_visits = -1
        for mv, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = mv

        if best_move is None:
            best_move = self.rng.choice(moves)

        self.last_info = {
            "time_ms": max(1, int((time.perf_counter() - t0) * 1000)),
            "nodes": iters,          # iterations as node proxy
            "depth": max_depth_seen, # tree depth reached (not rollout depth)
            "iters": iters,
            "exploration_c": self.exploration_c,
            "rollout_max_plies": self.rollout_max_plies,
            "rollout_top_k": self.rollout_top_k,
            "rollout_center_bias": self.rollout_center_bias,
            "rollout_check_win_block": self.rollout_check_win_block,
            "expansion_policy": self.expansion_policy,
        }
        return best_move

