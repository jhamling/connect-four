
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    wins_for_root_player: float = 0.0  # count wins from root player's perspective

    def __post_init__(self) -> None:
        self.children = {}
        self.untried_moves = self.state.board.valid_moves()


@dataclass
class MCTSAgent:
    """
    UCT Monte Carlo Tree Search agent (time-budgeted).

    Notes:
    - Uses deepcopy transitions (safe, not fastest).
    - Tracks last_info for league stats.
    """
    name: str = "MCTS"
    time_limit_sec: float = 0.10
    exploration_c: float = 1.4
    rollout_max_plies: int = 80
    seed: int = 0

    def __post_init__(self) -> None:
        import random
        self.rng = random.Random(self.seed)
        self.last_info = {}

    def _terminal_value(self, state: GameState, root_player: str) -> Optional[float]:
        w = check_winner(state.board)
        if w is not None:
            if w == root_player:
                return 1.0
            return 0.0
        if is_draw(state.board):
            return 0.5
        return None

    def _select_child_uct(self, node: _Node, root_player: str) -> _Node:
        # UCT: Q/N + c * sqrt(ln(N)/n)
        best_score = -1e18
        best_child = None

        for move, child in node.children.items():
            if child.visits == 0:
                return child

            exploit = child.wins_for_root_player / child.visits
            explore = self.exploration_c * math.sqrt(math.log(max(1, node.visits)) / child.visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child

        return best_child  # type: ignore

    def _expand(self, node: _Node) -> _Node:
        move = node.untried_moves.pop(self.rng.randrange(len(node.untried_moves)))

        s2 = copy.deepcopy(node.state)
        s2.board.drop(move, s2.current)
        s2.current = other(s2.current)

        child = _Node(state=s2, player_to_move=s2.current, parent=node, parent_move=move)
        node.children[move] = child
        return child

    def _rollout(self, state: GameState, root_player: str) -> float:
        s = copy.deepcopy(state)

        for _ in range(self.rollout_max_plies):
            val = self._terminal_value(s, root_player)
            if val is not None:
                return val

            moves = s.board.valid_moves()
            if not moves:
                return 0.5

            # rollout policy: small center bias + immediate win if found
            move = self._rollout_policy(s, moves)
            s.board.drop(move, s.current)
            s.current = other(s.current)

        # if rollout truncated, treat as draw-ish
        return 0.5

    def _rollout_policy(self, s: GameState, moves: List[int]) -> int:
        me = s.current
        opp = other(me)

        # immediate win
        for c in moves:
            b2 = copy.deepcopy(s.board)
            b2.drop(c, me)
            if check_winner(b2) == me:
                return c

        # immediate block
        for c in moves:
            b2 = copy.deepcopy(s.board)
            b2.drop(c, opp)
            if check_winner(b2) == opp:
                return c

        # center bias
        center = 3
        moves_sorted = sorted(moves, key=lambda c: abs(c - center))
        top_k = min(3, len(moves_sorted))
        return self.rng.choice(moves_sorted[:top_k])

    def _backprop(self, node: _Node, result: float) -> None:
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.wins_for_root_player += result
            cur = cur.parent

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

        deadline = t0 + max(0.001, self.time_limit_sec)

        while time.perf_counter() < deadline:
            node = root
            depth = 1

            # 1) Selection
            while node.untried_moves == [] and node.children:
                node = self._select_child_uct(node, root_player)
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

        # choose best move by most visits
        best_move = None
        best_visits = -1
        for mv, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = mv

        if best_move is None:
            best_move = self.rng.choice(moves)

        self.last_info = {
            "time_ms": max(1, int((time.perf_counter() - t0) * 1000)),
            "nodes": iters,     # treat iterations as "nodes" for now
            "depth": max_depth_seen,
        }
        return best_move
