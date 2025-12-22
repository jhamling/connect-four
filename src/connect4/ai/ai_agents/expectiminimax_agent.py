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
class ExpectiMiniMaxAgent:
    """
    ExpectiMinimax (adversary is "usually optimal" but sometimes noisy).

    Knobs that create meaningfully distinct variants (like MM d/t/temp):
      - depth: search depth cap
      - time_limit_sec: per-move budget
      - p_best: probability opponent plays the minimizing (best) response; otherwise mixes alternatives
      - temperature: softens BOTH players' move selection among near-best moves
          * 0 => strictly best (max) / strictly worst (min)
          * >0 => allow near-best / near-worst alternatives
      - opp_mix: how opponent behaves when not playing best
          * "avg_rest": average of non-best moves (your current behavior)
          * "uniform": uniform random over all moves
          * "softmin": softmin over moves (more realistic "blunder model")
      - move_order: affects performance under time limits
          * "center" (default), "random", "none"
      - seed: deterministic randomness for reproducibility (affects tie-breaking + random ordering)

    Notes:
      - We keep your eval/terminal conventions.
      - last_info includes extra knobs for league print.
    """
    name: str = "ExpectiMinimax"

    depth: int = 6
    time_limit_sec: float = 0.20

    p_best: float = 0.80
    temperature: float = 0.0

    opp_mix: str = "avg_rest"   # "avg_rest" | "uniform" | "softmin"
    move_order: str = "center"  # "center" | "random" | "none"

    seed: int = 0
    rng: random.Random = field(init=False)

    last_info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

        if self.depth < 1:
            self.depth = 1
        if self.time_limit_sec < 0:
            self.time_limit_sec = 0.0

        # clamp probabilities / knobs
        if self.p_best < 0.0:
            self.p_best = 0.0
        if self.p_best > 1.0:
            self.p_best = 1.0

        if self.temperature < 0.0:
            self.temperature = 0.0

        if self.opp_mix not in {"avg_rest", "uniform", "softmin"}:
            self.opp_mix = "avg_rest"
        if self.move_order not in {"center", "random", "none"}:
            self.move_order = "center"

    # ---------- helpers ----------
    def _terminal(self, board, perspective: Player) -> float | None:
        w = check_winner(board)
        if w == perspective:
            return 1_000_000.0
        if w is not None:
            return -1_000_000.0
        if is_draw(board):
            return 0.0
        return None

    def _ordered_moves(self, board, moves: list[Move]) -> list[Move]:
        if self.move_order == "none":
            return moves
        if self.move_order == "random":
            ms = moves[:]
            self.rng.shuffle(ms)
            return ms

        center = board.cols // 2
        return sorted(moves, key=lambda m: abs(int(m) - center))

    def _choose_from_near_best_max(self, scored: list[tuple[Move, float]]) -> tuple[Move, float]:
        # maximize; allow near-best within temperature
        best = max(v for (_, v) in scored)
        if self.temperature <= 0.0:
            # deterministic tie-break with rng among equals
            best_moves = [m for (m, v) in scored if v == best]
            m = self.rng.choice(best_moves)
            return m, best

        thr = best - float(self.temperature)
        cand = [(m, v) for (m, v) in scored if v >= thr]
        if not cand:
            cand = [(m, v) for (m, v) in scored if v == best]
        m, v = self.rng.choice(cand)
        return m, v

    def _choose_from_near_worst_min(self, scored: list[tuple[Move, float]]) -> tuple[Move, float]:
        # minimize; allow near-worst within temperature (i.e., values close to min)
        worst = min(v for (_, v) in scored)
        if self.temperature <= 0.0:
            worst_moves = [m for (m, v) in scored if v == worst]
            m = self.rng.choice(worst_moves)
            return m, worst

        thr = worst + float(self.temperature)
        cand = [(m, v) for (m, v) in scored if v <= thr]
        if not cand:
            cand = [(m, v) for (m, v) in scored if v == worst]
        m, v = self.rng.choice(cand)
        return m, v

    def _softmin_expected(self, vals: list[float], tau: float) -> float:
        # tau small => sharper (more like min). tau large => closer to average.
        # numerically stable softmin
        if not vals:
            return 0.0
        tau = max(1e-9, tau)
        m = min(vals)
        exps = [pow(2.718281828, -(v - m) / tau) for v in vals]
        z = sum(exps)
        if z <= 0:
            return sum(vals) / len(vals)
        probs = [e / z for e in exps]
        return sum(p * v for p, v in zip(probs, vals))

    # ---------- search ----------
    def choose_move(self, state: GameState) -> Move:
        board = state.board
        me: Player = state.current

        moves = board.valid_moves()
        if not moves:
            raise ValueError("No valid moves.")

        start = time.perf_counter()
        deadline = start + max(0.01, float(self.time_limit_sec))

        moves = self._ordered_moves(board, moves)

        best_move = moves[0]
        best_score = -inf
        nodes = 0
        depth_reached = 1

        # root: choose move that maximizes expected value
        root_scored: list[tuple[Move, float]] = []
        for m in moves:
            if time.perf_counter() >= deadline:
                break
            board.drop(m, me)
            v, n, dr = self._opp_expect(board, _other(me), self.depth - 1, me, deadline)
            board.undo(m)

            nodes += n
            depth_reached = max(depth_reached, dr + 1)

            root_scored.append((m, v))
            if v > best_score:
                best_score = v
                best_move = m

        if root_scored:
            # apply temperature at root too (adds variant behavior)
            chosen, chosen_v = self._choose_from_near_best_max(root_scored)
            best_move, best_score = chosen, chosen_v

        elapsed = time.perf_counter() - start
        self.last_info = {
            "depth": min(self.depth, depth_reached),
            "nodes": nodes,
            "eval": int(best_score) if best_score not in (inf, -inf) else best_score,
            "move_col": int(best_move) + 1,
            "time_ms": max(1, int(elapsed * 1000)),
            "time_limit_ms": int(self.time_limit_sec * 1000),
            "p_best": self.p_best,
            "temperature": self.temperature,
            "opp_mix": self.opp_mix,
            "move_order": self.move_order,
        }
        return best_move

    def _max_value(
        self, board, to_play: Player, depth: int, perspective: Player, deadline: float
    ) -> tuple[float, int, int]:
        """
        Returns (value, nodes, depth_reached_from_here)
        depth_reached_from_here counts plies explored from this node (>=1).
        """
        if time.perf_counter() >= deadline:
            return float(evaluate(board, perspective)), 1, 1

        term = self._terminal(board, perspective)
        if term is not None:
            return term, 1, 1
        if depth == 0:
            return float(evaluate(board, perspective)), 1, 1

        v = -inf
        nodes = 1
        best_child_depth = 1

        moves = board.valid_moves()
        moves = self._ordered_moves(board, moves)

        scored: list[tuple[Move, float]] = []

        for m in moves:
            if time.perf_counter() >= deadline:
                break
            board.drop(m, to_play)
            child, cn, dr = self._opp_expect(board, _other(to_play), depth - 1, perspective, deadline)
            board.undo(m)

            nodes += cn
            best_child_depth = max(best_child_depth, dr + 1)

            scored.append((m, child))
            v = max(v, child)

        if scored and self.temperature > 0.0:
            # optional softmax-like behavior via "near best" random choice
            _, v = self._choose_from_near_best_max(scored)

        return v, nodes, best_child_depth

    def _opp_expect(
        self, board, opp: Player, depth: int, perspective: Player, deadline: float
    ) -> tuple[float, int, int]:
        """
        Opponent node:
          With prob p_best, choose minimizing move (or near-worst if temperature>0).
          Otherwise:
            - avg_rest: average non-best moves (your original)
            - uniform: average over all moves (uniform random opponent)
            - softmin: softmin expected value over moves

        Returns (expected_value, nodes, depth_reached_from_here)
        """
        if time.perf_counter() >= deadline:
            return float(evaluate(board, perspective)), 1, 1

        term = self._terminal(board, perspective)
        if term is not None:
            return term, 1, 1
        if depth == 0:
            return float(evaluate(board, perspective)), 1, 1

        moves = board.valid_moves()
        if not moves:
            return float(evaluate(board, perspective)), 1, 1

        moves = self._ordered_moves(board, moves)

        vals: list[tuple[Move, float]] = []
        nodes = 1
        best_child_depth = 1

        for m in moves:
            if time.perf_counter() >= deadline:
                break
            board.drop(m, opp)
            child, cn, dr = self._max_value(board, _other(opp), depth - 1, perspective, deadline)
            board.undo(m)

            nodes += cn
            best_child_depth = max(best_child_depth, dr + 1)

            vals.append((m, child))

        if not vals:
            return float(evaluate(board, perspective)), nodes, best_child_depth

        # "best" opponent move minimizes perspective
        best_move, best_val = self._choose_from_near_worst_min(vals) if self.temperature > 0.0 else (
            min(vals, key=lambda t: t[1])[0],
            min(v for (_, v) in vals),
        )

        # non-best behavior
        if self.opp_mix == "uniform":
            # uniform random over all moves (expected = average of all vals)
            avg_all = sum(v for (_, v) in vals) / len(vals)
            expected = (self.p_best * best_val) + ((1.0 - self.p_best) * avg_all)
            return expected, nodes, best_child_depth

        if self.opp_mix == "softmin":
            # softmin over all moves when not best; tau tied to temperature (or a small default)
            tau = max(1.0, float(self.temperature)) if self.temperature > 0.0 else 3.0
            soft = self._softmin_expected([v for (_, v) in vals], tau=tau)
            expected = (self.p_best * best_val) + ((1.0 - self.p_best) * soft)
            return expected, nodes, best_child_depth

        # default: avg_rest (your original)
        # remove one instance of best (by identity if possible)
        rest_vals = []
        used_best = False
        for (m, v) in vals:
            if (not used_best) and (m == best_move) and (v == best_val):
                used_best = True
                continue
            rest_vals.append(v)

        if not rest_vals:
            return best_val, nodes, best_child_depth

        avg_rest = sum(rest_vals) / len(rest_vals)
        expected = (self.p_best * best_val) + ((1.0 - self.p_best) * avg_rest)
        return expected, nodes, best_child_depth

