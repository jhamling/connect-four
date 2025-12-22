
from __future__ import annotations

import csv
import inspect
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Tuple

from connect4.ai.tt import TranspositionTable
from connect4.core.board import Board
from connect4.core.rules import check_winner, is_draw
from connect4.game.state import GameState


def other(p: str) -> str:
    return "O" if p == "X" else "X"


# -----------------------------
# Terminal formatting (ANSI)
# -----------------------------
class _Ansi:
    def __init__(self) -> None:
        self.enabled = (
            os.environ.get("NO_COLOR") is None
            and os.environ.get("TERM") not in (None, "", "dumb")
        )

    def _wrap(self, s: str, code: str) -> str:
        if not self.enabled:
            return s
        return f"\x1b[{code}m{s}\x1b[0m"

    def bold(self, s: str) -> str: return self._wrap(s, "1")
    def dim(self, s: str) -> str: return self._wrap(s, "2")

    def red(self, s: str) -> str: return self._wrap(s, "31")
    def green(self, s: str) -> str: return self._wrap(s, "32")
    def yellow(self, s: str) -> str: return self._wrap(s, "33")
    def blue(self, s: str) -> str: return self._wrap(s, "34")
    def magenta(self, s: str) -> str: return self._wrap(s, "35")
    def cyan(self, s: str) -> str: return self._wrap(s, "36")


A = _Ansi()


def _hr(char: str = "─", n: int = 96) -> str:
    return char * n


# -----------------------------
# Team / Aggregation
# -----------------------------
@dataclass(frozen=True)
class Team:
    name: str
    make: Callable[[], object]  # must be picklable (use functools.partial, not lambda)


@dataclass
class Agg:
    games: int = 0
    points: float = 0.0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    moves: int = 0
    time_ms: int = 0
    nodes: int = 0
    depth_sum: int = 0


# -----------------------------
# Safe agent factory (filters kwargs by __init__ signature)
# Avoids "unexpected keyword argument" for agents that don't implement a knob.
# -----------------------------
def _make_agent(cls, kwargs: dict) -> object:
    sig = inspect.signature(cls)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return cls(**filtered)


def _team(cls, *, name: str, **kwargs) -> Team:
    # kwargs MUST be picklable; cls must be top-level importable.
    return Team(name, partial(_make_agent, cls, {"name": name, **kwargs}))


def _seed(agent, seed: int) -> None:
    if hasattr(agent, "rng"):
        try:
            agent.rng.seed(seed)
        except Exception:
            pass


# -----------------------------
# Game loop
# -----------------------------
def play_headless(agent_x, agent_o, seed_base: int = 0) -> Tuple[str, Dict[str, Dict[str, int]]]:
    """
    Headless game loop.
    Returns outcome: "X", "O", or "D"
    Also returns per-side move stats: {"X": {...}, "O": {...}}
    """
    state = GameState(board=Board(), current="X", last_status="")
    stats = {
        "X": {"moves": 0, "time_ms": 0, "nodes": 0, "depth": 0},
        "O": {"moves": 0, "time_ms": 0, "nodes": 0, "depth": 0},
    }

    # deterministic seeds per game for reproducibility
    _seed(agent_x, seed_base + 101)
    _seed(agent_o, seed_base + 202)

    # reset TT once per game (only if agent uses it)
    if hasattr(agent_x, "tt"):
        agent_x.tt = TranspositionTable()
    if hasattr(agent_o, "tt"):
        agent_o.tt = TranspositionTable()

    # --- opening randomization (2 plies) ---
    rng = random.Random(seed_base)
    for _ in range(2):
        moves = state.board.valid_moves()
        if not moves:
            break
        move = rng.choice(moves)
        state.board.drop(move, state.current)
        state.current = other(state.current)
    # --------------------------------------

    while True:
        w = check_winner(state.board)
        if w is not None:
            return w, stats
        if is_draw(state.board):
            return "D", stats

        agent = agent_x if state.current == "X" else agent_o
        move = agent.choose_move(state)

        info = getattr(agent, "last_info", None) or {}
        side_stats = stats[state.current]
        side_stats["moves"] += 1

        ms = int(info.get("time_ms", 0))
        side_stats["time_ms"] += max(1, ms)

        side_stats["nodes"] += int(info.get("nodes", 0))
        side_stats["depth"] += int(info.get("depth", 0))

        state.board.drop(move, state.current)
        state.current = other(state.current)


def add_result(agg_a: Agg, agg_b: Agg, outcome: str, a_is_x: bool) -> None:
    """
    outcome is "X", "O", or "D" for the game.
    a_is_x says whether team A played as X.
    """
    agg_a.games += 1
    agg_b.games += 1

    if outcome == "D":
        agg_a.draws += 1
        agg_b.draws += 1
        agg_a.points += 0.5
        agg_b.points += 0.5
        return

    a_won = (outcome == "X" and a_is_x) or (outcome == "O" and not a_is_x)
    if a_won:
        agg_a.wins += 1
        agg_b.losses += 1
        agg_a.points += 1.0
    else:
        agg_b.wins += 1
        agg_a.losses += 1
        agg_b.points += 1.0


# -----------------------------
# Multiprocessing: batched worker
# -----------------------------
def _run_pairings_batch(args):
    """
    Execute a batch of pairings in one process to reduce IPC/submit overhead.

    args = (batch_items, games_per_pair)

    batch_items is list of tuples:
      (A_name, B_name, A_make, B_make, base_seed)

    Returns:
      list of tuples:
        (A_name, B_name, a_is_x, outcome, stats)
    """
    (batch_items, games_per_pair) = args
    out = []

    for (A_name, B_name, A_make, B_make, base_seed) in batch_items:
        for g in range(games_per_pair):
            if g % 2 == 0:
                ax = A_make()
                bo = B_make()
                outcome, stats = play_headless(ax, bo, seed_base=(base_seed + g))
                out.append((A_name, B_name, True, outcome, stats))   # A is X
            else:
                bx = B_make()
                ao = A_make()
                outcome, stats = play_headless(bx, ao, seed_base=(base_seed + g))
                out.append((A_name, B_name, False, outcome, stats))  # A is O

    return out


def _chunked(lst, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# -----------------------------
# Scoring
# -----------------------------
def _ppg(a: Agg) -> float:
    return (a.points / a.games) if a.games else 0.0


def _avg_ms_per_move(a: Agg) -> float:
    return (a.time_ms / a.moves) if a.moves else 0.0


def _wilson_lcb(p: float, n: int, z: float) -> float:
    """
    Wilson score interval lower bound for a proportion p in [0,1] with n trials.
    We apply it to PPG (wins=1, draws=0.5, losses=0) as a practical ranking LCB.
    """
    if n <= 0:
        return 0.0
    p = max(0.0, min(1.0, p))
    z2 = z * z
    denom = 1.0 + (z2 / n)
    center = p + (z2 / (2.0 * n))
    rad = z * math.sqrt(max(0.0, (p * (1.0 - p) + (z2 / (4.0 * n))) / n))
    return max(0.0, (center - rad) / denom)


def _strength_score(a: Agg, z: float) -> float:
    return _wilson_lcb(_ppg(a), a.games, z)


def _compute_factor(ms_per_move: float, ms_target: float, alpha: float, min_factor: float) -> float:
    """
    Sublinear, capped speed factor:
      factor = 1 / (1 + alpha * sqrt(ms/ms_target))
      and clamp to >= min_factor to prevent speed dominating.
    """
    ms = max(1e-9, float(ms_per_move))
    tgt = max(1e-9, float(ms_target))
    factor = 1.0 / (1.0 + float(alpha) * math.sqrt(ms / tgt))
    return max(float(min_factor), factor)


def _efficiency_score(a: Agg, z: float, ms_target: float, alpha: float, min_factor: float) -> float:
    s = _strength_score(a, z)
    ms = _avg_ms_per_move(a)
    f = _compute_factor(ms, ms_target=ms_target, alpha=alpha, min_factor=min_factor)
    return s * f


def _pareto_frontier(rows: List[tuple[str, Agg]], z: float) -> List[tuple[str, Agg]]:
    """
    Frontier on (strength_score, -avg_ms_per_move): keep rows that are not dominated
    (no other agent is >= strength and <= ms with at least one strict).
    """
    scored = []
    for name, a in rows:
        scored.append((name, a, _strength_score(a, z), _avg_ms_per_move(a)))

    # Sort by strength desc, then ms asc
    scored.sort(key=lambda t: (-t[2], t[3]))

    frontier = []
    best_ms = float("inf")
    for name, a, s, ms in scored:
        if ms < best_ms - 1e-9:
            frontier.append((name, a))
            best_ms = ms
    return frontier


# -----------------------------
# League
# -----------------------------
def league_auto_prune(
    teams: List[Team],
    games_per_pair: int = 2,
    stage_pairings_per_team: int = 10,
    min_games_before_prune: int = 20,
    keep_fraction: float = 0.50,
    final_keep: int = 30,
    seed: int = 1234,
    max_workers: int | None = None,
    batch_pairings: int = 12,         # tune: 8..32
    max_stages: int | None = None,    # None = no cap
    prune_z: float = 1.28,            # stability penalty (LCB)
    ms_target: float = 50.0,          # compute scaling
    speed_alpha: float = 0.35,        # compute penalty strength (sublinear)
    speed_min_factor: float = 0.75,   # cap: speed cannot boost score more than ~33%
    export_csv: bool = True,
) -> None:
    """
    Produces *two* leaderboards:
      - Strength: Wilson LCB of PPG (speed ignored)
      - Efficiency: Strength * capped sublinear speed factor

    Pruning uses Efficiency by default (keeps strong + not-crazy-slow),
    without allowing speed to dominate.
    """
    rng = random.Random(seed)
    agg: Dict[str, Agg] = {t.name: Agg() for t in teams}

    def print_tables(title: str, roster: List[Team], top_n: int = 20) -> None:
        rows = [(t.name, agg[t.name]) for t in roster]

        def fmt_wdl(a: Agg) -> str:
            return f"{a.wins}-{a.draws}-{a.losses}"

        # Strength
        by_strength = sorted(rows, key=lambda r: _strength_score(r[1], prune_z), reverse=True)
        # Efficiency
        by_eff = sorted(
            rows,
            key=lambda r: _efficiency_score(r[1], prune_z, ms_target, speed_alpha, speed_min_factor),
            reverse=True,
        )
        frontier = _pareto_frontier(rows, prune_z)

        print("\n" + A.bold(f"=== {title} ==="))
        print(A.dim(_hr()))

        def _print_block(block_title: str, ranking: List[tuple[str, Agg]]) -> None:
            print(A.cyan(A.bold(block_title)))
            print(A.dim("rank  agent                             strength    eff_score   ppg    g    avg_ms/mv   W-D-L"))
            for i, (name, a) in enumerate(ranking[:top_n], start=1):
                s = _strength_score(a, prune_z)
                e = _efficiency_score(a, prune_z, ms_target, speed_alpha, speed_min_factor)
                ppg = _ppg(a)
                ms = _avg_ms_per_move(a)
                wdl = fmt_wdl(a)

                # small, readable coloring
                s_txt = A.magenta(f"{s:0.6f}")
                e_txt = A.magenta(f"{e:0.6f}")
                ppg_txt = A.green(f"{ppg:0.3f}") if ppg >= 0.75 else (A.yellow(f"{ppg:0.3f}") if ppg >= 0.5 else A.red(f"{ppg:0.3f}"))
                wdl_txt = f"{A.green(str(a.wins))}-{A.yellow(str(a.draws))}-{A.red(str(a.losses))}"

                print(
                    f"{i:>4}  {name:<32} "
                    f"{s_txt}  {e_txt}  {ppg_txt}  {a.games:>4}  {ms:>9.1f}  {wdl_txt}"
                )
            print(A.dim(_hr()))

        _print_block("Top by Strength (speed ignored)", by_strength)
        _print_block("Top by Efficiency (strength with capped speed factor)", by_eff)

        # Frontier is typically short; show up to top_n
        print(A.cyan(A.bold("Pareto Frontier (no agent is both stronger and faster)")))
        print(A.dim("agent                             strength    avg_ms/mv   ppg    g    W-D-L"))
        for (name, a) in frontier[:top_n]:
            s = _strength_score(a, prune_z)
            ms = _avg_ms_per_move(a)
            ppg = _ppg(a)
            wdl_txt = f"{A.green(str(a.wins))}-{A.yellow(str(a.draws))}-{A.red(str(a.losses))}"
            print(f"{name:<32} {A.magenta(f'{s:0.6f}')}  {ms:>9.1f}  {ppg:0.3f}  {a.games:>4}  {wdl_txt}")
        print(A.dim(_hr()))

    def schedule_stage(roster: List[Team]) -> List[tuple[Team, Team]]:
        if len(roster) < 2:
            return []

        need = {t.name: stage_pairings_per_team for t in roster}
        name_to_team = {t.name: t for t in roster}

        pool = []
        for name, k in need.items():
            pool.extend([name] * k)

        rng.shuffle(pool)

        used_pairs = set()
        matches: list[tuple[Team, Team]] = []

        i = 0
        while i + 1 < len(pool):
            a = pool[i]
            b = pool[i + 1]
            i += 2
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            if key in used_pairs:
                continue
            used_pairs.add(key)
            matches.append((name_to_team[a], name_to_team[b]))

        target_matches = (len(roster) * stage_pairings_per_team) // 2
        tries = 0
        while len(matches) < target_matches and tries < target_matches * 20:
            tries += 1
            A_, B_ = rng.sample(roster, 2)
            key = (A_.name, B_.name) if A_.name < B_.name else (B_.name, A_.name)
            if key in used_pairs:
                continue
            used_pairs.add(key)
            matches.append((A_, B_))

        return matches

    def _apply_game_result(A_name: str, B_name: str, a_is_x: bool, outcome: str, stats: Dict[str, Dict[str, int]]):
        add_result(agg[A_name], agg[B_name], outcome, a_is_x=a_is_x)

        if a_is_x:
            # A is X, B is O
            agg[A_name].moves += stats["X"]["moves"]
            agg[A_name].time_ms += stats["X"]["time_ms"]
            agg[A_name].nodes += stats["X"]["nodes"]
            agg[A_name].depth_sum += stats["X"]["depth"]

            agg[B_name].moves += stats["O"]["moves"]
            agg[B_name].time_ms += stats["O"]["time_ms"]
            agg[B_name].nodes += stats["O"]["nodes"]
            agg[B_name].depth_sum += stats["O"]["depth"]
        else:
            # B is X, A is O
            agg[B_name].moves += stats["X"]["moves"]
            agg[B_name].time_ms += stats["X"]["time_ms"]
            agg[B_name].nodes += stats["X"]["nodes"]
            agg[B_name].depth_sum += stats["X"]["depth"]

            agg[A_name].moves += stats["O"]["moves"]
            agg[A_name].time_ms += stats["O"]["time_ms"]
            agg[A_name].nodes += stats["O"]["nodes"]
            agg[A_name].depth_sum += stats["O"]["depth"]

    if max_workers is None:
        max_workers = min(os.cpu_count() or 2, 6)

    roster = teams[:]
    stage = 1

    print(A.bold(f"Initial roster: {len(roster)} teams"))
    print(A.dim(_hr()))
    print(
        f"Stage settings: pairings/team={stage_pairings_per_team}, games/pair={games_per_pair}, "
        f"keep_fraction={keep_fraction}, final_keep={final_keep}, workers={max_workers}, batch_pairings={batch_pairings}"
    )
    print(
        f"Scoring:\n"
        f"  Strength = WilsonLCB(PPG), z={prune_z}\n"
        f"  Efficiency = Strength * max({speed_min_factor}, 1/(1+alpha*sqrt(ms/ms_target)))  "
        f"(alpha={speed_alpha}, ms_target={ms_target})"
    )
    print(A.dim(_hr()))

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        # ---------- STAGED PRUNING ----------
        while len(roster) > final_keep:
            if max_stages is not None and stage > max_stages:
                print(A.yellow(f"Reached max stages ({max_stages}) — stopping early."))
                break

            print("\n" + A.bold(f"--- STAGE {stage} (roster={len(roster)}) ---"))
            matches = schedule_stage(roster)
            print(f"Stage {stage}: scheduled {len(matches)} pairings")

            batch_items = []
            for idx, (A_team, B_team) in enumerate(matches, start=1):
                base_seed = seed + stage * 1_000_000 + idx * 10_000
                batch_items.append((A_team.name, B_team.name, A_team.make, B_team.make, base_seed))

            futures = []
            for chunk in _chunked(batch_items, batch_pairings):
                futures.append(ex.submit(_run_pairings_batch, (chunk, games_per_pair)))

            for fut in as_completed(futures):
                results = fut.result()
                for (A_name, B_name, a_is_x, outcome, stats) in results:
                    _apply_game_result(A_name, B_name, a_is_x, outcome, stats)

            print_tables(f"After Stage {stage}", roster, top_n=20)

            # prune if enough data
            if all(agg[t.name].games >= min_games_before_prune for t in roster):
                keep_n = max(final_keep, int(math.ceil(len(roster) * keep_fraction)))
                keep_n = min(keep_n, len(roster))

                # prune by efficiency (strong + not insanely slow)
                by_eff = sorted(
                    roster,
                    key=lambda t: _efficiency_score(agg[t.name], prune_z, ms_target, speed_alpha, speed_min_factor),
                    reverse=True,
                )
                new_roster = by_eff[:keep_n]

                if len(new_roster) == len(roster):
                    print(A.yellow("No pruning occurred — stopping early."))
                    roster = new_roster
                    break

                roster = new_roster
                print(A.bold(f"Pruned to {len(roster)} teams (kept top {keep_n} by Efficiency)"))
            else:
                min_g = min(agg[t.name].games for t in roster)
                print(A.yellow(f"Not pruning yet: minimum games/team = {min_g}, need >= {min_games_before_prune}"))

            stage += 1
            if len(roster) < 2:
                break

        # ---------- FINAL ROUND-ROBIN (PARALLEL) ----------
        print("\n" + A.bold(f"=== FINAL ROUND-ROBIN (roster={len(roster)}) ==="))

        pair_items = []
        n = len(roster)
        for i in range(n):
            for j in range(i + 1, n):
                A_team = roster[i]
                B_team = roster[j]
                base_seed = seed + 9_000_000 + i * 10_000 + j * 100
                pair_items.append((A_team.name, B_team.name, A_team.make, B_team.make, base_seed))

        final_futures = []
        for chunk in _chunked(pair_items, batch_pairings):
            final_futures.append(ex.submit(_run_pairings_batch, (chunk, games_per_pair)))

        for fut in as_completed(final_futures):
            results = fut.result()
            for (A_name, B_name, a_is_x, outcome, stats) in results:
                _apply_game_result(A_name, B_name, a_is_x, outcome, stats)

        # Winners:
        winner_strength = max(roster, key=lambda t: _strength_score(agg[t.name], prune_z))
        winner_eff = max(
            roster,
            key=lambda t: _efficiency_score(agg[t.name], prune_z, ms_target, speed_alpha, speed_min_factor),
        )

        def _print_winner(title: str, team: Team) -> None:
            a = agg[team.name]
            print("\n" + A.bold(title))
            print(A.dim(_hr()))
            print(team.name)
            print(
                f"  strength={_strength_score(a, prune_z):.6f}  "
                f"eff={_efficiency_score(a, prune_z, ms_target, speed_alpha, speed_min_factor):.6f}\n"
                f"  ppg={_ppg(a):.3f}  W-D-L={a.wins}-{a.draws}-{a.losses}  games={a.games}\n"
                f"  avg_ms/move={_avg_ms_per_move(a):.1f}  moves={a.moves}  nodes={a.nodes}"
            )

        _print_winner("WINNER (Strength)", winner_strength)
        _print_winner("WINNER (Efficiency)", winner_eff)

        print_tables("Final (post round-robin)", roster, top_n=30)

        # ---------- CSV EXPORT ----------
        if export_csv:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = f"league_results_{ts}.csv"
            rows = [(t.name, agg[t.name]) for t in teams]  # export ALL teams, not just survivors

            with open(out_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "name",
                    "games", "wins", "draws", "losses",
                    "points", "ppg",
                    "strength_wilson_lcb",
                    "avg_ms_per_move",
                    "efficiency_score",
                    "moves", "time_ms", "nodes", "avg_depth",
                ])
                for name, a in rows:
                    ppg = _ppg(a)
                    s = _strength_score(a, prune_z)
                    ms = _avg_ms_per_move(a)
                    e = _efficiency_score(a, prune_z, ms_target, speed_alpha, speed_min_factor)
                    avg_depth = (a.depth_sum / a.moves) if a.moves else 0.0
                    w.writerow([
                        name,
                        a.games, a.wins, a.draws, a.losses,
                        a.points, round(ppg, 6),
                        round(s, 6),
                        round(ms, 3),
                        round(e, 6),
                        a.moves, a.time_ms, a.nodes, round(avg_depth, 3),
                    ])

            print("\n" + A.bold("Export"))
            print(A.dim(_hr()))
            print(f"Wrote CSV: {out_path}")


# -----------------------------
# Roster builder (with safe kwargs filtering)
# -----------------------------
def build_roster() -> List[Team]:
    """
    League roster builder.

    Goal:
      - Include ALL agents
      - Include MULTIPLE variants per agent type
      - Avoid crashes if an agent doesn't implement a knob:
        _team(...) filters kwargs by __init__ signature.
    """
    from connect4.ai.random_agent import RandomAgent
    from connect4.ai.weighted_random_agent import WeightedRandomAgent
    from connect4.ai.greedy_agent import GreedyAgent
    from connect4.ai.tactical_greedy_agent import TacticalGreedyAgent
    from connect4.ai.tactical_agent import TacticalAgent
    from connect4.ai.heuristic_agent import HeuristicAgent
    from connect4.ai.beam_agent import BeamAgent
    from connect4.ai.beam2_agent import Beam2Agent
    from connect4.ai.mcts_agent import MCTSAgent
    from connect4.ai.expectiminimax_agent import ExpectiMiniMaxAgent
    from connect4.ai.minimax_agent import MinimaxAgent

    teams: List[Team] = []

    # Baseline
    teams.append(_team(RandomAgent, name="Random"))

    # WeightedRandom variants
    for seed in [0, 1, 2]:
        teams.append(_team(WeightedRandomAgent, name=f"WeightedRandom seed{seed}", seed=seed))

    # Greedy variants
    for temp in [0, 5, 15]:
        for tl in [0.0, 0.01, 0.03]:
            teams.append(_team(GreedyAgent, name=f"Greedy temp{temp} t{int(tl*1000)}ms", temperature=temp, time_limit_sec=tl))

    # TacticalGreedy variants
    for temp in [0, 5, 15]:
        for tl in [0.0, 0.01, 0.03]:
            teams.append(_team(TacticalGreedyAgent, name=f"TacticalGreedy temp{temp} t{int(tl*1000)}ms", temperature=temp, time_limit_sec=tl))

    # TacticalAgent variants
    for center_bias in [True, False]:
        for seed in [0, 1, 2]:
            cb = "center" if center_bias else "nocenter"
            teams.append(_team(TacticalAgent, name=f"Tactical {cb} seed{seed}", center_bias=center_bias, seed=seed))

    # HeuristicAgent profiles
    heuristic_profiles = [
        (4, 2, 8, 9, 0),
        (6, 2, 8, 10, 1),
        (4, 3, 10, 9, 2),
        (3, 2, 7, 12, 3),
    ]
    for (cw, tw, thw, oppp, seed) in heuristic_profiles:
        teams.append(
            _team(
                HeuristicAgent,
                name=f"Heuristic cw{cw} tw{tw} th{thw} opp{oppp} seed{seed}",
                seed=seed,
                center_weight=cw,
                two_weight=tw,
                three_weight=thw,
                opp_three_penalty=oppp,
            )
        )

    # BeamAgent variants
    for d in [4, 6]:
        for w in [2, 3, 5]:
            for tl in [0.05, 0.12]:
                for temp in [0, 10]:
                    teams.append(
                        _team(
                            BeamAgent,
                            name=f"Beam d{d} w{w} t{int(tl*1000)}ms temp{temp}",
                            depth=d,
                            width=w,
                            time_limit_sec=tl,
                            temperature=temp,
                            seed=0,  # filtered out if BeamAgent doesn't accept seed
                        )
                    )

    # Beam2Agent variants
    # IMPORTANT: Your posted Beam2Agent does NOT have root_order.
    # Using _team(...) means extra knobs won't crash; they will be ignored if not supported.
    for d in [3, 4, 5]:
        for bw in [4, 8, 12]:
            for tl in [0.0, 0.03, 0.08]:
                for temp in [0, 6, 15]:
                    for (winb, lossp) in [(50_000, 50_000), (25_000, 25_000)]:
                        teams.append(
                            _team(
                                Beam2Agent,
                                name=f"Beam2 d{d} w{bw} t{int(tl*1000)}ms temp{temp} wb{winb} lp{lossp}",
                                depth=d,
                                beam_width=bw,
                                time_limit_sec=tl,
                                temperature=temp,
                                win_bonus=winb,
                                loss_penalty=lossp,
                                seed=0,
                                root_order="center",  # will be ignored unless your class adds it later
                            )
                        )

    # MCTS variants
    for tl in [0.03, 0.08, 0.15]:
        for c in [1.2, 1.4, 2.0]:
            for rm in [40, 80]:
                teams.append(
                    _team(
                        MCTSAgent,
                        name=f"MCTS t{int(tl*1000)}ms c{c} r{rm}",
                        time_limit_sec=tl,
                        exploration_c=c,
                        rollout_max_plies=rm,
                        seed=0,
                    )
                )

    # ExpectiMinimax variants
    for d in [4, 6]:
        for tl in [0.08, 0.20]:
            for p in [0.70, 0.85, 0.95]:
                teams.append(_team(ExpectiMiniMaxAgent, name=f"ExpectiMM d{d} t{int(tl*1000)}ms p{int(p*100)}", depth=d, time_limit_sec=tl, p_best=p))

    # Minimax variants
    for d in [3, 5, 7, 9, 11]:
        for tl in [0.05, 0.10, 0.20, 0.35, 0.75]:
            for temp in [0, 25, 75]:
                teams.append(_team(MinimaxAgent, name=f"MM d{d} t{int(tl*1000)}ms temp{temp}", depth=d, time_limit_sec=tl, temperature=temp))

    return teams


def main() -> None:
    roster = build_roster()
    print(A.bold(f"Roster size: {len(roster)} teams"))

    gpp = input("Games per pairing (default 1): ").strip()
    games_per_pair = int(gpp) if gpp else 1

    ppt = input("Pairings per team per stage (default 4): ").strip()
    stage_pairings_per_team = int(ppt) if ppt else 4

    mg = input("Min games/team before prune (default 8): ").strip()
    min_games_before_prune = int(mg) if mg else 8

    kf = input("Keep fraction each prune (default 0.60): ").strip()
    keep_fraction = float(kf) if kf else 0.60

    fk = input("Final keep (default 3): ").strip()
    final_keep = int(fk) if fk else 3

    sd = input("Seed (default 1234): ").strip()
    seed = int(sd) if sd else 1234

    mw = input("Max workers (default = cpu cores, capped at 6): ").strip()
    max_workers = int(mw) if mw else None

    bp = input("Batch pairings per worker task (default 12): ").strip()
    batch_pairings = int(bp) if bp else 12

    ms = input("Max stages (blank = no cap, recommended 3 for testing): ").strip()
    max_stages = int(ms) if ms else None

    z = input("Z for Wilson LCB (default 1.28): ").strip()
    prune_z = float(z) if z else 1.28

    mt = input("ms_target (default 50): ").strip()
    ms_target = float(mt) if mt else 50.0

    sa = input("speed_alpha (default 0.35): ").strip()
    speed_alpha = float(sa) if sa else 0.35

    sm = input("speed_min_factor (default 0.75): ").strip()
    speed_min_factor = float(sm) if sm else 0.75

    league_auto_prune(
        roster,
        games_per_pair=games_per_pair,
        stage_pairings_per_team=stage_pairings_per_team,
        min_games_before_prune=min_games_before_prune,
        keep_fraction=keep_fraction,
        final_keep=final_keep,
        seed=seed,
        max_workers=max_workers,
        batch_pairings=batch_pairings,
        max_stages=max_stages,
        prune_z=prune_z,
        ms_target=ms_target,
        speed_alpha=speed_alpha,
        speed_min_factor=speed_min_factor,
        export_csv=True,
    )


if __name__ == "__main__":
    main()
