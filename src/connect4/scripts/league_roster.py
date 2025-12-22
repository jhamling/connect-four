from __future__ import annotations

import inspect
from functools import partial
from typing import List

from .league_types import Team


def _make_agent(cls, kwargs: dict) -> object:
    sig = inspect.signature(cls)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return cls(**filtered)


def _team(cls, *, name: str, **kwargs) -> Team:
    return Team(name, partial(_make_agent, cls, {"name": name, **kwargs}))


def build_roster() -> List[Team]:

    from connect4.ai.ai_agents.random_agent import RandomAgent
    from connect4.ai.ai_agents.greedy_agent import GreedyAgent
    from connect4.ai.ai_agents.tactical_agent import TacticalAgent
    from connect4.ai.ai_agents.tactical_greedy_agent import TacticalGreedyAgent
    from connect4.ai.ai_agents.beam_agent import BeamAgent
    from connect4.ai.ai_agents.beam2_agent import Beam2Agent
    from connect4.ai.ai_agents.minimax_agent import MinimaxAgent
    from connect4.ai.ai_agents.expectiminimax_agent import ExpectiMiniMaxAgent
    from connect4.ai.ai_agents.weighted_random_agent import WeightedRandomAgent
    from connect4.ai.ai_agents.heuristic_agent import HeuristicAgent
    from connect4.ai.ai_agents.mcts_agent import MCTSAgent

    teams: List[Team] = []

    teams.append(_team(RandomAgent, name="Random"))

    for seed in [0, 1, 2]:
        teams.append(_team(WeightedRandomAgent, name=f"WeightedRandom seed{seed}", seed=seed))

    for temp in [0, 5, 15]:
        for tl in [0.0, 0.01, 0.03]:
            teams.append(_team(GreedyAgent, name=f"Greedy temp{temp} t{int(tl*1000)}ms", temperature=temp, time_limit_sec=tl))

    for temp in [0, 5, 15]:
        for tl in [0.0, 0.01, 0.03]:
            teams.append(_team(TacticalGreedyAgent, name=f"TacticalGreedy temp{temp} t{int(tl*1000)}ms", temperature=temp, time_limit_sec=tl))

    for center_bias in [True, False]:
        for seed in [0, 1, 2]:
            cb = "center" if center_bias else "nocenter"
            teams.append(_team(TacticalAgent, name=f"Tactical {cb} seed{seed}", center_bias=center_bias, seed=seed))

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
                            seed=0,
                        )
                    )

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
                                root_order="center",
                            )
                        )

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

    for d in [4, 6]:
        for tl in [0.08, 0.20]:
            for p in [0.70, 0.85, 0.95]:
                teams.append(_team(ExpectiMiniMaxAgent, name=f"ExpectiMM d{d} t{int(tl*1000)}ms p{int(p*100)}", depth=d, time_limit_sec=tl, p_best=p))

    for d in [3, 5, 7, 9, 11]:
        for tl in [0.05, 0.10, 0.20, 0.35, 0.75]:
            for temp in [0, 25, 75]:
                teams.append(_team(MinimaxAgent, name=f"MM d{d} t{int(tl*1000)}ms temp{temp}", depth=d, time_limit_sec=tl, temperature=temp))

    return teams
