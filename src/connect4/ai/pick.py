from __future__ import annotations

import random


def random_ai_agent():
    """
    Randomly select one AI agent configuration from your available agent families.
    Keeps compute reasonable while still showcasing variety.
    """
       
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

    factories = [
        lambda: RandomAgent(),

        lambda: GreedyAgent(
            name=f"Greedy (temp {random.choice([0, 25, 75])})",
            temperature=random.choice([0, 25, 75]),
            time_limit_sec=0.0,
        ),

        # WIP
        lambda: TacticalAgent(),

        lambda: TacticalGreedyAgent(
            name=f"TacticalGreedy (temp {random.choice([0, 25, 75])})",
            temperature=random.choice([0, 25, 75]),
            time_limit_sec=0.0,
        ),

        lambda: BeamAgent(
            name="BeamSearch",
            depth=random.choice([5, 7, 9]),
            width=random.choice([2, 3, 4]),
            time_limit_sec=random.choice([0.10, 0.20, 0.35]),
            temperature=random.choice([0, 25, 75]),
        ),

        # WIP
        lambda: Beam2Agent(),

        lambda: MinimaxAgent(
            name="Minimax",
            depth=random.choice([5, 7, 9]),
            time_limit_sec=random.choice([0.10, 0.20, 0.35]),
            temperature=random.choice([0, 25, 75]),
        ),

        lambda: ExpectiMiniMaxAgent(
            name="ExpectiMinimax",
            depth=random.choice([5, 7, 9]),
            time_limit_sec=random.choice([0.15, 0.25, 0.40]),
            p_best=random.choice([0.60, 0.80, 0.95]),
        ),

        # WIP
        lambda: WeightedRandomAgent(),

        # WIP
        lambda: HeuristicAgent(),

        # WIP
        lambda: MCTSAgent(),
    ]

    agent = random.choice(factories)()

    bits = []
    if hasattr(agent, "depth"):
        bits.append(f"d{getattr(agent, 'depth')}")
    if hasattr(agent, "time_limit_sec"):
        bits.append(f"{int(getattr(agent, 'time_limit_sec') * 1000)}ms")
    if hasattr(agent, "width"):
        bits.append(f"w{getattr(agent, 'width')}")
    if hasattr(agent, "temperature"):
        bits.append(f"temp{getattr(agent, 'temperature')}")
    if hasattr(agent, "p_best"):
        bits.append(f"p{getattr(agent, 'p_best'):.2f}")

    if bits:
        agent.name = f"{agent.name} ({' '.join(bits)})"

    return agent
