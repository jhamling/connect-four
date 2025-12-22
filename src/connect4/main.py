from __future__ import annotations

import random
import time

from connect4.game.controller import run_game
from connect4.types import Move
from connect4.game.state import GameState


class HumanAgent:
    name = "Human"

    def choose_move(self, state: GameState) -> Move:
        raise RuntimeError("HumanAgent.choose_move should never be called.")


def random_ai_agent():
    """
    Randomly select one AI agent configuration from your available agent families.
    Keeps compute reasonable while still showcasing variety.
    """
    from connect4.ai.random_agent import RandomAgent
    
    from connect4.ai.greedy_agent import GreedyAgent
    from connect4.ai.tactical_agent import TacticalAgent 
    from connect4.ai.tactical_greedy_agent import TacticalGreedyAgent

    from connect4.ai.beam_agent import BeamAgent
    from connect4.ai.beam2_agent import Beam2Agent 
   
    from connect4.ai.minimax_agent import MinimaxAgent
    from connect4.ai.expectiminimax_agent import ExpectiMiniMaxAgent
    
    from connect4.ai.weighted_random_agent import WeightedRandomAgent 
    
    from connect4.ai.heuristic_agent import HeuristicAgent
    
    factories = [
        # Pure random baseline
        lambda: RandomAgent(),

        # Cheap + fast agents (with controlled randomness)
        lambda: GreedyAgent(
            name=f"Greedy (temp {random.choice([0, 25, 75])})",
            temperature=random.choice([0, 25, 75]),
            time_limit_sec=0.0,
        ),

        # WIP
        lambda: TacticalAgent(

            ),


        lambda: TacticalGreedyAgent(
            name=f"TacticalGreedy (temp {random.choice([0, 25, 75])})",
            temperature=random.choice([0, 25, 75]),
            time_limit_sec=0.0,
        ),

        # Beam search (varied width/depth/time)
        lambda: BeamAgent (
            name="BeamSearch",
            depth=random.choice([5, 7, 9]),
            width=random.choice([2, 3, 4]),
            time_limit_sec=random.choice([0.10, 0.20, 0.35]),
            temperature=random.choice([0, 25, 75]),
        ),

        # WIP
        lambda: Beam2Agent(
        
        ),

        # Minimax (varied depth/time/temp)
        lambda: MinimaxAgent(
            name="Minimax",
            depth=random.choice([5, 7, 9]),
            time_limit_sec=random.choice([0.10, 0.20, 0.35]),
            temperature=random.choice([0, 25, 75]),
        ),

        # Expectiminimax (varied depth/time/p_best)
        lambda: ExpectiMiniMaxAgent(
            name="ExpectiMinimax",
            depth=random.choice([5, 7, 9]),
            time_limit_sec=random.choice([0.15, 0.25, 0.40]),
            p_best=random.choice([0.60, 0.80, 0.95]),
        ),

        # WIP
        lambda: WeightedRandomAgent(

        ),
        
        # WIP
        lambda: HeuristicAgent(

        ),
    ]

    agent = random.choice(factories)()

    # Make the name more informative if the agent exposes key knobs
    # (This does not break anything if attributes don't exist.)
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


def main() -> None:
    print("Select mode:")
    print("1) Human vs Human")
    print("2) Human vs AI (random model)")
    print("3) Run AI League (auto-prune tournament)")

    choice = input("Choice: ").strip()

    if choice == "1":
        p1 = HumanAgent()
        p2 = HumanAgent()
        print(f"\nStarting game: {p1.name} vs {p2.name}")
        print("Game will start in 3 seconds...\n")
        time.sleep(3)
        run_game(p1, p2)
        return

    if choice == "2":
        human = HumanAgent()
        ai = random_ai_agent()
        print(f"\nStarting game: {human.name} vs {ai.name}")
        print("Game will start in 3 seconds...\n")
        time.sleep(3)
        run_game(human, ai)
        return

    if choice == "3":
        print("\nStarting AI League Tournament in 3 seconds...\n")
        time.sleep(3)
        from connect4.scripts.league import main as league_main

        league_main()
        return

    print("\nInvalid choice. Defaulting to Human vs Human.\n")
    time.sleep(3)
    run_game(HumanAgent(), HumanAgent())


if __name__ == "__main__":
    main()

