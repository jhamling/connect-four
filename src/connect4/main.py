
from __future__ import annotations

from connect4.game.controller import run_game
from connect4.ai.random_agent import RandomAgent
from connect4.ai.minimax_agent import MinimaxAgent
from connect4.types import Move
from connect4.game.state import GameState


class HumanAgent:
    name = "Human"

    def choose_move(self, state: GameState) -> Move:
        raise RuntimeError("HumanAgent.choose_move should never be called.")


def main() -> None:
    print("Select mode:")
    print("1) Human vs Human")
    print("2) Human vs Random AI")
    print("3) Human vs Minimax AI (Easy)")
    print("4) Human vs Minimax AI (Medium)")
    print("5) Human vs Minimax AI (Hard)")
    print("6) AI vs AI (Minimax vs Minimax)")

    choice = input("Choice: ").strip()

    if choice == "2":
        run_game(HumanAgent(), RandomAgent())
        return

    if choice == "3":
        run_game(HumanAgent(), MinimaxAgent(name="Minimax AI (Easy)", depth=3))
        return

    if choice == "4":
        run_game(HumanAgent(), MinimaxAgent(name="Minimax AI (Medium)", depth=5))
        return

    if choice == "5":
        run_game(HumanAgent(), MinimaxAgent(name="Minimax AI (Hard)", depth=7))
        return

    if choice == "6":
        ai_x = MinimaxAgent(name="Minimax X", depth=7)
        ai_o = MinimaxAgent(name="Minimax O", depth=7)
        run_game(ai_x, ai_o, show_thinking=False)
        return

    run_game(HumanAgent(), HumanAgent())


if __name__ == "__main__":
    main()

