from __future__ import annotations

from connect4.game.controller import run_game
from connect4.ai.random_agent import RandomAgent
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
    choice = input("Choice: ").strip()

    if choice == "2":
        run_game(HumanAgent(), RandomAgent())
    else:
        run_game(HumanAgent(), HumanAgent())


if __name__ == "__main__":
    main()
