from __future__ import annotations

import time

from connect4.game.controller import run_game
from connect4.ai.pick import random_ai_agent
from connect4.ui.human import HumanAgent


def run_menu() -> None:
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
        from connect4.scripts.league_main import main as league_main
        league_main()
        return

    print("\nInvalid choice. Defaulting to Human vs Human.\n")
    time.sleep(3)
    run_game(HumanAgent(), HumanAgent())
