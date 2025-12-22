from __future__ import annotations

from connect4.types import Move
from connect4.game.state import GameState


class HumanAgent:
    name = "Human"

    def choose_move(self, state: GameState) -> Move:
        raise RuntimeError("HumanAgent.choose_move should never be called.")
