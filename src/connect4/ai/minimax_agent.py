from __future__ import annotations

from connect4.ai.base import Agent
from connect4.game.state import GameState
from connect4.types import Move


class MinimaxAgent:
    """
    Included now so the file structure is stable.
    """
    name = "Minimax AI"

    def __init__(self, depth: int = 5) -> None:
        self.depth = depth

    def choose_move(self, state: GameState) -> Move:
        raise NotImplementedError("Minimax not implemented yet.")
