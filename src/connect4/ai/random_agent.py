from __future__ import annotations
import random

from connect4.ai.base import Agent
from connect4.game.state import GameState
from connect4.types import Move


class RandomAgent:
    name = "Random AI"

    def choose_move(self, state: GameState) -> Move:
        moves = state.board.valid_moves()
        if not moves:
            raise ValueError("No valid moves.")
        return random.choice(moves)
