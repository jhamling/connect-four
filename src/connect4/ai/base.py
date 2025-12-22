from __future__ import annotations
from typing import Protocol

from connect4.game.state import GameState
from connect4.types import Move


class Agent(Protocol):
    name: str

    def choose_move(self, state: GameState) -> Move:
        ...
