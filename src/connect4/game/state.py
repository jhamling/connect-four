from __future__ import annotations
from dataclasses import dataclass

from connect4.core.board import Board
from connect4.types import Player


@dataclass(slots=True)
class GameState:
    board: Board
    current: Player
    last_status: str = "Player X starts."
