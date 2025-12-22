from __future__ import annotations
from typing import Optional

from connect4.core.board import Board
from connect4.core.rules import check_winner, is_draw
from connect4.types import Player


def winner(board: Board) -> Optional[Player]:
    return check_winner(board)


def draw(board: Board) -> bool:
    return is_draw(board)
