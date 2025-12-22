from __future__ import annotations
from typing import Optional, List, Tuple

from connect4.core.board import Board
from connect4.core.rules import check_winner_with_line, is_draw
from connect4.types import Player

Coord = Tuple[int, int]


def winner_with_line(board: Board) -> Optional[Tuple[Player, List[Coord]]]:
    return check_winner_with_line(board)


def draw(board: Board) -> bool:
    return is_draw(board)

