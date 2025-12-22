from __future__ import annotations
from typing import Optional, List, Tuple

from connect4.config import ROWS, COLS
from connect4.types import Player
from connect4.core.board import Board

Coord = Tuple[int, int]  # (row, col)


def check_winner_with_line(board: Board) -> Optional[Tuple[Player, List[Coord]]]:
    g = board.grid

    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            p = g[r][c]
            if p and p == g[r][c + 1] == g[r][c + 2] == g[r][c + 3]:
                return p, [(r, c + i) for i in range(4)]

    # Vertical
    for r in range(ROWS - 3):
        for c in range(COLS):
            p = g[r][c]
            if p and p == g[r + 1][c] == g[r + 2][c] == g[r + 3][c]:
                return p, [(r + i, c) for i in range(4)]

    # Diagonal down-right
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            p = g[r][c]
            if p and p == g[r + 1][c + 1] == g[r + 2][c + 2] == g[r + 3][c + 3]:
                return p, [(r + i, c + i) for i in range(4)]

    # Diagonal up-right
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            p = g[r][c]
            if p and p == g[r - 1][c + 1] == g[r - 2][c + 2] == g[r - 3][c + 3]:
                return p, [(r - i, c + i) for i in range(4)]

    return None


def check_winner(board: Board) -> Optional[Player]:
    res = check_winner_with_line(board)
    return res[0] if res else None


def is_draw(board: Board) -> bool:
    return board.is_full() and check_winner(board) is None
