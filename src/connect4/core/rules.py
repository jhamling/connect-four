from __future__ import annotations
from typing import Optional

from connect4.config import ROWS, COLS
from connect4.types import Player
from connect4.core.board import Board


def check_winner(board: Board) -> Optional[Player]:
    g = board.grid

    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            p = g[r][c]
            if p and p == g[r][c + 1] == g[r][c + 2] == g[r][c + 3]:
                return p

    # Vertical
    for r in range(ROWS - 3):
        for c in range(COLS):
            p = g[r][c]
            if p and p == g[r + 1][c] == g[r + 2][c] == g[r + 3][c]:
                return p

    # Diagonal down-right
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            p = g[r][c]
            if p and p == g[r + 1][c + 1] == g[r + 2][c + 2] == g[r + 3][c + 3]:
                return p

    # Diagonal up-right
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            p = g[r][c]
            if p and p == g[r - 1][c + 1] == g[r - 2][c + 2] == g[r - 3][c + 3]:
                return p

    return None


def is_draw(board: Board) -> bool:
    return board.is_full() and check_winner(board) is None
