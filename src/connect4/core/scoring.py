from __future__ import annotations
from connect4.core.board import Board
from connect4.types import Player


def evaluate(board: Board, player: Player) -> int:
    opp: Player = "O" if player == "X" else "X"

    center_col = board.cols // 2
    score_player = 0
    score_opp = 0

    for r in range(board.rows):
        if board.grid[r][center_col] == player:
            score_player += 3
        elif board.grid[r][center_col] == opp:
            score_opp += 3

    return score_player - score_opp
