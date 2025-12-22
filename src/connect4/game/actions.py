from __future__ import annotations
from connect4.core.board import Board
from connect4.types import Player, Move


def apply_move(board: Board, move: Move, player: Player) -> int:
    return board.drop(move, player)


def undo_move(board: Board, move: Move) -> None:
    board.undo(move)
