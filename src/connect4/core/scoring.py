from __future__ import annotations
from typing import List, Tuple, Optional

from connect4.core.board import Board
from connect4.types import Player, Cell
from connect4.core.rules import check_winner


def _other(p: Player) -> Player:
    return "O" if p == "X" else "X"


def _is_playable_empty(board: Board, r: int, c: int) -> bool:
    """
    An empty cell is playable if it is on the bottom row or there is a piece below it.
    This is the key "gravity-aware" rule for Connect 4.
    """
    if board.grid[r][c] is not None:
        return False
    return (r == board.rows - 1) or (board.grid[r + 1][c] is not None)


def _score_window(board: Board, coords: List[Tuple[int, int]], player: Player) -> int:
    opp = _other(player)
    cells: List[Cell] = [board.grid[r][c] for (r, c) in coords]

    p_count = cells.count(player)
    o_count = cells.count(opp)
    e_count = cells.count(None)

    # mixed window: both players present => no direct line potential
    if p_count > 0 and o_count > 0:
        return 0

    # Find empty squares and whether they are playable
    empties = [(r, c) for (r, c), v in zip(coords, cells) if v is None]
    playable_empties = [pos for pos in empties if _is_playable_empty(board, pos[0], pos[1])]

    score = 0

    # Player lines
    if p_count == 4:
        score += 10_000
    elif p_count == 3 and e_count == 1:
        # big reward only if the missing spot can actually be played now
        score += 250 if len(playable_empties) == 1 else 40
    elif p_count == 2 and e_count == 2:
        score += 18
    elif p_count == 1 and e_count == 3:
        score += 2

    # Opponent threats (penalize more)
    if o_count == 4:
        score -= 10_000
    elif o_count == 3 and e_count == 1:
        score -= 280 if len(playable_empties) == 1 else 50
    elif o_count == 2 and e_count == 2:
        score -= 20

    return score


def evaluate(board: Board, player: Player) -> int:
    w = check_winner(board)
    if w == player:
        return 1_000_000
    if w is not None:
        return -1_000_000

    score = 0

    # center column preference (strong in Connect 4)
    center = board.cols // 2
    for r in range(board.rows):
        if board.grid[r][center] == player:
            score += 6
        elif board.grid[r][center] == _other(player):
            score -= 6

    rows, cols = board.rows, board.cols

    # Horizontal
    for r in range(rows):
        for c in range(cols - 3):
            coords = [(r, c + i) for i in range(4)]
            score += _score_window(board, coords, player)

    # Vertical
    for r in range(rows - 3):
        for c in range(cols):
            coords = [(r + i, c) for i in range(4)]
            score += _score_window(board, coords, player)

    # Diagonal down-right
    for r in range(rows - 3):
        for c in range(cols - 3):
            coords = [(r + i, c + i) for i in range(4)]
            score += _score_window(board, coords, player)

    # Diagonal up-right
    for r in range(3, rows):
        for c in range(cols - 3):
            coords = [(r - i, c + i) for i in range(4)]
            score += _score_window(board, coords, player)

    return score
