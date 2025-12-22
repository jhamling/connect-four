from __future__ import annotations

from connect4.config import CLEAR_SCREEN
from connect4.core.board import Board
from connect4.types import Cell
from connect4.ui.colors import c, BOLD, DIM, FG_CYAN, FG_GRAY, FG_RED, FG_YELLOW


def _piece(cell: Cell) -> str:
    if cell is None:
        return c("·", FG_GRAY)
    if cell == "X":
        return c("X", FG_RED)
    return c("O", FG_YELLOW)


def clear_screen() -> None:
    if CLEAR_SCREEN:
        print("\033[2J\033[H", end="")


def render(board: Board, status: str = "") -> None:
    clear_screen()

    print(c("CONNECT 4", BOLD))
    if status:
        print(c(status, FG_CYAN))
    else:
        print()

    # column numbers
    nums = "   " + " ".join(str(i + 1) for i in range(board.cols))
    print(c(nums, DIM))

    # grid
    for r in range(board.rows):
        row = " | " + " ".join(_piece(board.grid[r][cidx]) for cidx in range(board.cols)) + " |"
        print(row)

    print(c("   " + "—" * (2 * board.cols - 1), DIM))
    print(c("   Enter 1-7 to drop. Enter q to quit.", DIM))
