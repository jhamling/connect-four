from __future__ import annotations
from typing import Optional, Iterable, Tuple, Set

from connect4.config import CLEAR_SCREEN
from connect4.core.board import Board
from connect4.types import Cell
from connect4.ui.colors import c, BOLD, DIM, FG_CYAN, FG_GRAY, FG_RED, FG_YELLOW, REVERSE

Coord = Tuple[int, int]


def _piece(cell: Cell) -> str:
    if cell is None:
        return c("·", FG_GRAY)
    if cell == "X":
        return c("X", FG_RED)
    return c("O", FG_YELLOW)


def clear_screen() -> None:
    if CLEAR_SCREEN:
        print("\033[2J\033[H", end="")


def render(board: Board, status: str = "", highlight: Optional[Iterable[Coord]] = None) -> None:
    clear_screen()

    hl: Set[Coord] = set(highlight) if highlight else set()

    print(c("CONNECT 4", BOLD))
    if status:
        print(c(status, FG_CYAN))
    else:
        print()

    nums = "   " + " ".join(str(i + 1) for i in range(board.cols))
    print(c(nums, DIM))

    for r in range(board.rows):
        parts = []
        for cidx in range(board.cols):
            p = _piece(board.grid[r][cidx])
            if (r, cidx) in hl:
                # reverse-video highlight on the cell character
                p = c(p, REVERSE) if "\033[" not in p else f"{REVERSE}{p}{c('', '')}"
                # simpler/safer approach: wrap with REVERSE and RESET
                p = f"{REVERSE}{p}{'\033[0m'}"
            parts.append(p)

        row = " | " + " ".join(parts) + " |"
        print(row)

    print(c("   " + "—" * (2 * board.cols - 1), DIM))
    print(c("   Enter 1-7 to drop. Enter q to quit.", DIM))

