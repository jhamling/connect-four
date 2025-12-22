from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

from connect4.config import ROWS, COLS
from connect4.types import Cell, Player, Move


@dataclass(slots=True)
class Board:
    rows: int = ROWS
    cols: int = COLS
    grid: List[List[Cell]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.grid:
            self.grid = [[None for _ in range(self.cols)] for _ in range(self.rows)]

    def copy(self) -> "Board":
        b = Board(self.rows, self.cols)
        b.grid = [row[:] for row in self.grid]
        return b

    def valid_moves(self) -> List[Move]:
        return [Move(c) for c in range(self.cols) if self.grid[0][c] is None]

    def is_full(self) -> bool:
        return all(self.grid[0][c] is not None for c in range(self.cols))

    def drop(self, col: Move, player: Player) -> int:
        c = int(col)
        if c < 0 or c >= self.cols:
            raise ValueError("Column out of range.")
        if self.grid[0][c] is not None:
            raise ValueError("Column is full.")

        for r in range(self.rows - 1, -1, -1):
            if self.grid[r][c] is None:
                self.grid[r][c] = player
                return r

        raise ValueError("Column is full.")  # defensive

    def undo(self, col: Move) -> None:
        """
        Remove the top-most piece from a column.
        Useful for AI search.
        """
        c = int(col)
        for r in range(self.rows):
            if self.grid[r][c] is not None:
                self.grid[r][c] = None
                return
        raise ValueError("Cannot undo: column is empty.")
