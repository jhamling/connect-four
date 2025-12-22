
# src/connect4/core/board.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

from connect4.config import ROWS, COLS
from connect4.types import Cell, Player, Move
from connect4.core.zobrist import piece_key


@dataclass(slots=True)
class Board:
    rows: int = ROWS
    cols: int = COLS
    grid: List[List[Cell]] = field(default_factory=list)
    zhash: int = 0  # incremental Zobrist hash (pieces only)

    def __post_init__(self) -> None:
        if not self.grid:
            self.grid = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        # If a board is created with an existing grid, compute zhash once.
        # (Normal flow starts empty so this is basically free.)
        self._recompute_zhash()

    def _recompute_zhash(self) -> None:
        h = 0
        for r in range(self.rows):
            for c in range(self.cols):
                p = self.grid[r][c]
                if p is not None:
                    h ^= piece_key(r, c, p)
        self.zhash = h

    def copy(self) -> "Board":
        b = Board(self.rows, self.cols)
        b.grid = [row[:] for row in self.grid]
        b.zhash = self.zhash
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
                # XOR in the new piece
                self.zhash ^= piece_key(r, c, player)
                return r

        raise ValueError("Column is full.")  # defensive

    def undo(self, col: Move) -> None:
        """
        Remove the top-most piece from a column.
        Useful for AI search.
        """
        c = int(col)
        for r in range(self.rows):
            p = self.grid[r][c]
            if p is not None:
                self.grid[r][c] = None
                # XOR out the removed piece
                self.zhash ^= piece_key(r, c, p)
                return
        raise ValueError("Cannot undo: column is empty.")
