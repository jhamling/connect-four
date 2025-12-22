from __future__ import annotations
from typing import Optional

from connect4.types import Move


def parse_move(raw: str, cols: int) -> Optional[Move]:
    s = raw.strip().lower()
    if s in {"q", "quit", "exit"}:
        return None
    if not s.isdigit():
        raise ValueError("Invalid input. Enter a number or q.")
    col = int(s) - 1
    if col < 0 or col >= cols:
        raise ValueError(f"Column must be between 1 and {cols}.")
    return Move(col)
