# src/connect4/types.py

from __future__ import annotations
from typing import Literal, Optional, NewType

Player = Literal["X", "O"]
Cell = Optional[Player]
Move = NewType("Move", int)   # column index 0..6
