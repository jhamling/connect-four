from __future__ import annotations
import random
from typing import Dict, Tuple

from connect4.config import ROWS, COLS
from connect4.types import Player

# Deterministic for reproducibility
_rng = random.Random(1337)

# Per-square piece keys
Z: Dict[Tuple[int, int, Player], int] = {}
for r in range(ROWS):
    for c in range(COLS):
        Z[(r, c, "X")] = _rng.getrandbits(64)
        Z[(r, c, "O")] = _rng.getrandbits(64)

# Side-to-move toggle (useful if you want the hash to include turn)
SIDE_TO_MOVE_O: int = _rng.getrandbits(64)


def piece_key(r: int, c: int, p: Player) -> int:
    return Z[(r, c, p)]


def side_key(to_play: Player) -> int:
    # Only need one toggle; X == 0, O == SIDE_TO_MOVE_O
    return SIDE_TO_MOVE_O if to_play == "O" else 0

