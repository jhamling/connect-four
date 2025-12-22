# src/connect4/config.py

from __future__ import annotations

ROWS = 6
COLS = 7
CONNECT_N = 4

# UI toggles
USE_COLOR = True
CLEAR_SCREEN = True

# “AI thinking” effect
AI_THINKING_SPINNER = True
AI_THINK_DELAY_SEC = 1  # short pause so AI moves aren’t instant

# AI defaults (we will use later for minimax)
MINIMAX_DEPTH = 5
