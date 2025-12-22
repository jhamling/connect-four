from __future__ import annotations
from connect4.config import USE_COLOR

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
REVERSE = "\033[7m"  # swaps fg/bg; good generic highlight

FG_RED = "\033[31m"
FG_YELLOW = "\033[33m"
FG_CYAN = "\033[36m"
FG_GRAY = "\033[90m"


def c(s: str, code: str) -> str:
    if not USE_COLOR:
        return s
    return f"{code}{s}{RESET}"

