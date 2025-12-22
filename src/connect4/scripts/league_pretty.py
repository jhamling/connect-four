from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Iterable, List, Sequence


def term_width(default: int = 120) -> int:
    try:
        return shutil.get_terminal_size(fallback=(default, 24)).columns
    except Exception:
        return default


def clamp(s: str, width: int) -> str:
    """Hard clamp with ellipsis to keep columns aligned."""
    if width <= 0:
        return ""
    if len(s) <= width:
        return s
    if width <= 1:
        return s[:width]
    return s[: width - 1] + "…"


def hr(char: str = "─", width: int | None = None) -> str:
    w = width or term_width()
    return char * max(10, w)


# -----------------------------
# ANSI (safe + optional)
# -----------------------------
class Ansi:
    def __init__(self) -> None:
        self.enabled = (
            os.environ.get("NO_COLOR") is None
            and os.environ.get("TERM") not in (None, "", "dumb")
        )

    def _w(self, s: str, code: str) -> str:
        if not self.enabled:
            return s
        return f"\x1b[{code}m{s}\x1b[0m"

    def b(self, s: str) -> str: return self._w(s, "1")
    def dim(self, s: str) -> str: return self._w(s, "2")

    def red(self, s: str) -> str: return self._w(s, "31")
    def green(self, s: str) -> str: return self._w(s, "32")
    def yellow(self, s: str) -> str: return self._w(s, "33")
    def cyan(self, s: str) -> str: return self._w(s, "36")
    def mag(self, s: str) -> str: return self._w(s, "35")


A = Ansi()


@dataclass(frozen=True)
class Col:
    title: str
    width: int
    align: str = "left"  # left|right


def format_row(values: Sequence[str], cols: Sequence[Col]) -> str:
    out: List[str] = []
    for v, c in zip(values, cols):
        s = clamp(str(v), c.width)
        if c.align == "right":
            out.append(s.rjust(c.width))
        else:
            out.append(s.ljust(c.width))
    return "  ".join(out)


def print_table(
    title: str,
    cols: Sequence[Col],
    rows: Iterable[Sequence[str]],
    *,
    width: int | None = None,
) -> None:
    w = width or term_width()
    print(A.cyan(A.b(title)))
    header = format_row([c.title for c in cols], cols)
    print(A.dim(header))
    print(A.dim(hr("─", w)))
    for r in rows:
        print(format_row(r, cols))
    print(A.dim(hr("─", w)))
