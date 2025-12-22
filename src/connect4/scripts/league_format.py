from __future__ import annotations

import os


class _Ansi:
    def __init__(self) -> None:
        self.enabled = (
            os.environ.get("NO_COLOR") is None
            and os.environ.get("TERM") not in (None, "", "dumb")
        )

    def _wrap(self, s: str, code: str) -> str:
        if not self.enabled:
            return s
        return f"\x1b[{code}m{s}\x1b[0m"

    def bold(self, s: str) -> str: return self._wrap(s, "1")
    def dim(self, s: str) -> str: return self._wrap(s, "2")

    def red(self, s: str) -> str: return self._wrap(s, "31")
    def green(self, s: str) -> str: return self._wrap(s, "32")
    def yellow(self, s: str) -> str: return self._wrap(s, "33")
    def blue(self, s: str) -> str: return self._wrap(s, "34")
    def magenta(self, s: str) -> str: return self._wrap(s, "35")
    def cyan(self, s: str) -> str: return self._wrap(s, "36")


A = _Ansi()


def hr(char: str = "─", n: int = 96) -> str:
    return char * n
