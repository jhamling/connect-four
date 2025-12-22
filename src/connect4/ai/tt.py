from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class TTEntry:
    depth: int
    score: float


class TranspositionTable:
    def __init__(self) -> None:
        self._d: dict[tuple[int, str], TTEntry] = {}

    def get(self, key_hash: int, to_play: str, depth: int) -> float | None:
        entry = self._d.get((key_hash, to_play))
        if entry is None:
            return None
        # Only reuse if stored result is at least as deep
        if entry.depth >= depth:
            return entry.score
        return None

    def put(self, key_hash: int, to_play: str, depth: int, score: float) -> None:
        existing = self._d.get((key_hash, to_play))
        if existing is None or depth >= existing.depth:
            self._d[(key_hash, to_play)] = TTEntry(depth=depth, score=score)
