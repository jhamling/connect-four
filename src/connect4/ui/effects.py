from __future__ import annotations
import sys
import time

from connect4.config import AI_THINKING_SPINNER, AI_THINK_DELAY_SEC


def ai_thinking(label: str = "AI is thinking") -> None:
    """
    Small user-visible delay + optional spinner so AI moves are not instant.
    """
    if AI_THINK_DELAY_SEC <= 0:
        return

    if not AI_THINKING_SPINNER:
        time.sleep(AI_THINK_DELAY_SEC)
        return

    frames = ["|", "/", "-", "\\"]
    start = time.time()
    i = 0
    while (time.time() - start) < AI_THINK_DELAY_SEC:
        sys.stdout.write(f"\r{label}... {frames[i % len(frames)]}")
        sys.stdout.flush()
        time.sleep(0.08)
        i += 1
    sys.stdout.write("\r" + (" " * (len(label) + 10)) + "\r")
    sys.stdout.flush()
