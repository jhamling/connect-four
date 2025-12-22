from __future__ import annotations

from connect4.game.controller import run_game
from connect4.ai.minimax_agent import MinimaxAgent
from connect4.core.board import Board
from connect4.game.state import GameState
from connect4.core.rules import check_winner


def play_single_game(ai_x, ai_o) -> str | None:
    state = GameState(board=Board(), current="X", last_status="")
    run_game(ai_x, ai_o, show_thinking=False)

    return check_winner(state.board)


def tournament(num_games: int = 5) -> None:
    wins = {"X": 0, "O": 0, "draw": 0}

    for i in range(num_games):
        ai_x = MinimaxAgent(name="Minimax X", depth=7)
        ai_o = MinimaxAgent(name="Minimax O", depth=7)

        winner = play_single_game(ai_x, ai_o)

        if winner is None:
            wins["draw"] += 1
        else:
            wins[winner] += 1

        print(f"Game {i + 1}/{num_games} complete")

    print("\n=== TOURNAMENT RESULTS ===")
    print(f"X wins:    {wins['X']}")
    print(f"O wins:    {wins['O']}")
    print(f"Draws:     {wins['draw']}")


if __name__ == "__main__":
    tournament(5)
