from __future__ import annotations

from connect4.ai.base import Agent
from connect4.core.board import Board
from connect4.game.state import GameState
from connect4.game.results import winner, draw
from connect4.ui.render import render
from connect4.ui.prompts import parse_move
from connect4.ui.effects import ai_thinking
from connect4.types import Player, Move


def other(player: Player) -> Player:
    return "O" if player == "X" else "X"


def run_game(agent_x: Agent, agent_o: Agent) -> None:
    state = GameState(board=Board(), current="X", last_status="Player X starts.")

    while True:
        render(state.board, state.last_status)

        w = winner(state.board)
        if w is not None:
            render(state.board, f"Player {w} wins!")
            return
        if draw(state.board):
            render(state.board, "Draw game.")
            return

        current_agent = agent_x if state.current == "X" else agent_o

        try:
            # Human agent is indicated by name convention in main; simplest:
            if current_agent.name == "Human":
                raw = input(f"Player {state.current} move: ")
                move = parse_move(raw, state.board.cols)
                if move is None:
                    render(state.board, "Game quit.")
                    return
            else:
                ai_thinking(f"{current_agent.name}")
                move = current_agent.choose_move(state)

            state.board.drop(move, state.current)
            state.current = other(state.current)
            state.last_status = f"Player {state.current}'s turn."

        except ValueError as e:
            state.last_status = str(e)
