from __future__ import annotations

from connect4.ai.base import Agent
from connect4.core.board import Board
from connect4.game.state import GameState
from connect4.game.results import winner_with_line, draw
from connect4.ui.render import render
from connect4.ui.prompts import parse_move
from connect4.ui.effects import ai_thinking
from connect4.types import Player, Move


def other(player: Player) -> Player:
    return "O" if player == "X" else "X"


def run_game(agent_x: Agent, agent_o: Agent, show_thinking: bool = True) -> None:
    state = GameState(board=Board(), current="X", last_status="Player X starts.")

    winning_line = None

    while True:
        render(state.board, state.last_status, highlight=winning_line)

        w = winner_with_line(state.board)
        if w is not None:
            player, line = w
            winning_line = line
            render(state.board, f"Player {player} wins!", highlight=winning_line)
            return

        if draw(state.board):
            render(state.board, "Draw game.")
            return

        current_agent = agent_x if state.current == "X" else agent_o

        try:
            if current_agent.name == "Human":
                raw = input(f"Player {state.current} move: ")
                move = parse_move(raw, state.board.cols)
                if move is None:
                    render(state.board, "Game quit.")
                    return
            else:
                if show_thinking:
                    ai_thinking(f"{current_agent.name}")

                move = current_agent.choose_move(state)

                # Show search stats if available
                info = getattr(current_agent, "last_info", None)
                if info:
                    state.last_status = (
                        f"{current_agent.name} chose {info.get('move_col')} | "
                        f"d={info.get('depth')} | "
                        f"nodes={info.get('nodes')} | "
                        f"tt={info.get('tt_hits')} | "
                        f"cut={info.get('cutoffs')} | "
                        f"eval={info.get('eval')} | "
                        f"{info.get('time_ms')}ms"
                    )

            # Apply the move (works for BOTH human and AI)
            state.board.drop(move, state.current)
            state.current = other(state.current)

            # Keep AI stats line if it exists; otherwise show generic turn line
            if not state.last_status or "chose" not in state.last_status:
                state.last_status = f"Player {state.current}'s turn."
            else:
                state.last_status += f" | Next: Player {state.current}"

            winning_line = None

        except ValueError as e:
            state.last_status = str(e)

