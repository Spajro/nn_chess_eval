import os
import sys
import time
import chess
import chess.pgn

from src.loading.raw_data_gather import gather
from src.patches import GAMES_PATCH, FENS_PATH
from chess.pgn import GameNode, read_game


def load(k: int) -> list[tuple[GameNode, str]]:
    pgn = open(GAMES_PATCH, encoding="utf-8")
    result = []

    game = read_game(pgn)
    count = 0
    while game is not None and count < k:
        if game.variations:
            result.append((game, game.headers["Result"]))
        game = read_game(pgn)
        count += 1

    print("Games count: ", count)
    pgn.close()
    return result


def games_to_unique_fens(games: list[tuple[GameNode, str]]) -> dict[str, tuple[int, int, int]]:
    result = {}
    for game, outcome in games:
        for fen in generate_fen_for_moves(generate_moves_for_games(game)):
            if fen in result:
                result[fen] = append_outcome(outcome, result[fen])
            else:
                result[fen] = append_outcome(outcome, (0, 0, 0))
    return result


def generate_moves_for_games(game: chess.pgn.GameNode) -> list[chess.pgn.ChildNode]:
    result = []
    while game.variations:
        result.append(game.variations[0])
        game = game.variations[0]
    return result


def generate_fen_for_moves(moves: list[chess.pgn.ChildNode]) -> list[str]:
    result = []
    board = chess.Board()
    for move in moves:
        board.push_uci(move.uci())
        result.append(board.fen())
    return result


def filter_non_quiet_position(fens: dict[str, tuple[int, int, int]]) -> dict[str, tuple[int, int, int]]:
    return {fen: fens[fen] for fen in fens if is_quiet(fen)}


def is_quiet(fen: str) -> bool:
    board = chess.Board(fen)
    for move in board.generate_legal_moves():
        if board.piece_at(move.to_square) is not None:
            return False
    return True


def append_outcome(outcome: str, status: tuple[int, int, int]) -> tuple[int, int, int]:
    if outcome == "1-0":
        return status[0] + 1, status[1], status[2]
    elif outcome == "0-1":
        return status[0], status[1], status[2] + 1
    elif outcome == "1/2-1/2":
        return status[0], status[1] + 1, status[2]
    else:
        print("Error")
        return status


if len(sys.argv) < 2:
    print("Usage: python prepare_fens.py <size> [--merge]")
    exit(0)

gather("https://database.lichess.org/standard",
       "lichess_db_standard_rated_2014-05.pgn.zst",
       GAMES_PATCH)

SIZE = int(sys.argv[1])

t1 = time.time()
unique_fens = games_to_unique_fens(load(SIZE))
print("Unique fens count: ", len(unique_fens))
filtered_fens = filter_non_quiet_position(unique_fens)
print("Filtered Fens count: ", len(filtered_fens))
print("Fen gather time: ", time.time() - t1)

with open(FENS_PATH, "a", newline='\n') as file:
    file.writelines(
        [fen + ',' + str(filtered_fens[fen][0]) + ',' + str(filtered_fens[fen][1]) + ',' + str(filtered_fens[fen][2]) + '\n' for fen in filtered_fens])
