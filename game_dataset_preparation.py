import sys
import time

import chess
import chess.pgn
import concurrent
from src.data_loading import save_dataset_to_csv
from src.raw_data_gather import gather
from src.patches import GAMES_DATASET_PATCH, GAMES_PATCH, STOCKFISH_PATH
from stockfish import Stockfish
from concurrent.futures import ThreadPoolExecutor


def load(k: int) -> [chess.pgn.GameNode]:
    pgn = open(GAMES_PATCH, encoding="utf-8")
    result = []

    game = chess.pgn.read_game(pgn)
    count = 0
    while game is not None and count < k:
        if game.variations:
            result.append(game)
        game = chess.pgn.read_game(pgn)
        count += 1

    print("Games count: ", count)
    pgn.close()
    return result


def generate_moves_for_games(game: chess.pgn.GameNode) -> [str]:
    result = []
    while game.variations:
        result.append(game.variations[0])
        game = game.variations[0]
    return result


def generate_fen_for_moves(moves: [str]) -> [str]:
    result = []
    board = chess.Board()
    for move in moves:
        board.push_uci(move.uci())
        result.append(board.fen())
    return result


def games_to_unique_fens(games: [chess.pgn.GameNode]) -> {str}:
    return {fen
            for game in games
            for fen in generate_fen_for_moves(generate_moves_for_games(game))}


def is_quiet(fen: str) -> bool:
    board = chess.Board(fen)
    for move in board.generate_legal_moves():
        if board.piece_at(move.to_square) is not None:
            return False
    return True


def filter_non_quiet_position(fens: {str}) -> {str}:
    return [fen for fen in fens if is_quiet(fen)]


def evaluate_fen(fen: str, stockfish_path: str) -> dict:
    stockfish = Stockfish(stockfish_path)
    stockfish.set_fen_position(fen)
    return stockfish.get_evaluation()


def evaluate_fens(fens: [str], stockfish_path: str) -> [(str, dict)]:
    with concurrent.futures.ThreadPoolExecutor(10) as executor:
        futures = [(fen, executor.submit(evaluate_fen, fen, stockfish_path)) for fen in fens]
    return [(f, e.result()) for f, e in futures]


def generate_dataset(size, stockfish_path) -> [(str, float)]:
    t1 = time.time()
    fens = games_to_unique_fens(load(size))
    print("Fens count: ", len(fens))
    filtered_fens = filter_non_quiet_position(fens)
    print("Filtered Fens count: ", len(filtered_fens))
    print("Fen gather time: ", time.time() - t1)
    t2 = time.time()
    data = [(f, e["value"]) for f, e in evaluate_fens(filtered_fens, stockfish_path) if e["type"] == "cp"]
    print("Eval time: ", time.time() - t2)
    return data


if len(sys.argv) < 2:
    print("Usage: python game_dataset_preparation.py <size>")
    exit(0)

gather("https://database.lichess.org/standard",
       "lichess_db_standard_rated_2014-05.pgn.zst",
       GAMES_PATCH)

SIZE = int(sys.argv[1])
dataset = generate_dataset(SIZE, STOCKFISH_PATH)
print("Dataset size: ", len(dataset))

save_dataset_to_csv(dataset, GAMES_DATASET_PATCH)
