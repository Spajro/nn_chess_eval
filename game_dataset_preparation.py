import csv
import multiprocessing
import sys
import time

import chess
import chess.pgn
import concurrent

from src.loading.data_loading import save_dataset_to_csv
from src.loading.raw_data_gather import gather
from src.patches import GAMES_DATASET_PATCH, GAMES_PATCH, STOCKFISH_PATH, FENS_PATH
from stockfish import Stockfish
from concurrent.futures import ThreadPoolExecutor


def generate_dataset(size, stockfish_path) -> [(str, float)]:
    t1 = time.time()
    fens = games_to_unique_fens(load(size))
    print("Fens count: ", len(fens))
    filtered_fens = filter_non_quiet_position(fens)
    print("Filtered Fens count: ", len(filtered_fens))
    print("Fen gather time: ", time.time() - t1)
    save_dataset_to_csv(filtered_fens, FENS_PATH)
    t2 = time.time()
    evaluate_fens(filtered_fens, stockfish_path)
    print("Eval time: ", time.time() - t2)


def games_to_unique_fens(games: [chess.pgn.GameNode]) -> {str}:
    return {fen
            for game in games
            for fen in generate_fen_for_moves(generate_moves_for_games(game))}


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


def filter_non_quiet_position(fens: {str}) -> {str}:
    return [fen for fen in fens if is_quiet(fen)]


def is_quiet(fen: str) -> bool:
    board = chess.Board(fen)
    for move in board.generate_legal_moves():
        if board.piece_at(move.to_square) is not None:
            return False
    return True


def evaluate_fens(fens: [str], stockfish_path: str) -> [(str, dict)]:
    queue = multiprocessing.Queue()
    with concurrent.futures.ThreadPoolExecutor(10) as executor:
        for fen in fens:
            executor.submit(evaluate_fen, fen, stockfish_path, queue)
        with open(GAMES_DATASET_PATCH, "a", newline='') as file:
            counter = 0
            writer = csv.writer(file)
            t0 = time.time()
            while counter < len(fens):
                counter += 1
                writer.writerow(queue.get())
                t1 = time.time()
                eta = (t1 - t0) * (len(fens) - counter) / counter
                print(str(counter) + "/" + str(len(fens)) + " t:" + str(t1 - t0) + " eta: " + str(eta))


def evaluate_fen(fen: str, stockfish_path: str, queue: multiprocessing.Queue):
    stockfish = Stockfish(stockfish_path)
    stockfish.set_fen_position(fen)
    e = stockfish.get_evaluation()
    if e["type"] == "cp":
        queue.put((fen, e["value"]))
    else:
        queue.put((fen, "M" + str(e["value"])))


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


if len(sys.argv) < 2:
    print("Usage: python game_dataset_preparation.py <size>")
    exit(0)

gather("https://database.lichess.org/standard",
       "lichess_db_standard_rated_2014-05.pgn.zst",
       GAMES_PATCH)

SIZE = int(sys.argv[1])
generate_dataset(SIZE, STOCKFISH_PATH)
