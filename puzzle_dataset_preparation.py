import sys

import chess
import concurrent
from src.loading.data_loading import save_dataset_to_csv
from src.loading.raw_data_gather import gather
from src.patches import PUZZLE_DATASET_PATCH, PUZZLE_PATCH, STOCKFISH_PATH
from stockfish import Stockfish
from concurrent.futures import ThreadPoolExecutor

gather("https://database.lichess.org", "lichess_db_puzzle.csv.zst", PUZZLE_PATCH)


class Puzzle:
    def __init__(self, row: str):
        fields = row.split(',')
        self.fen = fields[1]
        self.moves = fields[2].split(" ")
        self.tags = fields[7].split(" ")

    def __str__(self):
        return "{fen: " + self.fen + " ,tags: [" + ", ".join(self.tags) + "],moves: [" + ",".join(self.moves) + "]}"


def load(k: int) -> [Puzzle]:
    f = open(PUZZLE_PATCH)
    f.readline()
    result = []
    for i in range(k):
        result.append(Puzzle(f.readline()))
    f.close()
    return result


def generate_positions_for_puzzle(puzzle: Puzzle) -> [(str, [str])]:
    return [(puzzle.fen, puzzle.moves[:i]) for i in range(len(puzzle.moves) + 1)]


def generate_fen_for_position(position: (str, [str])) -> str:
    board = chess.Board(position[0])
    for move in position[1]:
        board.push_uci(move)
    if board.is_game_over():
        return 'FINISHED'
    return board.fen()


def puzzles_to_fens(puzzles: [Puzzle]) -> [str]:
    return [generate_fen_for_position(position)
            for puzzle in puzzles
            for position in generate_positions_for_puzzle(puzzle)]


def filter_finished_fens(fens: [str]) -> [str]:
    return [f for f in fens if f != 'FINISHED']


def evaluate_fen(fen: str, stockfish_path: str, ) -> dict:
    stockfish = Stockfish(stockfish_path)
    stockfish.set_fen_position(fen)
    return stockfish.get_evaluation()


def evaluate_fens(fens: [str], stockfish_path: str) -> [(str, dict)]:
    with concurrent.futures.ThreadPoolExecutor(10) as executor:
        futures = [(fen, executor.submit(evaluate_fen, fen, stockfish_path)) for fen in fens]
    return [(f, e.result()) for f, e in futures]


def generate_dataset(size, stockfish_path) -> [(str, float)]:
    return [(f, e["value"]) for f, e in
            evaluate_fens(filter_finished_fens(puzzles_to_fens(load(size))), stockfish_path) if e["type"] == "cp"]


if len(sys.argv) < 2:
    print("Usage: python puzzle_dataset_preparation.py <size>")
    exit(0)
SIZE = int(sys.argv[1])
dataset = generate_dataset(SIZE, STOCKFISH_PATH)
print("Dataset size: ", len(dataset))

save_dataset_to_csv(dataset, PUZZLE_DATASET_PATCH)