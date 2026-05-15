import csv
import multiprocessing
import os.path
import time
import concurrent

from src.core import format_time
from src.patches import GAMES_DATASET_PATCH, FENS_PATH, STOCKFISH_PATH
from stockfish import Stockfish
from concurrent.futures import ThreadPoolExecutor


def evaluate_fen(fen: str, stockfish_path: str, queue: multiprocessing.Queue):
    stockfish = Stockfish(stockfish_path)
    stockfish.set_fen_position(fen)
    e = stockfish.get_evaluation()
    if e["type"] == "cp":
        queue.put((fen, e["value"]))
    else:
        queue.put((fen, "M" + str(e["value"])))


if not os.path.exists(FENS_PATH):
    print("FENS not exist")
    exit(-1)
with open(FENS_PATH, "r") as f:
    all_fens = {fen for fen in f.readlines()}

if not os.path.exists(GAMES_DATASET_PATCH):
    evaluated_fens = set()
else:
    with open(GAMES_DATASET_PATCH, "r") as f:
        reader = csv.reader(f, delimiter=',')
        evaluated_fens = [fen for fen, val in reader]

fens = all_fens.difference(evaluated_fens)

t0 = time.time()
queue = multiprocessing.Queue()
with concurrent.futures.ThreadPoolExecutor(20) as executor:
    for fen in fens:
        executor.submit(evaluate_fen, fen, STOCKFISH_PATH, queue)
    with open(GAMES_DATASET_PATCH, "a", newline='') as file:
        counter = 0
        writer = csv.writer(file)
        t0 = time.time()
        while counter < len(fens):
            counter += 1
            writer.writerow(queue.get())
            if counter % 1000 == 0:
                deltat = time.time() - t0
                eta = deltat * (len(fens) - counter) / counter
                print(str(counter) + "/" + str(len(fens)) + " t:" + format_time(deltat) + " eta: " + format_time(eta))
print("Eval time: ", time.time() - t0)
