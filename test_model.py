import argparse

import chess
import torch
from torch import nn

from src.loading.data_loading import wdl_to_cp, load_dataset_with_stats
from src.loading.data_loading_halfkp import features_to_tensor, board_to_feature_set
from src.models.models import get_model
from src.patches import TEST_DATASET_PATCH
from src.rdzawa_bestia_eval import evaluate


def test(evaluate, data, ranges):
    results = []
    counts = []
    for _ in range(len(ranges) + 1):
        results.append(0)
        counts.append(0)

    for fen, stats, val in data:
        if val[0] == 'M':  # TODO
            continue
        else:
            val = float(val)
        board = chess.Board(fen)
        r = evaluate(board)
        dif = abs(r - val)

        out_of_range = True
        for i in range(len(ranges)):
            if ranges[i] > abs(val):
                results[i] += dif
                counts[i] += 1
                out_of_range = False
                break

        if out_of_range:
            results[len(ranges)] += dif
            counts[len(ranges)] += 1

    return results, counts


def log(ranges, results, counts):
    print('F      ', sum(results) / sum(counts), sum(counts))

    last_range = 0
    for i in range(len(ranges)):
        if counts[i] == 0:
            print(str(last_range) + ' - ' + str(ranges[i]), 0, counts[i])
        else:
            print(str(last_range) + ' - ' + str(ranges[i]), results[i] / counts[i], counts[i])
        last_range = ranges[i]
    if counts[len(ranges)] == 0:
        print(str(last_range) + ' - ' + 'inf', 0, counts[len(ranges)])
    else:
        print(str(last_range) + ' - ' + 'inf', results[len(ranges)] / counts[len(ranges)], counts[len(ranges)])


def evaluate_model(model: nn.Module, board: chess.Board, device: str):
    white_features, black_features = board_to_feature_set(board)
    white_tensor = features_to_tensor(white_features, device)
    black_tensor = features_to_tensor(black_features, device)
    return model.forward([(white_tensor, black_tensor)], torch.tensor(board.turn).reshape(1, -1))


def eval_fen(model, fen: str, device):
    white_features, black_features = board_to_feature_set(chess.Board(fen))
    white_tensor = features_to_tensor(white_features, device).reshape(1, -1)
    black_tensor = features_to_tensor(black_features, device).reshape(1, -1)
    color = torch.tensor(chess.Board(fen).turn).to(device).reshape(1, -1)
    return wdl_to_cp(model.forward([(white_tensor, black_tensor)], color)).item()


parser = argparse.ArgumentParser(description='Halfkp NNUE test')
parser.add_argument('name', type=str, help='checkpoint to test')
parser.add_argument('--model', type=str, default="nnue", help='model to train')
parser.add_argument('--device', type=str, default='cpu', help='cuda:X or cpu')
args = parser.parse_args()

model_name = args.model
name = args.name
device = args.device

checkpoint = torch.load(name)
model = get_model(model_name).to(device)
model.load_state_dict(checkpoint['model'])
data = load_dataset_with_stats(TEST_DATASET_PATCH)
torch.set_printoptions(sci_mode=False)

fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
fen2 = "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1"
fen3 = "rnbqkbnr/pp1ppppp/2p5/8/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 1"
fen4 = "rnbqkbnr/pp1ppppp/2p5/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 1"
fen5 = "8/6p1/p3Q1kp/1P1p2Bq/1p6/5N2/1PP2PPP/R4RK1 b - - 0 26"
print("BENCHMARKS:")
print(eval_fen(model, fen1, device), fen1)
print(eval_fen(model, fen2, device), fen2)
print(eval_fen(model, fen3, device), fen3)
print(eval_fen(model, fen4, device), fen4)
print(eval_fen(model, fen5, device), fen5)

rngs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]

print("RDZAWA BESTIA")
r1, c1 = test(evaluate, data, rngs)
log(rngs, r1, c1)

print("CHECKPOINT MODEL")
r2, c2 = test(lambda x: wdl_to_cp(evaluate_model(model, x, device)).item(), data, rngs)
log(rngs, r2, c2)
