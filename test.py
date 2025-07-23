import argparse

import chess
import torch
from torch import nn

from src.loading.data_loading import load_data_from_file, wdl_to_cp
from src.loading.data_loading_halfkp import features_to_tensor, board_to_feature_set
from src.models.models import get_model
from src.patches import TEST_DATASET_PATCH
from src.rdzawa_bestia_eval import evaluate

P1, P2, P3 = 100, 300, 500


def test(evaluate):
    result, result1, result2, result3, result4 = 0, 0, 0, 0, 0,
    count, count1, count2, count3, count4 = 0, 0, 0, 0, 0,
    for fen, val in data:
        board = chess.Board(fen)
        r = evaluate(board)
        dif = abs(r - val)
        result += dif
        count += 1

        if abs(val) < P1:
            result1 += dif
            count1 += 1
        elif abs(val) < P2:
            result2 += dif
            count2 += 1
        elif abs(val) < P3:
            result3 += dif
            count3 += 1
        else:
            result4 += dif
            count4 += 1
    return result, count, result1, count1, result2, count2, result3, count3, result4, count4


def log(result, count, result1, count1, result2, count2, result3, count3, result4, count4):
    print('F      ', result / count, count)
    print('0,' + str(P1), result1 / count1, count1)
    print(str(P1) + ',' + str(P2), result2 / count2, count2)
    print(str(P2) + ',' + str(P3), result3 / count3, count3)
    print(str(P3) + ',inf', result4 / count4, count4)


def evaluate_model(model: nn.Module, board: chess.Board, device: str):
    white_features, black_features = board_to_feature_set(board)
    white_tensor = features_to_tensor(white_features, device)
    black_tensor = features_to_tensor(black_features, device)
    return model.forward((white_tensor, black_tensor), torch.tensor(board.turn))


def eval_fen(model, fen: str, device):
    white_features, black_features = board_to_feature_set(chess.Board(fen))
    white_tensor = features_to_tensor(white_features, device).reshape(1, -1)
    black_tensor = features_to_tensor(black_features, device).reshape(1, -1)
    color = torch.tensor(chess.Board(fen).turn).to(device)
    return wdl_to_cp(model.forward((white_tensor, black_tensor), color)).item()


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
data = load_data_from_file(TEST_DATASET_PATCH)
torch.set_printoptions(sci_mode=False)

fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
fen2 = "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 1"
fen3 = "rnbqkbnr/pp1ppppp/2p5/8/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 1"
fen4 = "rnbqkbnr/pp1ppppp/2p5/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 1"
print("BENCHMARKS:")
print(eval_fen(model, fen1, device), fen1)
print(eval_fen(model, fen2, device), fen2)
print(eval_fen(model, fen3, device), fen3)
print(eval_fen(model, fen4, device), fen4)

print("RDZAWA BESTIA")
r, c, r1, c1, r2, c2, r3, c3, r4, c4 = test(evaluate)
log(r, c, r1, c1, r2, c2, r3, c3, r4, c4)

print("CHECKPOINT MODEL")
r, c, r1, c1, r2, c2, r3, c3, r4, c4 = test(lambda x: wdl_to_cp(evaluate_model(model, x, device)).item())
log(r, c, r1, c1, r2, c2, r3, c3, r4, c4)
