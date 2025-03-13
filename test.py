import argparse

import chess
import torch
from torch import nn

from src.rdzawa_bestia_eval import evaluate
from src.data_loading import load_data_from_file, wdl_to_cp
from src.data_loading_halfkp import M, feature_set_to_tensor, board_to_feature_set
from src.patches import TEST_DATASET_PATCH

P1, P2, P3 = 100, 300, 500

x0 = 2 * M
x1 = 2 ** 8
x2 = 2 ** 5


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(x0, x1)
        self.layer2 = nn.Linear(x1, x2)
        self.layer3 = nn.Linear(x2, 1)
        self.classifier = nn.Sequential(self.layer1,
                                        nn.ReLU(),
                                        self.layer2,
                                        nn.ReLU(),
                                        self.layer3)

    def forward(self, X):
        return self.classifier.forward(X)


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


def evaluate_model(model, board):
    tensor=feature_set_to_tensor(board_to_feature_set(board))
    return model.forward(tensor)


parser = argparse.ArgumentParser(description='Halfkp NNUE test')
parser.add_argument('name', type=str, help='checkpoint to test')
args = parser.parse_args()

name = args.name
checkpoint = torch.load(name)
model=Model().cuda()
model.load_state_dict(checkpoint['model'])
data = load_data_from_file(TEST_DATASET_PATCH)


print("RDZAWA BESTIA")
r, c, r1, c1, r2, c2, r3, c3, r4, c4 = test(evaluate)
log(r, c, r1, c1, r2, c2, r3, c3, r4, c4)

print("CHECKPOINT MODEL")
r, c, r1, c1, r2, c2, r3, c3, r4, c4 = test(lambda x: wdl_to_cp(evaluate_model(model, x)).item())
log(r, c, r1, c1, r2, c2, r3, c3, r4, c4)
