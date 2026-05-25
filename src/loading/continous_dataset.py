import csv
import chess

from src.loading.halfkp import batch_to_tensors, fen_to_stm, board_to_feature_set
from src.loading.data_loading import cp_to_wdl, Dataset
from torch import Tensor


class ContinuousHalfKpDataset(Dataset):
    def __init__(self, file_path: str, batch_size: int, device: str):
        self.path = file_path
        self.size = batch_size
        self.device = device
        self.__reader = iter(csv.reader(open(self.path, 'r'), delimiter=','))

    def __iter__(self):
        while True:
            index = 0
            data = []
            while index < self.size:
                fen, w, d, l, val = next(self.__reader, ("nofen", "w", "d", "l", "val"))
                if fen == "nofen":
                    self.__reader = iter(csv.reader(open(self.path, 'r'), delimiter=','))
                    continue
                data.append((fen, (int(w), int(d), int(l)), val))
                index += 1
            batch, color, stats, truth = data_to_batch(data, self.size, self.device)
            yield batch_to_tensors(batch, self.device), color, stats, truth

    def skip(self, count):
        for i in range(count * self.size):
            fen, w, d, l, val = next(self.__reader, ("nofen", "w", "d", "l", "val"))
            if fen == "nofen":
                self.__reader = iter(csv.reader(open(self.path, 'r'), delimiter=','))
                self.skip((count * self.size - i) % i)
                break

    def batch_size(self):
        return self.size


def data_to_batch(data: list[tuple[str, tuple[int, int, int], str]],
                  batch_size: int,
                  device: str
                  ) -> tuple[list[tuple[list[int], list[int]]], Tensor, Tensor, Tensor]:
    if batch_size != len(data):
        print("ERROR")
        print(batch_size, len(data))
        exit(-1)
    batch = []
    color = []
    interpolation = []
    truth = []
    for fen, stats, value in data:
        if value[0] == 'M':
            if value[1] == '-':
                value = 0.0 + 1e-5
            else:
                value = 1.0 - 1e-5
        else:
            value = cp_to_wdl(round(float(value)))

        stm = fen_to_stm(fen)
        if stm == chess.BLACK:
            value = 1.0 - value

        white_features, black_features = board_to_feature_set(chess.Board(fen))
        batch.append((white_features, black_features))
        color.append(stm)
        truth.append(value)

        stats_sum = stats[0] + stats[1] + stats[2]
        if stats_sum > 0:
            interpolation.append(stats[0] / stats_sum)
        else:
            interpolation.append(0.0)

    return batch, Tensor(color).to(device), Tensor(interpolation).to(device), Tensor(truth).to(device)
