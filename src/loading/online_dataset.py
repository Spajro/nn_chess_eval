import csv
import chess

from src.loading.halfkp import batch_to_tensors, fen_to_stm, board_to_feature_set
from src.loading.data_loading import cp_to_wdl, Dataset
from torch import Tensor


class HalfKpDataset(Dataset):
    def __init__(self, file_path: str, batch_size: int, device: str):
        self.path = file_path
        self.size = batch_size
        self.device = device
        self.len = 1

    def __iter__(self):
        with open(self.path, 'r') as f:
            reader = iter(csv.reader(f, delimiter=','))
            while True:
                index = 0
                data = []
                while index < self.size:
                    fen, w, d, l, val = next(reader, ("nofen", "w", "d", "l", "val"))
                    if fen == "nofen":
                        return
                    data.append((fen, (int(w), int(d), int(l)), val))
                    index+=1
                batch, color, stats, truth = data_to_batch(data, self.size, self.device)
                self.len+=1
                yield batch_to_tensors(batch, self.device), color, stats, truth

    def __len__(self):
        return self.len

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
        if value[0] == 'M':  # TODO
            continue
        else:
            value = round(float(value))
        stm = fen_to_stm(fen)
        white_features, black_features = board_to_feature_set(chess.Board(fen))

        batch.append((white_features, black_features))
        color.append(stm)
        stats_sum = stats[0] + stats[1] + stats[2]
        if stats_sum > 0:
            interpolation.append(stats[0] / stats_sum)
        else:
            interpolation.append(0.0)
        if stm == chess.WHITE:
            truth.append(cp_to_wdl(value))
        else:
            truth.append(cp_to_wdl(-1.0 * value))

    return batch, Tensor(color).to(device), Tensor(interpolation).to(device), Tensor(truth).to(device)
