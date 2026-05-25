import chess

from src.loading.halfkp import fen_to_stm, board_to_feature_set, batch_to_tensors
from src.loading.data_loading import cp_to_wdl, Dataset, load_dataset_with_stats
from torch import Tensor


class HalfKpDataset(Dataset):
    def __init__(self, file_path: str, batch_size: int, device: str):
        self.data = dataset_to_batches(load_dataset_with_stats(file_path), batch_size, device)
        self.size = batch_size
        self.device = device

    def __iter__(self):
        for batch, color, stats, truth in self.data:
            yield batch_to_tensors(batch, self.device), color, stats, truth

    def __len__(self):
        return len(self.data)

    def batch_size(self):
        return self.size


def dataset_to_batches(dataset: list[tuple[str, tuple[int, int, int], str]],
                       batch_size: int,
                       device: str
                       ) -> list[tuple[list[tuple[list[int], list[int]]], Tensor, Tensor, Tensor]]:
    batches = []
    index = 0
    while index + batch_size <= len(dataset):
        batch = []
        color = []
        interpolation = []
        truth = []
        max_index = index + batch_size
        while index < max_index:
            fen = dataset[index][0]
            stats = dataset[index][1]
            value = dataset[index][2]
            if value[0] == 'M':
                if value[1] == '-':
                    value = 0.0+1e-5
                else:
                    value = 1.0-1e-5
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

            index += 1
        batches.append((batch, Tensor(color).to(device), Tensor(interpolation).to(device), Tensor(truth).to(device)))

    return batches
