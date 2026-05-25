import csv
import os
import torch

from src.patches import FENS_PATH


class Dataset:
    def __iter__(self):
        pass

    def __len__(self):
        pass

    def batch_size(self):
        pass


def cp_to_wdl(value: int) -> float:
    return torch.sigmoid(torch.tensor(value) / 410).item()


def wdl_to_cp(value: torch.Tensor) -> torch.Tensor:
    return torch.logit(value) * 410


def save_dataset(dataset: list[tuple[str, float]], filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(dataset)


def load_dataset(file_path: str) -> list[tuple[str, float]]:
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        return [(fen, float(val)) for fen, val in reader if val[0] != 'M']


def save_dataset_with_stats(dataset: list[tuple[str, tuple[int, int, int], str]], filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([(fen, w, d, l, val) for fen, (w, d, l), val in dataset])


def load_dataset_with_stats(file_path: str) -> list[tuple[str, tuple[int, int, int], str]]:
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        return [(fen, (int(w), int(d), int(l)), val) for fen, w, d, l, val in reader]


def load_fens_with_stats() -> list[tuple[str, tuple[int, int, int]]]:
    with open(FENS_PATH, "r") as f:
        reader = csv.reader(f, delimiter=',')
        return [(fen, (int(w), int(d), int(l))) for fen, w, d, l in reader]


def dataset_to_batches(dataset: list[tuple[torch.Tensor, torch.Tensor]],
                       batch_size: int
                       ) -> list[tuple[torch.Tensor, torch.Tensor]]:
    batches = []
    index = 0
    while index + batch_size <= len(dataset):
        batch = []
        truth = []
        max_index = index + batch_size
        while index < max_index:
            batch.append(dataset[index][0])
            truth.append(dataset[index][1])
            index += 1
        batches.append((torch.stack(batch).cuda(), torch.tensor(truth).cuda()))

    return batches
