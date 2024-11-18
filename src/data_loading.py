import csv
import os
import torch


def cp_to_wdl(value: int) -> float:
    return value / 410


def wdl_to_cp(value):
    return value * 410


def save_dataset_to_csv(dataset, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(dataset)


def load_data_from_file(file_path: str) -> [(str, float)]:
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        return [(fen, float(val)) for fen, val in reader]


def dataset_to_batches(dataset: [(torch.Tensor, torch.Tensor)], batch_size) -> [(torch.Tensor, torch.Tensor)]:
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
