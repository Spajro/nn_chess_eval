import csv
import chess
import torch


def bitboard_to_tensor(bitboard: int) -> torch.Tensor:
    li = [1.0 if digit == '1' else 0.0 for digit in bin(bitboard)[2:]]
    li = [0.0 for _ in range(64 - len(li))] + li
    return torch.tensor(li).reshape((8, 8))


def fen_to_tensor(fen: str) -> [torch.Tensor]:
    board = chess.Board(fen)
    return torch.stack([
        bitboard_to_tensor(board.occupied_co[chess.WHITE]),
        bitboard_to_tensor(board.occupied_co[chess.BLACK]),
        bitboard_to_tensor(board.pawns),
        bitboard_to_tensor(board.kings),
        bitboard_to_tensor(board.queens),
        bitboard_to_tensor(board.knights),
        bitboard_to_tensor(board.bishops),
        bitboard_to_tensor(board.rooks)
    ])


def data_to_tensors(data: (str, float)) -> (torch.Tensor, torch.Tensor):
    return [(fen_to_tensor(fen), torch.tensor(value / 100, dtype=torch.float)) for fen, value in data]


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


def load_dataset(file_path: str, batch_size) -> [(torch.Tensor, torch.Tensor)]:
    return dataset_to_batches(data_to_tensors(load_data_from_file(file_path)), batch_size)
