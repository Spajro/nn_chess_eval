import chess
import torch

from src.loading.data_loading import dataset_to_batches, load_data_from_file, cp_to_wdl, Dataset


class Dataset3D(Dataset):
    def __init__(self, file_path: str, batch_size):
        self.data = dataset_to_batches(data_to_tensors(load_data_from_file(file_path)), batch_size)
        self.size = batch_size

    def __iter__(self):
        for batch, truth in self.data:
            yield batch, truth

    def __len__(self):
        return len(self.data)

    def batch_size(self):
        return self.size


def bitboard_to_tensor(bitboard: int) -> torch.Tensor:
    li = [1.0 if digit == '1' else 0.0 for digit in bin(bitboard)[2:]]
    li = [0.0 for _ in range(64 - len(li))] + li
    return torch.tensor(li).reshape((8, 8))


def fen_to_tensor(fen: str) -> torch.Tensor:
    board = chess.Board(fen)
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    return torch.stack([
        bitboard_to_tensor(white & board.kings),
        bitboard_to_tensor(white & board.queens),
        bitboard_to_tensor(white & board.rooks),
        bitboard_to_tensor(white & board.bishops),
        bitboard_to_tensor(white & board.knights),
        bitboard_to_tensor(white & board.pawns),
        bitboard_to_tensor(black & board.kings),
        bitboard_to_tensor(black & board.queens),
        bitboard_to_tensor(black & board.rooks),
        bitboard_to_tensor(black & board.bishops),
        bitboard_to_tensor(black & board.knights),
        bitboard_to_tensor(black & board.pawns)
    ]).cuda()


def data_to_tensors(data: (str, float)) -> (torch.Tensor, torch.Tensor):
    return [(fen_to_tensor(fen), torch.tensor(cp_to_wdl(value), dtype=torch.float)) for fen, value in data]
