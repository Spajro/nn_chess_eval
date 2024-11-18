import chess
import torch

from src.data_loading import cp_to_wdl, load_data_from_file, dataset_to_batches


def bitboard_to_tensor(bitboard: int) -> torch.Tensor:
    li = [1.0 if digit == '1' else 0.0 for digit in bin(bitboard)[2:]]
    li = [0.0 for _ in range(64 - len(li))] + li
    return torch.tensor(li)


def fen_to_tensor(fen: str) -> torch.Tensor:
    board = chess.Board(fen)
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    return torch.cat([
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
    ])


def data_to_tensors(data: (str, float)) -> (torch.Tensor, torch.Tensor):
    return [(fen_to_tensor(fen), torch.tensor(cp_to_wdl(value), dtype=torch.float)) for fen, value in data]


def load_dataset(file_path: str, batch_size) -> [(torch.Tensor, torch.Tensor)]:
    return dataset_to_batches(data_to_tensors(load_data_from_file(file_path)), batch_size)
