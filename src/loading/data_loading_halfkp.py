import chess

from src.loading.data_loading import cp_to_wdl, Dataset, load_dataset_with_stats
from torch import Tensor, zeros

FEATURES_COUNT = 40960


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
            if value[0] == 'M':  # TODO
                continue
            else:
                value = round(float(value))
            stm = fen_to_stm(fen)
            white_features, black_features = board_to_feature_set(chess.Board(fen))

            batch.append((white_features, black_features))
            color.append(stm)
            interpolation.append(stats[0] / (stats[0] + stats[1] + stats[2]))
            if stm == chess.WHITE:
                truth.append(cp_to_wdl(value))
            else:
                truth.append(cp_to_wdl(-1.0 * value))

            index += 1
        batches.append((batch, Tensor(color).to(device), Tensor(interpolation).to(device), Tensor(truth).to(device)))

    return batches


def fen_to_stm(fen: str) -> chess.Color:
    return 'w' in fen


def batch_to_tensors(batch: list[tuple[list[int], list[int]]], device: str) -> list[tuple[Tensor, Tensor]]:
    return [(features_to_tensor(white_features, device), features_to_tensor(black_features, device)) for
            white_features, black_features in batch]


def features_to_tensor(features: list[int], device: str) -> Tensor:
    tensor = zeros(FEATURES_COUNT).to(device)
    for feature in features:
        tensor[feature] = 1
    return tensor


def board_to_feature_set(board: chess.Board) -> tuple[list[int], list[int]]:
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    white_features = []
    black_features = []

    for (piece_type, piece_color, piece_square) in gather_pieces_from_board(board):
        if piece_type != chess.KING:
            (white_idx, black_idx) = generate_indexes(piece_type, piece_color, piece_square, white_king, black_king)
            white_features.append(white_idx)
            black_features.append(black_idx)
    return white_features, black_features


def gather_pieces_from_board(board: chess.Board):
    result = []
    for square in chess.SQUARES:
        opt_piece = board.piece_at(square)
        if opt_piece is not None:
            color = board.color_at(square)
            result.append((opt_piece.piece_type, color, square))
    return result


def generate_indexes(piece_type: chess.PieceType,
                     piece_color: chess.Color,
                     piece_square: chess.Square,
                     white_king: chess.Square,
                     black_king: chess.Square):
    white_idx = piece_square + (white_king * 10 + (piece_type - 1) * 2 + piece_color) * 64
    black_idx = piece_square + (black_king * 10 + (piece_type - 1) * 2 + (not piece_color)) * 64

    return white_idx, black_idx
