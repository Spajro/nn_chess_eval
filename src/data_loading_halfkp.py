import chess
import torch

from src.data_loading import cp_to_wdl, load_data_from_file, Dataset

M = 40960


class HalfKpDataset(Dataset):
    def __init__(self, file_path: str, batch_size):
        self.data = dataset_to_batches(data_to_tensors(load_data_from_file(file_path)), batch_size)
        self.batch_size = batch_size

    def __iter__(self):
        for batch, truth in self.data:
            yield batch_to_tensors(batch), truth

    def __len__(self):
        return len(self.data)

    def batch_size(self):
        return self.batch_size


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
    p_idx = piece_type * 2 + piece_color
    white_idx = piece_square + (p_idx + white_king * 10) * 64
    black_idx = piece_square + (p_idx + black_king * 10) * 64
    return white_idx, black_idx


def board_to_feature_set(board: chess.Board):
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    features = []

    for (piece_type, piece_color, piece_square) in gather_pieces_from_board(board):
        (white_idx, black_idx) = generate_indexes(piece_type, piece_color, piece_square, white_king, black_king)
        features.append(white_idx)
        features.append(black_idx)
    return features


def feature_set_to_tensor(features):
    tensor = torch.zeros(2 * M).cuda()
    for feature in features:
        tensor[feature] = 1
    return tensor


def data_to_tensors(data: (str, float)) -> ([int], torch.Tensor):
    return [(board_to_feature_set(chess.Board(fen)),
             torch.tensor(cp_to_wdl(value), dtype=torch.float)) for fen, value in data]


def dataset_to_batches(dataset: [([int], torch.Tensor)], batch_size) -> [(torch.Tensor, torch.Tensor)]:
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
        batches.append((batch, torch.tensor(truth).cuda()))

    return batches


def batch_to_tensors(batch: [[int]]) -> torch.Tensor:
    result = []
    for features in batch:
        result.append(feature_set_to_tensor(features))
    return torch.stack(result).cuda()
