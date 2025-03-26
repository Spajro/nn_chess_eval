import chess
import torch

from src.loading.data_loading import cp_to_wdl, load_data_from_file, Dataset

FEATURES_COUNT = 40960


class HalfKpDataset(Dataset):
    def __init__(self, file_path: str, batch_size, device):
        self.data = dataset_to_batches(load_data_from_file(file_path), batch_size, device)
        self.size = batch_size
        self.device = device

    def __iter__(self):
        for batch, color, truth in self.data:
            yield batch_to_tensors(batch, self.device), color, truth

    def __len__(self):
        return len(self.data)

    def batch_size(self):
        return self.size


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
    white_idx = piece_square + (white_king * 10 + piece_type * 2 + piece_color) * 64
    black_idx = piece_square + (black_king * 10 + piece_type * 2 + (not piece_color)) * 64
    return white_idx, black_idx


def board_to_feature_set(board: chess.Board):
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    features = []

    for (piece_type, piece_color, piece_square) in gather_pieces_from_board(board):
        if piece_type != chess.KING:
            (white_idx, black_idx) = generate_indexes(piece_type, piece_color, piece_square, white_king, black_king)
            features.append(white_idx)
            features.append(black_idx)
    return features


def feature_set_to_tensor(features, device):
    tensor = torch.zeros(2 * FEATURES_COUNT).to(device)
    for feature in features:
        tensor[feature] = 1
    return tensor


def fen_to_stm(fen: str) -> chess.Color:
    return 'w' in fen


def data_to_tensors(data: (str, float)) -> ([int], chess.Color, float):
    return [(board_to_feature_set(chess.Board(fen)), fen_to_stm(fen), cp_to_wdl(value)) for fen, value in data]


def dataset_to_batches(dataset: [([int], float)],
                       batch_size: int,
                       device: str
                       ) -> [(torch.Tensor, torch.Tensor, torch.Tensor)]:
    batches = []
    index = 0
    while index + batch_size <= len(dataset):
        batch = []
        color = []
        truth = []
        max_index = index + batch_size
        while index < max_index:
            fen = dataset[index][0]
            value = dataset[index][1]
            stm = fen_to_stm(fen)

            batch.append(board_to_feature_set(chess.Board(fen)))
            color.append(stm)
            truth.append(cp_to_wdl(value))

            index += 1
        batches.append((batch, torch.tensor(color).to(device), torch.tensor(truth).to(device)))

    return batches


def batch_to_tensors(batch: [[int]], device) -> torch.Tensor:
    result = []
    for features in batch:
        result.append(feature_set_to_tensor(features, device))
    return torch.stack(result).to(device)
