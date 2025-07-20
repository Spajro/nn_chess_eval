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
            white_features, black_features = board_to_feature_set(chess.Board(fen))

            batch.append((white_features, black_features))
            color.append(stm)
            truth.append(cp_to_wdl(value))

            index += 1
        batches.append((batch, torch.tensor(color).to(device), torch.tensor(truth).to(device)))

    return batches


def fen_to_stm(fen: str) -> chess.Color:
    return 'w' in fen


def batch_to_tensors(batch: [[int], [int]], device) -> (torch.Tensor, torch.Tensor):
    white_result = []
    black_result = []
    for white_features, black_features in batch:
        white_result.append(features_to_tensor(white_features, device))
        black_result.append(features_to_tensor(black_features, device))
    return torch.stack(white_result).to(device)


def features_to_tensor(features, device):
    tensor = torch.zeros(FEATURES_COUNT).to(device)
    for feature in features:
        tensor[feature] = 1
    return tensor


def board_to_feature_set(board: chess.Board) -> ([int], [int]):
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
