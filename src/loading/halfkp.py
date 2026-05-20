import chess
from torch import Tensor, zeros

FEATURES_COUNT = 40960


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
