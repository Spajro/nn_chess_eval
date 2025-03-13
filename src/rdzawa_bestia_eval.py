import chess

#@formatter:off
KING_SQUARE_TABLE= [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]
QUEEN_SQUARE_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

ROOK_SQUARE_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0,
]

BISHOP_SQUARE_TABLE= [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

KNIGHT_SQUARE_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

PAWN_SQUARE_TABLE= [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

# @formatter:on


def evaluate(board: chess.Board):
    white_value = get_pieces_value(board, chess.WHITE)
    black_value = get_pieces_value(board, chess.BLACK)
    return (white_value
            - black_value
            + get_position_cumulative_value(board, chess.WHITE)
            - get_position_cumulative_value(board, chess.BLACK))


def get_pieces_value(board, color) -> int:
    return (100 * chess.popcount(board.pieces_mask(chess.PAWN, color))
            + 320 * chess.popcount(board.pieces_mask(chess.KNIGHT, color))
            + 330 * chess.popcount(board.pieces_mask(chess.BISHOP, color))
            + 500 * chess.popcount(board.pieces_mask(chess.ROOK, color))
            + 900 * chess.popcount(board.pieces_mask(chess.QUEEN, color)))


def get_position_cumulative_value(board, color) -> int:
    king_value = sum([get_square_value(square, KING_SQUARE_TABLE, color) for square in
                      chess.scan_forward(board.pieces_mask(chess.KING, color))])
    pawn_value = sum([get_square_value(square, PAWN_SQUARE_TABLE, color) for square in
                      chess.scan_forward(board.pieces_mask(chess.PAWN, color))])
    knight_value = sum([get_square_value(square, KNIGHT_SQUARE_TABLE, color) for square in
                        chess.scan_forward(board.pieces_mask(chess.KNIGHT, color))])
    bishop_value = sum([get_square_value(square, BISHOP_SQUARE_TABLE, color) for square in
                        chess.scan_forward(board.pieces_mask(chess.BISHOP, color))])
    rook_value = sum([get_square_value(square, ROOK_SQUARE_TABLE, color) for square in
                      chess.scan_forward(board.pieces_mask(chess.ROOK, color))])
    queen_value = sum([get_square_value(square, QUEEN_SQUARE_TABLE, color) for square in
                       chess.scan_forward(board.pieces_mask(chess.QUEEN, color))])
    return king_value + pawn_value + knight_value + bishop_value + rook_value + queen_value


def get_square_value(square, table, color) -> int:
    if color == chess.BLACK:
        square = flip_square_vertical(square)
    return table[chess.SQUARES.index(square)]


def flip_square_vertical(square):
    row = square % 8
    return square - row + (7 - row)
