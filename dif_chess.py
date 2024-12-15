import torch

def invalid_move_loss(move_sequence, board_state, device='cpu'):
    """
    Calculates a loss for an invalid move sequence in a chess game.

    Args:
        move_sequence: A string representing the move sequence in algebraic notation 
                       (e.g., "1.e4 e5 2.d4 d5 3.Nc3").
        board_state: A 2D list representing the board state.
        device: The device to perform calculations on (e.g., 'cpu', 'cuda').

    Returns:
        A scalar loss value. Higher values indicate more invalid moves.
    """
    loss = 0.0
    current_board = board_state.copy() if board_state else create_initial_board() 
    moves = move_sequence.split() 

    for i in range(0, len(moves), 2): 
        if i + 1 < len(moves):  # Check for out-of-bounds index
            try:
                white_move = moves[i]
                black_move = moves[i+1]

                from_square, to_square = parse_move_string(white_move) 
                from_row, from_col = from_square // 8, from_square % 8
                to_row, to_col = to_square // 8, to_square % 8
                move = (from_row, from_col, to_row, to_col)

                if not is_valid_move(move, current_board): 
                    loss += 10.0  # High penalty for invalid moves
                else:
                    # Update the board only if the move is valid
                    current_board[to_row][to_col] = current_board[from_row][from_col]
                    current_board[from_row][from_col] = '.' 

                from_square, to_square = parse_move_string(black_move) 
                from_row, from_col = from_square // 8, from_square % 8
                to_row, to_col = to_square // 8, to_square % 8
                move = (from_row, from_col, to_row, to_col)

                if not is_valid_move(move, current_board): 
                    loss += 10.0  # High penalty for invalid moves
                else:
                    # Update the board only if the move is valid
                    current_board[to_row][to_col] = current_board[from_row][from_col]
                    current_board[from_row][from_col] = '.' 

            except ValueError:  # Handle invalid move strings
                loss += 100.0  # Even higher penalty for invalid move strings

    return torch.tensor(loss, device=device)

# ... (Rest of the functions: create_initial_board(), parse_move_string(), is_valid_move(), etc.)

def create_initial_board():
    """
    Creates the initial chessboard state.

    Returns:
        A 2D list representing the initial board state.
    """
    return [
        ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
        ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
        ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
    ]

def parse_move_string(move_str):
    """
    Parses a move string in algebraic notation (e.g., "e2e4") 
    into from_square and to_square integers.

    Args:
        move_str: The move string in algebraic notation.

    Returns:
        A tuple of integers (from_square, to_square).
    """
    if len(move_str) != 4:
        raise ValueError("Invalid move string: Incorrect length")

    file_to_index = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    rank_to_index = {'1': 7, '2': 6, '3': 5, '4': 4, '5': 3, '6': 2, '7': 1, '8': 0}

    from_file = move_str[0]
    from_rank = move_str[1]
    to_file = move_str[2]
    to_rank = move_str[3]

    if from_file not in file_to_index or from_rank not in rank_to_index:
        raise ValueError("Invalid move string: Invalid from square")

    if to_file not in file_to_index or to_rank not in rank_to_index:
        raise ValueError("Invalid move string: Invalid to square")

    from_square = rank_to_index[from_rank] * 8 + file_to_index[from_file]
    to_square = rank_to_index[to_rank] * 8 + file_to_index[to_file]

    return from_square, to_square

def is_valid_move(move, board_state):
    """
    Checks if a given move is valid on the current board state.

    Args:
        move: A tuple representing the move (from_row, from_col, to_row, to_col).
        board_state: A 2D list representing the board state.

    Returns:
        True if the move is valid, False otherwise.
    """
    from_row, from_col, to_row, to_col = move

    # Check if the move is within the board bounds
    if not (0 <= from_row < 8 and 0 <= from_col < 8 and 0 <= to_row < 8 and 0 <= to_col < 8):
        return False

    # Get the piece at the starting position
    piece = board_state[from_row][from_col]

    # Handle empty squares
    if piece == '.':
        return False

    # Determine the piece type
    piece_type = piece.lower()

    # Validate moves based on piece type
    if piece_type == 'p':  # Pawn
        return is_valid_pawn_move(move, board_state, piece)
    elif piece_type == 'r':  # Rook
        return is_valid_rook_move(move, board_state)
    elif piece_type == 'n':  # Knight
        return is_valid_knight_move(move)
    elif piece_type == 'b':  # Bishop
        return is_valid_bishop_move(move, board_state)
    elif piece_type == 'q':  # Queen
        return is_valid_queen_move(move, board_state)
    elif piece_type == 'k':  # King
        return is_valid_king_move(move, board_state)

    return False

def is_valid_pawn_move(move, board_state, piece):
    """
    Checks if a pawn move is valid.

    Args:
        move: A tuple representing the move (from_row, from_col, to_row, to_col).
        board_state: A 2D list representing the board state.
        piece: The color of the pawn ('P' or 'p').

    Returns:
        True if the move is valid, False otherwise.
    """
    from_row, from_col, to_row, to_col = move

    # Pawn direction
    direction = -1 if piece == 'P' else 1

    # One step forward
    if from_col == to_col and to_row == from_row + direction and board_state[to_row][to_col] == '.':
        return True

    # Two steps forward from starting position
    if from_col == to_col and to_row == from_row + 2 * direction and from_row in (1, 6) and \
       board_state[to_row][to_col] == '.' and board_state[from_row + direction][to_col] == '.':
        return True

    # Capture diagonally
    if abs(to_col - from_col) == 1 and to_row == from_row + direction and \
       (board_state[to_row][to_col] != '.' and board_state[to_row][to_col].islower() if piece == 'P' 
        else board_state[to_row][to_col].isupper() if piece == 'p' else False):
        return True

    return False

def is_valid_rook_move(move, board_state):
    """
    Checks if a rook move is valid.

    Args:
        move: A tuple representing the move (from_row, from_col, to_row, to_col).
        board_state: A 2D list representing the board state.

    Returns:
        True if the move is valid, False otherwise.
    """
    from_row, from_col, to_row, to_col = move

    # Rook moves only horizontally or vertically
    if from_row == to_row:  # Horizontal move
        for col in range(min(from_col, to_col) + 1, max(from_col, to_col)): 
            if board_state[from_row][col] != '.': 
                return False
    elif from_col == to_col:  # Vertical move
        for row in range(min(from_row, to_row) + 1, max(from_row, to_row)):
            if board_state[row][from_col] != '.':
                return False
    else:
        return False

    return True  # If no obstructions were found

def is_valid_bishop_move(move, board_state):
    """
    Checks if a bishop move is valid.

    Args:
        move: A tuple representing the move (from_row, from_col, to_row, to_col).
        board_state: A 2D list representing the board state.

    Returns:
        True if the move is valid, False otherwise.
    """
    from_row, from_col, to_row, to_col = move

    # Bishop moves diagonally
    if abs(from_row - to_row) != abs(from_col - to_col):
        return False

    # Check for obstructions on the diagonal
    row_step = 1 if to_row > from_row else -1
    col_step = 1 if to_col > from_col else -1
    row, col = from_row + row_step, from_col + col_step
    while row != to_row and col != to_col:
        if board_state[row][col] != '.':
            return False
        row += row_step
        col += col_step

    return True

def is_valid_knight_move(move):
    """
    Checks if a knight move is valid.

    Args:
        move: A tuple representing the move (from_row, from_col, to_row, to_col).

    Returns:
        True if the move is valid, False otherwise.
    """
    from_row, from_col, to_row, to_col = move
    row_diff = abs(from_row - to_row)
    col_diff = abs(from_col - to_col)
    return (row_diff == 2 and col_diff == 1) or (row_diff == 1 and col_diff == 2)


def is_valid_king_move(move, board_state):
    """
    Checks if a king move is valid.

    Args:
        move: A tuple representing the move (from_row, from_col, to_row, to_col).
        board_state: A 2D list representing the board state.

    Returns:
        True if the move is valid, False otherwise.
    """
    from_row, from_col, to_row, to_col = move
    row_diff = abs(from_row - to_row)
    col_diff = abs(from_col - to_col)
    return (row_diff <= 1 and col_diff <= 1) 

# Example usage
move_sequence = "1.e4 e5 2.d4 d5 3.Nc3" 
board_state = create_initial_board()  # Initialize the board
loss = invalid_move_loss(move_sequence, board_state)
print(f"Invalid Move Loss: {loss.item()}")