import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(side, board, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(side, board, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return (value, [ move ], { encode(*move): {} })
    else:
        return (evaluate(board), [], {})

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.

def minimax(side, board, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
        Return the resulting optimal list of moves (including moves by both white and black) as moveList
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
        the complete tree of evaluated moves as moveTree
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''

    # You will certainly want to implement minimax as a recursive function.
    # You will certainly want to use the function generateMoves to generate all moves that are legal in the current game state, 
    # and you will certainly want to use makeMove to find the newside, newboard, and newflags 
    # that result from making each move. 
    # When you get to depth==0, you will certainly want to use evaluate(board) in order to compute the heuristic value of the resulting board.

    # When you get to depth==0, you will certainly want to use evaluate(board) in order to compute the heuristic value of the resulting board.
    # If depth==1 and side==False, then you should just find one move, from the current board, that maximizes the value of the resulting board.
    # If depth==2 and side==False, then you should find a white move, and the immediate following black move.
    # If depth==3 and side==False, then you should find a white, black, white sequence of moves.

    # the white player is Max, the black player is Min
    # side. PyChess keeps track of whose turn it is by using a boolean called side:
    # side==False if Player 0(White) should play next.
    # side==True if Player 1(Black) should play next.
    # If side==True, in current move, you should choose a path through this tree that minimizes the heuristic value of the final board,
    # knowing that your opponent will be trying to maximize value in the next move;
    # conversely if side==False.

    if depth == 0:
        return evaluate(board), [], {}

    moves = [ move for move in generateMoves(side, board, flags) ]
    # iterate the moves to find the minimum
    move_dic = {}
    best_minimum_value = float('inf')
    best_maximum_value = float('-inf')

    best_move = None
    best_move_list = None
    for move in moves:
        temp_side, temp_board, temp_flags = makeMove(side, board, move[0], move[1], flags, move[2])
        parent_value, parent_move_list, parent_move_tree = minimax(temp_side, temp_board, temp_flags, depth - 1)
        move_dic[encode(*move)] = parent_move_tree

        # current move is balck, choose minimum
        if side == True and parent_value < best_minimum_value:
            best_minimum_value = parent_value
            best_move = move
            best_move_list = parent_move_list
        # current move is white, choose maximum
        if side == False and parent_value > best_maximum_value:
            best_maximum_value = parent_value
            best_move = move
            best_move_list = parent_move_list

    return_list = []
    if best_move is not None:
        return_list.append(best_move)
    if best_move_list is not None and len(best_move_list) != 0:
        flatten_best_move_list = [val for val in best_move_list]
        return_list = return_list + (flatten_best_move_list)

    evaluation = best_minimum_value if (side == True) else best_maximum_value
    return evaluation, return_list, move_dic

def alphabeta(side, board, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    if depth == 0:
        return evaluate(board), [], {}
    
    current_alpha = alpha # 3
    current_beta = beta # inf
    # current move is balck, choose minimum, can only update beta
    moves = [ move for move in generateMoves(side, board, flags) ]
    # iterate the moves to find the minimum
    move_dic = {}
    best_move = None
    best_move_list = None
    for move in moves:
        temp_side, temp_board, temp_flags = makeMove(side, board, move[0], move[1], flags, move[2])
        best_value, child_move_list, child_move_tree = alphabeta(temp_side, temp_board, temp_flags, depth - 1, alpha=current_alpha, beta=current_beta)
        move_dic[encode(*move)] = child_move_tree
        # print("depth at minimum = ", depth, "best_value = ", best_value, ", current_beta=", current_beta, ", current_alpha=", current_alpha, ", iterate at encode(*move) = ", encode(*move))

        if side == True and best_value < current_beta:
            # move to the child
            current_beta = best_value
            best_move = move
            best_move_list = child_move_list

        if side == False and best_value > current_alpha:
            # take child move
            current_alpha = best_value
            best_move = move
            best_move_list = child_move_list

        if current_beta <= current_alpha:
            # prune
            return best_value, child_move_list, move_dic

    return_list = []
    if best_move is not None:
        return_list.append(best_move)
    if best_move_list is not None and len(best_move_list) != 0:
        flatten_best_move_list = [val for val in best_move_list]
        return_list = return_list + (flatten_best_move_list)

    evaluation = current_beta if (side == True) else current_alpha
    return evaluation, return_list, move_dic
    

def stochastic(side, board, flags, depth, breadth, chooser):
    '''
    Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
    Return: (value, moveList, moveTree)
      value (float): average board value of the paths for the best-scoring move
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
      breadth: number of different paths 
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    raise NotImplementedError("you need to write this!")
