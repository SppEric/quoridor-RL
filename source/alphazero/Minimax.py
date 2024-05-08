import numpy as np

class Minimax():
    """
    This class handles the Minimax tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.max_depth = args.maxDepthMinimax
        self.origPlayer = None
        print("Minimax depth: ", self.max_depth)

    def getActionMinimax(self, state, curPlayer):
        self.origPlayer = curPlayer
        _, action, win = self.minimax(state, self.max_depth, curPlayer)
        return action, win

    def minimax(self, state, depth, curPlayer, alpha=float('-inf'), beta=float('inf'), maximizing_player=True):
        """
        This function performs minimax search starting from state.
        Returns:
            best move from the current state
        """

        # replace utility, is_winner, get_legal_moves, and make_move with the appropriate functions
        # call this recursive function from a separate function
        # make sure to use curPlayer to account for who is playing
        canonicalBoard = self.game.getCanonicalForm(state, curPlayer)
        winner = self.game.getGameEnded(state, self.origPlayer)
        # if winner is 1, that means curPlayer won
        # if winner is -1, that means curPlayer lost
        # if winner is 0, that means the game is still ongoing
        if depth == 0 or winner != 0:
            const = -1 if curPlayer == self.origPlayer else 1
            value = (depth+1) * const * winner
            if winner == 0:
                value = (1 / float(state.length_to_goal(self.game.getPlayerName(curPlayer))))

            return value, None, winner

        best_move = None
        actual_winner = 0
        valid_moves = np.nonzero(self.game.getValidMoves(state, curPlayer))[0]
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in valid_moves:
                next_state, next_player = self.game.getNextState(state, curPlayer, move)
                eval, _, win = self.minimax(next_state, depth - 1, next_player, alpha, beta, not maximizing_player)
                # print(state)
                if eval >= max_eval:
                    max_eval = eval
                    best_move = move
                    actual_winner = win
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break

            return max_eval, best_move, actual_winner
    
        else:
            min_eval = float('inf')
            for move in valid_moves:
                next_state, next_player = self.game.getNextState(state, curPlayer, move)
                eval, _, win = self.minimax(next_state, depth - 1, next_player, alpha, beta, not maximizing_player)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                    actual_winner = win
                beta = min(beta, eval)
                if beta <= alpha:
                    break

            return min_eval, best_move, actual_winner
            

        
        

