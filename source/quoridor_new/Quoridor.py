from alphazero.Game import Game
from game import QuoridorGame
import numpy as np
from actions import MoveAction, WallAction
import copy

class Quoridor(Game):
    def __init__(self, n):
        # this is not used oopsie
        self.n = n
        self.game = QuoridorGame()

        # The agents all have the same persepctive even though
        # the code doesn't seem like it. Might change this???
        
    def getInitBoard(self):
        state = self.game.state
        return state

    def getBoardSize(self):
        return self.game.board_size
    
    def getActionSize(self):
        return len(self.game.static_actions.all_actions)
    
    def getNextState(self, board, player, action):
        # action will be an index in the list of all possible elements
        # need to convert this to a wall/move action for the agent to take
        # we can assume it is valid/legal

        agent = self.game.agents["B" if player == 1 else "T"]
        # convert to action
        # we know that 0-8 are move actions, 9-72 are wall actions
        action = self.game.static_actions.all_actions[action]
        
        next_board, next_state_vector, reward = agent.take_action(board, action)
        next_board_copy = copy.deepcopy(next_board)

        return next_board_copy, -player


    def getValidMoves(self, board, player):
        # THIS MIGHT SLOW US DOWN. SHOULD PROBABLY MAKE A FASTER VERSION
        agent = "B" if player == 1 else "T"
        valid = np.zeros(self.getActionSize())

        for i, action in enumerate(self.game.static_actions.all_actions):
            if board.is_legal_action(action, agent):
                valid[i] = 1
                
        return valid
    
    def getGameEnded(self, board, player):
        player_name = "B" if player == 1 else "T"
        if board.winner == None:
            return 0
        if board.winner == player_name:
            return 1
        else:
            return -1
        
    def getCanonicalForm(self, board, player):
        return self.game.agents["B" if player == 1 else "T"].get_perspective_state(board)
    
    def stringRepresentation(self, canonicalBoard):
        # I have no idea what this is for
        # Need to turn game state into a string. 
        # Defaulted to agent B perspective
        return str(canonicalBoard)      
    
        

    
