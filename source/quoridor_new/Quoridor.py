from alphazero.Game import Game
from game import QuoridorGame
import numpy as np
from actions import MoveAction, WallAction

class Quoridor(Game):
    def __init__(self, n):
        # this is not used oopsie
        self.n = n
        self.game = QuoridorGame()

        # The agents all have the same persepctive even though
        # the code doesn't seem like it. Might change this???
        
    def getInitBoard(self):
        state = self.game.state
        return self.game.agents["B"].get_perspective_state(state)

    def getBoardSize(self):
        return self.game.board_size
    
    def getActionSize(self):
        return len(self.game.static_actions.all_actions)
    
    def getNextState(self, board, player, action):
        # action will be an index in the list of all possible elements
        # need to convert this to a wall/move action for the agent to take
        # we can assume it is valid/legal

        agent = self.game.agents["B" if player == 1 else "T"]
        state = self.game.state

        # convert to action
        # we know that 0-8 are move actions, 9-72 are wall actions
        action = self.game.static_actions.all_actions[action]
        
        next_state, reward = agent.take_action(state, action)

        return next_state, -player, reward


    def getValidMoves(self, board, player):
        # THIS MIGHT SLOW US DOWN. SHOULD PROBABLY MAKE A FASTER VERSION
        agent = self.game.agents["B" if player == 1 else "T"]
        state = self.game.state
        valid = np.zeros(self.getActionSize())

        for i, action in enumerate(self.game.static_actions.all_actions):
            if state.is_legal_action(action, agent):
                valid[i] = 1
                
        return valid
    
    def getGameEnded(self, board, player):
        state = self.game.state
        player_name = "B" if player == 1 else "T"
        if state.winner == None:
            return 0
        if state.winner == player_name:
            return 1
        else:
            return -1
        
    def getCanonicalForm(self, board, player):
        return self.game.agents["B" if player == 1 else "T"].get_perspective_state(self.game.state)
    
    def stringRepresentation(self, board):
        # I have no idea what this is for
        # Need to turn game state into a string. 
        # Defaulted to agent B perspective
        state_array = self.game.agents["B"].get_perspective_state(self.game.state)
        return str(state_array)      
    
        

    
