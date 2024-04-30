from alphazero.Game import Game
from game import QuoridorGame

class Quoridor(Game):
    def __init__(self, n):
        # this is not used oopsie
        self.n = n
        self.game = QuoridorGame()
        
    def getInitBoard(self):
        state = self.game.state
        return self.game.agents["B"].get_perspective_state(state)

    def getBoardSize(self):
        return self.game.board_size
    
    def getActionSize(self):
        return len(self.game.static_actions.all_actions)
    
    def getNextState(self, board, player, action):
        state = self.game.state

    
        

    
