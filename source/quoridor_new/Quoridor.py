from alphazero.Game import Game
from game import QuoridorGame
import numpy as np
from actions import MoveAction, WallAction
from point import Point
import copy

class Quoridor(Game):
    def __init__(self, n):
        # this is not used oopsie
        self.n = n
        self.game = QuoridorGame()

        # The agents all have the same persepctive even though
        # the code doesn't seem like it. Might change this???
    
    def getPlayerName(self, player):
        return "B" if player == 1 else "T"
    
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
        board = copy.deepcopy(board)

        agent = self.game.agents["B" if player == 1 else "T"]
        # convert to action
        # we know that 0-8 are move actions, 9-72 are wall actions
        # print("action index", action)
        action = self.game.static_actions.all_actions[action]

        name = "B" if player == 1 else "T"
        if not board.is_legal_action(action, name):
            print("illegal action:")
            if isinstance(action, MoveAction):
                print("move action", action.direction)
            else:
                if board.wall_counts[name] == 0:
                    print("illegal because no walls left for", name)
                print("wall action", action.position, action.orientation)
        
        next_board, next_state_vector, reward = agent.take_action(board, action)

        return next_board, -player


    def getValidMoves(self, board, player):
        # NOTE: THIS MIGHT SLOW US DOWN. SHOULD PROBABLY MAKE A FASTER VERSION
        agent = "B" if player == 1 else "T"
        valid = np.zeros(self.getActionSize())

        for i, action in enumerate(self.game.static_actions.all_actions):
            if board.is_legal_action(action, agent):
                valid[i] = 1
        
        return valid
    
    def getProbableWalls(self, board, player, valids):
        possible_walls = self.game.static_actions.all_actions[8:]

        probable_wall_positions = []
        

        # Add walls near enemy
        opposite_player = "T" if player == 1 else "B"
        enemy_x, enemy_y = board.agent_positions[opposite_player].X, board.agent_positions[opposite_player].Y
        probable_wall_positions += [(enemy_x, enemy_y, "H"), 
                                    (enemy_x, enemy_y - 1, "H"), 
                                    (enemy_x - 1, enemy_y, "H"),
                                    (enemy_x - 1, enemy_y - 1, "H"),
                                    (enemy_x, enemy_y, "V"),
                                    (enemy_x, enemy_y - 1, "V"),
                                    (enemy_x - 1, enemy_y, "V"),
                                    (enemy_x - 1, enemy_y - 1, "V")]
        

        # # Add walls near other walls
        # for wall in possible_walls:
        #     x, y = wall.position.X, wall.position.Y
        #     if board.walls[x][y] == "H":
        #         probable_walls_positions += [(x - 1, y - 1, "V"), 
        #                                      (x - 1, y, "V"), 
        #                                      (x - 1, y + 1, "V"),
        #                                      (x, y - 1, "V"),
        #                                      (x, y + 1, "V"),
        #                                      (x + 1, y - 1, "V"),
        #                                      (x + 1, y, "V"),
        #                                      (x + 1, y + 1, "V"),]
        
        probable_walls = np.zeros(self.getActionSize())
        # board_max = self.getBoardSize()[0] - 1
        for i, wall in enumerate(possible_walls):
            x, y = wall.position.X, wall.position.Y
            if (x, y, wall.orientation) in probable_wall_positions:
                probable_walls[i + 8] = 1

        return probable_walls

        
                
                
                
    def getShortestPathAction(self, board, player):
        player_name = "B" if player == 1 else "T"
        point = board.path_to_goal(player)
        if point == None:
            print("No path to goal for", player_name)
            return None
        # get difference between agent and point
        action_x = point.X - board.agent_positions[player_name].X
        action_y = point.Y - board.agent_positions[player_name].Y
        action = MoveAction(Point(action_x, action_y))
        a = board.static_actions.get_index_of_action(action)
        return a
            
    
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
    
        

    
