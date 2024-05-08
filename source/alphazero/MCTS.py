import copy
import math
import numpy as np
EPS = 1e-8
DEBUGGING = False
HEURISTICS = True
TAKE_SHORTEST_PATH_PROB = 0.5
class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

        self.sH = {}        # state history, to detect cycle in search()

    def getActionProb(self, board, curPlayer, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            v = self.search(board, curPlayer, canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if np.sum(counts) == 0: return counts
        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs

        counts = [x**(1./temp) for x in counts]

        valids = self.game.getValidMoves(board,curPlayer)
        counts = counts * valids
        if np.sum(counts) == 0:
            print("All valid moves were masked, do workaround.")
            counts = valids
        probs = [x/float(sum(counts)) for x in counts]
        return probs


    def search(self, board, curPlayer, canonicalBoard, counter=0):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        
        if counter == 0:
            self.sH = {}
        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(board, curPlayer)
        if self.Es[s]!=0:
            # terminal node
            return -self.Es[s]
        

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(board, curPlayer)
            self.Ps[s] = self.Ps[s]*valids      # masking invalid moves
            
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0

            # NOTE: STARTED AS -v
            return -v
        
        valids = self.Vs[s]

        ### THIS PART IS CHANGED FROM THE ORIGINAL CODE
        ### NEW ###
        bypass = False
        p_mask = None
        if HEURISTICS:
            if np.random.random() > TAKE_SHORTEST_PATH_PROB:
                # Place a wall w/ 1-prob_constant chance
                p_mask = self.game.getProbableWalls(board, curPlayer, valids)
                test_ps = self.Ps[s] * p_mask 
                if np.sum(test_ps) == 0:
                    p_mask = None
                    # a = self.game.getShortestPathAction(board, curPlayer)
                    # if a is not None:
                    #     bypass = True
            else:
                a = self.game.getShortestPathAction(board, curPlayer)
                if a is not None:
                    bypass = True

        ### END OF NEW ###

        cur_best = -float('inf')
        best_act = -1
        state_revisit_penalty = 1
        # pick the action with the highest upper confidence bound
        if not bypass:
            for a in range(self.game.getActionSize()):
                if p_mask is not None and p_mask[a] == 0:
                    continue
                if valids[a]:
                    if (s,a) in self.Qsa:
                        u = self.Qsa[(s,a)] + self.args.cpuct*self.Ps[s][0][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                        # if s in self.sH:
                        #     u -= state_revisit_penalty
                    else:
                        u = self.args.cpuct*self.Ps[s][0][a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?
                        # if s in self.sH:
                        #     u -= state_revisit_penalty
                    if u > cur_best:
                        cur_best = u
                        best_act = a
            a = best_act

        ### END OF CHANGED PART###

        if s in self.sH or counter > 20: # cycle
            return state_revisit_penalty
        
        self.sH[s] = 1
        next_board, next_player = self.game.getNextState(board, curPlayer, a)
        next_s = self.game.getCanonicalForm(next_board, next_player)

        v = self.search(next_board, next_player, next_s, counter+1)
        if v == 0:
            self.Ns[s] -= 1 if self.Ns[s] > 0 else 0
            return 0

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return -v
    

