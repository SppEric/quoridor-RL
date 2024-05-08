from collections import deque
import copy
from alphazero.Arena import Arena
from alphazero.MCTS import MCTS
from alphazero.Minimax import Minimax
import numpy as np
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
import tensorflow as tf

DEBUGGING = False
HEURISTICS = False

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, nnet, args, sweep_config):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game, sweep_config, args.wandb)  # the competitor network
        self.tempnet = self.nnet.__class__(self.game, sweep_config, args.wandb)  # the next most recent network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.minimax = Minimax(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0
        
        while True and episodeStep<200:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board,self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)


            pi = self.mcts.getActionProb(board, self.curPlayer, canonicalBoard, temp=temp)

            if np.sum(pi) == 0: break

            # sym = self.game.getSymmetries(canonicalBoard, pi)
            # for b,p in sym:
            #self.game.print_board(canonicalBoard)

            # can only pick if action is valid
            action = np.random.choice(len(pi), p=pi)
            
            
            trainExamples.append([canonicalBoard, self.curPlayer, pi, None])
            
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)
            
            r = self.game.getGameEnded(board, self.curPlayer)

            if r!=0:
                return [(x[0],x[2],r*x[1]) for x in trainExamples]
        #return [(x[0],x[2],0) for x in trainExamples]
        return []

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters+1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i>1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                # eps_time = AverageMeter()
                # bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()

                for eps in range(self.args.numEps):
                    print("Episode", eps)  
                    self.mcts = MCTS(self.game, self.nnet, self.args)   # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                    # bookkeeping + plot progress
                    # eps_time.update(time.time() - end)
                    end = time.time()
                    # bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg,
                                                                                                            #    total=bar.elapsed_td, eta=bar.eta_td)
                    # bar.next()
                # bar.finish()

                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)
                trainStats = [0,0,0]
                for _,_,res in iterationTrainExamples:
                    trainStats[res] += 1
                print(trainStats)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i-1)

            # shuffle examlpes before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp')
            pmcts = MCTS(self.game, self.pnet, self.args)

            # self.nnet.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x, y, z: np.argmax(pmcts.getActionProb(x, y, z, temp=0)),
                          lambda x, y, z: np.argmax(nmcts.getActionProb(x, y, z, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins+nwins > 0 and float(nwins)/(pwins+nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                #self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='best')
                self.nnet = self.tempnet
                
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best')
                self.tempnet.load_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))

    def getNextAction(self, board, curPlayer, canonicalBoard):
        win = 0
        if self.args.minimax and 0 in board.wall_counts.values():
            action, win = self.minimax.getActionMinimax(board, curPlayer)
        
        # If minimax does not get to terminal state, use MCTS
        if (not self.args.minimax) or (win != 1):
            
            # p, v = self.nnet.predict(canonicalBoard)
            # valids = self.game.getValidMoves(board, curPlayer)
            # p = p*valids      # masking invalid moves

            # action = np.argmax(p)
            action = np.argmax(self.mcts.getActionProb(board, curPlayer, canonicalBoard))

        return action
    
    
    def getNextActionNoMini(self, board, curPlayer, canonicalBoard):
        action = np.argmax(self.mcts.getActionProb(board, curPlayer, canonicalBoard))
        return action



    def getRandomAction(self, board, curPlayer, canonicalBoard):
        valids = self.game.getValidMoves(board, curPlayer)
        return np.random.choice(np.nonzero(valids)[0])
    
    def getMoveForwardAction(self, board, curPlayer, canonicalBoard):
        valids = self.game.getValidMoves(board, curPlayer)
        if valids[2] == 1:
            return 2
        return np.random.choice(np.nonzero(valids)[0])
    
    def testPlayers(self):
        
        arena = Arena(lambda x, y, z: self.getNextAction(x, y, z), 
                      lambda x, y, z: self.getMoveForwardAction(x, y, z), self.game, display=print)

        # arena = Arena(lambda x, y, z: np.argmax(self.mcts.getActionProb(x, y, z)), 
        #               lambda x, y, z: self.getRandomAction(x, y, z), self.game, display=print)

        print("PLAYING GAMES!")
        wins = 0
        losses = 0
        draws = 0
        i = 0
        while i < 10:
            outcome = arena.playGame(verbose=True)
            if outcome == 1:
                wins += 1
            elif outcome == -1:
                losses += 1
            else:
                draws += 1
            i += 1

            
        print("Wins: ", wins)
        print("Losses: ", losses)
        print("Draws: ", draws)

        

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration)

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        #modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = './temp/checkpoint_8.examples'
        # Make it automatic self.args.load_folder_file[0]+".examples"
        m = 0
        for file in os.listdir('./temp'):
            # pick one with largest number
            if file.endswith(".examples"):
                if int(file.split('_')[1].split('.')[0]) > m:
                    examplesFile = os.path.join('./temp', file)
                    m = int(file.split('_')[1].split('.')[0])

        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
