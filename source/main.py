from alphazero.Coach import Coach
from alphazero.Arena import Arena
from quoridor_new.Quoridor import Quoridor
from quoridor_new.QuoridorNetWrapper import QuoridorNetWrapper as nn
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import constants


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


args = dotdict({
    'testbest': False,
    'minimax': False,
    'wandb': True,
    'numIters': 10, #10
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 200000,
    'maxDepthMinimax': 4, 
    'numMCTSSims': 30, #30
    'arenaCompare': 40, #40
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp','best'),
    'numItersForTrainExamplesHistory': 20,
})

sweep_config = {
  'method': 'random', 
  'metric': {
      'name': 'val_loss',
      'goal': 'minimize'
  },
  'parameters': {
      'batch_size': {
          'values': [32, 64, 128]
      },
      'learning_rate':{
          'values': [0.1, 0.05]
      },
      'epochs': {
          'values': [25, 50]
      },
        'fc_layer_sizes': {
            'value': [128, 64, 32]
        },

  }
}

def train():
    if args.wandb:
        wandb.init()

    g = Quoridor(constants.BOARD_SIZE)
    nnet = nn(g, sweep_config, args.wandb)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args, sweep_config)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()

    try:
        c.learn()
    except ValueError:
        print("Training stopped early")
        pass


def testbest():
    g = Quoridor(constants.BOARD_SIZE)
    nnet = nn(g, sweep_config, args.wandb)
    nnet.load_checkpoint(args.load_folder_file[0], 'best')
    
    c = Coach(g, nnet, args, sweep_config)
    c.testPlayers()


if __name__=="__main__":
    if args.testbest:
        print("Testing best model")
        testbest()
    else:
        print("Starting sweep")
        if args.wandb:
            sweep_id = wandb.sweep(sweep_config, project="quoridor-RL")
            wandb.agent(sweep_id, function=train, count=15)
        else:
            train()



