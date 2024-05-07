from alphazero.Coach import Coach
from alphazero.Arena import Arena
from quoridor_new.Quoridor import Quoridor
from quoridor_new.QuoridorNetWrapper import QuoridorNetWrapper as nn
import wandb
import constants


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


args = dotdict({
    'testbest': False,
    'minimax': False,
    'wandb': True,
    'numIters': 150, #10
    'numEps': 50,
    'tempThreshold': 15,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 200000,
    'maxDepthMinimax': 4, 
    'numMCTSSims': 20, #30
    'arenaCompare': 20, #40
    'cpuct': 1,

    'checkpoint': './temp_heuristic/',
    'load_model': False,
    'load_folder_file': ('./temp_heuristic','best'),
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
          'values': [32]
      },
      'learning_rate':{
          'values': [0.1]
      },
      'epochs': {
          'values': [50]
      },
        'fc_layer_sizes': {
            'value': [128, 64, 32]
      },
      'conv_layer_sizes': {
            'value': [128, 256]
      },
        'dropout': {
            'value': [0.3]
     },
     'conv': {
            'value': False
     },
  }
}

def error_resistant_train(g, nnet, load):
    # try:
    if load:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    c = Coach(g, nnet, args, sweep_config)
    if load:
        c.loadTrainExamples()
    c.learn()
    # except ValueError:
    #     print("Training stopped early")
    #     error_resistant_train(g, nnet, True)


def train():
    if args.wandb:
        wandb.init()

    g = Quoridor(constants.BOARD_SIZE)
    nnet = nn(g, sweep_config, args.wandb)

    error_resistant_train(g, nnet, args.load_model)

def testbest():
    g = Quoridor(constants.BOARD_SIZE)
    nnet = nn(g, sweep_config, args.wandb)
    nnet.load_checkpoint(args.load_folder_file[0], 'best')
    # if args.minimax:
    #     nnet.load_checkpoint(args.load_folder_file_minimax[0], 'best')

    c = Coach(g, nnet, args, sweep_config)
    c.testPlayers()


if __name__=="__main__":
    if args.testbest:
        print("Testing best model")
        testbest()
    else:
        print("Starting sweep")
        if args.wandb:
            if sweep_config['parameters']['conv']['value']:
                sweep_id = wandb.sweep(sweep_config, project="quoridor-RL-conv")
            else:
                sweep_id = wandb.sweep(sweep_config, project="quoridor-RL")
            wandb.agent(sweep_id, function=train, count=5)
        else:
            train()


