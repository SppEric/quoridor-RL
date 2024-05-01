from alphazero.Coach import Coach
from quoridor_new.Quoridor import Quoridor
from quoridor_new.QuoridorNetWrapper import QuoridorNetWrapper as nn
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


args = dotdict({
    'wandb': True,
    'numIters': 10, #1000
    'numEps': 50, #100
    'tempThreshold': 15,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 50,
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp','5x5best.pth.tar'),
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
          'values': [32, 64, 128, 256]
      },
      'learning_rate':{
          'values': [0.01, 0.005, 0.001, 0.0005, 0.0001]
      },
      'epochs': {
          'value': 50
      },
      'channel_sizes': {
            'value': [512, 512, 512]
        },
        'fc_layer_sizes': {
            'value': [64]
        },
        'dropout': {
            'value': 0.4
        },
        'optimizer': {
            'value': 'adam'
        }

  }
}

def train():
    if args.wandb:
        wandb.init()

    g = Quoridor(5)
    nnet = nn(g, sweep_config, args.wandb)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args, sweep_config)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()



if __name__=="__main__":
    if args.wandb:
        sweep_id = wandb.sweep(sweep_config, project="quoridor-RL")
        wandb.agent(sweep_id, function=train, count=1)
    else:
        train()



