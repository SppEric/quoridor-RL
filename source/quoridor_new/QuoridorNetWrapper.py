from alphazero.NeuralNet import NeuralNet
import tensorflow as tf
import numpy as np
from QuoridorNeuralNet import QuoridorNeuralNet
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

default_config = {
    'channel_sizes' : [512, 512, 512],          # Expecting 3 conv, 2 fc always?
    'fc_layer_sizes' : [128], 
    'dropout' : 0.4,
    'optimizer' : 'adam',
}

# Start a run, tracking hyperparameters
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",

    # track hyperparameters and run metadata with wandb.config
    config={
        "layer_1": 512,
        "activation_1": "relu",
        "dropout": random.uniform(0.01, 0.80),
        "layer_2": 10,
        "activation_2": "softmax",
        "optimizer": "sgd",
        "loss": "sparse_categorical_crossentropy",
        "metric": "accuracy",
        "epoch": 8,
        "batch_size": 256
    }
)

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
  }
}

class QuoridorNetWrapper(NeuralNet):
    def __init__(self, game, config=default_config):
        # How to interface with weights and biases?
        # For now, temporary model architecture
        self.config = config

        # Set up optimizer and loss
        self.loss_fn = self.quoridor_loss
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size
        
        # Initialize model
        self.model = QuoridorNeuralNet(game, config)


    def train(self, examples):
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (board, pi, v). pi is the MCTS informed policy vector for
                      the given board, and v is its value. The examples has
                      board in its canonical form.
        """
        # Input looks like many of (canonicalBoard, pi, reward * self.curPlayer)
        for epoch in range(self.config.epochs):
            print('EPOCH ::: ' + str(epoch+1))

            batch_idx = 0
            while batch_idx < int(len(examples)/self.config.batch_size):
                sample_ids = np.random.randint(len(examples), size=self.config.batch_size)
                boards, target_pis, target_vs = list(zip(*[examples[i] for i in sample_ids]))
                with tf.GradientTape() as tape:
                    pred_pis, pred_vs = self.model(boards)
                    loss = self.loss_fn(pred_pis, target_pis, pred_vs, target_vs)

            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        


    def predict(self, board):
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
        pass

    def save_checkpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        pass

    def load_checkpoint(self, folder, filename):
        """
        Loads parameters of the neural network from folder/filename
        """
        pass

    def quoridor_loss(self, pred_pis, target_pis, pred_vs, target_vs):
        # Initialize loss functions
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        mse = tf.keras.losses.mean_squared_error

        # Calculate loss
        loss_pi = cce(target_pis, pi)
        loss_v = mse(target_vs, tf.reshape(self.v, shape=[-1,]))

        return loss_pi + loss_v
from NeuralNet import NeuralNet
import tensorflow as tf
import numpy as np
from QuoridorNeuralNet import QuoridorNeuralNet
import os

default_config = {
    'channel_sizes' : [512, 512, 512],          # Expecting 3 conv, 2 fc always?
    'fc_layer_sizes' : [128, 64], 
    'dropout' : 0.4,
    'optimizer' : 'adam',
    'learning_rate' : 0.001,
    'epochs' : 10,
    'batch_size': 32,
}

class QuoridorNetWrapper(NeuralNet):
    def __init__(self, game, config=default_config):
        # Set up from config
        self.config = config
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size

        # We use a custom loss function, written below
        self.loss_fn = self.quoridor_loss

        # Initialize model
        self.model = QuoridorNeuralNet(game, config)


    def train(self, examples):
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (board, pi, v). pi is the MCTS informed policy vector for
                      the given board, and v is its value. The examples has
                      board in its canonical form.
        """
        # Input looks like many of (canonicalBoard, pi, reward * self.curPlayer)
        for epoch in range(self.config.epochs):
            print('EPOCH ::: ' + str(epoch+1))

            batch_idx = 0
            while batch_idx < int(len(examples)/self.config.batch_size):
                # Create inputs
                sample_ids = np.random.randint(len(examples), size=self.config.batch_size)
                boards, target_pis, target_vs = list(zip(*[examples[i] for i in sample_ids]))

                # Run model
                with tf.GradientTape() as tape:
                    pred_pis, pred_vs = self.model(boards)
                    loss = self.loss_fn(pred_pis, target_pis, pred_vs, target_vs)

                # Logging
                if batch_idx % 5 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (batch_idx, float(loss))
                    )
                    print("Seen so far: %d samples" % ((batch_idx + 1) * self.config.batch_size))

                # Increment batch stuff
                batch_idx += 1

            # Backprop
            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

            # Do logging for end of epoch stuff
            # Idk what to track here
                

    def predict(self, board):
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
        board = tf.expand_dims(board, 0)
        pi, v = self.model(board)
        probs = tf.nn.softmax(pi) # Not sure if we want probabilities or just the policy vector
        return probs, v

    def save_checkpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) in
        folder/filename

        Make sure to remember to use .keras filetype in filename!
        
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(f"Checkpoint Directory does not exist! Making directory {folder}")
            os.makedirs(folder)
        else:
            print("Found directory!")
        print("Saving model...")

        self.model.save(filepath)

    def load_checkpoint(self, folder, filename):
        """
        Loads parameters of the neural network from folder/filename
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise(f"No model found at {filepath}")
        else:
            self.model = tf.keras.models.load_model(filepath)
            self.model.summary()


    def quoridor_loss(self, pred_pis, target_pis, pred_vs, target_vs):
        # Initialize loss functions
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        mse = tf.keras.losses.mean_squared_error

        # Calculate loss
        loss_pi = cce(target_pis, pred_pis)
        loss_v = mse(target_vs, tf.reshape(pred_vs, shape=[-1,]))

        return loss_pi + loss_v