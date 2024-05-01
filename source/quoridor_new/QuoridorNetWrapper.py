from alphazero.NeuralNet import NeuralNet
import tensorflow as tf
import numpy as np
import random
from quoridor_new.QuoridorNeuralNet import QuoridorNeuralNet
import wandb
from wandb.keras import WandbCallback
import os



class QuoridorNetWrapper(NeuralNet):
    def __init__(self, game, config, is_wandb):
        # Set up from config
        self.config = config['parameters']
        if is_wandb:
            self.optimizer = wandb.config.optimizer
            self.batch_size = wandb.config.batch_size
            self.epochs = wandb.config.epochs
            print("BATCHSIZE IS", self.batch_size)
        else:
            self.optimizer = config.optimizer
            self.batch_size = config.batch_size
        self.wandb = is_wandb

        # We use a custom loss function, written below
        self.loss_fn = self.quoridor_loss

        # Initialize model
        self.model = QuoridorNeuralNet(game, self.config)


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
        for epoch in range(self.epochs):
            print('EPOCH ::: ' + str(epoch+1))
            epoch_loss = []
            batch_idx = 0
            while batch_idx < int(len(examples)/self.batch_size):
                # Create inputs
                sample_ids = np.random.randint(len(examples), size=self.batch_size)
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
                    print("Seen so far: %d samples" % ((batch_idx + 1) * self.batch_size))

                # Increment batch stuff
                batch_idx += 1

                # Backprop
                epoch_loss.append(loss)
                grads = tape.gradient(loss, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

            # Do logging for end of epoch stuff
            # Idk what to track here
            if self.wandb:
                wandb.log({'epochs': epoch,
                    'loss': np.mean(epoch_loss)})
                

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