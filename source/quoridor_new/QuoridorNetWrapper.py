from NeuralNet import NeuralNet
import tensorflow as tf
import numpy as np
from QuoridorNeuralNet import QuoridorNeuralNet

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