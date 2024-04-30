import tensorflow as tf
from keras import layers


class QuoridorNeuralNet(tf.keras.Model):
        def __init__(self, game, config):
            # Renaming functions
            Relu = tf.nn.relu
            Tanh = tf.nn.tanh
            BatchNormalization = tf.layers.batch_normalization
            Dropout = tf.layers.dropout
            Dense = tf.layers.dense

            # Grab game board properties
            self.board_x, self.board_y = game.getBoardSize()
            self.action_size = game.getActionSize()

            # Initialize model according to config parameters
            
            self.pi = layers.Dense(self.action_size)
            self.v = layers.Dense(1)

        def call(self, boards):
            x = self.all_the_other_stuff(boards)
            pis = self.pi(x)
            vs = tf.math.tanh(self.v(pis))

            # Returning both policy and values
            return pis, vs