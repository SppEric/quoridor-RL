import tensorflow as tf
from tensorflow import keras
from keras import layers, initializers
import wandb
from quoridor_new.CustomLoss import quoridor_loss

class QuoridorNeuralNet(keras.Model):
    def __init__(self, board_x, board_y, action_size, config):
        super(QuoridorNeuralNet, self).__init__()
        self.board_x, self.board_y = board_x, board_y
        self.action_size = action_size
        self.config = config
        initializer = initializers.HeNormal()
        # self.conv1 = layers.Conv2D(self.config['conv_layer_sizes']['value'][0], 3, padding='same', activation='relu', kernel_initializer=initializer)
        # self.conv2 = layers.Conv2D(self.config['conv_layer_sizes']['value'][1], 3, padding='same', activation='relu', kernel_initializer=initializer)
        # self.conv3 = layers.Conv2D(self.config['conv_layer_sizes']['value'][2], 3, padding='same', activation='relu', kernel_initializer=initializer)
        self.fc1 = layers.Dense(self.config['fc_layer_sizes']['value'][0], activation='relu')
        self.fc2 = layers.Dense(self.config['fc_layer_sizes']['value'][1], activation='relu')
        self.fc3 = layers.Dense(self.config['fc_layer_sizes']['value'][2], activation='relu')
        self.pi = layers.Dense(self.action_size, activation='softmax')
        self.v = layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        pis = self.pi(x)
        vs = tf.math.tanh(self.v(x))
        return pis, vs

    def get_config(self):
        return {
            'board_x': self.board_x,
            'board_y': self.board_y,
            'action_size': self.action_size,
            'config': self.config
        }

    @classmethod
    def from_config(cls, config):
        return cls(config['board_x'], config['board_y'], config['action_size'], config['config'])

# Saving the model


class QuoridorConvNet(keras.Model):
    def __init__(self, board_x, board_y, action_size, config):
        super(QuoridorConvNet, self).__init__()
        self.board_x, self.board_y = board_x, board_y
        self.action_size = action_size
        self.config = config
        initializer = initializers.HeNormal()
        
        self.conv1 = layers.Conv2D(self.config['conv_layer_sizes']['value'][0], 3, padding='same', 
                                   activation='relu', kernel_initializer=initializer)
        self.conv2 = layers.Conv2D(self.config['conv_layer_sizes']['value'][1], 3, padding='same', 
                                   activation='relu', kernel_initializer=initializer)
        self.maxpool = layers.MaxPooling2D()
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(self.config['fc_layer_sizes']['value'][0], activation='relu')
        self.pi = layers.Dense(self.action_size, activation='softmax')
        self.v = layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        pi = self.pi(x)
        v = tf.math.tanh(self.v(x))
        return pi, v

    def get_config(self):
        return {
            'board_x': self.board_x,
            'board_y': self.board_y,
            'action_size': self.action_size,
            'config': self.config
        }

    @classmethod
    def from_config(cls, config):
        return cls(config['board_x'], config['board_y'], config['action_size'], config['config'])


