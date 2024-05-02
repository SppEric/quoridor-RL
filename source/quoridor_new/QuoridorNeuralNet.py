import tensorflow as tf
import keras
from keras import layers
from quoridor_new.CustomLoss import quoridor_loss



class QuoridorNeuralNet(tf.keras.Model):
        def __init__(self, game, config):
            super().__init__()
            # Grab game board properties
            self.board_x, self.board_y = game.getBoardSize()
            self.action_size = game.getActionSize()

            # Initialize model according to config parameters
            self.config = config
            # self.conv1 = layers.Conv2D(self.config['channel_sizes']['value'][0], 3, padding='same', activation='relu')
            # self.conv2 = layers.Conv2D(self.config['channel_sizes']['value'][1], 3, padding='same', activation='relu')
            # self.conv3 = layers.Conv2D(self.config['channel_sizes']['value'][2], 3, padding='same', activation='relu')
            # self.dropout = layers.Dropout(self.config['dropout']['value'])
            # self.batch_norm = layers.BatchNormalization()
            self.fc1 = layers.Dense(self.config['fc_layer_sizes']['value'][0], activation='relu')
            self.fc2 = layers.Dense(self.config['fc_layer_sizes']['value'][0], activation='relu')
            self.fc3 = layers.Dense(self.config['fc_layer_sizes']['value'][0], activation='relu')
            self.pi = layers.Dense(self.action_size, activation='relu')
            self.v = layers.Dense(1)
            

        def call(self, boards):
            # x = self.conv1(boards)
            # x = self.conv2(x)
            # x = self.conv3(x)
            # x = self.dropout(x)
            # x = self.batch_norm(x)
            x = self.fc1(boards)
            x = self.fc2(x)
            x = self.fc3(x)
            pis = self.pi(x)
            vs = tf.math.tanh(self.v(pis))

            # Returning both policy and values
            return pis, vs
        
        def get_config(self):
            config = super().get_config()
            config.update(
                 {
                    "config" : self.config,
                    "loss" : quoridor_loss
                }
            )

            return {**config}
        
        def from_config(cls, config):
            config["config"] = keras.saving.deserialize_keras_object(config["config"])
            config["loss"] = keras.saving.deserialize_keras_object(config["loss"])

            return cls(**config)   
             
             