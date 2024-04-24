import tensorflow as tf

LAYER_SIZE = 350
TENSORFLOW_CHECKPOINT_FOLDER = 'tensorflow_checkpoint'
TENSORFLOW_SAVE_FILE = 'agent'


class DeepQLearningModel(tf.keras.Model):
    """ Neural network to implement deep Q-learning with memory """
    def __init__(self, num_states, num_actions):
        super(DeepQLearningModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(LAYER_SIZE, activation='relu')
        self.dense2 = tf.keras.layers.Dense(LAYER_SIZE, activation='relu')
        self.logits = tf.keras.layers.Dense(num_actions)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.logits(x)


class Model:
    def __init__(self, num_states, num_actions, batch_size, restore):
        self.num_states = num_states
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.model = DeepQLearningModel(num_states, num_actions)
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        self.recent_loss = None  # Initialize recent_loss

        if restore:
            self.load()
        else:
            # No explicit initialization is necessary as it is handled by Keras
            pass

    def save(self):
        """ Save model parameters to file """
        self.model.save_weights(TENSORFLOW_CHECKPOINT_FOLDER + '/' + TENSORFLOW_SAVE_FILE)
        print("Model saved to ", TENSORFLOW_CHECKPOINT_FOLDER + '/' + TENSORFLOW_SAVE_FILE)
        
    def load(self):
        """ Load model parameters from file """
        self.model.load_weights(TENSORFLOW_CHECKPOINT_FOLDER + '/' + TENSORFLOW_SAVE_FILE)
        print("Model loaded from ", TENSORFLOW_CHECKPOINT_FOLDER + '/' + TENSORFLOW_SAVE_FILE)

    def predict_one(self, state):
        """ Run the state through the model and return the predicted q values """
        return self.model(state.reshape(1, self.num_states)).numpy()
    
    def predict_batch(self, states):
        """ Run a batch of states through the model and return a batch of q values. """
        return self.model(states).numpy()
    
    def train_batch(self, x_batch, y_batch):
        """ Trains the model with a batch of X (state) -> Y (reward) examples """
        loss = self.model.train_on_batch(x_batch, y_batch)
        self.recent_loss = loss  # Store the loss after training
        return loss

    def get_recent_loss(self):
        """ Returns the loss from the most recent batch training. """
        return self.recent_loss

