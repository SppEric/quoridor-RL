from alphazero.NeuralNet import NeuralNet
import tensorflow as tf
import numpy as np
from quoridor_new.QuoridorNeuralNet import QuoridorNeuralNet, QuoridorConvNet
from quoridor_new.CustomLoss import quoridor_loss
import wandb
from wandb.keras import WandbCallback
import os



class QuoridorNetWrapper(NeuralNet):
    def __init__(self, game, config, is_wandb):
        # Set up from config
        self.config = config['parameters']
        if is_wandb:
            self.optimizer =  tf.keras.optimizers.SGD(learning_rate=wandb.config.learning_rate) # NOTE: Make this an if statement later wandb.config.optimizer
            self.batch_size = wandb.config.batch_size
            self.epochs = wandb.config.epochs
        else:
            print("NOT USING WANDB")
            self.optimizer =  tf.keras.optimizers.SGD(learning_rate=self.config['learning_rate']['values'][0])
            self.batch_size = self.config['batch_size']['values'][0]
            self.epochs = self.config['epochs']['values'][0]
        self.wandb = is_wandb
        self.config['wandb'] = is_wandb
        

        print(self.epochs, "EPOCHS")
        # We use a custom loss function, written below
        self.loss_fn = quoridor_loss
        board_x, board_y = game.getBoardSize()
        action_size = game.getActionSize()

        # Initialize model
        if self.config['conv']['value']:
            self.model = QuoridorConvNet(board_x, board_y, action_size, self.config)
        else:
            self.model = QuoridorNeuralNet(board_x, board_y, action_size, self.config)
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_fn,
                           )


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
            # print('EPOCH ::: ' + str(epoch+1))
            epoch_loss = []
            batch_idx = 0
            while batch_idx < int(len(examples)/self.batch_size):
                # Create inputs
                sample_ids = np.random.randint(len(examples), size=self.batch_size)
                boards, target_pis, target_vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = tf.stack(boards)
                target_pis = tf.stack(target_pis)
                target_vs = tf.stack(target_vs)

                # Run model
                with tf.GradientTape() as tape:
                    pred_pis, pred_vs = self.model(boards)
                    loss = self.loss_fn(pred_pis, target_pis, pred_vs, target_vs)
                
                # Logging
                epoch_loss.append(loss)
                # if batch_idx % 2 == 0:
                    # print(
                    #     "Training loss (for one batch) at step %d: %.4f"
                    #     % (batch_idx, float(loss))
                    # )
                    # print("Seen so far: %d samples" % ((batch_idx + 1) * self.batch_size))

                # Increment batch stuff
                batch_idx += 1

                # Backprop
                grads = tape.gradient(loss, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                

            
            # Do logging for end of epoch stuff
            total_loss = np.mean(epoch_loss)
            print("EPOCH LOSS: ", total_loss)
            if self.wandb:
                wandb.log({
                    'epochs': epoch,
                    'loss': total_loss
                })
                

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
        filepath = os.path.join(folder, filename) + '.keras'
        if not os.path.exists(folder):
            print(f"Checkpoint Directory does not exist! Making directory {folder}")
            os.makedirs(folder)
        else:
            print("Found directory!")
        print("Saving model...")
        
        # self.model.compile(optimizer='adam', loss=quoridor_loss)
        self.model.save(filepath)
        # self.model.save(filepath, save_format='tf')

    def load_checkpoint(self, folder, filename):
        """
        Loads parameters of the neural network from folder/filename
        """
        filepath = os.path.join(folder, filename) + '.keras'
        print(filepath)
        if not os.path.exists(filepath):
            raise(f"No model found at {filepath}")
        else:
            if self.config['conv']['value']:
                custom_objects = {"QuoridorConvNet": QuoridorConvNet, "quoridor_loss": quoridor_loss}
            else:
                custom_objects = {"QuoridorNeuralNet": QuoridorNeuralNet, "quoridor_loss": quoridor_loss}

            # Example of loading a model
            self.model = tf.keras.models.load_model(filepath, custom_objects=custom_objects)
            # self.model = tf.keras.models.load_model(filepath, compile=False)
            # optimizer =  tf.keras.optimizers.SGD(learning_rate=0.001)
            # self.model.compile(loss=quoridor_loss, optimizer=optimizer)
            if self.wandb:
                self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=wandb.config.learning_rate), loss=quoridor_loss)
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']['values'][0]), loss=quoridor_loss)
            self.model.summary()