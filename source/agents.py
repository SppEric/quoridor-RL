import numpy as np
import tensorflow as tf
import random
import os
import math

from point import Point
from actions import StaticActions, MoveAction, WallAction

from memory import Memory, MemoryInstance

import constants
from constants import BoardElement

class Agent:
    """ Parent class of TopAgent and BottomAgent. Handles action selection and Q-learning updates.
        Differences between agents are managed by overridden methods for state perspective and action transformations.
    """
    def __init__(self, static_actions, model, name):
        self.memory = Memory(constants.MEMORY_SIZE)
        self.model = model
        self.static_actions = static_actions
        self.exploration_probability = constants.STARTING_EXPLORATION_PROBABILITY
        self.steps = 1
        self.game_loss = 0
        self.recent_loss = 0
        self.recent_loss_counter = 0
        self.name = name

    def take_action(self, board_state, only_inference, valid_human_action=None):
        state_vector = self.get_perspective_state(board_state)
        action_index = None

        if valid_human_action is None:
            if only_inference or random.random() > self.exploration_probability:
                action_index = self.greedy_action(state_vector, board_state)
            else:
                action_index = self.random_action(board_state)
            if action_index is None:
                return None  # Indicate no valid action available
            action = self.static_actions.all_actions[action_index]
        else:
            action = self.action_to_global_and_back(valid_human_action)
            action_index = self.static_actions.get_index_of_action(action)

        state_action = self.action_to_global_and_back(action)
        reward = board_state.apply_action(self.name, state_action)
        next_state_vector = self.get_perspective_state(board_state)
        self.memory.add_sample(MemoryInstance(state_vector, action_index, reward, next_state_vector))
        self.q_learn()

        self.steps += 1
        self.update_exploration_probability()

        return reward

    def update_exploration_probability(self):
        self.exploration_probability = constants.ENDING_EXPLORATION_PROBABILITY + \
                                       (constants.STARTING_EXPLORATION_PROBABILITY - constants.ENDING_EXPLORATION_PROBABILITY) \
                                       * math.exp(-constants.EXPLORATION_PROBABILITY_DECAY * self.steps)

    def random_action(self, board_state):
        actions = self.static_actions.move_actions if random.random() < constants.MOVE_ACTION_PROBABILITY else self.static_actions.all_actions
        action_indexes = list(range(len(actions)))
        random.shuffle(action_indexes)
        return self.first_legal_action(action_indexes, board_state)

    def greedy_action(self, state_vector, board_state):
        q_values = np.asarray(self.model.predict_one(np.expand_dims(state_vector, 0))).flatten()
        action_indexes = np.argsort(-q_values)
        return self.first_legal_action(action_indexes, board_state)

    def first_legal_action(self, action_indexes, board_state):
        for action_index in action_indexes:
            if self.is_legal_action(action_index, board_state):
                return action_index

    def is_legal_action(self, action_index, board_state):
        action = self.static_actions.all_actions[action_index]
        state_action = self.action_to_global_and_back(action)
        return board_state.is_legal_action(state_action, self.name)

    def get_perspective_state(self, board_state):
        full_grid_size = board_state.full_grid_size
        grid = board_state.build_grid(BoardElement.AGENT_BOT, BoardElement.AGENT_TOP)
        vector = [grid[x][y] for y in range(full_grid_size) for x in range(full_grid_size)]
        vector.extend([board_state.wall_counts[BoardElement.AGENT_BOT], board_state.wall_counts[BoardElement.AGENT_TOP]])
        return np.array(vector)

    def action_to_global_and_back(self, agent_action):
        return agent_action

    def q_learn(self):
        batch = self.memory.sample(self.model.batch_size)
        if not batch:
            return
        
        # Create arrays for states and next_states from the batch
        states = np.array([m.state for m in batch])
        next_states = np.array([m.next_state for m in batch if m.next_state is not None])

        # Predict Q values for current states and next states
        q_s_a = self.model.predict_batch(states)
        q_s_a_d = self.model.predict_batch(next_states)

        # Prepare target Q values array
        y = np.copy(q_s_a)
        for i, memory in enumerate(batch):
            state, action, reward, next_state = memory.state, memory.action, memory.reward, memory.next_state
            # Update the Q value for the action taken
            if next_state is None:
                y[i, action] = reward  # If final state, Q-value is the immediate reward
            else:
                y[i, action] = reward + constants.GAMMA * np.max(q_s_a_d[i])  # Bellman equation

        # Train the model with states and target Q-values
        self.model.train_batch(states, y)
        self.game_loss = self.model.get_recent_loss()
        self.recent_loss += self.game_loss
        self.recent_loss_counter += 1


    def get_recent_loss(self):
        if self.recent_loss_counter == 0:
            return 0
        recent_loss = self.recent_loss / self.recent_loss_counter
        self.recent_loss = 0
        self.recent_loss_counter = 0
        return recent_loss



class AZAgent:
    """ Parent class of TopAgent and BottomAgent. Handles action selection and Q-learning updates.
        Differences between agents are managed by overridden methods for state perspective and action transformations.
    """
    def __init__(self, static_actions, name):
        self.static_actions = static_actions
        self.exploration_probability = constants.STARTING_EXPLORATION_PROBABILITY
        self.name = name

    def take_action(self, board_state, valid_human_action=None):
        state_vector = self.get_perspective_state(board_state)
        action_index = None

        # if valid_human_action is None:
        #     if only_inference or random.random() > self.exploration_probability:
        #         action_index = self.greedy_action(state_vector, board_state)
        #     else:
        #         action_index = self.random_action(board_state)
        #     if action_index is None:
        #         return None  # Indicate no valid action available
        #     action = self.static_actions.all_actions[action_index]
        # else:
        action = self.action_to_global_and_back(valid_human_action)
        action_index = self.static_actions.get_index_of_action(action)

        state_action = self.action_to_global_and_back(action)
        reward = board_state.apply_action(self.name, state_action)
        next_state_vector = self.get_perspective_state(board_state)

        return board_state, next_state_vector, reward

    def random_action(self, board_state):
        actions = self.static_actions.move_actions if random.random() < constants.MOVE_ACTION_PROBABILITY else self.static_actions.all_actions
        action_indexes = list(range(len(actions)))
        random.shuffle(action_indexes)
        return self.first_legal_action(action_indexes, board_state)

    def first_legal_action(self, action_indexes, board_state):
        for action_index in action_indexes:
            if self.is_legal_action(action_index, board_state):
                return action_index

    def is_legal_action(self, action_index, board_state):
        action = self.static_actions.all_actions[action_index]
        state_action = self.action_to_global_and_back(action)
        return board_state.is_legal_action(state_action, self.name)

    def get_perspective_state(self, board_state):
        full_grid_size = board_state.full_grid_size
        grid = board_state.build_grid(BoardElement.AGENT_BOT, BoardElement.AGENT_TOP)
        vector = [grid[x][y] for y in range(full_grid_size) for x in range(full_grid_size)]
        vector.extend([board_state.wall_counts[BoardElement.AGENT_BOT], board_state.wall_counts[BoardElement.AGENT_TOP]])
        return np.array(vector)

    def action_to_global_and_back(self, agent_action):
        return agent_action


class TopAZAgent(AZAgent):
    """ Agent that starts out at the top of the screen and has a perspective that the board is 
        flipped horizontally and vertically.
    """
    def __init__(self, static_actions):
        super().__init__(static_actions, BoardElement.AGENT_TOP)

    def get_perspective_state(self, board_state):
        """ Appends grid squares to the state vector in reversed fashion, effectively flipping the
            horizontal and vertical axes. Also appends BoardElement.AGENT_TOP's wall count before
            BoardElement.AGENT_BOT's wall count because the current agent must come first to preserve consistency.
        """
        full_grid_size = board_state.full_grid_size
        grid = board_state.build_grid(BoardElement.AGENT_TOP, BoardElement.AGENT_BOT)

        vector = [grid[x][y] for y in reversed(range(full_grid_size)) for x in reversed(range(full_grid_size))]
        vector.extend([board_state.wall_counts[BoardElement.AGENT_TOP], board_state.wall_counts[BoardElement.AGENT_BOT]])
        return np.array(vector)

    def action_to_global_and_back(self, agent_action):
        """ Actions are also flipped on both axes. """
        if isinstance(agent_action, MoveAction):
            state_action = MoveAction(Point(-agent_action.direction.X, -agent_action.direction.Y))
        else:
            agent_wall_pos = agent_action.position
            wall_pos = Point(constants.BOARD_SIZE - agent_wall_pos.X - 2, constants.BOARD_SIZE - agent_wall_pos.Y - 2)
            state_action = WallAction(wall_pos, agent_action.orientation)
        return state_action

class BottomAZAgent(AZAgent):
    """ Bottom agent has nothing to override because its perspective is the same as
        the board's and the typical human perspective.
    """
    def __init__(self, static_actions):
        super().__init__(static_actions, BoardElement.AGENT_BOT)

    def get_perspective_state(self, board_state):
        return super().get_perspective_state(board_state)

    def action_to_global_and_back(self, agent_action):
        return agent_action


class TopAgent(Agent):
    """ Agent that starts out at the top of the screen and has a perspective that the board is 
        flipped horizontally and vertically.
    """
    def __init__(self, static_actions, model):
        super().__init__(static_actions, model, BoardElement.AGENT_TOP)

    def get_perspective_state(self, board_state):
        """ Appends grid squares to the state vector in reversed fashion, effectively flipping the
            horizontal and vertical axes. Also appends BoardElement.AGENT_TOP's wall count before
            BoardElement.AGENT_BOT's wall count because the current agent must come first to preserve consistency.
        """
        full_grid_size = board_state.full_grid_size
        grid = board_state.build_grid(BoardElement.AGENT_TOP, BoardElement.AGENT_BOT)

        vector = [grid[x][y] for y in reversed(range(full_grid_size)) for x in reversed(range(full_grid_size))]
        vector.extend([board_state.wall_counts[BoardElement.AGENT_TOP], board_state.wall_counts[BoardElement.AGENT_BOT]])
        return np.array(vector)

    def action_to_global_and_back(self, agent_action):
        """ Actions are also flipped on both axes. """
        if isinstance(agent_action, MoveAction):
            state_action = MoveAction(Point(-agent_action.direction.X, -agent_action.direction.Y))
        else:
            agent_wall_pos = agent_action.position
            wall_pos = Point(constants.BOARD_SIZE - agent_wall_pos.X - 2, constants.BOARD_SIZE - agent_wall_pos.Y - 2)
            state_action = WallAction(wall_pos, agent_action.orientation)
        return state_action

class BottomAgent(Agent):
    """ Bottom agent has nothing to override because its perspective is the same as
        the board's and the typical human perspective.
    """
    def __init__(self, static_actions, model):
        super().__init__(static_actions, model, BoardElement.AGENT_BOT)

    def get_perspective_state(self, board_state):
        return super().get_perspective_state(board_state)

    def action_to_global_and_back(self, agent_action):
        return agent_action