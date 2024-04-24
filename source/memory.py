import random
import numpy as np

class MemoryInstance:
    """Remember a specific state -> action -> reward -> next_state training example."""
    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

    def __repr__(self):
        return f'MemoryInstance(state={self.state}, action={self.action}, reward={self.reward}, next_state={self.next_state})'

class Memory:
    """Memory to store experiences (training examples) that the agent has encountered."""
    def __init__(self, max_memory):
        self.max_memory = max_memory
        self.samples = []

    def add_sample(self, memory_instance):
        """Adds a memory_instance to the memory, removing the oldest if the memory exceeds max_memory."""
        self.samples.append(memory_instance)
        if len(self.samples) > self.max_memory:
            self.samples.pop(0)

    def sample(self, no_samples):
        """Randomly samples 'no_samples' instances from memory, or returns all if there are not enough."""
        if no_samples > len(self.samples):
            return self.samples
        else:
            return random.sample(self.samples, no_samples)

    def clear(self):
        """Clears all the memories."""
        self.samples.clear()
