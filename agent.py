# Do some random walk and build up a map of the environment
# The map is a 2D array of floats from 0 to 1.

import random
import numpy as np

class Agent:
    def __init__(self, position, size):
        self.position = position
        self.previous_direction = self.random_direction()
        self.map = np.zeros((size[0], size[1]))
        self.mask = np.zeros((size[0], size[1]))
        self.map[:] = np.nan

        self.messages = MessageQueue()
        self.vector = None

        self.vectors = dict()
    
    def random_walk(self):
        if (random.uniform(0, 1) < 0.7):
            self.previous_direction = self.random_direction()
        
        self.position = (min(max(self.previous_direction[0] + self.position[0], 1), len(self.map) - 2), min(max(self.previous_direction[1] + self.position[1], 1), len(self.map[0]) - 2))
        
        if (self.position[0] == 0 or self.position[0] == len(self.map) - 1 or self.position[1] == 0 or self.position[1] == len(self.map[0]) - 1):
            self.previous_direction = self.random_direction()

    def random_direction(self):
        n = random.randint(0, 3)
        return [
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1)
        ][n]

    def measure(self, measurements: np.ndarray):
        for i in range(len(measurements)):
            for j in range(len(measurements[0])):
                self.map[self.position[0] + i - 1][self.position[1] + j - 1] = measurements[i][j]
                self.mask[self.position[0] + i - 1][ self.position[1] + j - 1] = 1
        self.random_walk()
    
    def get_position(self):
        return self.position

    def get_map(self):
        return self.map
    
    def get_mask(self):
        return self.mask

    def get_generator(self):
        return self.generator

    def get_message(self):
        return self.messages.get_message()
    
    def has_messages(self):
        return self.messages.get_size() > 0

    def set_vector(self, vector):
        self.vector = vector

    def receive_vector(self, vector, agent):
        self.messages.add_message((vector, agent))
    
    def get_vector(self):
        return self.vector

    def add_vector(self, vector, agent):
        self.vectors[agent] = vector
    
    def get_average_vector(self):
        vectors = list(self.vectors.values()) + [self.vector]
        return np.mean(np.array(vectors), axis=0)
    

class MessageQueue:

    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)

    def get_size(self):
        return len(self.messages)

    def get_message(self):
        if len(self.messages) > 0:
            return self.messages.pop(0)
        else:
            return None
