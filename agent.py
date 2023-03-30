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
    
    def receive_observation(self, x_mb, m_mb):
        self.messages.add_message((x_mb, m_mb))

    def get_observation(self):
        return self.messages.get_message()
    
    def has_observations(self):
        return self.messages.get_size() > 0
    
    def fill_in(self, x_mb, m_mb):
        x_mb = x_mb.reshape(28, 28)
        m_mb = m_mb.reshape(28, 28)

        m_mb = m_mb - (m_mb * self.mask)
        self.mask = self.mask + m_mb
        self.map = self.map + (x_mb * m_mb)

    

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
