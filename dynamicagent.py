# Do some random walk and build up a map of the environment
# The map is a 2D array of floats from 0 to 1.

import random
import numpy as np

class DynamicAgent:
    def __init__(self, position, size, alpha=0.01, beta=0.9, starting_confidence=0.5):
        self.position = position
        self.previous_direction = self.random_direction()
        self.map = np.zeros((size[0], size[1]))
        self.mask = np.zeros((size[0], size[1]))
        self.map[:] = np.nan
        self.confidence = np.zeros((size[0], size[1]))
        self.decay = alpha
        self.beta = beta
        self.starting_confidence = starting_confidence

        self.messages = MessageQueue()
        self.vector = None
    
        self.vectors = dict()
        self.vector_confidences = dict()
    
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
                self.confidence[self.position[0] + i - 1][ self.position[1] + j - 1] = 1
        self.random_walk()

    def update_confidence(self):
        self.confidence = self.confidence - self.confidence * self.decay
        self.mask = np.where(self.confidence > 0.1, self.mask, 0)

        to_delete = []
        for agent in self.vectors:
            self.vector_confidences[agent] = self.vector_confidences[agent] - self.vector_confidences[agent] * self.decay
            if self.vector_confidences[agent] < 0.2:
                to_delete.append(agent)

    def get_confidence(self):
        return self.confidence
    
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
        self.vector_confidences[agent] = self.starting_confidence
    
    def get_average_vector(self):
        weighted_numerator = np.zeros(self.vector.shape)
        for agent in self.vectors:
            weighted_numerator += self.vector_confidences[agent] * (self.vectors[agent] ** 2)
        weighted_denominator = np.zeros(self.vector.shape)
        for agent in self.vectors:
            weighted_denominator += self.vector_confidences[agent] * self.vectors[agent]
        weighted_denominator += 1
        
        return ((1 - self.beta) * (weighted_numerator / weighted_denominator)) + (self.beta * self.vector)
    

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
