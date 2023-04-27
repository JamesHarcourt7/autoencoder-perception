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
    
    '''
    def get_average_vector(self):
        vectors = list(self.vectors.values()) + [self.vector]
        return np.mean(np.array(vectors), axis=0)
    '''

    def get_average_vector(self):
        weighted_numerator = np.zeros(self.vector.shape)
        for agent in self.vectors:
            weighted_numerator += self.vectors[agent] ** 2
        weighted_numerator += self.vector
        weighted_denominator = np.zeros(self.vector.shape)
        for agent in self.vectors:
            weighted_denominator += self.vectors[agent]
        weighted_denominator += 1
        
        return weighted_numerator / weighted_denominator
    

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

class Agent2:
    def __init__(self, position, size):
        self.position = position
        self.previous_direction = self.random_direction()
        self.map = np.zeros((size[0], size[1]))
        self.mask = np.zeros((size[0], size[1]))
        self.map[:] = np.nan

        self.messages1 = MessageQueue()
        self.messages2 = MessageQueue()
        self.vector1 = None
        self.vector2 = None

        self.vectors1 = dict()
        self.vectors2 = dict()
        self.beta = 0.8
    
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

    def get_message1(self):
        return self.messages1.get_message()
    
    def get_message2(self):
        return self.messages2.get_message()
    
    def has_messages1(self):
        return self.messages1.get_size() > 0
    
    def has_messages2(self):
        return self.messages2.get_size() > 0

    def set_vector1(self, vector):
        self.vector1 = vector

    def set_vector2(self, vector):
        self.vector2 = vector

    def receive_vector1(self, vector, agent):
        self.messages1.add_message((vector, agent))
    
    def receive_vector2(self, vector, agent):
        self.messages2.add_message((vector, agent))
    
    def get_vector1(self):
        return self.vector1
    
    def get_vector2(self):
        return self.vector2

    def add_vector1(self, vector, agent):
        self.vectors1[agent] = vector

    def add_vector2(self, vector, agent):
        self.vectors2[agent] = vector
    
    '''
    def get_average_vector(self):
        vectors = list(self.vectors.values()) + [self.vector]
        return np.mean(np.array(vectors), axis=0)
    '''

    def get_average_vector1(self):
        weighted_numerator = np.zeros(self.vector1.shape)
        for agent in self.vectors1:
            weighted_numerator += self.vectors1[agent] ** 2
        weighted_denominator = np.zeros(self.vector1.shape)
        for agent in self.vectors1:
            weighted_denominator += self.vectors1[agent]
        weighted_denominator += 1
        
        return ((1 - self.beta) * weighted_numerator / weighted_denominator) + (self.beta * self.vector1)
    
    def get_average_vector2(self):
        weighted_numerator = np.zeros(self.vector2.shape)
        for agent in self.vectors2:
            weighted_numerator += self.vectors2[agent] ** 2
        weighted_denominator = np.zeros(self.vector2.shape)
        for agent in self.vectors2:
            weighted_denominator += self.vectors2[agent]
        weighted_denominator += 1
        
        return ((1 - self.beta) * weighted_numerator / weighted_denominator) + (self.beta * self.vector2)
