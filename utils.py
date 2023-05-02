import numpy as np
import time


def normalization (data):
  # Parameters
  _, dim = data.shape
  norm_data = data.copy()
  
  min_val = np.zeros(dim)
  max_val = np.zeros(dim)
  
  # For each dimension
  for i in range(dim):
    min_val[i] = np.nanmin(norm_data[:,i])
    norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
    max_val[i] = np.nanmax(norm_data[:,i])
    norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)   
      
  return norm_data


def simulate_agent_on_samples(matrices):
  results = []
  masks = []
  for matrix in matrices:
    matrix = matrix.reshape(28, 28)
    mask = np.zeros((28, 28)).astype(int)

    # pick random number of walk steps 
    steps = np.random.randint(20, 600)
    # pick random starting point
    pos = np.random.randint(0, 27), np.random.randint(0, 27)

    previous_direction = (0, 0)

    for _ in range(steps):
      for i in range(3):
        for j in range(3):
          mask[pos[0] + i - 1][pos[1] + j - 1] = 1
      
      if (np.random.uniform(0, 1) < 0.7):
        previous_direction = random_direction()
    
      pos = (min(max(previous_direction[0] + pos[0], 1), matrix.shape[0] - 2), min(max(previous_direction[1] + pos[1], 1), matrix.shape[1] - 2))
      
      if (pos[0] == 0 or pos[0] == matrix.shape[0] - 1 or pos[1] == 0 or pos[1] == matrix.shape[1] - 1):
          previous_direction = random_direction()

    matrix = np.where(mask == 1, matrix, np.nan)

    masks.append(mask.reshape(28, 28, 1))
    results.append(matrix.reshape(28, 28, 1))

  return np.array(results), np.array(masks)


def random_direction():
    n = np.random.randint(0, 4)
    return [
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1)
    ][n]
