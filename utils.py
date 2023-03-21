import numpy as np
import time


def normalization (data, parameters=None):
  '''Normalize data in [0, 1] range.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  '''

  # Parameters
  _, dim = data.shape
  norm_data = data.copy()
  
  if parameters is None:
  
    # MixMax normalization
    min_val = np.zeros(dim)
    max_val = np.zeros(dim)
    
    # For each dimension
    for i in range(dim):
      min_val[i] = np.nanmin(norm_data[:,i])
      norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
      max_val[i] = np.nanmax(norm_data[:,i])
      norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)   
      
    # Return norm_parameters for renormalization
    norm_parameters = {'min_val': min_val,
                       'max_val': max_val}

  else:
    min_val = parameters['min_val']
    max_val = parameters['max_val']
    
    # For each dimension
    for i in range(dim):
      norm_data[:,i] = norm_data[:,i] - min_val[i]
      norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)  
      
    norm_parameters = parameters    
      
  return norm_data, norm_parameters


def renormalization (norm_data, norm_parameters):
  '''Renormalize data from [0, 1] range to the original range.
  
  Args:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  
  Returns:
    - renorm_data: renormalized original data
  '''
  
  min_val = norm_parameters['min_val']
  max_val = norm_parameters['max_val']

  _, dim = norm_data.shape
  renorm_data = norm_data.copy()
    
  for i in range(dim):
    renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
    renorm_data[:,i] = renorm_data[:,i] + min_val[i]
    
  return renorm_data


def rounding (imputed_data, data_x):
  '''Round imputed data for categorical variables.
  
  Args:
    - imputed_data: imputed data
    - data_x: original data with missing values
    
  Returns:
    - rounded_data: rounded imputed data
  '''
  
  _, dim = data_x.shape
  rounded_data = imputed_data.copy()
  
  for i in range(dim):
    temp = data_x[~np.isnan(data_x[:, i]), i]
    # Only for the categorical variable
    if len(np.unique(temp)) < 20:
      rounded_data[:, i] = np.round(rounded_data[:, i])
      
  return rounded_data


def rmse_loss (ori_data, imputed_data, data_m):
  '''Compute RMSE loss between ori_data and imputed_data
  
  Args:
    - ori_data: original data without missing values
    - imputed_data: imputed data
    - data_m: indicator matrix for missingness
    
  Returns:
    - rmse: Root Mean Squared Error
  '''
  
  ori_data, norm_parameters = normalization(ori_data)
  imputed_data, _ = normalization(imputed_data, norm_parameters)
    
  # Only for missing values
  nominator = np.sum(((1-data_m) * ori_data - (1-data_m) * imputed_data)**2)
  denominator = np.sum(1-data_m)
  
  rmse = np.sqrt(nominator/float(denominator))
  
  return rmse


def binary_sampler(p, rows, cols):
  '''Sample binary random variables.
  
  Args:
    - p: probability of 1
    - rows: the number of rows
    - cols: the number of columns
    
  Returns:
    - binary_random_matrix: generated binary random matrix.
  '''
  unif_random_matrix = np.random.uniform(0., 1., size = [rows, cols])
  binary_random_matrix = 1*(unif_random_matrix < p)
  return binary_random_matrix


def uniform_sampler(low, high, rows, cols):
  '''Sample uniform random variables.
  
  Args:
    - low: low limit
    - high: high limit
    - rows: the number of rows
    - cols: the number of columns
    
  Returns:
    - uniform_random_matrix: generated uniform random matrix.
  '''
  return np.random.uniform(low, high, size = [rows, cols])       


def sample_batch_index(total, batch_size):
  '''Sample index of the mini-batch.
  
  Args:
    - total: total number of samples
    - batch_size: batch size
    
  Returns:
    - batch_idx: batch index
  '''
  total_idx = np.random.permutation(total)
  batch_idx = total_idx[:batch_size]
  return batch_idx

def remove_regions(x, min_size, max_size):
  # Determine the size of the 3D matrix
  no, dim = x.shape
  x = x.reshape(no, int(np.sqrt(dim)), int(np.sqrt(dim)))
  _, n_rows, n_cols = x.shape

  # Create a copy of the input 3D matrix
  result = np.copy(x)

  masks = []

  # Iterate over each 2D matrix in the 3D matrix
  for i in range(no):

    # Choose a random point within the image
    p_j, p_k = np.random.randint(0, n_rows), np.random.randint(0, n_cols)

    # Choose the start indices based on the random point
    start_j = max(0, p_j - np.random.randint(min_size, max_size + 1) // 2)
    start_k = max(0, p_k - np.random.randint(min_size, max_size + 1) // 2)

    # Determine the end indices for the region of removed elements
    end_j = min(n_rows - 1, start_j + np.random.randint(min_size, max_size + 1))
    end_k = min(n_cols - 1, start_k + np.random.randint(min_size, max_size + 1))


    # Create a mask with NaN values where the region is removed
    mask = np.ones((n_rows, n_cols))
    mask[start_j:end_j, start_k:end_k] = np.nan

    # Replace the removed region with NaN values in the current matrix
    result[i] = np.where(np.isnan(mask), np.nan, result[i])

    # Create mask based off of the region of removed elements
    masks.append(np.logical_not(np.isnan(result[i])).astype(int))

  ms = np.array(masks)
  ms = ms.reshape(no, dim)
  result = result.reshape(no, dim)

  return result, ms


def keep_region(matrices, percentage):
    results = []
    for matrix in matrices:
      '''
      # Randomly choose a starting point for the connected region
      i, j = np.random.randint(matrix.shape[0]), np.random.randint(matrix.shape[1])

      # Calculate radius of region to keep
      area = matrix.shape[0] * matrix.shape[1] * percentage
      radius = np.sqrt(area / np.pi)
      
      # Create a mask with NaN values where the region is removed
      mask = np.ones(matrix.shape)
      for x in range(matrix.shape[0]):
          for y in range(matrix.shape[1]):
              if np.sqrt((x - i)**2 + (y - j)**2) < radius:
                  mask[x, y] = np.nan

      # Replace the non-removed region with NaN values in the current matrix
      result = np.where(np.isnan(mask), matrix, np.nan)
      results.append(result)'''
      # Randomly choose a starting point for the connected region
      i, j = np.random.randint(matrix.shape[0]), np.random.randint(matrix.shape[1])

      # Calculate radius of region to keep
      area = matrix.shape[0] * matrix.shape[1] * percentage
      radius = np.sqrt(area / np.pi)

      # Set NaN values in the current matrix
      mask = np.sqrt((np.arange(matrix.shape[0])[:, np.newaxis] - i) ** 2 + (np.arange(matrix.shape[1]) - j) ** 2) < radius
      matrix[mask] = np.nan
      results.append(matrix)

    return np.array(results)

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
