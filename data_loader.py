# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Data loader for UCI letter, spam and MNIST datasets.
'''

# Necessary packages
import numpy as np
from utils import binary_sampler, keep_region
from keras.datasets import mnist


def data_loader(data_name, miss_rate=0.2):
  '''Loads datasets and introduce missingness.
  
  Args:
    - data_name: letter, spam, or mnist
    - miss_rate: the probability of missing components
    
  Returns:
    data_x: original data
    miss_data_x: data with missing values
    data_m: indicator matrix for missing components
  '''
  
  # Load data
  (data_x, _), _ = mnist.load_data()
  data_x = np.concatenate((data_x, data_x, data_x, data_x, data_x), axis=0)
  data_x = np.reshape(np.asarray(data_x), [data_x.shape[0], 784]).astype(float)

  # Shuffle data
  np.random.seed(1234)
  idx = np.random.permutation(data_x.shape[0])
  data_x = data_x[idx, :]

  # Parameters
  no, dim = data_x.shape
  
  # Introduce missing data
  square_x = np.reshape(np.copy(data_x), [no, 28, 28])

  if data_name == 'patch':
    # Patch
    miss_data_x = keep_region(square_x, 0.2)
    miss_data_x = np.reshape(miss_data_x, [no, dim])
    data_m = 1 - np.isnan(miss_data_x)
  elif data_name == 'adaptive_patch':
    # Adaptive patch
    pass
  elif data_name == 'strip':
    # Strip
    pass
  elif data_name == 'adaptive_strip':
    # Adaptive strip
    pass
  else:
    # Removed
    data_m = binary_sampler(1-miss_rate, no, dim)
    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0] = np.nan

  return data_x, miss_data_x, data_m
