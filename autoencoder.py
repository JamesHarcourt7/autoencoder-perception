import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import MaxPooling2D, Conv2DTranspose, Conv2D, Concatenate
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from utils import simulate_agent_on_samples

print('GPU: ', tf.config.experimental.list_physical_devices('GPU'))

# MNIST digits classification dataset
# This is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000
# Find more at http://yann.lecun.com/exdb/mnist/
# path: path where to cache the dataset locally (relative to ~/.keras/datasets).
# Loads the MNIST dataset.

def load_data():
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
  x_train = (x_train.astype(np.float32))/255.0
  
  x_train = x_train.reshape(60000, 28, 28, 1)
  return (x_train, y_train, x_test, y_test)


from inspect import modulesbyfile
# Feed generator noise samples and will produce digit
# Question for later is how to get this to imputate missing data rather than produce instances...
def create_encoder():

    X = Input(shape=(28, 28, 1))
    M = Input(shape=(28, 28, 1))

    m = MaxPooling2D(pool_size=(4, 4))(M)
    m = Flatten()(m)

    x = Conv2D(8, kernel_size=4, padding='same', activation='relu')(X)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Conv2D(16, kernel_size=4, padding='same', activation='relu')(x)
    x = Conv2D(32, kernel_size=4, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)

    c = Concatenate()([x, m])

    c = Dense(64, activation='relu')(c)
    c = Dense(40, activation='relu')(c) # Compress to 40 features, to learn the important features -> increase if you want to learn more features (more sharpness)
    c = Dense(49, activation='relu')(c)
    c = Reshape((7, 7, 1))(c)
    c = Conv2DTranspose(32, kernel_size=4, padding='same', activation='relu')(c)
    c = UpSampling2D(size=(2, 2))(c)
    c = Conv2DTranspose(16, kernel_size=4, padding='same', activation='relu')(c)
    #x = Conv2DTranspose(8, kernel_size=4, padding='same', activation='relu')(x)
    c = UpSampling2D(size=(2, 2))(c)
    c = Conv2DTranspose(1, kernel_size=4, padding='same', activation='sigmoid')(c)

    model = Model(inputs=[X, M], outputs=c)
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    
    return model


def generate_real_samples(dataset, n_samples):
  # choose random instances
  ix = np.random.randint(0, dataset.shape[0], n_samples)
  # retrieve selected images
  X = dataset[ix]
  # generate 'real' class labels (1)
  y = np.ones((n_samples, 1))
  return X, y


def generate_missing_samples(dataset, n_samples):
  # choose random instances
  ix = np.random.randint(0, dataset.shape[0], n_samples)
  # retrieve selected images
  X = dataset[ix]
  X, M = simulate_agent_on_samples(X)
  return X, M, dataset[ix]


def summarize_performance(epoch, encoder, dataset, n_samples=100):
   # generate points in latent space
  x_input, masks, complete = generate_missing_samples(dataset, n_samples)
  # predict outputs
  x = np.nan_to_num(x_input, 0)
  X = encoder.predict([x, masks])

  plt.figure(figsize=(30,10))
  # plot images
  for i in range(100):
    # define subplot
    plt.subplot(10, 30, 3 * i + 1)
    # turn off axis
    plt.axis('off')
    # plot raw pixel data
    plt.imshow(X[i])

    plt.subplot(10, 30, 3 * i + 2)
    # turn off axis
    plt.axis('off')
    # plot raw pixel data
    plt.imshow(x_input[i])

    plt.subplot(10, 30, 3 * i + 3)
    # turn off axis
    plt.axis('off')
    # plot raw pixel data
    plt.imshow(masks[i])

  # save plot to file
  filename = 'generated_plot_e%03d.png' % (epoch+1)
  plt.savefig(filename)
  plt.close()

def plot_history(loss):
 plt.plot(loss, label='loss')
 plt.legend()
 filename = 'plot_line_plot_loss.png'
 plt.savefig(filename)
 plt.close()
 print('Saved %s' % (filename))


def train():
  # Load data
  (X, _, _, _) = load_data()
  print(np.min(X), np.max(X))

  # Create ecoder
  encoder = create_encoder()

  loss1 = list()

  for e in range(200):
    print("Epoch: ", e)
    missing, masks, complete = generate_missing_samples(X, 2048)
    x = np.nan_to_num(missing, 0)

    loss = encoder.fit([x, masks], complete, epochs=1, verbose=1)
    loss1.append(loss.history['loss'][0])
    
    if e % 10 == 0:
      print('Summarizing performance')
      summarize_performance(e, encoder, X)
      plot_history(loss1)
  
  encoder.save('models/encoder.h5')

(X, _, _, _) = load_data()
missing, masks, complete = generate_missing_samples(X, 2048)
x = np.nan_to_num(missing, 0)

plt.figure(figsize=(60, 20))
# plot images
for i in range(100):
  # define subplot
  plt.subplot(10, 30, 3 * i + 3)
  # turn off axis
  plt.axis('off')
  # plot raw pixel data
  plt.imshow(missing[i])

  plt.subplot(10, 30, 3 * i + 2)
  # turn off axis
  plt.axis('off')
  # plot raw pixel data
  plt.imshow(masks[i])

  plt.subplot(10, 30, 3 * i + 1)
  # turn off axis
  plt.axis('off')
  # plot raw pixel data
  plt.imshow(complete[i])

# save plot to file
filename = 'zzz.png'
plt.savefig(filename)


