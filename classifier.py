import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import MaxPooling2D, Conv2DTranspose, Conv2D, Concatenate
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import to_categorical

print('GPU: ', tf.config.experimental.list_physical_devices('GPU'))


# Loads the MNIST dataset.
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    x_train = (x_train.astype(np.float32))

    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)) / 255.0
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)) / 255.0
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test


def create_classifier():
    X = Input(shape=(28, 28, 1))

    x = Conv2D(32, kernel_size=3, padding='same', activation='relu')(X)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)

    x = Dense(128, activation='relu')(x)
    
    x = Dense(10, activation='softmax')(x)

    model = Model(inputs=[X], outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01, momentum=0.9), metrics=['accuracy'])

    return model


def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = np.ones((n_samples, 1))
    return X, y


def train():
    # Load data
    (X_train, X_labels, Y_test, Y_labels) = load_data()

    # Create classifier
    classifier = create_classifier()
    loss1 = list()

    # Train encoder/decoder pair on normal MNIST data to learn latent space
    for e in range(1001):
        print("Epoch: ", e)
        ix = np.random.randint(0, X_train.shape[0], 32)
        x = X_train[ix]
        y = X_labels[ix]

        loss = classifier.fit(x, y)
        loss1.append(loss.history['loss'])
    
        if e % 100 == 0:
            _, acc = classifier.evaluate(Y_test, Y_labels, verbose=0)
            print('Accuracy: %.3f' % (acc * 100.0))
    
    plt.figure()
    plt.plot(loss1, label='train')
    plt.legend()
    plt.savefig('loss.png')

    classifier.save('models2/classifier.h5')

train()
