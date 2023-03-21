import numpy as np
import matplotlib.pyplot as plt
import pygame
import sys
import os
import datetime
import tensorflow
from utils import normalization
from keras.datasets import mnist
import keras

from agent import Agent


def generate_image(truth, measurement, prediction, logdir, dim=(1,3), figsize=(10,10)):
    plt.figure(figsize=figsize)
    plt.subplot(dim[0], dim[1], 1)
    plt.imshow(truth.reshape(28, 28), interpolation='nearest')
    plt.subplot(dim[0], dim[1], 2)
    plt.imshow(measurement.reshape(28, 28), interpolation='nearest')
    plt.subplot(dim[0], dim[1], 3)
    plt.imshow(prediction.reshape(28, 28), interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(logdir + '/prediction.png')
    plt.close('all')


def accuracy(truth, prediction):
    return -np.sum(np.power(truth-prediction, 2))#np.mean(truth == prediction)


def main(steps, visualise, model_path='removed'):
    # Create the environment
    (data_x, _), _ = mnist.load_data()
    data_x = np.reshape(np.asarray(data_x), [60000, 28*28]).astype(float)
    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)
    env = norm_data_x[0]
    print(np.max(env), np.min(env))

    generator = keras.models.load_model("models/no-mask/encoder.h5")

    # Create agent
    agent = Agent((14, 14), (28, 28))

    if visualise:
        pygame.init()
        screen = pygame.display.set_mode((1120, 280))
        pygame.display.set_caption("MNIST")
        clock = pygame.time.Clock()

        # Run the agent in the environment
        for step in range(steps):
            pos = agent.get_position()

            env_square = env.reshape(28, 28)
            measurements = env_square[pos[0]-1:pos[0]+2, pos[1]-1:pos[1]+2]

            agent.measure(measurements)
            x_mb = agent.get_map().reshape(1, 28, 28, 1)
            x_mb = np.nan_to_num(x_mb, 0)
            m_mb = agent.get_mask().reshape(1, 28, 28, 1)

            # Get the prediction
            prediction = np.array(generator([x_mb]))
            prediction = m_mb * x_mb + (1-m_mb) * prediction
            acc = accuracy(env, prediction)
            print("Prediction Accuracy: ", acc)

            
            env_square = env_square.T
            x_mb = x_mb.reshape(28, 28).T
            m_mb = m_mb.reshape(28, 28).T
            prediction = prediction.reshape(28, 28).T

            clock.tick(60)
            print("Step: ", step)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            screen.fill((0, 0, 0))
            x_square = x_mb.reshape(28, 28)
            for i in range(len(env_square)):
                    for j in range(len(env_square[0])):
                        for x in range(10):
                            for y in range(10):
                                if (np.isnan(env_square[i][j])):
                                    screen.set_at((i*10+x, j*10+y), (191,79,92))
                                else:
                                    screen.set_at((i*10+x, j*10+y), (env_square[i][j]*255, env_square[i][j]*255, env_square[i][j]*255))
            for i in range(len(x_square)):
                for j in range(len(x_square[0])):
                    for x in range(10):
                        for y in range(10):
                            if (np.isnan(x_square[i][j])):
                                screen.set_at((i*10+x+280, j*10+y), (191,79,92))
                            else:
                                screen.set_at((i*10+x+280, j*10+y), (x_square[i][j]*255, x_square[i][j]*255, x_square[i][j]*255))
            m_square = m_mb.reshape(28, 28)
            for i in range(len(m_square)):
                for j in range(len(m_square[0])):
                    for x in range(10):
                        for y in range(10):
                            screen.set_at((i*10+x+560, j*10+y), (m_square[i][j]*255, m_square[i][j]*255, m_square[i][j]*255))
            p_square = prediction.reshape(28, 28)
            for i in range(len(p_square)):
                for j in range(len(p_square[0])):
                    for x in range(10):
                        for y in range(10):
                            if (np.isnan(p_square[i][j])):
                                screen.set_at((i*10+x+840, j*10+y), (191,79,92))
                            else:
                                screen.set_at((i*10+x+840, j*10+y), (p_square[i][j]*255, p_square[i][j]*255, p_square[i][j]*255))
            
            for x in range(10):
                for y in range(10):
                    screen.set_at((pos[1]*10+x+280, pos[0]*10+y), (255, 0, 0))
            # add eyes to agent
            pygame.draw.circle(screen, (255, 255, 255), (pos[1]*10+280+7, pos[0]*10+2), 2)
            pygame.draw.circle(screen, (255, 255, 255), (pos[1]*10+280+2, pos[0]*10+2), 2)
            pygame.draw.circle(screen, (0, 0, 0), (pos[1]*10+280+7, pos[0]*10+2), 1)
            pygame.draw.circle(screen, (0, 0, 0), (pos[1]*10+280+2, pos[0]*10+2), 1)
            pygame.display.update()
    else:
        # Run the agent in the environment
        accuracies = []
        percentages_explored = []

        # Create new log directory
        log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        os.makedirs(log_dir)

        # Create summary writer
        acc, exp = list(), list()

        for step in range(steps):
            pos = agent.get_position()

            env_square = env.reshape(28, 28)
            measurements = env_square[pos[0]-1:pos[0]+2, pos[1]-1:pos[1]+2]

            agent.measure(measurements)
            raw_x_mb = agent.get_map()
            x_mb = np.nan_to_num(np.copy(raw_x_mb), 0)
            m_mb = agent.get_mask()

            # Get the prediction
            prediction = generator.predict(x_mb)
            prediction = m_mb * x_mb + (1-m_mb) * prediction
            acc = accuracy(env, prediction)
            #print("Prediction Accuracy: ", acc)

            accuracies.append(acc)
            percentage_explored = np.count_nonzero(~np.isnan(raw_x_mb)) / 784
            percentages_explored.append(percentage_explored)

        print("Prediction Accuracy: ", accuracy(env, prediction))
        generate_image(env, m_mb, prediction, log_dir)

        # Save the accuracies and percentages explored on same figure
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(accuracies)
        plt.legend()
        plt.xlabel("Steps")
        plt.ylabel("Accuracy")
        plt.title("Accuracy")
        plt.ylim(np.min(accuracies), 0)

        plt.subplot(1, 2, 2)
        plt.plot(percentages_explored, color="green")
        plt.legend()
        plt.xlabel("Steps")
        plt.ylabel("Percentage Explored")
        plt.title("Percentage Explored")
        plt.ylim(0, 1)

        plt.tight_layout()
        plt.savefig(log_dir + "accuracies.png")
        plt.close('all')

if __name__ == "__main__":
    visualise = False
    if len(sys.argv) > 1:
        visualise = sys.argv[1] == "visualise"
    main(1000, visualise)
