import numpy as np
import matplotlib.pyplot as plt
import pygame
import sys
import os
import datetime
from utils import normalization
from keras.datasets import mnist
import keras
import csv
from skimage.metrics import structural_similarity as ssim

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
    mse_score = np.mean(np.power(truth-prediction, 2))
    psnr_score = 20 * np.log10(np.max(truth) / np.sqrt(mse_score))
    ssim_score = ssim(truth, prediction, data_range=truth.max() - truth.min())

    return mse_score, psnr_score, ssim_score


def main(steps, visualise, model='none', log_dir=None):
    # Create the environment
    (data_x, _), _ = mnist.load_data()
    data_x = np.reshape(np.asarray(data_x), [60000, 784]).astype(float)
    norm_data, _ = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)
    data_index = 0
    env = norm_data_x[data_index]
    print(np.max(env), np.min(env))

    if model == "none":
        generator = None
    else:
        path = os.path.join(os.getcwd(), "models", model, "encoder.h5")
        generator = keras.models.load_model(path)

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
            if generator is None:
                prediction = x_mb
            else:
                prediction = np.array(generator([x_mb, m_mb]))
            prediction = m_mb * x_mb + (1-m_mb) * prediction
            
            mse_score, psnr_score, ssim_score = accuracy(env.reshape(784), prediction.reshape(784))
            
            print("\nStep: ", step)
            print("MSE: ", mse_score)
            print("PSNR: ", psnr_score)
            print("SSIM: ", ssim_score)
            
            env_square = env_square.T
            x_mb = x_mb.reshape(28, 28).T
            m_mb = m_mb.reshape(28, 28).T
            prediction = prediction.reshape(28, 28).T

            clock.tick(60)
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
        mse_scores = []
        psnr_scores = []
        ssim_scores = []
        percentages_explored = []

        # Create new log directory
        if log_dir is None:
            log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
            os.makedirs(log_dir)

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
            if generator is None:
                prediction = x_mb 
            else:
                if model is None or model == 'no-mask':
                    prediction = np.array(generator([x_mb]))
                else:
                    prediction = np.array(generator([x_mb, m_mb]))
            prediction = m_mb * x_mb + (1-m_mb) * prediction

            mse_score, psnr_score, ssim_score = accuracy(env.reshape(784), prediction.reshape(784))
            mse_scores.append(mse_score)
            psnr_scores.append(psnr_score)
            ssim_scores.append(ssim_score)
            percentage_explored = np.sum(m_mb.reshape(784)) / 784
            percentages_explored.append(percentage_explored)

        print("Prediction Accuracy: {} {} {}".format(mse_score, psnr_score, ssim_score))
        generate_image(env, m_mb, prediction, log_dir)

        # Save to csv
        with open(log_dir + "/accuracies.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows([["Data Index", data_index],
                              ["Network", model],
                              ["MSE"] + mse_scores,
                              ["PSNR"] + psnr_scores,
                              ["SSIM"] + ssim_scores,
                              ["Percentage Explored"] + percentages_explored,
                              ])

        # Save the accuracies and percentages explored on same figure
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(mse_scores)
        plt.xlabel("Steps")
        plt.ylabel("Accuracy")
        plt.title("Accuracy")

        plt.subplot(1, 2, 2)
        plt.plot(percentages_explored, color="green")
        plt.xlabel("Steps")
        plt.ylabel("Percentage Explored")
        plt.title("Percentage Explored")
        plt.ylim(0, 1)

        plt.tight_layout()
        plt.savefig(log_dir + "/accuracies.png")
        plt.close('all')

if __name__ == "__main__":
    visualise = False
    output_dir = None
    model = None
    if len(sys.argv) > 1:
        visualise = sys.argv[1] == "visualise"
    if len(sys.argv) > 2:
        model = sys.argv[2]
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    main(1000, visualise, model, output_dir)
