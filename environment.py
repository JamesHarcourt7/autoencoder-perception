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


def accuracy(truth, prediction):
    mse_score = np.mean(np.power(truth-prediction, 2))
    psnr_score = 20 * np.log10(np.max(truth) / np.sqrt(mse_score))
    ssim_score = ssim(truth, prediction, data_range=truth.max() - truth.min())

    return mse_score, psnr_score, ssim_score


def main(steps, visualise, model='none', log_dir=None):
    # Create the environment
    (data_x, _), _ = mnist.load_data()
    data_x = np.reshape(np.asarray(data_x), [60000, 784]).astype(float)
    norm_data = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)
    data_index = 5016
    env = norm_data_x[data_index]
    print(np.max(env), np.min(env))

    if model == "none":
        encoder = None
        decoder = None
    elif model in ["dims-old", "mask-old", "no-mask"]:
        path = os.path.join(os.getcwd(), "models", model, "encoder.h5")
        encoder = keras.models.load_model(path)
        decoder = None
    else:
        path = os.path.join(os.getcwd(), "models2", model, "encoder.h5")
        encoder = keras.models.load_model(path)
        path = os.path.join(os.getcwd(), "models2", model, "decoder.h5")
        decoder = keras.models.load_model(path)

    # Create agent
    agent = Agent((14, 14), (28, 28))

    if visualise:
        pygame.init()
        font = pygame.font.SysFont('Comic Sans MS', 24)
        screen = pygame.display.set_mode((1120, 310))
        pygame.display.set_caption("DL Augmented Single-Agent Exploration")
        clock = pygame.time.Clock()

        # Run the agent in the environment
        for step in range(steps):
            pygame.display.set_caption("DL Augmented Single-Agent Exploration t={}".format(step))

            pos = agent.get_position()

            env_square = env.reshape(28, 28)
            measurements = env_square[pos[0]-1:pos[0]+2, pos[1]-1:pos[1]+2]

            agent.measure(measurements)
            x_mb = agent.get_map().reshape(1, 28, 28, 1)
            x_mb = np.nan_to_num(x_mb, 0)
            m_mb = agent.get_mask().reshape(1, 28, 28, 1)

            # Get the prediction
            if encoder is None:
                prediction = x_mb
            elif decoder is None:
                if model == "no-mask":
                    prediction = np.array(encoder([x_mb]))
                else:
                    prediction = np.array(encoder([x_mb, m_mb]))
            else:
                encoded = encoder([x_mb, m_mb])
                prediction = np.array(decoder(encoded))
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

            clock.tick(8)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            screen.fill((255, 255, 255))
            x_square = x_mb.reshape(28, 28)
            text_surface = font.render('Ground Truth', False, (0, 0, 0))
            screen.blit(text_surface, (0, 0))
            for i in range(len(env_square)):
                for j in range(len(env_square[0])):
                    colour = env_square[i][j]*255
                    pygame.draw.rect(screen, (colour, colour, colour), (i*10, j*10 + 30, 10, 10))
            text_surface = font.render('Observation', False, (0, 0, 0))
            screen.blit(text_surface, (280, 0))
            for i in range(len(x_square)):
                for j in range(len(x_square[0])):
                    colour = x_square[i][j]*255
                    pygame.draw.rect(screen, (colour, colour, colour), (i*10+280, j*10 + 30, 10, 10))
            m_square = m_mb.reshape(28, 28)
            text_surface = font.render('Mask', False, (0, 0, 0))
            screen.blit(text_surface, (560, 0))
            for i in range(len(m_square)):
                for j in range(len(m_square[0])):
                    colour = m_square[i][j]*255
                    pygame.draw.rect(screen, (colour, colour, colour), (i*10+560, j*10 + 30, 10, 10))
            p_square = prediction.reshape(28, 28)
            text_surface = font.render('Prediction', False, (0, 0, 0))
            screen.blit(text_surface, (840, 0))
            for i in range(len(p_square)):
                for j in range(len(p_square[0])):
                    colour = p_square[i][j]*255
                    modifier = (m_square[i][j] + 1)/2
                    pygame.draw.rect(screen, (modifier * colour, colour, modifier * colour), (i*10+840, j*10 + 30, 10, 10))
            
            # Draw the agent
            pygame.draw.rect(screen, (255, 0, 0), (pos[1]*10+280, pos[0]*10+30, 10, 10))
            # add eyes to agent
            pygame.draw.circle(screen, (255, 255, 255), (pos[1]*10+280+7, pos[0]*10+32), 2)
            pygame.draw.circle(screen, (255, 255, 255), (pos[1]*10+280+2, pos[0]*10+32), 2)
            pygame.draw.circle(screen, (0, 0, 0), (pos[1]*10+280+7, pos[0]*10+32), 1)
            pygame.draw.circle(screen, (0, 0, 0), (pos[1]*10+280+2, pos[0]*10+32), 1)
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
            if step % 100 == 0:
                print("Step: {}".format(step))

            pos = agent.get_position()

            env_square = env.reshape(28, 28)
            measurements = env_square[pos[0]-1:pos[0]+2, pos[1]-1:pos[1]+2]

            agent.measure(measurements)
            x_mb = agent.get_map().reshape(1, 28, 28, 1)
            x_mb = np.nan_to_num(x_mb, 0)
            m_mb = agent.get_mask().reshape(1, 28, 28, 1)

            # Get the prediction
            if encoder is None:
                prediction = x_mb
            elif decoder is None:
                if model == "no-mask":
                    prediction = np.array(encoder([x_mb]))
                else:
                    prediction = np.array(encoder([x_mb, m_mb]))
            else:
                encoded = encoder([x_mb, m_mb])
                prediction = np.array(decoder(encoded))
            prediction = m_mb * x_mb + (1-m_mb) * prediction

            mse_score, psnr_score, ssim_score = accuracy(env.reshape(784), prediction.reshape(784))
            mse_scores.append(mse_score)
            psnr_scores.append(psnr_score)
            ssim_scores.append(ssim_score)
            percentage_explored = np.sum(m_mb.reshape(784)) / 784
            percentages_explored.append(percentage_explored)

        print("Prediction Accuracy: {} {} {}".format(mse_score, psnr_score, ssim_score))

        # Save to csv
        with open(log_dir + "/accuracies.csv", "a") as f:
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
    # example usage for visualisation:
    # python maenvironment.py visualise mask
    # otherwise:
    # python maenvironment.py no-visualise mask <output_dir>
    visualise = False
    output_dir = None
    model = None
    if len(sys.argv) > 1:
        visualise = sys.argv[1] == "visualise"
    if len(sys.argv) > 2:
        model = sys.argv[2]
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]

    for _ in range(20):
        main(1000, visualise, model, output_dir)
