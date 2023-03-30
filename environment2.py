# Multi-agent environment

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


'''

What do we want to show here?
- MSE, PSNR, SSIM over time: show how the agent improves in predictions -> but which agent?
    Do we want to show the average over all agents?
- Show the predictions of all agents over time as well as average. as a GIF?
- Graphical interaction between agents + direction of interaction (who is sending data to who)
- Record size of data sent between agents over time - cumulative frequency distribution

'''

agent_colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)]
communication_radius = 3

def accuracy(truth, prediction):
    mse_score = np.mean(np.power(truth-prediction, 2))
    if mse_score == 0:
        psnr_score = 100
    else:
        psnr_score = 20 * np.log10(np.max(truth) / np.sqrt(mse_score))
    ssim_score = ssim(truth, prediction, data_range=truth.max() - truth.min())

    return mse_score, psnr_score, ssim_score


def main(steps, visualise, n_agents=2, model="none", log_dir=None):
    # Create the environment
    (data_x, _), _ = mnist.load_data()
    data_x = np.reshape(np.asarray(data_x), [60000, 784]).astype(float)
    norm_data, _ = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)
    data_index = 0
    env = norm_data_x[data_index]
    env_square = env.reshape(28, 28)
    tenv_square = env_square.T

    print("MODEL: ", model, "")
    if model == "none":
        generator = None
    else:
        print(os.getcwd())
        path = os.path.join(os.getcwd(), "models", model, "encoder.h5")
        generator = keras.models.load_model(path)

    # Create agents in random positions
    size = 28
    agents = [Agent((np.random.randint(1, size - 1), np.random.randint(1, size - 1)), (size, size)) for _ in range(n_agents)]

    if visualise:
        pygame.init()
        screen = pygame.display.set_mode((1120, 560))
        pygame.display.set_caption("DL Augmented Multi-Agent Exploration")
        clock = pygame.time.Clock()

        # Run the agents in the environment
        for step in range(steps):
            clock.tick(3)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            screen.fill((0, 0, 0))

            for i in range(len(tenv_square)):
                for j in range(len(tenv_square[0])):
                    colour = tenv_square[i][j]*255
                    pygame.draw.rect(screen, (colour, colour, colour), (i*10, j*10, 10, 10))

            global_mask = np.zeros((28, 28))
            average_prediction = None
            communications = list()
            
            for i in range(n_agents):
                agent = agents[i]
                pos = agent.get_position()
                
                measurements = env_square[pos[0]-1:pos[0]+2, pos[1]-1:pos[1]+2]

                agent.measure(measurements)
                x_mb = agent.get_map().reshape(1, 28, 28, 1)
                x_mb = np.nan_to_num(x_mb, 0)
                m_mb = agent.get_mask().reshape(1, 28, 28, 1)

                # Share obvervations with other agents
                for j in range(n_agents):
                    if i != j:
                        distance = np.linalg.norm(np.array(pos) - np.array(agents[j].get_position()))
                        if distance <= communication_radius:
                            agents[j].receive_observation(x_mb, m_mb)
                            communications.append((i, j))

                # Check recieved observations from other agents
                while agent.has_observations():
                    obs = agent.get_observation()
                    # Fill in the gaps
                    agent.fill_in(obs[0], obs[1])

                # Get the prediction
                if generator is None:
                    prediction = x_mb
                else:
                    prediction = np.array(generator([x_mb, m_mb]))
                prediction = m_mb * x_mb + (1-m_mb) * prediction

                # Update global prediction
                if average_prediction is None:
                    average_prediction = prediction
                else:
                    average_prediction = average_prediction + prediction

                # Update global mask
                square_mask = m_mb.reshape(28, 28)
                global_mask = np.where(square_mask == 1, 1, global_mask)

                x_offset = 280 * i
                
                p_square = prediction.reshape(28, 28).T
                for i in range(len(p_square)):
                    for j in range(len(p_square[0])):
                        colour = p_square[i][j]*255
                        pygame.draw.rect(screen, (colour, colour, colour), (i*10+x_offset, j*10+280, 10, 10))
            
            # Update global prediction
            average_prediction /= n_agents

            # Global observation 
            m_square = global_mask.reshape(28, 28).T
            x_square = tenv_square * m_square
            for i in range(len(x_square)):
                for j in range(len(x_square[0])):
                    colour = x_square[i][j]*255
                    pygame.draw.rect(screen, (colour, colour, colour), (i*10+280, j*10, 10, 10))
            
            for i in range(n_agents):
                pos = agents[i].get_position()
                colour = agent_colours[i]

                # Draw the agent
                pygame.draw.rect(screen, colour, (pos[1]*10+280, pos[0]*10, 10, 10))
                # add eyes to agent
                pygame.draw.circle(screen, (255, 255, 255), (pos[1]*10+280+7, pos[0]*10+2), 2)
                pygame.draw.circle(screen, (255, 255, 255), (pos[1]*10+280+2, pos[0]*10+2), 2)
                pygame.draw.circle(screen, (0, 0, 0), (pos[1]*10+280+7, pos[0]*10+2), 1)
                pygame.draw.circle(screen, (0, 0, 0), (pos[1]*10+280+2, pos[0]*10+2), 1)

            # Draw communications
            for i in range(len(communications)):
                a1 = agents[communications[i][0]]
                a2 = agents[communications[i][1]]
                p1 = a1.get_position()
                p2 = a2.get_position()
                pygame.draw.line(screen, (255, 255, 255), (p1[1]*10+280+5, p1[0]*10+5), (p2[1]*10+280+5, p2[0]*10+5), 1)
            
            # Global mask
            for i in range(len(m_square)):
                for j in range(len(m_square[0])):
                    colour = m_square[i][j]*255
                    pygame.draw.rect(screen, (colour, colour, colour), (i*10+560, j*10, 10, 10))
            
            # Global prediction
            p_square = average_prediction.reshape(28, 28).T
            for i in range(len(p_square)):
                for j in range(len(p_square[0])):
                    colour = p_square[i][j]*255
                    pygame.draw.rect(screen, (colour, colour, colour), (i*10+840, j*10, 10, 10))

            pygame.display.update()
    else:
        # Run the agent in the environment
        mse_scores = {i : [] for i in range(n_agents)}
        psnr_scores = {i : [] for i in range(n_agents)}
        ssim_scores = {i : [] for i in range(n_agents)}
        percentages_explored = {i : [] for i in range(n_agents)}
        average_mse = []
        average_psnr = []
        average_ssim = []
        average_percentage_explored = []
        average_communication_overhead = []

        # Create new log directory
        if log_dir is None:
            log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
            os.makedirs(log_dir)

        # Run the agents in the environment
        for step in range(steps):
            global_mask = np.zeros((28, 28))
            average_prediction = None
            communications = list()
            communication_overhead = 0
            
            for i in range(n_agents):
                agent = agents[i]
                pos = agent.get_position()
                
                measurements = env_square[pos[0]-1:pos[0]+2, pos[1]-1:pos[1]+2]

                agent.measure(measurements)
                x_mb = agent.get_map().reshape(1, 28, 28, 1)
                x_mb = np.nan_to_num(x_mb, 0)
                m_mb = agent.get_mask().reshape(1, 28, 28, 1)

                # Share obvervations with other agents
                for j in range(n_agents):
                    if i != j:
                        distance = np.linalg.norm(np.array(pos) - np.array(agents[j].get_position()))
                        if distance <= communication_radius:
                            agents[j].receive_observation(x_mb, m_mb)
                            communications.append((i, j))
                            communication_overhead += x_mb.size + m_mb.size

                # Check recieved observations from other agents
                while agent.has_observations():
                    obs = agent.get_observation()
                    # Fill in the gaps
                    agent.fill_in(obs[0], obs[1])

                # Get the prediction
                if generator is None:
                    prediction = x_mb
                else:
                    prediction = np.array(generator([x_mb, m_mb]))
                prediction = m_mb * x_mb + (1-m_mb) * prediction

                mse_score, psnr_score, ssim_score = accuracy(env.reshape(784), prediction.reshape(784))
                mse_scores[i].append(mse_score)
                psnr_scores[i].append(psnr_score)
                ssim_scores[i].append(ssim_score)
                percentage_explored = np.sum(m_mb.reshape(784)) / 784
                percentages_explored[i].append(percentage_explored)

                # Update global prediction
                if average_prediction is None:
                    average_prediction = prediction
                else:
                    average_prediction = average_prediction + prediction

                # Update global mask
                square_mask = m_mb.reshape(28, 28)
                global_mask = np.where(square_mask == 1, 1, global_mask)
            
            # Update global prediction
            average_prediction = average_prediction / n_agents

            mse_score, psnr_score, ssim_score = accuracy(env.reshape(784), average_prediction.reshape(784))
            average_mse.append(mse_score)
            average_psnr.append(psnr_score)
            average_ssim.append(ssim_score)
            percentage_explored = np.sum(m_mb.reshape(784)) / 784
            average_percentage_explored.append(percentage_explored)

            # Update communication overhead
            average_communication_overhead.append(communication_overhead / n_agents)

        print("Prediction Accuracy: {} {} {}".format(average_mse[-1], average_psnr[-1], average_ssim[-1]))

        # Save to csv
        with open(log_dir + "/accuracies.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows([["Data Index", data_index],
                              ["Network", model],
                              ["MSE"] + average_mse,
                              ["PSNR"] + average_psnr,
                              ["SSIM"] + average_ssim,
                              ["Percentage Explored"] + average_percentage_explored]
                                + [["Agent {}".format(i), "MSE"] + mse_scores[i] for i in range(n_agents)]
                                + [["Agent {}".format(i), "PSNR"] + psnr_scores[i] for i in range(n_agents)]
                                + [["Agent {}".format(i), "SSIM"] + ssim_scores[i] for i in range(n_agents)]
                                + [["Agent {}".format(i), "Percentage Explored"] + percentages_explored[i] for i in range(n_agents)]
                                + [["Overhead"] + average_communication_overhead])


if __name__ == "__main__":
    visualise = False
    output_dir = None
    model = "none"
    n_agents = 2
    if len(sys.argv) > 1:
        visualise = sys.argv[1] == "visualise"
    if len(sys.argv) > 2:
        n_agents = int(sys.argv[2])
    if len(sys.argv) > 3:
        model = sys.argv[3]
    if len(sys.argv) > 4:
        output_dir = sys.argv[4]
    main(1000, visualise, n_agents, model, output_dir)
