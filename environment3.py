# Dynamic environment

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

from dynamicagent import DynamicAgent as Agent


'''
Multiple agents
- Environment will change after 100 steps
- Agents will have to learn to adapt to the new environment
- Agents will need confidence map instead of mask
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


def main(steps, visualise, n_agents=2, model="none", log_dir=None, idx1=1, idx2=6, decay=0.01, beta=0.2, starting_confidence=0.16):
    # Create the environment
    (data_x, _), _ = mnist.load_data()
    data_x = np.reshape(np.asarray(data_x), [60000, 784]).astype(float)
    norm_data, _ = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)
    data_index = idx1
    env = norm_data_x[data_index]
    env_square = env.reshape(28, 28)
    tenv_square = env_square.T

    print("MODEL: ", model, "")
    if model == "none":
        encoder = None
        decoder = None
    else:
        path = os.path.join(os.getcwd(), "models2", model, "encoder.h5")
        encoder = keras.models.load_model(path)
        path = os.path.join(os.getcwd(), "models2", model, "decoder.h5")
        decoder = keras.models.load_model(path)

    # Get classifier
    path = os.path.join(os.getcwd(), "models2", "classifier.h5")
    classifier = keras.models.load_model(path)
    decisions = dict()
    time_decisions = list()

    # Create agents in random positions
    size = 28
    agents = [Agent((np.random.randint(1, size - 1), np.random.randint(1, size - 1)), (size, size), decay, beta, starting_confidence) for _ in range(n_agents)]

    if visualise:
        pygame.init()
        screen = pygame.display.set_mode((1120, 840))
        pygame.display.set_caption("DL Augmented Multi-Agent Exploration")
        clock = pygame.time.Clock()

        # Run the agents in the environment
        for step in range(steps):
            if step == 500:
                data_index = idx2
                env = norm_data_x[data_index]
                env_square = env.reshape(28, 28)
                tenv_square = env_square.T

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
            
            # observe and send information
            for i in range(n_agents):
                agent = agents[i]
                pos = agent.get_position()
                agent.update_confidence()
                
                measurements = env_square[pos[0]-1:pos[0]+2, pos[1]-1:pos[1]+2]

                agent.measure(measurements)
                x_mb = agent.get_map().reshape(1, 28, 28, 1)
                x_mb = np.nan_to_num(x_mb, 0)
                m_mb = agent.get_mask().reshape(1, 28, 28, 1)

                # Share obvervations with other agents
                if encoder is None:
                    vector = x_mb.reshape(784)
                    agent.set_vector(vector)
                else:
                    vector = np.array(encoder([x_mb, m_mb]))
                    agent.set_vector(vector)

                vector_to_send = agent.get_average_vector()
                for j in range(n_agents):
                    if i != j:
                        distance = np.linalg.norm(np.array(pos) - np.array(agents[j].get_position()))
                        if distance <= communication_radius:
                            agents[j].receive_vector(vector_to_send, i)
                            communications.append((i, j))
            
            # receive information
            for i in range(n_agents):
                agent = agents[i]
                # Check recieved vectors from other agents
                count = 0
                while agent.has_messages():
                    count += 1
                    vec, j = agent.get_message()
                    # Average with agent's own vector
                    agent.add_vector(vec, j)

            # predict
            for i in range(n_agents):
                agent = agents[i]
                x_mb = agent.get_map().reshape(1, 28, 28, 1)
                x_mb = np.nan_to_num(x_mb, 0)
                m_mb = agent.get_mask().reshape(1, 28, 28, 1)

                # Get the prediction
                if encoder is None:
                    prediction = agent.get_average_vector().reshape((1, 28, 28, 1))
                else:
                    avg_vector = agent.get_average_vector()
                    prediction = np.array(decoder([avg_vector])) # avg vector is a combination of the agent's own vector and the vectors of other agents
                prediction = m_mb * x_mb + (1-m_mb) * prediction

                # Update agent's decision on classification
                decision = np.argmax(classifier(prediction))
                decisions[i] = decision

                # Update global prediction
                if average_prediction is None:
                    average_prediction = prediction
                else:
                    average_prediction = average_prediction + prediction

                # Update global mask
                square_mask = m_mb.reshape(28, 28)
                global_mask = np.where(square_mask == 1, 1, global_mask)

                if i < 4:
                    x_offset = 280 * i
                    agent_colour = agent_colours[i % len(agent_colours)]

                    p_square = prediction.reshape(28, 28).T
                    for i in range(len(p_square)):
                        for j in range(len(p_square[0])):
                            colour = p_square[i][j]*255
                            pygame.draw.rect(screen, (colour, colour, colour), (i*10+x_offset, j*10+280, 10, 10))
                    
                    confidence = agent.get_confidence()
                    c_square = confidence.reshape(28, 28).T
                    for i in range(len(c_square)):
                        for j in range(len(c_square[0])):
                            colour = c_square[i][j]*255
                            pygame.draw.rect(screen, (colour, colour, colour), (i*10+x_offset, j*10+560, 10, 10))
                    # Draw the agent
                    pygame.draw.rect(screen, agent_colour, (x_offset, 280, 10, 10))
                    # add eyes to agent
                    pygame.draw.circle(screen, (255, 255, 255), (x_offset+7, 282), 2)
                    pygame.draw.circle(screen, (255, 255, 255), (x_offset+2, 282), 2)
                    pygame.draw.circle(screen, (0, 0, 0), (x_offset+7, 282), 1)
                    pygame.draw.circle(screen, (0, 0, 0), (x_offset+2, 282), 1)
            
            time_decisions.append(decisions.copy())

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
                colour = agent_colours[i % len(agent_colours)]

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
            if step == 500:
                data_index = idx2
                env = norm_data_x[data_index]
                env_square = env.reshape(28, 28)
                tenv_square = env_square.T
                
            global_mask = np.zeros((28, 28))
            average_prediction = None
            communications = list()
            communication_overhead = 0
            
            # observe and send information
            for i in range(n_agents):
                agent = agents[i]
                pos = agent.get_position()
                agent.update_confidence()
                
                measurements = env_square[pos[0]-1:pos[0]+2, pos[1]-1:pos[1]+2]

                agent.measure(measurements)
                x_mb = agent.get_map().reshape(1, 28, 28, 1)
                x_mb = np.nan_to_num(x_mb, 0)
                m_mb = agent.get_mask().reshape(1, 28, 28, 1)

                # Share obvervations with other agents
                if encoder is None:
                    vector = x_mb.reshape(784)
                    agent.set_vector(vector)
                else:
                    vector = np.array(encoder([x_mb, m_mb]))
                    agent.set_vector(vector)

                vector_to_send = agent.get_average_vector()
                for j in range(n_agents):
                    if i != j:
                        distance = np.linalg.norm(np.array(pos) - np.array(agents[j].get_position()))
                        if distance <= communication_radius:
                            agents[j].receive_vector(vector_to_send, i)
                            communications.append((i, j))
                            communication_overhead += vector_to_send.size + 1

            # receive information
            for i in range(n_agents):
                agent = agents[i]
                # Check recieved vectors from other agents
                count = 0
                while agent.has_messages():
                    count += 1
                    vec, j = agent.get_message()
                    # Average with agent's own vector
                    agent.add_vector(vec, j)
            
             # predict
            for i in range(n_agents):
                agent = agents[i]
                x_mb = agent.get_map().reshape(1, 28, 28, 1)
                x_mb = np.nan_to_num(x_mb, 0)
                m_mb = agent.get_mask().reshape(1, 28, 28, 1)

                # Get the prediction
                if encoder is None:
                    prediction = agent.get_average_vector().reshape((1, 28, 28, 1))
                else:
                    avg_vector = agent.get_average_vector()
                    prediction = np.array(decoder([avg_vector])) # avg vector is a combination of the agent's own vector and the vectors of other agents
                prediction = m_mb * x_mb + (1-m_mb) * prediction

                # Update agent's decision on classification
                decision = np.argmax(classifier(prediction))
                decisions[i] = decision

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
            
            time_decisions.append(decisions.copy())

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
        '''
        with open(log_dir + "/accuracies.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows([["Data Indexes", idx1, idx2],
                              ["Decay", decay],
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
        '''
        with open(log_dir + "/accuracies.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows([["Data Indexes", idx1, idx2],
                              ["Decay", decay],
                              ["Beta", beta],
                              ["MSE"] + average_mse]
                                + [["Agent {}".format(i), "MSE"] + mse_scores[i] for i in range(n_agents)])
            
        with open(log_dir + "/decisions.csv", "w") as f:
            data = [[decisions[k] for k in range(0, n_agents)] for decisions in time_decisions]
            writer = csv.writer(f)
            writer.writerows(data)


'''
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
'''
'''
if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    output_dir = "beta_tests"
    idx1 = 1
    idx2 = 6
    model = 'mask'
    for _ in range(10):
        for decay in [0.01, 0.02, 0.05]:
            for starting_confidence in [0.2, 0.4, 0.6, 0.8]:
                for beta in [0.2, 0.4, 0.6, 0.8]:
                    for x in range(10):
                        # make new output directory
                        out = output_dir + "/" + str(starting_confidence).replace('.', '') + '_' + str(beta).replace('.', '') + '_' + str(decay).replace('.', '') + "_" + str(idx1) + "_" + str(idx2) + "_" + str(x)
                        os.makedirs(out)

                        main(1000, False, 4, model, out, idx1, idx2, decay, beta, starting_confidence)
'''
if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    output_dir = "timing_tests"
    idx1 = 1
    idx2 = 6
    model = 'mask'
    for n in range(50, 101, 10):
        if n == 0:
            n = 1
        for x in range(25):
            # make new output directory
            out = output_dir + "/" + str(n) + "_" + str(x)
            os.makedirs(out)

            main(1000, False, n, model, out, idx1, idx2)
'''
if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    label1 = np.random.choice(np.arange(10), 1)
    idx1 = np.random.choice(np.where(y_train == label1)[0], 1)
    idx2 = np.random.choice(np.where(y_train != label1)[0], 1)
    print(idx1, idx2)
    main(1000, True, 4, "none", "logs", idx1, idx2)
'''
