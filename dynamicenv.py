# Dynamic environment

import numpy as np
import pygame
import os
import datetime
from utils import normalization
import keras
import csv
from skimage.metrics import structural_similarity as ssim
from load_mnist import load_data as load_mnist

from dynamicagent import DynamicAgent2 as Agent


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


def main(steps, visualise, n_agents=2, idx1=1, idx2=6, digit1=0, digit2=1, decay=0.01, beta=0.8, starting_confidence=0.2):
    # Create the environment
    (data_x, _), _ = load_mnist()
    data_x = np.reshape(np.asarray(data_x), [60000, 784]).astype(float)
    norm_data, _ = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)
    data_index = idx1
    env = norm_data_x[data_index]
    env_square = env.reshape(28, 28)
    tenv_square = env_square.T

    model = "mask"
    path = os.path.join(os.getcwd(), "models2", model, "encoder.h5")
    encoder = keras.models.load_model(path)
    path = os.path.join(os.getcwd(), "models2", model, "decoder.h5")
    decoder = keras.models.load_model(path)

    # Get classifier
    path = os.path.join(os.getcwd(), "models2", "classifier.h5")
    classifier = keras.models.load_model(path)
    decisions = dict()
    time_decisions = list()
    decisions2 = dict()
    time_decisions2 = list()

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
                vector = x_mb.reshape(784)
                agent.set_vector1(vector)

                vector = np.array(encoder([x_mb, m_mb]))
                agent.set_vector2(vector)

                vector_to_send1 = agent.get_average_vector1()
                vector_to_send2 = agent.get_average_vector2()
                for j in range(n_agents):
                    if i != j:
                        distance = np.linalg.norm(np.array(pos) - np.array(agents[j].get_position()))
                        if distance <= communication_radius:
                            agents[j].receive_vector1(vector_to_send1, i)
                            agents[j].receive_vector2(vector_to_send2, i)
                            communications.append((i, j))
            
            # receive information
            for i in range(n_agents):
                agent = agents[i]
                # Check recieved vectors from other agents
                while agent.has_messages1():
                    vec, j = agent.get_message1()
                    # Average with agent's own vector
                    agent.add_vector1(vec, j)
                while agent.has_messages2():
                    vec, j = agent.get_message2()
                    # Average with agent's own vector
                    agent.add_vector2(vec, j)

            # predict
            for i in range(n_agents):
                agent = agents[i]
                x_mb = agent.get_map().reshape(1, 28, 28, 1)
                x_mb = np.nan_to_num(x_mb, 0)
                m_mb = agent.get_mask().reshape(1, 28, 28, 1)

                # Get the prediction
                avg_vector = agent.get_average_vector1()
                prediction = avg_vector.reshape((1, 28, 28, 1))
                
                avg_vector = agent.get_average_vector2()
                prediction2 = np.array(decoder([avg_vector])) # avg vector is a combination of the agent's own vector and the vectors of other agents
                
                prediction = m_mb * x_mb + (1-m_mb) * prediction
                prediction2 = m_mb * x_mb + (1-m_mb) * prediction2

                # Update agent's decision on classification
                output = classifier(prediction)
                decision1 = np.argmax(output)
                if output[0][decision1] < 0.5: # inlcude this in the report
                    decision1 = -1
                decisions[i] = decision1

                output = classifier(prediction2)
                decision2 = np.argmax(output)
                if output[0][decision2] < 0.5: # inlcude this in the report
                    decision2 = -1
                decisions2[i] = decision2

                # Update global prediction
                if average_prediction is None:
                    average_prediction = prediction
                else:
                    average_prediction = average_prediction + prediction

                # Update global mask
                square_mask = m_mb.reshape(28, 28)
                global_mask = np.where(square_mask == 1, 1, global_mask)

                if i < 2:
                    x_offset = 280 * i
                    agent_colour = agent_colours[i % len(agent_colours)]
                    m_mb = agent.get_mask()
                    m_square = m_mb.reshape(28, 28).T

                    p_square = prediction.reshape(28, 28).T
                    for i in range(len(p_square)):
                        for j in range(len(p_square[0])):
                            colour = p_square[i][j]*255
                            pygame.draw.rect(screen, (colour, colour, colour), (i*10+x_offset, j*10+280, 10, 10))

                    # Draw the agent
                    pygame.draw.rect(screen, agent_colour, (x_offset, 280, 10, 10))
                    # add eyes to agent
                    pygame.draw.circle(screen, (255, 255, 255), (x_offset+7, 282), 2)
                    pygame.draw.circle(screen, (255, 255, 255), (x_offset+2, 282), 2)
                    pygame.draw.circle(screen, (0, 0, 0), (x_offset+7, 282), 1)
                    pygame.draw.circle(screen, (0, 0, 0), (x_offset+2, 282), 1)

                    p_square = prediction2.reshape(28, 28).T
                    for i in range(len(p_square)):
                        for j in range(len(p_square[0])):
                            colour = p_square[i][j]*255
                            modifier = (m_square[i][j] + 1) / 2 
                            pygame.draw.rect(screen, (colour * modifier, colour, colour * modifier), (i*10+x_offset + 560, j*10+280, 10, 10))
            
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
        mse_scores2 = {i : [] for i in range(n_agents)}

        # Run the agents in the environment
        for step in range(steps):
            if step == 500:
                data_index = idx2
                env = norm_data_x[data_index]
                env_square = env.reshape(28, 28)
                tenv_square = env_square.T
                
            global_mask = np.zeros((28, 28))
            average_prediction = None
            
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
                vector = x_mb.reshape(784)
                agent.set_vector1(vector)

                vector = np.array(encoder([x_mb, m_mb]))
                agent.set_vector2(vector)

                vector_to_send1 = agent.get_average_vector1()
                vector_to_send2 = agent.get_average_vector2()
                for j in range(n_agents):
                    if i != j:
                        distance = np.linalg.norm(np.array(pos) - np.array(agents[j].get_position()))
                        if distance <= communication_radius:
                            agents[j].receive_vector1(vector_to_send1, i)
                            agents[j].receive_vector2(vector_to_send2, i)
            
            # receive information
            for i in range(n_agents):
                agent = agents[i]
                # Check recieved vectors from other agents
                while agent.has_messages1():
                    vec, j = agent.get_message1()
                    # Average with agent's own vector
                    agent.add_vector1(vec, j)
                while agent.has_messages2():
                    vec, j = agent.get_message2()
                    # Average with agent's own vector
                    agent.add_vector2(vec, j)
            
            # predict
            for i in range(n_agents):
                agent = agents[i]
                x_mb = agent.get_map().reshape(1, 28, 28, 1)
                x_mb = np.nan_to_num(x_mb, 0)
                m_mb = agent.get_mask().reshape(1, 28, 28, 1)

                # Get the prediction
                avg_vector = agent.get_average_vector1()
                prediction = avg_vector.reshape((1, 28, 28, 1))
                
                avg_vector = agent.get_average_vector2()
                prediction2 = np.array(decoder([avg_vector])) # avg vector is a combination of the agent's own vector and the vectors of other agents
                
                prediction = m_mb * x_mb + (1-m_mb) * prediction
                prediction2 = m_mb * x_mb + (1-m_mb) * prediction2

                # Update agent's decision on classification
                output = classifier(prediction)
                decision1 = np.argmax(output)
                if output[0][decision1] < 0.5: # inlcude this in the report
                    decision1 = -1
                decisions[i] = decision1

                output = classifier(prediction2)
                decision2 = np.argmax(output)
                if output[0][decision2] < 0.5: # inlcude this in the report
                    decision2 = -1
                decisions2[i] = decision2

                mse_score, _, _ = accuracy(env.reshape(784), prediction.reshape(784))
                mse_score2, _, _ = accuracy(env.reshape(784), prediction2.reshape(784))
                mse_scores[i].append(mse_score)
                mse_scores2[i].append(mse_score2)
            
            time_decisions.append(decisions.copy())
            time_decisions2.append(decisions2.copy())

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
        with open("scenario3proper/baseline/accuracies.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerows([["Data Indexes", idx1, idx2],
                              ["n", n_agents]]
                                + [["Agent {}".format(i), "MSE"] + mse_scores[i] for i in range(n_agents)])
            
        with open("scenario3proper/mask/accuracies.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerows([["Data Indexes", idx1, idx2],
                              ["n", n_agents]]
                                + [["Agent {}".format(i), "MSE"] + mse_scores2[i] for i in range(n_agents)])
            
        with open("scenario3proper/baseline/decisions.csv", "a") as f:
            data = [[decisions[k] for k in range(0, n_agents)] for decisions in time_decisions]
            writer = csv.writer(f)
            writer.writerow(["n", n_agents, digit1, digit2])
            writer.writerows(data)

        with open("scenario3tuning/mask/decisions.csv", "a") as f:
            data = [[decisions[k] for k in range(0, n_agents)] for decisions in time_decisions2]
            writer = csv.writer(f)
            writer.writerow(["n", n_agents, digit1, digit2])
            writer.writerows(data)

'''
if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_mnist()
    label1 = 0
    label2 = 1
    idx1 = np.random.choice(np.where(y_train == label1)[0], 1)
    idx2 = np.random.choice(np.where(y_train == label2)[0], 1)
    main(1000, True, 40, idx1, idx2, label1, label2, 0.02, 0.8, 0.2)
'''
'''
if __name__ == '__main__':
    # Tuning alpha, beta and theta
    (X_train, y_train), (X_test, y_test) = load_mnist()
    label1 = 0
    label2 = 1
    idx1 = 1
    idx2 = 6
    for alpha in [0.01, 0.02, 0.05]:
        for beta in [0.4, 0.6, 0.8]:
            for theta_max in [0.4, 0.6, 0.8]:
                main(1000, False, 10, idx1, idx2, label1, label2, alpha, beta, theta_max)
'''
if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_mnist()
    label1 = 0
    label2 = 1
    idx1 = 1
    idx2 = 6
    main(1000, False, 40, idx1, idx2, label1, label2, 0.02, 0.8, 0.2)
