import numpy as np
import pygame
import os
from utils import normalization
import keras
import csv
from skimage.metrics import structural_similarity as ssim
from load_mnist import load_data as load_mnist

from agent import Agent2 as Agent


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


def main(steps, visualise, n_agents=2, idx1=1, digit="0"):
    # Create the environment
    (data_x, _), _ = load_mnist()
    data_x = np.reshape(np.asarray(data_x), [60000, 784]).astype(float)
    norm_data = normalization(data_x)
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
    agents = [Agent((np.random.randint(1, size - 1), np.random.randint(1, size - 1)), (size, size)) for _ in range(n_agents)]

    if visualise:
        pygame.init()
        font = pygame.font.SysFont('Comic Sans MS', 24)
        screen = pygame.display.set_mode((1120, 930))
        pygame.display.set_caption("DL Augmented Multi-Agent Exploration")
        clock = pygame.time.Clock()

        # Run the agents in the environment
        for step in range(steps):
            pygame.display.set_caption("DL Augmented Multi-Agent Exploration t={}".format(step))

            clock.tick(3)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            screen.fill((255, 255, 255))

            text_surface = font.render('Ground Truth', False, (0, 0, 0))
            screen.blit(text_surface, (0, 0))
            for i in range(len(tenv_square)):
                for j in range(len(tenv_square[0])):
                    colour = tenv_square[i][j]*255
                    pygame.draw.rect(screen, (colour, colour, colour), (i*10, j*10+30, 10, 10))

            global_mask = np.zeros((28, 28))
            average_prediction = None
            average_prediction2 = None
            communications = list()
            
            # observe and send information
            for i in range(n_agents):
                agent = agents[i]
                pos = agent.get_position()
                
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
                    average_prediction2 = prediction2
                else:
                    average_prediction = average_prediction + prediction
                    average_prediction2 = average_prediction2 + prediction2

                # Update global mask
                square_mask = m_mb.reshape(28, 28)
                global_mask = np.where(square_mask == 1, 1, global_mask)

                if i < 2:
                    x_offset = 280 * i
                    agent_colour = agent_colours[i]
                    m_mb = agent.get_mask()
                    m_square = m_mb.reshape(28, 28).T

                    p_square = prediction.reshape(28, 28).T
                    text_surface = font.render('Baseline Agent {}'.format(i), False, (0, 0, 0))
                    screen.blit(text_surface, (x_offset + 30, 280 + 30))
                    for x in range(len(p_square)):
                        for j in range(len(p_square[0])):
                            colour = p_square[x][j]*255
                            modifier = (m_square[x][j] + 2) / 3
                            pygame.draw.rect(screen, (colour, colour * modifier, colour * modifier), (x*10+x_offset, j*10+280 + 60, 10, 10))

                    text_surface = font.render('Decision: {}'.format(decision1), False, (255, 255, 255))
                    screen.blit(text_surface, (x_offset, 280 + 60))

                    # Draw the agent
                    pygame.draw.rect(screen, agent_colour, (x_offset + 560, 280 + 30, 30, 30))
                    # add eyes to agent
                    pygame.draw.circle(screen, (255, 255, 255), (x_offset+21+560, 286 + 30), 6)
                    pygame.draw.circle(screen, (255, 255, 255), (x_offset+6+560, 286 + 30), 6)
                    pygame.draw.circle(screen, (0, 0, 0), (x_offset+21+560, 286 + 30), 3)
                    pygame.draw.circle(screen, (0, 0, 0), (x_offset+6+560, 286 + 30), 3)

                    text_surface = font.render('Predictive Agent {}'.format(i), False, (0, 0, 0))
                    screen.blit(text_surface, (x_offset + 560 + 30, 280 + 30))
                    p_square = prediction2.reshape(28, 28).T
                    for i in range(len(p_square)):
                        for j in range(len(p_square[0])):
                            colour = p_square[i][j]*255
                            modifier = (m_square[i][j] + 1) / 2 
                            pygame.draw.rect(screen, (colour * modifier, colour, colour * modifier), (i*10+x_offset + 560, j*10+280 + 60, 10, 10))

                    text_surface = font.render('Decision: {}'.format(decision2), False, (255, 255, 255))
                    screen.blit(text_surface, (x_offset + 560, 280 + 60))

                    # Draw the agent
                    pygame.draw.rect(screen, agent_colour, (x_offset, 280 + 30, 30, 30))
                    # add eyes to agent
                    pygame.draw.circle(screen, (255, 255, 255), (x_offset+21, 286 + 30), 6)
                    pygame.draw.circle(screen, (255, 255, 255), (x_offset+6, 286 + 30), 6)
                    pygame.draw.circle(screen, (0, 0, 0), (x_offset+21, 286 + 30), 3)
                    pygame.draw.circle(screen, (0, 0, 0), (x_offset+6, 286 + 30), 3)

                    text_surface = font.render('Observation', False, (0, 0, 0))
                    screen.blit(text_surface, (x_offset + 30, 560 + 60))
                    x_square = x_mb.reshape((28, 28)).T
                    for i in range(len(x_square)):
                        for j in range(len(x_square[0])):
                            colour = x_square[i][j]*255
                            pygame.draw.rect(screen, (colour, colour, colour), (i*10+x_offset, j*10+560 + 90, 10, 10))
                    
                     # Draw the agent
                    pygame.draw.rect(screen, agent_colour, (x_offset, 560 + 60, 30, 30))
                    # add eyes to agent
                    pygame.draw.circle(screen, (255, 255, 255), (x_offset+21, 560 + 6 + 60), 6)
                    pygame.draw.circle(screen, (255, 255, 255), (x_offset+6, 560 + 6 + 60), 6)
                    pygame.draw.circle(screen, (0, 0, 0), (x_offset+21, 560 + 6 + 60), 3)
                    pygame.draw.circle(screen, (0, 0, 0), (x_offset+6, 560 + 6 + 60), 3)

                    text_surface = font.render('Mask', False, (0, 0, 0))
                    screen.blit(text_surface, (x_offset + 560 + 30, 560 + 60))
                    x_square = m_mb.reshape((28, 28)).T
                    for i in range(len(x_square)):
                        for j in range(len(x_square[0])):
                            colour = x_square[i][j]*255
                            pygame.draw.rect(screen, (colour, colour, colour), (i*10+x_offset + 560, j*10+560 + 90, 10, 10))
                    
                    # Draw the agent
                    pygame.draw.rect(screen, agent_colour, (x_offset + 560, 560 + 60, 30, 30))
                    # add eyes to agent
                    pygame.draw.circle(screen, (255, 255, 255), (x_offset+560+21, 560 + 6 + 60), 6)
                    pygame.draw.circle(screen, (255, 255, 255), (x_offset+560+6, 560 + 6 + 60), 6)
                    pygame.draw.circle(screen, (0, 0, 0), (x_offset+560+21, 560 + 6 + 60), 3)
                    pygame.draw.circle(screen, (0, 0, 0), (x_offset+560+6, 560 + 6 + 60), 3)

            
            time_decisions.append(decisions.copy())
            time_decisions2.append(decisions2.copy())

            # Update global prediction
            average_prediction /= n_agents
            average_prediction2 /= n_agents

            # Global observation 
            m_square = global_mask.reshape(28, 28).T
            x_square = tenv_square * m_square
            text_surface = font.render('Global Observations', False, (0, 0, 0))
            screen.blit(text_surface, (280, 0))
            for i in range(len(x_square)):
                for j in range(len(x_square[0])):
                    colour = x_square[i][j]*255
                    pygame.draw.rect(screen, (colour, colour, colour), (i*10+280, j*10 + 30, 10, 10))
            
            for i in range(n_agents):
                pos = agents[i].get_position()
                if i == 0:
                    colour = agent_colours[0]
                elif i == 1:
                    colour = agent_colours[1]
                else:
                    colour = agent_colours[2]

                # Draw the agent
                pygame.draw.rect(screen, colour, (pos[1]*10+280, pos[0]*10 +30, 10, 10))
                # add eyes to agent
                pygame.draw.circle(screen, (255, 255, 255), (pos[1]*10+280+7, pos[0]*10+2 + 30), 2)
                pygame.draw.circle(screen, (255, 255, 255), (pos[1]*10+280+2 , pos[0]*10+2 + 30), 2)
                pygame.draw.circle(screen, (0, 0, 0), (pos[1]*10+280+7, pos[0]*10+2+30), 1 )
                pygame.draw.circle(screen, (0, 0, 0), (pos[1]*10+280+2, pos[0]*10+2+30), 1)

            # Draw communications
            for i in range(len(communications)):
                a1 = agents[communications[i][0]]
                a2 = agents[communications[i][1]]
                p1 = a1.get_position()
                p2 = a2.get_position()
                pygame.draw.line(screen, (255, 255, 255), (p1[1]*10+280+5, p1[0]*10+5 + 30), (p2[1]*10+280+5, p2[0]*10+5 + 30), 1)
            
            # Global baseline prediction
            text_surface = font.render('Baseline Average', False, (0, 0, 0))
            screen.blit(text_surface, (560, 0))
            p_square = average_prediction.reshape(28, 28).T
            for i in range(len(p_square)):
                for j in range(len(p_square[0])):
                    colour = p_square[i][j]*255
                    pygame.draw.rect(screen, (colour, colour, colour), (i*10+560, j*10 + 30, 10, 10))

            # Global prediction
            text_surface = font.render('Autoencoder Average', False, (0, 0, 0))
            screen.blit(text_surface, (840, 0))
            p_square = average_prediction2.reshape(28, 28).T
            for i in range(len(p_square)):
                for j in range(len(p_square[0])):
                    colour = p_square[i][j]*255
                    pygame.draw.rect(screen, (colour, colour, colour), (i*10+840, j*10 + 30, 10, 10))

            pygame.display.update()
    else:
        # Run the agents in the environment
        for step in range(steps):
            global_mask = np.zeros((28, 28))
            communications = list()
            
            # observe and send information
            for i in range(n_agents):
                agent = agents[i]
                pos = agent.get_position()
                
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
            
        with open("scenario2propernope/baseline/decisions.csv", "a") as f:
            data = [[decisions[k] for k in range(0, n_agents)] for decisions in time_decisions]
            writer = csv.writer(f)
            writer.writerow(["n", n_agents, digit])
            writer.writerows(data)

        with open("scenario2propernope/mask/decisions.csv", "a") as f:
            data = [[decisions[k] for k in range(0, n_agents)] for decisions in time_decisions2]
            writer = csv.writer(f)
            writer.writerow(["n", n_agents, digit])
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

'''
if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_mnist()
    start = time.time()
    for n in range(0, 51, 10):
        if n == 0:
            n = 1
        for digit in range(10):
            for _ in range(5):
                idx = np.where(y_train == digit)[0][0]
                print(time.time() - start, n, digit, idx)
                main(1000, False, n, idx, str(digit))

'''
if __name__ == '__main__':
    # example usage
    # python maenvironment.py
    (X_train, y_train), (X_test, y_test) = load_mnist()
    label1 = 5
    idx1 = 0
    main(1000, True, 20, idx1, str(label1))

