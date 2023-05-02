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
    data_index = 0
    env = norm_data_x[data_index]
    print(np.max(env), np.min(env))

    encoder1 = keras.models.load_model("models/no-mask/encoder.h5")
    encoder2 = keras.models.load_model("models2/mask/encoder.h5")
    decoder2 = keras.models.load_model("models2/mask/decoder.h5")
    encoder3 = keras.models.load_model("models2/dims/encoder.h5")
    decoder3 = keras.models.load_model("models2/dims/decoder.h5")
    encoder4 = keras.models.load_model("models2/split/encoder.h5")
    decoder4 = keras.models.load_model("models2/split/decoder.h5")
    encoder5 = keras.models.load_model("models2/split-dims/encoder.h5")
    decoder5 = keras.models.load_model("models2/split-dims/decoder.h5")

    # Create agent
    agent = Agent((14, 14), (28, 28))

    # Run the agent in the environment
    mse_scores1 = []
    mse_scores2 = []
    mse_scores3 = []
    mse_scores4 = []
    mse_scores5 = []

    # Create new log directory
    if log_dir is None:
        log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        os.makedirs(log_dir)
    
    for step in range(steps):
        if step % 100 == 0:
            print(step)
        pos = agent.get_position()

        env_square = env.reshape(28, 28)
        measurements = env_square[pos[0]-1:pos[0]+2, pos[1]-1:pos[1]+2]

        agent.measure(measurements)
        x_mb = agent.get_map().reshape(1, 28, 28, 1)
        x_mb = np.nan_to_num(x_mb, 0)
        m_mb = agent.get_mask().reshape(1, 28, 28, 1)

        prediction1 = np.array(encoder1([x_mb]))
        prediciton2 = np.array(decoder2(encoder2([x_mb, m_mb])))
        prediction3 = np.array(decoder3(encoder3([x_mb, m_mb])))
        prediction4 = np.array(decoder4(encoder4([x_mb, m_mb])))
        prediction5 = np.array(decoder5(encoder5([x_mb, m_mb])))

        prediction1 = m_mb * x_mb + (1-m_mb) * prediction1
        prediciton2 = m_mb * x_mb + (1-m_mb) * prediciton2
        prediction3 = m_mb * x_mb + (1-m_mb) * prediction3
        prediction4 = m_mb * x_mb + (1-m_mb) * prediction4
        prediction5 = m_mb * x_mb + (1-m_mb) * prediction5
        
        mse_score1, _, _ = accuracy(env.reshape(784), prediction1.reshape(784))
        mse_score2, _, _ = accuracy(env.reshape(784), prediciton2.reshape(784))
        mse_score3, _, _ = accuracy(env.reshape(784), prediction3.reshape(784))
        mse_score4, _, _ = accuracy(env.reshape(784), prediction4.reshape(784))
        mse_score5, _, _ = accuracy(env.reshape(784), prediction5.reshape(784))

        mse_scores1.append(mse_score1)
        mse_scores2.append(mse_score2)
        mse_scores3.append(mse_score3)
        mse_scores4.append(mse_score4)
        mse_scores5.append(mse_score5)
    
    # Save to csv
    with open(log_dir + "/accuracies.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerows([["No Mask"], mse_scores1,
                            ["1P 20HD"], mse_scores2,
                            ["1P 40HD"], mse_scores3,
                            ["2P 20HD"], mse_scores4,
                            ["2P 40HD"], mse_scores5])

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

    for x in range(20):
        main(1000, visualise, model, output_dir)
