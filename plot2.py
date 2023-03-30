import csv
import matplotlib.pyplot as plt
import numpy as np

model_lables = {"none": "Baseline", "no-mask": "No Mask", "mask": "Mask", "dims": "40 Hidden Dimensions"}


class Entry:

    def __init__(self, digit, model, mse, psnr, ssim, explorations, agents_mse, agents_psnr, agents_ssim, agents_explorations, overhead):
        self.digit = digit
        self.model = model
        self.mse = mse
        self.psnr = psnr
        self.ssim = ssim
        self.explorations = explorations
        self.agents_mse = agents_mse
        self.agents_psnr = agents_psnr
        self.agents_ssim = agents_ssim
        self.agents_explorations = agents_explorations
        self.overhead = overhead


with open('resultsfinal.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

    entries = list()

    for i in range(len(data)):
        if (data[i]):
            if data[i][0] == "Data Index":
                digit = data[i][1]
                model = data[i+1][1]
                mse = np.array(data[i+2][1:]).astype(np.float32)
                psnr = np.array(data[i+3][1:]).astype(np.float32)
                psnr = np.where(psnr == np.inf, 100, psnr)
                ssim = np.array(data[i+4][1:]).astype(np.float32)
                explorations = np.array(data[i+5][1:]).astype(np.float32)
                agents_mse = dict()
                agents_psnr = dict()
                agents_ssim = dict()
                agents_explorations = dict()
                offset = 0

                while (data[i+6+offset][0] != "Overhead"):
                    agent = data[i+6+offset][0]

                    if (data[i+6+offset][1] == "MSE"):
                        agents_mse[agent] = np.array(data[i+6+offset][2:]).astype(np.float32)
                    elif (data[i+6+offset][1] == "PSNR"):
                        val = np.array(data[i+6+offset][2:]).astype(np.float32)
                        agents_psnr[agent] = np.where(val == np.inf, 100, val)
                    elif (data[i+6+offset][1] == "SSIM"):
                        agents_ssim[agent] = np.array(data[i+6+offset][2:]).astype(np.float32)
                    else:
                        agents_explorations[agent] = np.array(data[i+6+offset][2:]).astype(np.float32)

                    offset += 1

                overhead = np.array(data[i+6+offset][1:]).astype(np.float32)

                entry = Entry(digit, model, mse, psnr, ssim, explorations, agents_mse, agents_psnr, agents_ssim, agents_explorations, overhead)
                entries.append(entry)


max_mse = np.max(np.array([list(entry.agents_mse.values()) for entry in entries]))
min_mse = 0
max_psnr = np.max(np.array([list(entry.agents_psnr.values()) for entry in entries]))
min_psnr = np.min(np.array([list(entry.agents_psnr.values()) for entry in entries]))
max_ssim = np.max(np.array([list(entry.agents_ssim.values()) for entry in entries]))
min_ssim = np.min(np.array([list(entry.agents_ssim.values()) for entry in entries]))
max_overhead = np.max(np.cumsum(np.mean([entry.overhead for entry in entries], axis=0)))
min_overhead = 0

# Group entries by digit
digits = dict()
for entry in entries:
    if entry.digit not in digits:
        digits[entry.digit] = list()
    digits[entry.digit].append(entry)

# Group by model
models = dict()
for digit in digits:
    if digit not in models:
        models[digit] = dict()
    for entry in digits[digit]:
        if entry.model not in models[digit]:
            models[digit][entry.model] = list()
        models[digit][entry.model].append(entry)

# Plot each model for each digit
for digit in models:
    for model in models[digit]:
        plt.figure(figsize=(25, 10))


        plt.subplot(2, 5, 1)
        for agent in models[digit][model][0].agents_mse:
            agent_means = np.mean(np.array([entry.agents_mse[agent] for entry in models[digit][model]]), axis=0)
            agent_stds = np.std(np.array([entry.agents_mse[agent] for entry in models[digit][model]]), axis=0)
            
            plt.plot(agent_means, label=agent)
            plt.fill_between(range(len(agent_means)), agent_means - agent_stds, agent_means + agent_stds, alpha=0.2)

        plt.legend(loc='upper right')
        plt.xlabel("Steps")
        plt.ylabel("MSE")
        plt.title("{}: Mean MSE".format(model_lables[model]))
        plt.ylim(min_mse, max_mse)

        plt.subplot(2, 5, 2)
        for agent in models[digit][model][0].agents_psnr:
            agent_means = np.mean(np.array([entry.agents_psnr[agent] for entry in models[digit][model]]), axis=0)
            agent_stds = np.std(np.array([entry.agents_psnr[agent] for entry in models[digit][model]]), axis=0)
            
            plt.plot(agent_means, label=agent)
            plt.fill_between(range(len(agent_means)), agent_means - agent_stds, agent_means + agent_stds, alpha=0.2)
        
        plt.legend(loc='upper left')
        plt.xlabel("Steps")
        plt.ylabel("PSNR")
        plt.title("{}: Mean PSNR".format(model_lables[model]))
        plt.ylim(min_psnr, max_psnr)

        plt.subplot(2, 5, 3)
        for agent in models[digit][model][0].agents_ssim:
            agent_means = np.mean(np.array([entry.agents_ssim[agent] for entry in models[digit][model]]), axis=0)
            agent_stds = np.std(np.array([entry.agents_ssim[agent] for entry in models[digit][model]]), axis=0)
            
            plt.plot(agent_means, label=agent)
            plt.fill_between(range(len(agent_means)), agent_means - agent_stds, agent_means + agent_stds, alpha=0.2)
        
        plt.legend(loc='lower right')
        plt.xlabel("Steps")
        plt.ylabel("SSIM")
        plt.title("{}: Mean SSIM".format(model_lables[model]))
        plt.ylim(min_ssim, max_ssim)

        plt.subplot(2, 5, 4)
        for agent in models[digit][model][0].agents_explorations:
            agent_means = np.mean(np.array([entry.agents_explorations[agent] for entry in models[digit][model]]), axis=0)
            agent_stds = np.std(np.array([entry.agents_explorations[agent] for entry in models[digit][model]]), axis=0)

            plt.plot(agent_means, label=agent)
            plt.fill_between(range(len(agent_means)), agent_means - agent_stds, agent_means + agent_stds, alpha=0.2)
        
        plt.legend(loc='upper left')
        plt.xlabel("Steps")
        plt.ylabel("Explorations")
        plt.title("{}: Mean Explorations".format(model_lables[model]))
        plt.ylim(0, 1)

        mean_mse = np.mean([entry.mse for entry in models[digit][model]], axis=0)
        std_dev_mse = np.std([entry.mse for entry in models[digit][model]], axis=0)
        mean_psnr = np.mean([entry.psnr for entry in models[digit][model]], axis=0)
        std_dev_psnr = np.std([entry.psnr for entry in models[digit][model]], axis=0)
        mean_ssim = np.mean([entry.ssim for entry in models[digit][model]], axis=0)
        std_dev_ssim = np.std([entry.ssim for entry in models[digit][model]], axis=0)
        mean_explorations = np.mean([entry.explorations for entry in models[digit][model]], axis=0)
        std_dev_explorations = np.std([entry.explorations for entry in models[digit][model]], axis=0)
        
        mean_overheads = np.mean([entry.overhead for entry in models[digit][model]], axis=0)
        cumulative_overheads = np.cumsum(mean_overheads)
        std_dev_overheads = np.std([entry.overhead for entry in models[digit][model]], axis=0)

        plt.subplot(2, 5, 6)
        plt.plot(mean_mse, label="{} Mean".format(model_lables[model]))
        plt.fill_between(np.arange(len(mean_mse)), mean_mse - std_dev_mse, mean_mse + std_dev_mse, alpha=0.5)
        plt.legend(loc='upper right')
        plt.xlabel("Steps")
        plt.ylabel("MSE")
        plt.title("{}: Mean MSE".format(model_lables[model]))
        plt.ylim(min_mse, max_mse)

        plt.subplot(2, 5, 7)
        plt.plot(mean_psnr, label="{} Mean".format(model_lables[model]))
        plt.fill_between(np.arange(len(mean_psnr)), mean_psnr - std_dev_psnr, mean_psnr + std_dev_psnr, alpha=0.5)
        plt.legend(loc='upper left')
        plt.xlabel("Steps")
        plt.ylabel("PSNR")
        plt.title("{}: Mean PSNR".format(model_lables[model]))
        plt.ylim(min_psnr, max_psnr)

        plt.subplot(2, 5, 8)
        plt.plot(mean_ssim, label="{} Mean".format(model_lables[model]))
        plt.fill_between(np.arange(len(mean_ssim)), mean_ssim - std_dev_ssim, mean_ssim + std_dev_ssim, alpha=0.5)
        plt.legend(loc='lower right')
        plt.xlabel("Steps")
        plt.ylabel("SSIM")
        plt.title("{}: Mean SSIM".format(model_lables[model]))
        plt.ylim(min_ssim, max_ssim)

        plt.subplot(2, 5, 9)
        plt.plot(mean_explorations, label="{} Mean".format(model_lables[model]))
        plt.fill_between(np.arange(len(mean_explorations)), mean_explorations - std_dev_explorations, mean_explorations + std_dev_explorations, alpha=0.5)
        plt.legend(loc='lower right')
        plt.xlabel("Steps")
        plt.ylabel("Percentage Explored")
        plt.title("{}: Percentage Explored".format(model_lables[model]))
        plt.ylim(0, 1)

        plt.subplot(2, 5, 10)
        plt.plot(cumulative_overheads, label="{} Mean".format(model_lables[model]))
        plt.fill_between(np.arange(len(cumulative_overheads)), cumulative_overheads - std_dev_overheads, cumulative_overheads + std_dev_overheads, alpha=0.5)
        plt.legend(loc='upper left')
        plt.xlabel("Steps")
        plt.ylabel("Communication Overhead")
        plt.title("{}: Cumulative Communication Overhead".format(model_lables[model]))
        plt.ylim(min_overhead, max_overhead)
        
        plt.savefig("digit_" + str(digit) + "_" + model + ".png")

'''
# plot the mean and std dev of the accuracies and explorations for each digit
for digit in models:
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    for model in models[digit]:
        #model_mean_accuracies = np.mean([entry.accuracies for entry in models[digit][model]], axis=0)
        model_mean_mse = np.mean([entry.mse for entry in models[digit][model]], axis=0)
        model_std_dev_mse = np.std([entry.mse for entry in models[digit][model]], axis=0)

        plt.plot(model_mean_mse, label=model_lables[model])
        plt.fill_between(np.arange(len(model_mean_mse)), model_mean_mse - model_std_dev_mse, model_mean_mse + model_std_dev_mse, alpha=0.5)

    plt.legend(loc='upper right')
    plt.xlabel("Steps")
    plt.ylabel("MSE")
    plt.title("Comparison of MSE")
    plt.ylim(min_mse, max_mse)

    plt.subplot(1, 4, 2)
    for model in models[digit]:
        model_mean_psnr = np.mean([entry.psnr for entry in models[digit][model]], axis=0)
        model_std_dev_psnr = np.std([entry.psnr for entry in models[digit][model]], axis=0)

        plt.plot(model_mean_psnr, label=model_lables[model])
        plt.fill_between(np.arange(len(model_mean_psnr)), model_mean_psnr - model_std_dev_psnr, model_mean_psnr + model_std_dev_psnr, alpha=0.5)
    
    plt.legend(loc='upper left')
    plt.xlabel("Steps")
    plt.ylabel("PSNR")
    plt.title("Comparison of PSNR")
    plt.ylim(min_psnr, max_psnr)

    plt.subplot(1, 4, 3)
    for model in models[digit]:
        model_mean_ssim = np.mean([entry.ssim for entry in models[digit][model]], axis=0)
        model_std_dev_ssim = np.std([entry.ssim for entry in models[digit][model]], axis=0)

        plt.plot(model_mean_ssim, label=model_lables[model])
        plt.fill_between(np.arange(len(model_mean_ssim)), model_mean_ssim - model_std_dev_ssim, model_mean_ssim + model_std_dev_ssim, alpha=0.5)
    
    plt.legend(loc='lower right')
    plt.xlabel("Steps")
    plt.ylabel("SSIM")
    plt.title("Comparison of SSIM")
    plt.ylim(min_ssim, max_ssim)

    plt.subplot(1, 4, 4)
    for model in models[digit]:
        model_mean_explorations = np.mean([entry.explorations for entry in models[digit][model]], axis=0)
        model_std_dev_explorations = np.std([entry.explorations for entry in models[digit][model]], axis=0)

        plt.plot(model_mean_explorations, label=model_lables[model])
        plt.fill_between(np.arange(len(model_mean_explorations)), model_mean_explorations - model_std_dev_explorations, model_mean_explorations + model_std_dev_explorations, alpha=0.5)

    plt.legend(loc='upper left')
    plt.xlabel("Steps")
    plt.ylabel("Percentage Explored")
    plt.title("Percentage Explored")
    plt.ylim(0, 1)

    plt.savefig("digit_" + str(digit) + "_all.png")
'''
