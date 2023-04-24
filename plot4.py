import csv
import matplotlib.pyplot as plt
import numpy as np

class Entry:

    def __init__(self, digit1, digit2, decay, beta, mse, agents_mse):
        self.digit1 = digit1
        self.digit2 = digit2
        self.decay = decay
        self.beta = beta
        self.mse = mse
        self.agents_mse = agents_mse


with open('resultsfinal.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

    entries = list()

    for i in range(len(data)):
        if (data[i]):
            if data[i][0] == "Data Indexes":
                digit1 = data[i][1]
                digit2 = data[i][2]
                decay = data[i+1][1]
                beta = data[i+2][1]
                mse = np.array(data[i+3][1:]).astype(np.float32)
                agents_mse = dict()
                offset = 0

                while len(data) > i+4+offset and (data[i+4+offset][0] != "Data Indexes"):
                    agent = data[i+4+offset][0]
                    agents_mse[agent] = np.array(data[i+4+offset][2:]).astype(np.float32)

                    offset += 1

                entry = Entry(digit1, digit2, decay, beta, mse, agents_mse)
                entries.append(entry)

max_mse = np.max(np.array([list(entry.mse) for entry in entries]))
min_mse = 0

# Group entries by decay
decays = dict()
for entry in entries:
    if entry.decay not in decays:
        decays[entry.decay] = list()
    decays[entry.decay].append(entry)

# Group entries by beta
betas = dict()
for entry in entries:
    if entry.beta not in betas:
        betas[entry.beta] = list()
    betas[entry.beta].append(entry)

# Plot each model for each decay
for decay in decays:
    plt.figure(figsize=(25, 10))


    plt.subplot(2, 5, 1)
    for agent in decays[decay][0].agents_mse:
        agent_means = np.mean(np.array([entry.agents_mse[agent] for entry in decays[decay]]), axis=0)
        agent_stds = np.std(np.array([entry.agents_mse[agent] for entry in decays[decay]]), axis=0)
        
        plt.plot(agent_means, label=agent)
        plt.fill_between(range(len(agent_means)), agent_means - agent_stds, agent_means + agent_stds, alpha=0.2)

    plt.legend(loc='upper right')
    plt.xlabel("Steps")
    plt.ylabel("MSE")
    plt.title("{}: Mean MSE".format(decay))
    plt.ylim(min_mse, max_mse)

    mean_mse = np.mean([entry.mse for entry in decays[decay]], axis=0)
    std_dev_mse = np.std([entry.mse for entry in decays[decay]], axis=0)

    plt.subplot(2, 5, 6)
    plt.plot(mean_mse, label="{} Mean".format(decay))
    plt.fill_between(np.arange(len(mean_mse)), mean_mse - std_dev_mse, mean_mse + std_dev_mse, alpha=0.5)
    plt.legend(loc='upper right')
    plt.xlabel("Steps")
    plt.ylabel("MSE")
    plt.title("{}: Mean MSE".format(decay))
    plt.ylim(min_mse, max_mse)
    
    plt.savefig("decay_" + str(decay) + ".png")

for beta in betas:
    plt.figure(figsize=(25, 10))


    plt.subplot(2, 5, 1)
    for agent in betas[beta][0].agents_mse:
        agent_means = np.mean(np.array([entry.agents_mse[agent] for entry in betas[beta]]), axis=0)
        agent_stds = np.std(np.array([entry.agents_mse[agent] for entry in betas[beta]]), axis=0)
        
        plt.plot(agent_means, label=agent)
        plt.fill_between(range(len(agent_means)), agent_means - agent_stds, agent_means + agent_stds, alpha=0.2)

    plt.legend(loc='upper right')
    plt.xlabel("Steps")
    plt.ylabel("MSE")
    plt.title("{}: Mean MSE".format(beta))
    plt.ylim(min_mse, max_mse)

    mean_mse = np.mean([entry.mse for entry in betas[beta]], axis=0)
    std_dev_mse = np.std([entry.mse for entry in betas[beta]], axis=0)

    plt.subplot(2, 5, 6)
    plt.plot(mean_mse, label="{} Mean".format(beta))
    plt.fill_between(np.arange(len(mean_mse)), mean_mse - std_dev_mse, mean_mse + std_dev_mse, alpha=0.5)
    plt.legend(loc='upper right')
    plt.xlabel("Steps")
    plt.ylabel("MSE")
    plt.title("{}: Mean MSE".format(beta))
    plt.ylim(min_mse, max_mse)
    
    plt.savefig("beta_" + str(beta) + ".png")

# plot the mean and std dev of the accuracies and explorations for each decay
plt.figure(figsize=(10, 7.5))

decay_list = sorted(decays.keys())

for i, decay in enumerate(decay_list):
    model_mean_mse = np.mean([entry.mse for entry in decays[decay]], axis=0)
    model_std_dev_mse = np.std([entry.mse for entry in decays[decay]], axis=0)

    plt.plot(model_mean_mse, label=decay)
    #plt.fill_between(np.arange(len(model_mean_mse)), model_mean_mse - model_std_dev_mse, model_mean_mse + model_std_dev_mse, alpha=0.5)

plt.legend(loc='upper right')
plt.xlabel("Time (steps)")
plt.ylabel("MSE")
plt.title("Comparison of MSE for different decay values")
plt.ylim(min_mse, max_mse)

plt.savefig("decay_all.png")

# plot the mean and std dev of the accuracies and explorations for each beta
plt.figure(figsize=(10, 7.5))

beta_list = sorted(list(betas.keys()))

for i, beta in enumerate(beta_list):
    print(len(betas[beta]))
    model_mean_mse = np.mean([entry.mse for entry in betas[beta]], axis=0)
    model_std_dev_mse = np.std([entry.mse for entry in betas[beta]], axis=0)

    plt.plot(model_mean_mse, label=beta)
    #plt.fill_between(np.arange(len(model_mean_mse)), model_mean_mse - model_std_dev_mse, model_mean_mse + model_std_dev_mse, alpha=0.5)

plt.legend(loc='upper right')
plt.xlabel("Time (steps)")
plt.ylabel("MSE")
plt.title("Comparison of MSE for different beta values")
plt.ylim(min_mse, max_mse)

plt.savefig("beta_all.png")

