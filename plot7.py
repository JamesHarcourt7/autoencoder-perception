import csv
import matplotlib.pyplot as plt
import numpy as np


class Entry:

    def __init__(self, alpha, beta, theta_max, mse):
        self.alpha = alpha
        self.beta = beta
        self.theta_max = theta_max
        self.mse = mse

'''
for alpha in [0.01, 0.02, 0.05]:
        for beta in [0.4, 0.6, 0.8]:
            for theta_max in [0.4, 0.6, 0.8]:
                main(1000, False, 10, idx1, idx2, label1, label2, alpha, beta, theta_max)
'''


with open('scenario3tune/mask/accuracies.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

    entries = list()

    '''
    Messed up formatting of results, so reconstructing in a crude way :(
    0.01 0.4 0.4 0
    0.01 0.4 0.6 0
    0.01 0.4 0.8 0
    0.01 0.6 0.4 0
    0.01 0.6 0.6 0
    0.01 0.6 0.8 0
    0.01 0.8 0.4 0
    0.01 0.8 0.6 0
    0.01 0.8 0.8 0
    0.02 0.4 0.4 0
    0.02 0.4 0.6 0
    0.02 0.4 0.8 0
    0.02 0.6 0.4 0
    0.02 0.6 0.6 0
    0.02 0.6 0.8 0
    0.02 0.8 0.4 0
    0.02 0.8 0.6 0
    0.02 0.8 0.8 0
    0.05 0.4 0.4 0
    0.05 0.4 0.6 0
    0.05 0.4 0.8 0
    0.05 0.6 0.4 0
    0.05 0.6 0.6 0
    0.05 0.6 0.8 0
    0.05 0.8 0.4 0
    0.05 0.8 0.6 0
    0.05 0.8 0.8 0

    0.01 0.4 0.4 1
    0.01 0.4 0.4 2
    0.01 0.4 0.4 3
    0.01 0.4 0.4 4
    0.01 0.4 0.6 1
    0.01 0.4 0.6 2
    0.01 0.4 0.6 3
    0.01 0.4 0.6 4
    0.01 0.4 0.8 1
    0.01 0.4 0.8 2
    0.01 0.4 0.8 3
    0.01 0.4 0.8 4
    0.01 0.6 0.4 1
    0.01 0.6 0.4 2
    0.01 0.6 0.4 3
    0.01 0.6 0.4 4
    0.01 0.6 0.6 1
    ...
    '''

    entries = list()

    for x, alpha in enumerate([0.01, 0.02, 0.05]):
        for y, beta in enumerate([0.4, 0.6, 0.8]):
            for z, theta_max in enumerate([0.4, 0.6, 0.8]):
                index = ((x * 9) + (y * 3) + z) * 12

                mse_values = list()

                first = index + 2
                for i in range(10):
                    agent = data[first + i][0]
                    agent_mse = np.array(data[first + i][2:]).astype(np.float32)
                    mse_values.append(agent_mse)

                
                mse_values = np.array(mse_values)
                mean_mse = np.mean(mse_values, axis=0)
                entry = Entry(alpha, beta, theta_max, mean_mse)
                entries.append(entry)

                
    for x, alpha in enumerate([0.01, 0.02, 0.05]):
        for y, beta in enumerate([0.4, 0.6, 0.8]):
            for z, theta_max in enumerate([0.4, 0.6, 0.8]):
                for i in range(4):
                    index = (27 * 12) + (((x * 27) + (y * 9) + (z * 3) + i) * 12)

                    mse_values = list()

                    first = index + 2
                    for i in range(10):
                        agent = data[first + i][0]
                        agent_mse = np.array(data[first + i][2:]).astype(np.float32)
                        mse_values.append(agent_mse)
                    
                    mse_values = np.array(mse_values)
                    mean_mse = np.mean(mse_values, axis=0)
                    entry = Entry(alpha, beta, theta_max, mean_mse)
                    entries.append(entry)
    
    print(len(entries))
    
    # Group entries by alpha, beta, theta_max
    abt = dict()
    for entry in entries:
        key = "alpha={}, beta={}, theta_max={}".format(entry.alpha, entry.beta, entry.theta_max)
        if key not in abt:
            abt[key] = list()
        abt[key].append(entry)


# plot the means of mse for each alpha, beta, theta_max
plt.figure(figsize=(13, 13))
for key in abt:
    entries = np.array([e.mse for e in abt[key]])

    plt.plot(np.mean(entries, axis=0), label=key)
    
plt.legend(loc='upper right')
plt.xlabel("Time (steps)")
plt.ylabel("MSE")

plt.savefig("scenario3_tuning.png")

# plot the means of mse for each alpha
plt.figure(figsize=(10, 10))
a = dict()

for key in abt:
    entries = abt[key]

    # group by alpha
    for entry in entries:
        key = entry.alpha
        if key not in a:
            a[key] = list()
        a[key].append(entry)
    
for key in a:
    entries = np.array([e.mse for e in a[key]])
    plt.plot(np.mean(entries, axis=0), label="alpha={}".format(key))
    std = np.std(entries, axis=0)
    plt.fill_between(range(len(std)), np.mean(entries, axis=0) - std, np.mean(entries, axis=0) + std, alpha=0.2)

plt.legend(loc='upper right')
plt.xlabel("Time (steps)")
plt.ylabel("MSE")

plt.savefig("scenario3_tuning_alpha.png")

# plot the means of mse for each beta
plt.figure(figsize=(10, 10))
b = dict()

for key in abt:
    entries = abt[key]

    # group by beta
    for entry in entries:
        key = entry.beta
        if key not in b:
            b[key] = list()
        b[key].append(entry)
    
for key in b:
    entries = np.array([e.mse for e in b[key]])
    plt.plot(np.mean(entries, axis=0), label="beta={}".format(key))
    std = np.std(entries, axis=0)
    plt.fill_between(range(len(std)), np.mean(entries, axis=0) - std, np.mean(entries, axis=0) + std, alpha=0.2)

plt.legend(loc='upper right')
plt.xlabel("Time (steps)")
plt.ylabel("MSE")

plt.savefig("scenario3_tuning_beta.png")

# plot the means of mse for each theta_max
plt.figure(figsize=(10, 10))
tm = dict()

for key in abt:
    entries = abt[key]
    
    # group by theta_max
    for entry in entries:
        key = entry.theta_max
        if key not in tm:
            tm[key] = list()
        tm[key].append(entry)
    
for key in tm:
    entries = np.array([e.mse for e in tm[key]])
    plt.plot(np.mean(entries, axis=0), label="theta_max={}".format(key))
    std = np.std(entries, axis=0)
    plt.fill_between(range(len(std)), np.mean(entries, axis=0) - std, np.mean(entries, axis=0) + std, alpha=0.2)

plt.legend(loc='upper right')
plt.xlabel("Time (steps)")
plt.ylabel("MSE")

plt.savefig("scenario3_tuning_theta_max.png")

# plot the means of mse for each alpha, beta
plt.figure(figsize=(10, 10))
ab = dict()

for key in abt:
    entries = abt[key]

    # group by alpha, beta
    for entry in entries:
        key = "alpha={}, beta={}".format(entry.alpha, entry.beta)
        if key not in ab:
            ab[key] = list()
        ab[key].append(entry)
    
for key in ab:
    entries = np.array([e.mse for e in ab[key]])
    plt.plot(np.mean(entries, axis=0), label=key)

plt.legend(loc='upper right')
plt.xlabel("Time (steps)")
plt.ylabel("MSE")

plt.savefig("scenario3_tuning_alpha_beta.png")

# plot the means of mse for each alpha, theta_max
plt.figure(figsize=(10, 10))
at = dict()

for key in abt:
    entries = abt[key]

    # group by alpha, theta_max
    for entry in entries:
        key = "alpha={}, theta_max={}".format(entry.alpha, entry.theta_max)
        if key not in at:
            at[key] = list()
        at[key].append(entry)
    
for key in at:
    entries = np.array([e.mse for e in at[key]])
    plt.plot(np.mean(entries, axis=0), label=key)

plt.legend(loc='upper right')
plt.xlabel("Time (steps)")
plt.ylabel("MSE")

plt.savefig("scenario3_tuning_alpha_theta_max.png")

# plot the means of mse for each beta, theta_max
plt.figure(figsize=(10, 10))
bt = dict()

for key in abt:
    entries = abt[key]

    # group by beta, theta_max
    for entry in entries:
        key = "beta={}, theta_max={}".format(entry.beta, entry.theta_max)
        if key not in bt:
            bt[key] = list()
        bt[key].append(entry)

for key in bt:
    entries = np.array([e.mse for e in bt[key]])
    plt.plot(np.mean(entries, axis=0), label=key)

plt.legend(loc='upper right')
plt.xlabel("Time (steps)")
plt.ylabel("MSE")

plt.savefig("scenario3_tuning_beta_theta_max.png")

# plot the means of mse for each alpha, beta, theta_max where alpha=0.01
plt.figure(figsize=(10, 10))
a1 = dict()

for key in abt:
    entries = abt[key]

    # group by alpha, beta, theta_max
    for entry in entries:
        if entry.alpha != 0.01:
            continue
        
        key = "alpha={}, beta={}, theta_max={}".format(entry.alpha, entry.beta, entry.theta_max)
        if key not in a1:
            a1[key] = list()
        a1[key].append(entry)
    
for key in a1:
    entries = np.array([e.mse for e in a1[key]])
    plt.plot(np.mean(entries, axis=0), label=key)

plt.legend(loc='upper right')
plt.xlabel("Time (steps)")
plt.ylabel("MSE")

plt.savefig("scenario3_tuning_alpha_0.01.png")

