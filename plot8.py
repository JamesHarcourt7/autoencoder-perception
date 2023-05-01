import matplotlib.pyplot as plt
import csv
import numpy as np


n_map1 = dict()
n_map2 = dict()
n_map3 = dict()
n_map4 = dict()
count = 0

threshold = 0.8

with open('scenario3proper/baseline/decisions.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

    for i in range(len(data)):
        if data[i][0] == 'n':
            
            n = int(data[i][1])

            count += 1
            digit1 = data[i][2]
            digit2 = data[i][3]

            if n not in n_map1:
                n_map1[n] = list()
            if n not in n_map2:
                n_map2[n] = list()
            
            for j in range(500):
                total = 0
                for decision in data[i + j + 1]:
                    if decision == digit1:
                        total += 1

                if total/n >= threshold or (total == 1 and n == 1):
                    n_map1[n].append(j)
                    print(n, digit1, j)
                    break
            
                if j == 499:
                    n_map1[n].append(500)

            for j in range(500):
                total = 0
                for decision in data[i + j + 501]:
                    if decision == digit2:
                        total += 1

                if total/n >= threshold or (total == 1 and n == 1):
                    n_map2[n].append(j)
                    print(n, digit2, j)
                    break

                if j == 499:
                    n_map2[n].append(500)
                    
print()

with open('scenario3proper/mask/decisions.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

    for i in range(len(data)):
        if data[i][0] == 'n':

            n = int(data[i][1])

            count += 1
            digit1 = data[i][2]
            digit2 = data[i][3]

            if n not in n_map3:
                n_map3[n] = list()
            if n not in n_map4:
                n_map4[n] = list()
            
            for j in range(500):
                total = 0
                for decision in data[i + j + 1]:
                    if decision == digit1:
                        total += 1

                if total/n >= threshold or (total == 1 and n == 1):
                    n_map3[n].append(j)
                    print(n, digit1, j)
                    break
            
                if j == 499:
                    n_map3[n].append(500)
                
            for j in range(500):
                total = 0
                for decision in data[i + j + 501]:
                    if decision == digit2:
                        total += 1

                if total/n >= threshold or (total == 1 and n == 1):
                    n_map4[n].append(j)
                    print(n, digit2, j)
                    break

                if j == 499:
                    n_map4[n].append(500)

print()
class Entry:

    def __init__(self, n, mse, digit1, digit2):
        self.n = n
        self.mse = mse
        self.digit1 = digit1
        self.digit2 = digit2

with open('scenario3proper/baseline/accuracies.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

    entries = list()

    for i in range(len(data)):
        if data[i][0] == 'Data Indexes':
            mse_values = list()

            digit1 = int(data[i][1])
            digit2 = int(data[i][2])
            n = int(data[i + 1][1])

            first = i + 2
            for j in range(n):
                agent = data[first + j][0]
                agent_mse = np.array(data[first + j][2:]).astype(np.float32)
                mse_values.append(agent_mse)
            
            mse_values = np.array(mse_values)
            average_mse = np.mean(mse_values, axis=0)

            n = int(data[first - 1][1])
            entries.append(Entry(n, average_mse, digit1, digit2))

with open('scenario3proper/mask/accuracies.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

    entries2 = list()

    for i in range(len(data)):
        if data[i][0] == 'Data Indexes':
            mse_values = list()

            digit1 = int(data[i][1])
            digit2 = int(data[i][2])
            n = int(data[i + 1][1])

            first = i + 2
            for j in range(n):
                agent = data[first + j][0]
                agent_mse = np.array(data[first + j][2:]).astype(np.float32)
                mse_values.append(agent_mse)
            
            mse_values = np.array(mse_values)
            average_mse = np.mean(mse_values, axis=0)

            n = int(data[first - 1][1])
            entries2.append(Entry(n, average_mse, digit1, digit2))


plt.figure(figsize=(7, 7))
print(count)

for n in n_map1:
    average1 = sum(n_map1[n])/len(n_map1[n])
    std_dev1 = (sum([(x - average1)**2 for x in n_map1[n]])/len(n_map1[n]))**0.5
    average2 = sum(n_map2[n])/len(n_map2[n])
    std_dev2 = (sum([(x - average2)**2 for x in n_map2[n]])/len(n_map2[n]))**0.5

    #print(n, average1, std_dev1, average2, std_dev2)

    # grouped bar chart

    plt.bar(n - 1.5, average1, width=3, yerr=std_dev1, label='Phase 1', color='lightblue', ecolor='cornflowerblue', capsize=5)
    plt.bar(n + 1.5, average2, width=3, yerr=std_dev2, label='Phase 2', color='deepskyblue', ecolor='cornflowerblue', capsize=5)


ticks = [i for i in range(10, 51, 10)]
ticks = np.array(ticks)

plt.ylabel('Time (Steps)')
plt.xlabel('Number of Agents')
plt.xticks(ticks)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.ylim(0, 500)
#plt.yscale('log')
plt.savefig('scenario3_baseline.png')

plt.figure(figsize=(7, 7))

for n in n_map3:
    average1 = sum(n_map3[n])/len(n_map3[n])
    std_dev1 = (sum([(x - average1)**2 for x in n_map3[n]])/len(n_map3[n]))**0.5
    average2 = sum(n_map4[n])/len(n_map4[n])
    std_dev2 = (sum([(x - average2)**2 for x in n_map4[n]])/len(n_map4[n]))**0.5

    #print(n, average1, std_dev1, average2, std_dev2)

    # grouped bar chart

    plt.bar(n - 1.5, average1, width=3, yerr=std_dev1, label='Phase 1', color='gold', ecolor='chocolate', capsize=5)
    plt.bar(n + 1.5, average2, width=3, yerr=std_dev2, label='Phase 2', color='orange', ecolor='chocolate', capsize=5)

plt.ylabel('Time (Steps)')
plt.xlabel('Number of Agents')
plt.xticks(ticks)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.ylim(0, 500)
#plt.yscale('log')
plt.savefig('scenario3_mask.png')

# plot mse vs n
# group by n
n_map = dict()
for entry in entries:
    if entry.n not in n_map:
        n_map[entry.n] = list()
    n_map[entry.n].append(entry)

plt.figure(figsize=(7, 7))

for n in n_map:
    es = np.array([e.mse for e in n_map[n]])

    plt.plot(np.mean(es, axis=0), label="{} Agents".format(n))

plt.ylabel('MSE')
plt.xlabel('Time (Steps)')
plt.legend()
plt.savefig('scenario3_mse_baseline.png')

# plot mse vs n for mask
# group by n
n_map = dict()
for entry in entries2:
    if entry.n not in n_map:
        n_map[entry.n] = list()
    n_map[entry.n].append(entry)

plt.figure(figsize=(7, 7))

for n in n_map:
    es = np.array([e.mse for e in n_map[n]])

    plt.plot(np.mean(es, axis=0), label="{} Agents".format(n))

plt.ylabel('MSE')
plt.xlabel('Time (Steps)')
plt.legend()
plt.savefig('scenario3_mse_mask.png')

# plot mse vs n
# group by starting digit and n
n_map = dict()
for entry in entries:
    key = (entry.digit1, entry.digit2, entry.n)
    if key not in n_map:
        n_map[key] = list()
    n_map[key].append(entry)
    
plt.figure(figsize=(7, 7))

for key in n_map:
    es = np.array([e.mse for e in n_map[key]])

    plt.plot(np.mean(es, axis=0), label="{}".format(key))

plt.ylabel('MSE')
plt.xlabel('Time (Steps)')
plt.legend()
plt.savefig('scenario3_digit_baseline.png')

# plot mse vs n for mask
# group by starting digit and n
n_map = dict()
for entry in entries2:
    key = (entry.digit1, entry.digit2, entry.n)
    if key not in n_map:
        n_map[key] = list()
    n_map[key].append(entry)

plt.figure(figsize=(7, 7))

for key in n_map:
    es = np.array([e.mse for e in n_map[key]])

    plt.plot(np.mean(es, axis=0), label="{}".format(key))

plt.ylabel('MSE')
plt.xlabel('Time (Steps)')
plt.legend()
plt.savefig('scenario3_digit_mask.png')

# plot mse for each n and include decision time
# group by n
n_map = dict()
for entry in entries:
    if entry.n not in n_map:
        n_map[entry.n] = list()
    n_map[entry.n].append(entry)

plt.figure(figsize=(25, 5))

for i, n in enumerate(n_map.keys()):
    es = np.array([e.mse for e in n_map[n]])
    std = np.std(es, axis=0)

    start = 0
    end = sum(n_map1[n])/len(n_map1[n])
    start2 = 501
    end2 = 500 + sum(n_map2[n])/len(n_map2[n])
    
    plt.subplot(1, 5, i + 1)
    
    plt.fill_betweenx([0, 0.15], start, end, alpha=0.2, color='green')
    plt.fill_betweenx([0, 0.15], start2, end2, alpha=0.2, color='green')

    plt.fill_between(range(len(np.mean(es, axis=0))), np.mean(es, axis=0) - std, np.mean(es, axis=0) + std, alpha=1.0, color='white')
    plt.plot(np.mean(es, axis=0), label="{} Agents".format(n))
    plt.fill_between(range(len(np.mean(es, axis=0))), np.mean(es, axis=0) - std - 0.0003, np.mean(es, axis=0) + std + 0.0003, alpha=0.5)
    plt.title("{} Agents".format(n))
    if (i == 0):
        plt.ylabel('MSE')
    if (i == 2):
        plt.xlabel('Time (Steps)')
    plt.ylim(0, 0.15)
    plt.xlim(0, 1000)

plt.savefig('scenario3_threshold_baseline.png')

# plot mse for each n and include decision time for mask
# group by n
n_map = dict()
for entry in entries2:
    if entry.n not in n_map:
        n_map[entry.n] = list()
    n_map[entry.n].append(entry)

plt.figure(figsize=(25, 5))

for i, n in enumerate(n_map.keys()):
    es = np.array([e.mse for e in n_map[n]])
    std = np.std(es, axis=0)

    start = 0
    end = sum(n_map3[n])/len(n_map3[n])
    start2 = 501
    end2 = 500 + sum(n_map4[n])/len(n_map4[n])
    
    plt.subplot(1, 5, i + 1)

    plt.fill_betweenx([0, 0.15], start, end, alpha=0.2, color='green')
    plt.fill_betweenx([0, 0.15], start2, end2, alpha=0.2, color='green')

    plt.fill_between(range(len(np.mean(es, axis=0))), np.mean(es, axis=0) - std, np.mean(es, axis=0) + std, alpha=1.0, color='white')
    plt.plot(np.mean(es, axis=0), label="{} Agents".format(n), color='chocolate')
    plt.fill_between(range(len(np.mean(es, axis=0))), np.mean(es, axis=0) - std, np.mean(es, axis=0) + std, alpha=0.5, color='orange')

    plt.title("{} Agents".format(n))
    if (i == 0):
        plt.ylabel('MSE')
    if (i == 2):
        plt.xlabel('Time (Steps)')
    plt.ylim(0, 0.15)
    plt.xlim(0, 1000)

plt.savefig('scenario3_threshold_mask.png')



