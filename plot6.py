import matplotlib.pyplot as plt
import csv
import numpy as np


n_map1 = dict()
n_map2 = dict()
count = 0

threshold = 0.8

with open('scenario2proper2/baseline/decisions.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

    for i in range(len(data)):
        if data[i][0] == 'n':
            
            n = int(data[i][1])
            if n == 1:
                continue

            count += 1
            digit = data[i][2]
            if n not in n_map1:
                n_map1[n] = list()
            if n not in n_map2:
                n_map2[n] = list()
            
            for j in range(1000):
                total = 0
                for decision in data[i + j + 1]:
                    if decision == digit:
                        total += 1

                if total/n >= threshold or (total == 1 and n == 1):
                    n_map1[n].append(j)
                    print(n, digit, j)
                    break
            
                if j == 999:
                    n_map1[n].append(1000)

print()
with open('scenario2proper2/mask/decisions.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

    for i in range(len(data)):
        if data[i][0] == 'n':

            n = int(data[i][1])
            if n == 1:
                continue

            count += 1

            digit = data[i][2]
            if n not in n_map1:
                n_map1[n] = list()
            if n not in n_map2:
                n_map2[n] = list()
            
            for j in range(1000):
                total = 0
                for decision in data[i + j + 1]:
                    if decision == digit:
                        total += 1

                if total/n >= threshold or (total == 1 and n == 1):
                    n_map2[n].append(j)
                    print(n, digit, j)
                    break
            
                if j == 999:
                    n_map2[n].append(1000)

plt.figure(figsize=(7, 7))
print(count)

for n in n_map1:
    average1 = sum(n_map1[n])/len(n_map1[n])
    std_dev1 = (sum([(x - average1)**2 for x in n_map1[n]])/len(n_map1[n]))**0.5

    if (len(n_map2[n]) != 0):
        average2 = sum(n_map2[n])/len(n_map2[n])
        std_dev2 = (sum([(x - average2)**2 for x in n_map2[n]])/len(n_map2[n]))**0.5

    #print(n, average1, std_dev1, average2, std_dev2)

    # grouped bar chart

    plt.bar(n - 1.5, average1, width=3, yerr=std_dev1, label='Baseline', color='lightblue', ecolor='cornflowerblue', capsize=5)
    plt.bar(n + 1.5, average2, width=3, yerr=std_dev2, label='Autoencoder', color='deepskyblue', ecolor='cornflowerblue', capsize=5)

# Line of best fit
ticks = [i for i in range(10, 41, 10)]
ticks = np.array(ticks)

plt.ylabel('Time (Steps)')
plt.xlabel('Number of Agents')
plt.xticks(ticks)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.ylim(0, 1000)
#plt.yscale('log')
plt.savefig('scenario2_decision6.png')

