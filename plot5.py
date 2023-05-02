import matplotlib.pyplot as plt
import csv


n_map1 = dict()
n_map2 = dict()

threshold = 0.5

with open('decisions_final.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

    for i in range(len(data)):
        if data[i][0] == 'n':
            n = int(data[i][1])
            if n not in n_map1:
                n_map1[n] = list()
            if n not in n_map2:
                n_map2[n] = list()
            
            for j in range(500):
                total = 0
                for decision in data[i + j + 1]:
                    if decision == '0':
                        total += 1
                
                print(total, n, total/n)

                if total/n >= threshold or (total == 1 and n == 1):
                    n_map1[n].append(j)
                    break
            
                if j == 499:
                    n_map1[n].append(500)
            
            for j in range(500):
                total = 0
                for decision in data[i + j + 501]:
                    if decision == '1':
                        total += 1

                if total/n >= threshold or (total == 1 and n == 1):
                    n_map2[n].append(j)
                    break

                if j == 499:
                    n_map2[n].append(500)

plt.figure(figsize=(7, 7))
print(n_map1[10], len(n_map1[10]))

for n in n_map1:
    average1 = sum(n_map1[n])/len(n_map1[n])
    std_dev1 = (sum([(x - average1)**2 for x in n_map1[n]])/len(n_map1[n]))**0.5

    average2 = sum(n_map2[n])/len(n_map2[n])
    std_dev2 = (sum([(x - average2)**2 for x in n_map2[n]])/len(n_map2[n]))**0.5

    #print(n, average1, std_dev1, average2, std_dev2)

    # grouped bar chart

    plt.bar(n - 1, average1, width=2, yerr=std_dev1, label='Phase 1', color='blue')
    plt.bar(n + 1, average2, width=2, yerr=std_dev2, label='Phase 2', color='lightblue')

plt.ylabel('Time (Steps)')
plt.xlabel('Number of Agents')
plt.title('Time to Consensus')
ticks = [i for i in range(0, 101, 10)]
ticks[0] = 1
plt.xticks(ticks)
#plt.yscale('log')
plt.savefig('decision_time.png')

