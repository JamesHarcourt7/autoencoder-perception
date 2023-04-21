import matplotlib.pyplot as plt
import csv
import numpy as np

with open('classify/decisions.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)
    line0 = [len([1 for n in d if n == '0']) for d in data]
    line1 = [len([1 for n in d if n == '1']) for d in data]

plt.figure(figsize=(25, 10))
plt.plot(line0, label='0')
plt.plot(line1, label='1')
plt.legend()
plt.savefig('classify/decisions.png')

