import matplotlib.pyplot as plt
import csv
import numpy as np

from scipy.interpolate import make_interp_spline, BSpline

plt.figure(figsize=(7, 7))

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    # https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int32(window_size))
        order = np.abs(np.int32(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

map = dict()
with open('losses.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

    label = "None"
    for i in range(len(data)):
        if i % 2 == 0:
            label = data[i][0]
        else:
            x = np.array(data[i]).astype(float)
            if label not in map:
                map[label] = list()
            
            map[label].append(x)

for label in map:
    x = np.mean(map[label], axis=0)
    xhat = savitzky_golay(x, 51, 3)
    plt.plot(range(x.size), xhat, label=label)
            

plt.ylabel('Loss (BCE)')
plt.xlabel('Time (epochs)')
plt.legend()
plt.savefig('losses.png')

