import numpy as np
r = np.load("ratios.npy")
x = np.array(range(len(r)))
x = np.array(range(len(r)))
import matplotlib.pyplot as plt
plt.plot(x,r)
plt.savefig("ratios.svg")
