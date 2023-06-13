import matplotlib.pyplot as plt
import sys
import numpy as np
import glob

files = glob.glob(sys.argv[1]+"*.npy")

for file in files:
    print(file)
    losses = np.load(file)
    x = np.array(range(len(losses)))
    print(min(losses))
    plt.plot(x,losses)

plt.savefig("loss.svg")
