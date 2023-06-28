from tropical_reconstruction.examples import RandomNeuralNetwork
from tropical_reconstruction.function.prune import prune_network
import numpy as np
import sys

# seed from Example in thesis: 1353678626

if len(sys.argv) > 1:
    seed = int(sys.argv[1])
else:
    seed = np.random.randint(2**32)

print(seed)
np.random.seed(seed)
N = RandomNeuralNetwork((2,4,2,1),MAX=3,convex=False).NN
res = prune_network(N, 6, 8000)
