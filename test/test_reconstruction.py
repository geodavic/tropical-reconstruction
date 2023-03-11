from examples import RandomNeuralNetwork
from function import test_equal
import tqdm


def test_composition(d, n, num):
    """Generate a random neural network N, convert it to a
    tropical polynomial f, and the convert f back to a neural
    network N2 (using TZLP framework). Then check N == N2 pointwise
    everywhere. Repeat num times.
    """
    for _ in tqdm.tqdm(range(num)):

        N = RandomNeuralNetwork((d, n, 1)).NN
        f = N.tropical()[0]
        N2 = f.neural_network(verbose=False)
        if N2 is not None:
            passed = test_equal(N, f, d) and test_equal(f, N2, d)
            if not passed:
                print("Failed equality test")
        else:
            print("Unable to solve for N")
            passed = False

        if not passed:
            break

    if not passed:
        print("Problem data:")
        print(N.weights)
        print(N.thresholds)
    else:
        print("All examples passed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=int, required=False, default=2)
    parser.add_argument("-n", type=int, required=False, default=4)
    parser.add_argument("-num", type=int, required=False, default=50)
    args = parser.parse_args()

    print(
        f"Testing TZLP solver on {args.num} random ({args.d},{args.n},1) architectures..."
    )
    test_composition(args.d, args.n, args.num)
