# Solving the reconstruction problem for tropical neural networks

This is a repository implementing the algorithms presented in "Tropical and Convex Geometry of Depth Two ReLU Neural Networks." We assume the user is familiar with the concepts and notation therein. 

# Installing

Install using pip:
```
pip install -e .
```

# Examples

Here are some example uses of the code that demonstrate capability.

## Tropical Polynomials

To create a tropical polynomial, use the `TropicalPolynomial` class in `function.py`, and pass the list of monomials and the list of their coefficients. Polynomials in arbitrary numbers of variables are allowed.

```python
>>> from tropical_recionstruction.function import TropicalPolynomial
>>> f = TropicalPolynomial([(0,0),(1,1),(2,1)],[1,2,3])
```

This is the polynomial `1+2xy+3x^2y`. To evaluate this polynomial at a point `[x,y]`:
```python
>>> f([1.2,5])
10.4
```

To get the list of monomials and coefficients as a dictionary:
```python
>>> f.poly()
{(0, 0): 1, (1, 1): 2, (2, 1): 3}
```

To get the coefficient of a given monomial:
```python
>>> f.coef((1,1))
2
>>> f.coef((1,0))
-inf
```

There is a larger example polynomial in `example.py`:

```python
>>> from tropical_reconstruction.examples import TropicalExampleB
>>> f = TropicalExampleB.f
>>> f.poly
{(0, 0): 10, (4, 0): 12, (2, 1): 9, (6, 1): 11, (2, 4): 14, (6, 4): 16, (4, 5): 13, (8, 5): 15, (0, 6): -5, (4, 6): -3, (2, 7): -6, (6, 7): -4, (2, 10): -1, (6, 10): 1, (4, 11): -2, (8, 11): 0}
```

## Polynomial Neural Networks

Recall a Polynomial Neural Network is a deep ReLU activated homogenous neural network with non-negative weights. These can be constructed through the `PolynomialNeuralNetwork` class in `function.py`. Simply specify the parameters A = (A^1,A^2,...,A^L) and t = (t^1,t^2,...,t^L) as numpy arrays. For example, the following is a network of architecture `(2,4,1)`:

```python
>>> from tropical_reconstruction.function import PolynomialNeuralNetwork
>>> import numpy as np
>>> A1 = np.array([[2,0],[2,1],[1,2],[0,2]]) # weights
>>> A2 = np.array([[2,1,2,3]]) # weights
>>> t1 = np.array([-1,1,-2,5]) # thresholds
>>> t2 = np.array([5]) # thresholds
>>> N = PolynomialNeuralNetwork([A1,A2],[t1,t2])
```

Evaluate the neural network at a point `[x,y]`:
```python
>>> N([3,5])
array([79])
```

To get the weights and thresholds of a network:
```python
>>> N.weights
[array([[2, 0],
       [2, 1],
       [1, 2],
       [0, 2]]), array([[2, 1, 2, 3]])]
>>> N.thresholds
[array([-1,  1, -2,  5]), array([5])]
```

You can get the `i`th component of a neural network, which will return another instance of `PolynomialNeuralNetwork`. In this case, `N` has output dimension 1, so the 1st component is `N` itself:

```python
>>> N1 = N.component(1)
>>> N == N1
True
```

## Converting Between Tropical Polynomials and Polynomial Neural Networks

To convert from a `PolynomialNeuralNetwork` to a `TropicalPolynomial`, use the `.tropical()` method, which returns a list of tropical polynomials (one for each out put component of the network):

```python
>>> from tropical_reconstruction.examples import NeuralNetworkExampleB
>>> N = NeuralNetworkExampleB.NN
>>> [f] = N.tropical()
>>> f.poly
{(0, 0): 10, (4, 0): 12, (2, 1): 9, (6, 1): 11, (2, 4): 14, (6, 4): 16, (4, 5): 13, (8, 5): 15, (0, 6): -5, (4, 6): -3, (2, 7): -6, (6, 7): -4, (2, 10): -1, (6, 10): 1, (4, 11): -2, (8, 11): 0}
```

We can verify that `f` and `N` are pointwise equal using the `test_equal` function (which samples 10000 points from the domain and checks if `f` and `N` are equal at these points).

```python
>>> from tropical_reconstruction.function import test_equal
>>> test_equal(f,N,2)
True
```


To convert from a `TropicalPolynomial` to a `PolynomialNeuralNetwork`, use the `.neural_network()` method. This is what uses the main `TZLP` algorithm, and by default it will print the steps of the algorithm.

```python
>>> from tropical_reconstruction.examples import TropicalExampleB
>>> f = TropicalExampleB.f
>>> N = f.neural_network()
Solving for primal variable x...
Checking that TLP is feasible at x...
Checking that SLP is feasible at x...
Success! Solution to the TZLP:
```

Once again, we can verify that `f` and `N` are pointwise equal:

```
>>> from tropical_reconstruction.function import test_equal
>>> test_equal(f,N,2)
True
```


# TODO

- Handle architectures of the form `(1,n,1)` in `TropicalPolynomial.neural_network()`
- Implement `_get_zonotope` method for `TropicalPolynomial`
- unit tests
- documentation for polytopes and optimization
