# Solving the reconstruction problem for tropical neural networks

This is a repository implementing the algorithms presented in "Zonotope Matching and Approximation for Neural Networks" We assume the user is familiar with the concepts and notation therein. 

# Installing

Install using pip:
```
pip install -e .
```

or using poetry:
```
poetry install
```

# Examples

Here are some example uses that demonstrate capability.

## Tropical Polynomials

To create a tropical polynomial, use the `TropicalPolynomial` class, and pass the list of monomials and the list of their coefficients. Polynomials in arbitrary numbers of variables are allowed.

```python
>>> from tropical_recionstruction.function import TropicalPolynomial
>>> f = TropicalPolynomial([(0,0),(1,1),(2,1)],[1,2,3])
```
(make sure you are in the root directory when you run this, or else it will not know what `function` is)

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

`TropicalPolynomial` objects can be (tropically) added and multiplied:

```python
>>> g = TropicalPolynomial([(1,2),(2,2)],[-1,-1])
>>> h = f+g
>>> h.poly
{(1, 2): -1, (2, 2): -1, (0, 0): 1, (1, 1): 2, (2, 1): 3}
>>> h = f*g
>>> h.poly
(2, 2): 0, (3, 3): 2, (4, 3): 2, (1, 2): 0, (2, 3): 1}
```

A tropical polynomial can be "simplified" by removing all inactive monomials, yielding a new polynomial that is pointwise equal:

```python
>>> h = TropicalPolynomial([(0,0),(2,0),(0,2),(1,1)],[1,1,1,0])
>>> h1 = h.simplify()
>>> h1.poly
{(0.0, 0.0): 1.0, (2.0, 0.0): 1.0, (0.0, 2.0): 1.0}
```

To compute a tropical power of a polynomial f^a, use `.power(a)`. Passing `lazy=True` will compute the simplification of the tropical power, denoted f^[a]. This is generally much faster.

## ReLU Neural Networks

Homogenous ReLU networks are implemented in `function.py`. Specify the paramters A = (A^1,A^2,...,A^L) and t = (t^1,t^2,...,t^L) as numpy arrays. Recall a Polynomial Neural Network is ReLU network with A >= 0. These can be constructed through the `PolynomialNeuralNetwork` class in `function.py`. Specify the parameters A = (A^1,A^2,...,A^L) and t = (t^1,t^2,...,t^L) as numpy arrays. For example, the following is a network of architecture `(2,4,1)`:

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

## Converting A ReLU Network to a Tropical Rational Function

To convert a `NeuralNetwork` object into a `TropicalRationalFunction`, use the `.tropical()` method, which returns a list of tropical rational functions (one for each component of the network).

```python
>>> from tropical_reconstruction.examples import NeuralNetworkExampleE
>>> N = NeuralNetworkExampleE
>>> [h] = N.tropical()
>>> h.numerator.poly
{(6.0, 4.0): 6.0, (2.0, 16.0): 3.0, (6.0, 12.0): 7.0, (2.0, 12.0): 8.5, (4.0, 5.0): 4.5, (0.0, 17.0): 1.5, (4.0, 13.0): 5.5, (0.0, 13.0): 7.0, (6.0, 6.0): 3.5, (2.0, 18.0): 0.5, (6.0, 14.0): 4.5, (2.0, 14.0): 6.0, (4.0, 7.0): 2.0, (0.0, 19.0): -1.0, (4.0, 15.0): 3.0, (0.0, 15.0): 4.5, (4.0, 8.0): 1.5, (8.0, 8.0): 0.0}
>>> h.denominator.poly
{(6.0, 4.0): 9.0, (2.0, 16.0): 6.0, (6.0, 12.0): 2.0, (2.0, 12.0): 11.0, (4.0, 5.0): 7.5, (0.0, 17.0): 4.5, (4.0, 13.0): 0.5, (0.0, 13.0): 9.5, (6.0, 6.0): 6.5, (2.0, 18.0): 3.5, (6.0, 14.0): -0.5, (2.0, 14.0): 8.5, (4.0, 7.0): 5.0, (0.0, 19.0): 2.0, (4.0, 15.0): -2.0, (0.0, 15.0): 7.0}
```

## Converting Between Tropical Polynomials and Polynomial Neural Networks

To convert from a `PolynomialNeuralNetwork` to a `TropicalPolynomial`, use the `.tropical()` method as well. This will return a list of tropical polynomials, one for each component.

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

# Zonotope Optimization

Use the script `scripts/example_training_run.py` to run a sample zonotope optimization run. This generates a random polytope P and runs the optimization algorithm with your specified parameters. For example:

```
python scripts/example_training_run.py --rank 5 --dimension 2 --steps 5000 --lr 0.01 --multiplicity_thresh 0.98 --render_last
 ```

The `--rank` parameter is the rank of the zonotope. Passing `--render_last` will generate an image `out.png` of the final zonotope overlaid on P.

# TODO

- Handle architectures of the form `(1,n,1)` in `TropicalPolynomial.neural_network()`
- Implement `_get_zonotope` method for `TropicalPolynomial`
- unit tests
- better documentation
