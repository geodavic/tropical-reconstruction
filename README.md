# Zonotope Matching and Approximation for Neural Networks

*Copyright George D. Torres 2023.*

Submitted as supplementary material for "Zonotope Matching and Approximation for Neural Networks", a thesis submitted in partial fulfillment for the requirements of Doctor of Philosophy at the University of Texas at Austin.

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
>>> f
1 + 2•xy + 3•x^2y
```

To evaluate this polynomial at a point `[x,y]`:
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
>>> f
10 + 12•x^4 + 9•x^2y + 11•x^6y + 14•x^2y^4 + 16•x^6y^4 + 13•x^4y^5 + 15•x^8y^5 + (-5)•y^6 + (-3)•x^4y^6 + (-6)•x^2y^7 + (-4)•x^6y^7 + (-1)•x^2y^10 + 1•x^6y^10 + (-2)•x^4y^11 + 0•x^8y^11
```

`TropicalPolynomial` objects can be (tropically) added and multiplied:

```python
>>> g = TropicalPolynomial([(1,2),(2,2)],[-1,-1])
>>> h = f+g
>>> h
(-1)•xy^2 + (-1)•x^2y^2 + 10 + 12•x^4 + 9•x^2y + 11•x^6y + 14•x^2y^4 + 16•x^6y^4 + 13•x^4y^5 + 15•x^8y^5 + (-5)•y^6 + (-3)•x^4y^6 + (-6)•x^2y^7 + (-4)•x^6y^7 + (-1)•x^2y^10 + 1•x^6y^10 + (-2)•x^4y^11 + 0•x^8y^11
>>> h = f*g
>>> h
9•x^2y^2 + 11•x^6y^2 + 8•x^4y^3 + 10•x^8y^3 + 13•x^4y^6 + 15•x^8y^6 + 12•x^6y^7 + 14•x^10y^7 + (-6)•x^2y^8 + (-4)•x^6y^8 + (-7)•x^4y^9 + (-5)•x^8y^9 + (-2)•x^4y^12 + 0•x^8y^12 + (-3)•x^6y^13 + (-1)•x^10y^13 + 9•xy^2 + 11•x^5y^2 + 8•x^3y^3 + 10•x^7y^3 + 13•x^3y^6 + 15•x^7y^6 + 12•x^5y^7 + 14•x^9y^7 + (-6)•xy^8 + (-4)•x^5y^8 + (-7)•x^3y^9 + (-5)•x^7y^9 + (-2)•x^3y^12 + 0•x^7y^12 + (-3)•x^5y^13 + (-1)•x^9y^13
```

A tropical polynomial can be "simplified" by removing all inactive monomials, yielding a new polynomial that is pointwise equal:

```python
>>> h = TropicalPolynomial([(0,0),(2,0),(0,2),(1,1)],[1,1,1,0])
>>> h1 = h.simplify()
>>> h1
1.0 + 1.0•x^2.0 + 1.0•y^2.0
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
>>> N = NeuralNetworkExampleE.NN
>>> [h] = N.tropical()
>>> h.numerator
6.0•x^6.0y^4.0 + 3.0•x^2.0y^16.0 + 7.0•x^6.0y^12.0 + 8.5•x^2.0y^12.0 + 4.5•x^4.0y^5.0 + 1.5•y^17.0 + 5.5•x^4.0y^13.0 + 7.0•y^13.0 + 3.5•x^6.0y^6.0 + 0.5•x^2.0y^18.0 + 4.5•x^6.0y^14.0 + 6.0•x^2.0y^14.0 + 2.0•x^4.0y^7.0 + (-1.0)•y^19.0 + 3.0•x^4.0y^15.0 + 4.5•y^15.0 + 1.5•x^4.0y^8.0 + 0.0•x^8.0y^8.0
>>> h.denominator
9.0•x^6.0y^4.0 + 6.0•x^2.0y^16.0 + 2.0•x^6.0y^12.0 + 11.0•x^2.0y^12.0 + 7.5•x^4.0y^5.0 + 4.5•y^17.0 + 0.5•x^4.0y^13.0 + 9.5•y^13.0 + 6.5•x^6.0y^6.0 + 3.5•x^2.0y^18.0 + (-0.5)•x^6.0y^14.0 + 8.5•x^2.0y^14.0 + 5.0•x^4.0y^7.0 + 2.0•y^19.0 + (-2.0)•x^4.0y^15.0 + 7.0•y^15.0
```

## Converting Between Tropical Polynomials and Polynomial Neural Networks

To convert from a `PolynomialNeuralNetwork` to a `TropicalPolynomial`, use the `.tropical()` method as well. This will return a list of tropical polynomials, one for each component.

```python
>>> from tropical_reconstruction.examples import NeuralNetworkExampleB
>>> N = NeuralNetworkExampleB.NN
>>> [f] = N.tropical()
>>> f
10.0 + (-2.0)•x^4.0y^11.0 + 0.0•x^8.0y^11.0 + (-1.0)•x^2.0y^10.0 + (-5.0)•y^6.0 + 13.0•x^4.0y^5.0 + 15.0•x^8.0y^5.0 + 14.0•x^2.0y^4.0 + 16.0•x^6.0y^4.0 + 11.0•x^6.0y + 12.0•x^4.0
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
