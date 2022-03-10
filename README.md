# Solving the reconstruction problem for tropical neural networks

This is a repository implementing the algorithms presented in "Tropical and Convex Geometry of Depth Two ReLU Neural Networks." We assume the user is familiar with the concepts and notation therein. 

# Installing

You will need python 3.5+ and the libraries `tqdm`,`scipy`, and `numpy`. Assuming you have python already installed, these libraries can be installed with:
```
python3 -m pip install -r requirements.txt
```

# Examples

Here are some example uses of the code that demonstrate capability. This depends on 

## Tropical Polynomials

To create a tropical polynomial, use the `TropicalPolynomial` class in `function.py`, and pass the list of monomials and the list of their coefficients.

```python
>>> from function import TropicalPolynomial
>>> f = TropicalPolynomial([(0,0),(1,1),(2,1)],[1,2,3])
```

This is the polynomial `1+2xy+3x^2y`. To evaluate this polynomial at a point:
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

## Polynomial Neural Networks



## Converting Between Tropical Polynomials and Polynomial Neural Networks
