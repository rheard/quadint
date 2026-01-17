This library provides a `QuadInt` class for dealing with Quadratic integers.

Additionally, a `complexint` class is provided by default where D == -1 obviously. 
    This class acts just like Python's built-in `complex` except it is backed by Python integers instead of doubles.
    This allows for infinite precision.

## Examples

```python
from quadint import complexint

a = complexint(1, 2)
b = complexint(3, 6)

c = a * b
print(c)  # Outputs "-9+12j"

print(type(c.real))  # Outputs "int"
```

```python
from quadint import make_quadint

Q2 = make_quadint(-2)

a = Q2(1, 2)
b = Q2(3, 6)

c = a * b
print(c)  # Outputs "(-21+12*sqrt(-2))"
```

## Disclaimer

This is intended for use with discrete mathematics, and ideally will be limited to the
    operations: add, sub, mul, and pow.

Trying to divide using this class, or using floats with this class, will (probably) result in integer conversion cutoff.

As an example of this problem, note the equivalences below:
```python
from quadint import complexint

a = complexint(3, 6)

print(a / 3)  # Outputs "(1+2j)
print(a / 3.5)  # Outputs "(1+2j)"

print(a + 1)  # Outputs "(4+6j)"
print(a + 1.5)  # Outputs "(4+6j)"
```