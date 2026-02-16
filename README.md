# quadint

Fast, integer-backed algebraic number types for **exact** arithmetic in imaginary quadratic integer rings.

- **`complexint`**: a Gaussian integer type that mirrors Python’s `complex`, but stores **`int`** components (no floating-point drift).
- **`QuadInt` / `QuadraticRing`**: a general quadratic-integer implementation for elements of the form  
  $(a + b\sqrt{D}) / \mathrm{den}$ with $den ∈ {1,2}$. 
  - By default, `QuadraticRing(D)` chooses `den = 2` when `D % 4 == 1`, otherwise `den = 1` (and you can override with `QuadraticRing(D, den=1)` / `den=2` to work in a non-default order).
- **`eisensteinint`**: Eisenstein integers in the ω-basis (`a + bω`, where `ω = (-1 + √-3)/2`).
- **`dualint`**: dual integers of the form `a + bε` where **`ε² = 0`** and **`ε != 0`**.
- **`splitint`**: split-complex (hyperbolic) integers of the form `a + bj` where **`j² = 1`** and **`j != 1`**.
 
Designed for discrete math, number theory tooling, and high-throughput exact computations (this project is built to compile cleanly with **mypyc**).

---

## Installation

```bash
python -m pip install quadint
```

---

## Quickstart (recommended): `complexint`

```python
from quadint import complexint

a = complexint(1, 2)
b = complexint(3, 6)

c = a * b
print(c)          # "(-9+12j)"  (exact, integer-backed)
print(c.real)     # -9
print(c.imag)     # 12
print(type(c.real))  # <class 'int'>

print(abs(a))     # 1^2 + 2^2 = 5  (norm)
```

`complexint` is ideal when you want something that *feels like* `complex`, but with infinite-precision integer components.

---

## Quadratic integers: `QuadraticRing`

Create a ring instance for a chosen discriminant parameter `D`, then construct values in that ring:

```python
from quadint import QuadraticRing

Q2 = QuadraticRing(-2)  # Z[√-2]

x = Q2(1, 2)           # (1 + 2*sqrt(-2))
y = Q2(3, 6)

print(x * y)           # "(-21+12*sqrt(-2))"
print(abs(x))          # norm: 1^2 - (-2)*2^2 = 9
```

Common operations include `+`, `-`, `*`, `**` (non-negative powers), `conjugate()`, and `abs()` (the norm).

---

## Eisenstein integers: `eisensteinint`

```python
from quadint.eisenstein import eisensteinint

z = eisensteinint(2, 3)   # 2 + 3ω
w = eisensteinint(1, -1)  # 1 - ω

print(z)
print(z * w)              # exact product in Z[ω]
print(abs(z))             # norm (integer)
```

Use `real` and `omega` to access the ω-basis components.

---

## Dual integers: `dualint`

```python
from quadint import dualint

z = dualint(2, 3)   # 2 + 3ε
w = dualint(1, -1)  # 1 - ε

print(z)
print(z * w)              # (2+1ε)
```

Use `real` and `dual` (or `epsilon`) to access the ε-basis components.

---
 
## Split-complex integers: `splitint`

Split-complex (a.k.a. *hyperbolic*) integers behave like `complexint`, except the generator satisfies **`j² = 1`** instead of **`j² = -1`**.

Unlike complex numbers, split-complex numbers have an **indefinite norm** and **zero divisors** (e.g. `(1+j)*(1-j) == 0`).

```python
from quadint.split import splitint

z = splitint(1, 1)    # 1 + 1j
w = splitint(1, -1)   # 1 - 1j

print(z * w)          # 0j   (zero divisor behavior)
```

---


## Division & interoperability notes

* This package is primarily intended for **exact, discrete** arithmetic (`+`, `-`, `*`, `**`, conjugation, norms).
* Division helpers (`divmod`, `//`, `%`, `/`) are implemented for **dual numbers** (`D=0`) and **split-complex integers** (`D=1`), and for the finite set of **norm-Euclidean** quadratic rings (including selected `D<0` and `D>0`) **at the default/maximal denominator**; outside these cases division may raise `NotImplementedError` (and `ZeroDivisionError` for zero divisors).
* **Floats and Python `complex` are accepted in some operations but are converted via `int(...)`, which truncates toward zero. If you care about rationals, avoid mixing in `float`.

Example of truncation behavior:

```python
from quadint import complexint

a = complexint(3, 6)

print(a / 3)     # "(1+2j)"
print(a / 3.5)   # "(1+2j)"  (3.5 -> 3 by int(...) conversion)

print(a + 1)     # "(4+6j)"
print(a + 1.5)   # "(4+6j)"  (1.5 -> 1)
```

---

## Minimal API overview

### Constructors

* `complexint(a: int = 0, b: int = 0)`
* `eisensteinint(a: int = 0, b: int = 0)` where `a + bω`
* `dualint(a: int = 0, b: int = 0)`
* `splitint(a: int = 0, b: int = 0)`
* `QuadraticRing(D: int = 0, den: int = None)`
  * If `den` is omitted (`None`), it defaults to `2` when `D % 4 == 1`, otherwise `1`.

### Ring instance (`QuadraticRing`)

* `Q(a: int = 0, b: int = 0) -> QuadInt` (constructs using the ring’s internal basis)
* `Q.from_ab(a: int, b: int) -> QuadInt` (construct with user coords, respecting `den`)
* `Q.from_obj(x) -> QuadInt` (embed `int`/`float`, and `complex` only when `D == -1`)

### Value type (`QuadInt`)

* `x.conjugate()`
* `abs(x)` (norm)
* `divmod(x, y)`, `x // y`, `x % y` (where supported)
* Iteration/indexing over the stored coefficients: `list(x)`, `x[0]`, `x[1]`
