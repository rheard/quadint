# quadint

Fast, integer-backed algebraic number types for **exact** arithmetic in imaginary quadratic integer rings.

- **`complexint`**: a Gaussian integer type that mirrors Python’s `complex`, but stores **`int`** components (no floating-point drift).
- **`QuadInt` / `QuadraticRing`**: a general quadratic-integer implementation for elements of the form  
  $(a + b\sqrt{D}) / \mathrm{den}$ with $den ∈ {1,2}$.
- **`eisensteinint`**: Eisenstein integers in the ω-basis (`a + bω`, where `ω = (-1 + √-3)/2`).
- **`dualint`**: dual integers of the form `a + bε` where **`ε² = 0`** and **`ε != 0`** (useful for exact first-order / automatic-differentiation-style arithmetic).

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

## Quadratic integers: `make_quadint`

Create a ring instance for a chosen discriminant parameter `D`, then construct values in that ring:

```python
from quadint import make_quadint

Q2 = make_quadint(-2)  # Z[√-2]

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

## Division & interoperability notes

* This package is primarily intended for **exact, discrete** arithmetic (`+`, `-`, `*`, `**`, conjugation, norms).
* Some division helpers exist (e.g. `divmod`, `//`, `%`) for **imaginary** quadratic rings, using a nearest-lattice approach; it may be `NotImplemented` for `D ≥ 0`, and behavior depends on the ring being Euclidean enough for your use case.
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
* `make_quadint(D: int) -> QuadraticRing`

### Ring instance (`QuadraticRing`)

* `Q(a: int = 0, b: int = 0) -> QuadInt` (constructs using the ring’s internal basis)
* `Q.from_ab(a: int, b: int) -> QuadInt` (construct with user coords, respecting `den`)
* `Q.from_obj(x) -> QuadInt` (embed `int`/`float`, and `complex` only when `D == -1`)

### Value type (`QuadInt`)

* `x.conjugate()`
* `abs(x)` (norm)
* `divmod(x, y)`, `x // y`, `x % y` (where supported)
* Iteration/indexing over the stored coefficients: `list(x)`, `x[0]`, `x[1]`
