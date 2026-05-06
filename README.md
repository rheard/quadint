# quadint

Fast, integer-backed algebraic number types for **exact** arithmetic in imaginary quadratic integer rings.

- **`complexint`**: a Gaussian integer type that mirrors Python’s `complex`, but stores **`int`** components (no floating-point drift).
- **`QuadInt` / `QuadraticRing`**: a general quadratic-integer implementation for elements of the form  
  $(a + b\sqrt{D}) / \mathrm{den}$ with $den ∈ {1,2}$. 
  - By default, `QuadraticRing(D)` chooses `den = 2` when `D % 4 == 1`, otherwise `den = 1` (and you can override with `QuadraticRing(D, den=1)` / `den=2` to work in a non-default order).
- **`eisensteinint`**: Eisenstein integers in the ω-basis (`a + bω`, where $ω = (-1 + \sqrt{-3})/2$).
- **`dualint`**: dual integers of the form `a + bε` where **`ε² = 0`** and **`ε != 0`**.
- **`splitint`**: split-complex (hyperbolic) integers of the form `a + bj` where **`j² = 1`** and **`j != 1`**.
 
Designed for discrete math, number theory tooling, and high-throughput exact computations (this project is built to compile cleanly with **mypyc**).

New helper methods on every quadratic integer value:

* `x.content()` — largest positive integer `n` such that `x = n*y` in the same ring.
* `x.factor_detail()` — structured factorization as `Factorization(unit, primes)`.
* `x.factor()` — a plain `{prime_like_factor: exponent}` mapping whose product is exactly `x`.
* `x.basis`, `x.basis_a`, `x.basis_b` — public/user-facing basis coordinates, which may differ from the internal `(a, b)` numerator coordinates.

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
* Division helpers (`divmod`, `//`, `%`, `/`) are implemented for the finite set of **norm-Euclidean** quadratic rings **at the default/maximal denominator**, and also for **dual** (`D=0`) and **split-complex integers** (`D=1`).
  * New: division is also available in selected **Euclidean-but-not-norm-Euclidean real quadratic maximal orders** via a Harper-style method (weighted Euclidean score + local quotient search). This currently covers:
    * `D=14,22,23,31,43,46,47,53,59,61,62,67,71,77,83,86,89,93,94,97`
    * and `D=69` via a dedicated Clark-style Euclidean function implementation.
    * Without `cypari`, Harper-style support is limited to the built-in hard-coded cases above (with `D < 100`); with `cypari` installed, additional admissible Harper-like cases may be discoverable.
* Factorization (`factor` / `factor_detail`) is currently implemented for:
  * `complexint` (`D=-1, den=1`),
  * `QuadraticRing(-2, den=1)`,
  * `eisensteinint` (`D=-3, den=2`),
  * and the Heegner maximal orders for `D=-7` and `D=-11`.
  Other rings may raise `NotImplementedError`.
* Floats and Python `complex` are accepted in some operations but are converted via `int(...)`, which truncates toward zero. If you care about rationals, avoid mixing in `float`.

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

## Basis-vector coordinates

`QuadInt` separates public basis coordinates from the internal numerator coordinates used by the arithmetic engine. 
  Internally, every value is still stored as `(a, b)` numerators for $(a + b\sqrt{D}) / \mathrm{den}$

For most quadratic integer types, the public basis is the identity basis, 
  so the coordinates you pass to the constructor are the same coordinates used internally. 
  Subclasses can override that by defining conversion matrices:

* `BASIS_TO_INTERNAL` maps constructor/user coordinates `(x, y)` into internal numerator coordinates `(a, b)`.
* `INTERNAL_TO_BASIS` and `INTERNAL_TO_BASIS_DEN` map internal numerator coordinates back to public basis coordinates.

This is mainly useful when the natural mathematical notation for a type is not the raw `1, √D` basis. 
  Eisenstein integers are the motivating example. Users write them as `a + bω`, where $ω = (-1 + \sqrt{-3})/2$,
  but the shared quadratic-integer engine stores values over `QuadraticRing(-3)` as $(a + b\sqrt{D}) / \mathrm{den}$.

So `eisensteinint(x, y)` converts from the public ω-basis to the internal numerator basis as:

```text
x + yω = ((2x - y) + y√-3) / 2
```

Example:

```python
from quadint.eisenstein import eisensteinint

z = eisensteinint(2, 3)

print(z)             # (2+3ω)
print(z.real)        # 2
print(z.omega)       # 3
print(z.basis)       # (2, 3)
print(tuple(z))      # (2, 3)

# Internal numerator coordinates are still available, but usually only useful
# for implementing rings/subclasses or debugging low-level arithmetic.
print(z.a, z.b, z.ring.den)  # 1 3 2
```

Prefer `basis`, `basis_a`, `basis_b`, and type-specific aliases such as `real` / `omega` when presenting values to users. Prefer the internal `.a` and `.b` fields only when implementing arithmetic, division, factorization, or another low-level ring operation.

---

## Sums of squares and quadratic-form decompositions: `quadint.sums`

`quadint.sums` provides small number-theory helpers for decomposing primes and integers into non-negative integer solutions of:

```text
x^2 + d*y^2 = n
```

The default `d=1` gives the classic sum-of-two-squares problem.

```python
from quadint.sums import decompose_prime, decompose_number

print(decompose_prime(19889))
# (17, 140)

print(decompose_number(19890))
# {(69, 123), (57, 129), (3, 141), (87, 111)}
```

Use `d` to solve related forms:

```python
from quadint.sums import decompose_prime, decompose_number

print(decompose_prime(19, d=3))
# (4, 1) because 4^2 + 3*1^2 == 19

print(decompose_number(12, d=3, no_trivial_solutions=False))
# {(0, 2), (3, 1)} because 0^2 + 3*2^2 == 12 and 3^2 + 3*1^2 == 12
```

### `decompose_prime(p, d=1, den=1)`

Return a non-negative pair `(x, y)` for a prime-like input where:

```text
x^2 + d*y^2 = den^2 * p
```

For normal public use, leave `den=1`. Passing `den=2` is mainly useful when working with denominator-2 quadratic orders, where the returned pair is in numerator coordinates.

```python
from quadint.sums import decompose_prime

print(decompose_prime(5))
# (1, 2)

print(decompose_prime(7, d=3))
# (2, 1)

print(decompose_prime(2, d=7, den=2))
# (1, 1) because 1^2 + 7*1^2 == 2^2 * 2
```

### `decompose_number(n, d=1, ...)`

Return all canonical non-negative integer pairs `(x, y)` satisfying:

```text
x^2 + d*y^2 = n
```

```python
from quadint.sums import decompose_number

print(decompose_number(325, no_trivial_solutions=False))
# {(1, 18), (6, 17), (10, 15)}
```

`decompose_number` accepts either an integer or a precomputed factorization dictionary:

```python
from quadint.sums import decompose_number

print(decompose_number({2: 1, 3: 2, 5: 1, 13: 1, 17: 1}))
# same result as decompose_number(19890)
```

Useful options:

* `d=1` by default; use another positive integer for `x^2 + d*y^2 = n`.
* `no_trivial_solutions=True` by default; set it to `False` to include solutions with a zero coordinate and symmetric `d=1` solutions such as `(0, 2)` for `n=4`.
* `check_count=N` returns an empty set early when the predicted number of solutions is below `N`.

Completeness is best-supported for the class-number-one Heegner values used by the package: `d in {1, 2, 3, 7, 11, 19, 43, 67, 163}`. Other `d` values may work, and results are still validated as true solutions, but completeness is not guaranteed.

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
* `x.units` (finite torsion unit subgroup exposed as a tuple)
* `x.content()`
* `x.factor_detail()` (returns `Factorization(unit, primes)`)
* `x.factor()` (returns plain `dict[QuadInt, int]`)
* `divmod(x, y)`, `x // y`, `x % y` (where supported)
* Iteration/indexing over the stored coefficients: `list(x)`, `x[0]`, `x[1]`


### `quadint.sums`

* `decompose_prime(p: int, d: int = 1, den: int = 1) -> tuple[int, int]`
* `decompose_number(n: int | dict[int, int], d: int = 1, check_count: int | None = None, *, limited_checks: bool = False, no_trivial_solutions: bool = True, warn: bool = True) -> set[tuple[int, int]]`
