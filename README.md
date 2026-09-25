# quadint

Fast, integer-backed algebraic number types for **exact** arithmetic in quadratic integer rings (imaginary and real), plus the dual and split-complex integers.

- **`complexint`**: a Gaussian integer type that mirrors Python’s `complex`, but stores **`int`** components (no floating-point drift).
- **`QuadInt` / `QuadraticRing`**: a general quadratic-integer implementation for elements of the form  
  $(a + b\sqrt{D}) / \mathrm{den}$ with $den ∈ {1,2}$. 
  - By default, `QuadraticRing(D)` chooses `den = 2` when `D % 4 == 1`, otherwise `den = 1` (and you can override with `QuadraticRing(D, den=1)` to work in the non-maximal order $\mathbb{Z}[\sqrt{D}]$). `den=2` is only allowed when `D % 4 == 1`, since otherwise these numbers are not closed under multiplication.
- **`eisensteinint`**: Eisenstein integers in the ω-basis (`a + bω`, where $ω = (-1 + \sqrt{-3})/2$).
- **`dualint`**: dual integers of the form `a + bε` where **`ε² = 0`** and **`ε != 0`**.
- **`splitint`**: split-complex (hyperbolic) integers of the form `a + bj` where **`j² = 1`** and **`j != 1`**.
 
Designed for discrete math, number theory tooling, and high-throughput exact computations (this project is built to compile cleanly with **mypyc**).

Helper methods on every quadratic integer value:

* `x.content()` — largest positive integer `n` such that `x = n*y` in the same ring.
* `x.gcd(y)`, `x.xgcd(y)`, `x.inv_mod(m)`, and `pow(x, e, m)` — gcds and modular arithmetic, in the rings with division (see below).
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

Common operations include `+`, `-`, `*`, `**` (non-negative powers, or negative ones with a modulus, as in `pow(x, -1, m)`), `conjugate()`, and `abs()` (the norm).

The two arguments are the numerators of $(a + b\sqrt{D}) / \mathrm{den}$. That only matters when `den == 2` (the default when `D % 4 == 1`), where `a` and `b` must have the same parity:

```python
from quadint import QuadraticRing

Z5 = QuadraticRing(5)         # den=2, so this is Z[(1 + √5)/2]

phi = Z5(1, 1)                # (1 + √5)/2, the golden ratio
print(phi * phi)              # (3+1*sqrt(5))/2
print(phi * phi == phi + 1)   # True
print(Z5(2, 0) == 1)          # True: 1 is 2/2 here, and Z5(1, 0) raises ValueError
print(Z5.from_ab(1, 1))       # (2+2*sqrt(5))/2, since from_ab takes a + b√D directly
print(Z5.fundamental_unit())  # (1+1*sqrt(5))/2
```

---

## Eisenstein integers: `eisensteinint`

```python
from quadint.eisenstein import eisensteinint

z = eisensteinint(2, 3)   # 2 + 3ω
w = eisensteinint(1, -1)  # 1 - ω

print(z)                  # (2+3ω)
print(z * w)              # (5+4ω), the exact product in Z[ω]
print(abs(z))             # 7, the norm a^2 - ab + b^2
```

Use `real` and `omega` to access the ω-basis components.

---

## Dual integers: `dualint`

```python
from quadint import dualint

z = dualint(2, 3)   # 2 + 3ε
w = dualint(1, -1)  # 1 - ε

print(z)            # (2+3ε)
print(z * w)        # (2+1ε)
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
* Division (`divmod`, `//`, `%`, and `/`, which gives the same rounded quotient as `//`) is implemented for the **Euclidean** maximal orders (the default `den`), and for the **dual** (`D=0`) and **split-complex** (`D=1`) integers. `ring.supports_division()` says whether a ring has it, and rings without it raise `NotImplementedError`. The Euclidean rings are:
  * the norm-Euclidean ones: `D=-1,-2,-3,-7,-11` and `D=2,3,5,6,7,11,13,17,19,21,29,33,37,41,57,73`,
  * `D=69`, via Clark's Euclidean function,
  * and real quadratic rings that are **Euclidean but not norm-Euclidean**, via a Harper-style method (a weighted Euclidean score plus a quotient search). Witnesses are built in for `D=14,22,23,31,43,46,47,53,59,61,62,67,71,77,83,86,89,93,94,97`, and any other real `D` whose maximal order has class number one is checked when the ring is created (`D=38`, `101`, `103`, and so on): it qualifies if its discriminant is at most 500, or if an admissible pair of witness primes turns up below 200.
* In the Harper-style rings, `divmod`, `//` and `%` can raise `NotImplementedError` for some inputs, because the weighted score is not a Euclidean function for every pair (every remainder of `1 + √14` modulo `2` has a larger weighted norm than `2`, for example). `gcd`, `xgcd`, `inv_mod` and `pow(x, e, m)` don't use that search, so they aren't affected.
* `gcd`, `xgcd` and `inv_mod` are available wherever division is, except in two rings that are not PIDs: the dual integers have none of them, and the `den=1` split-complex integers only have `gcd`. A gcd is only defined up to a unit, so it is normalized to a positive leading coefficient (the first quadrant in the Gaussian integers, the first sextant in the Eisenstein integers), and coprime elements have gcd `1`.
* Factorization (`factor` / `factor_detail`) is implemented for the imaginary quadratic fields with class number one:
  * `complexint` (`D=-1`), `QuadraticRing(-2)`, and `eisensteinint` (`D=-3`),
  * and the maximal orders for `D=-7,-11,-19,-43,-67,-163`.
  Other rings raise `NotImplementedError`.
* Floats and Python `complex` are accepted in some operations but are converted via `int(...)`, which truncates toward zero. If you care about rationals, avoid mixing in `float`.
  * Equality is the exception, and is exact: `complexint(1) == 1.9` is `False`. A value that equals a Python number also hashes like it, so `complexint(1)` and `1` are the same dict key (as `1` and `1.0` are), and likewise `complexint(1, 2)` and `1+2j`.

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

## Ideals and class numbers

`quadint` can work with integral ideals of quadratic orders.

```python
from quadint import QuadraticRing

O = QuadraticRing(-5)                         # Z[√-5]
I = O.ideal(3, O(1, 1))                       # (3, 1 + √-5)

print(I.is_prime(), I.is_principal())         # True False
print((I**2).principal_generator())           # (2-1*sqrt(-5)), so I**2 is the principal ideal (2 - √-5)
print([P.norm for P in O.ideal(6).factor()])  # [2, 2, 3, 3]: (6) factors into four prime ideals
print(O.class_number)                         # 2
```

Principal generators come from lattice reduction in imaginary rings and continued fractions in real ones, so this stays fast for large ideals. Class groups (`O.class_group`, `O.class_number`) are available for maximal orders, real and imaginary. Non-maximal orders such as `QuadraticRing(-3, den=1)` raise `NotImplementedError`.

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

### Eisenstein norm decompositions: `quadint.sums.eisenstein`

There is also a small companion module for decomposing Eisenstein norms of the form:

```text
a^2 - a*b + b^2 = n
```

This is mostly a fun helper built on the Eisenstein integer machinery rather than a central part of the package. It mirrors the main `quadint.sums` API:

```python
from quadint.sums.eisenstein import decompose_prime, decompose_number

print(decompose_prime(7))
# (1, 3)  # because 1^2 - 1*3 + 3^2 == 7

print(decompose_number(91, no_trivial_solutions=False))
# {(1, 10), (5, 11)}, the canonical pairs (a, b) with a^2 - a*b + b^2 == 91
```

`no_trivial_solutions=True` filters the obvious square-like rays where `a == 0`, `b == 0`, or `a == b`. As with the rest of `quadint.sums`, a factorization dictionary may be passed instead of an integer.

---

## Minimal API overview

### Constructors

* `complexint(a: int = 0, b: int = 0)`
* `eisensteinint(a: int = 0, b: int = 0)` where `a + bω`
* `dualint(a: int = 0, b: int = 0)`
* `splitint(a: int = 0, b: int = 0)`
* `QuadraticRing(D: int, den: int | None = None)`
  * If `den` is omitted (`None`), it defaults to `2` when `D % 4 == 1`, otherwise `1`. Passing `den=2` for any other `D` raises `ValueError`.

### Ring instance (`QuadraticRing`)

* `Q(a: int = 0, b: int = 0) -> QuadInt` (the numerators of $(a + b\sqrt{D}) / \mathrm{den}$, so with `den=2` they must have the same parity)
* `Q.from_ab(a: int, b: int) -> QuadInt` ($a + b\sqrt{D}$, whatever `den` is)
* `Q.from_obj(x) -> QuadInt` (embed `int`/`float`, and `complex` only in the Gaussian integers)
* `Q.ideal(*generators)`, `Q.prime_ideals_over(p)`, `Q.class_group`, `Q.class_number` (maximal orders)
* `Q.fundamental_unit()` (real rings), `Q.elements_with_norm(n)`, `Q.has_element_with_norm(n)`
* `Q.discriminant()`, `Q.supports_division()`, `Q.supports_factorization()`

### Value type (`QuadInt`)

* `x.conjugate()`
* `abs(x)` (norm)
* `x.units` (finite torsion unit subgroup exposed as a tuple)
* `x.is_unit()`, `x.is_irreducible()`
* `x.content()`
* `x.factor_detail()` (returns `Factorization(unit, primes)`)
* `x.factor()` (returns plain `dict[QuadInt, int]`)
* `divmod(x, y)`, `x // y`, `x % y` (where supported)
* `x.gcd(y)`, `x.xgcd(y)` (returns `(g, s, t)` with `s*x + t*y == g`), `x.inv_mod(m)`, `pow(x, e, m)` (where supported)
* `x.exact_div(y)` (the quotient if `y` divides `x`, otherwise `None`), `y.divides(x)`
* Iteration/indexing over the basis coordinates: `list(x)`, `x[0]`, `x[1]`

### Ideals (`Ideal`)

* `I.norm`, `x in I`, `I * J`, `I**k`, `I // J` (exact quotient), `I.divides(J)`, `I.conjugate()`
* `I.is_prime()`, `I.factor()` (prime ideals, with repeats), `I.is_principal()`, `I.principal_generator()`


### `quadint.sums`

* `decompose_prime(p: int, d: int = 1, den: int = 1) -> tuple[int, int]`
* `decompose_number(n: int | dict[int, int], d: int = 1, check_count: int | None = None, *, limited_checks: bool = False, no_trivial_solutions: bool = True, warn: bool = True) -> set[tuple[int, int]]`
