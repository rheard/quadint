from __future__ import annotations

from functools import cache
from itertools import product
from math import gcd, isqrt, pi, prod, sqrt
from typing import TYPE_CHECKING, ClassVar

from sympy import Matrix, factorint, gcdex, primerange
from sympy.matrices.normalforms import hermite_normal_form

from quadint.quad.int import QuadInt
from quadint.utils import _is_squarefree

if TYPE_CHECKING:
    from collections.abc import Iterator

    from quadint.quad.rings import QuadraticRing

_IDEAL_OP_TYPES = (complex, int, float, QuadInt)


def _coords(x: QuadInt) -> tuple[int, int]:
    """Return coordinates in the ring's integral basis (1, w)."""
    if x.ring.den == 1:
        return x.a, x.b

    return (x.a - x.b) // 2, x.b


def _from_coords(ring: QuadraticRing, c0: int, c1: int) -> QuadInt:
    """Create an element from coordinates in the ring's integral basis (1, w)."""
    cls = ring.DEFAULT_KLASS
    if ring.den == 1:
        return cls(c0, c1, ring, skip_basis=True)

    return cls(2 * c0 + c1, c1, ring, skip_basis=True)


def _coerce(ring: QuadraticRing, x: complex | int | float | QuadInt) -> QuadInt:
    """Coerce x into ring or raise TypeError."""
    y = ring.from_obj(x)
    if y is NotImplemented:
        raise TypeError(f"Cannot coerce {type(x).__name__} into {ring!r}")

    return y


def _canonical_hnf(a: int, b: int, c: int) -> tuple[int, int, int]:
    """Return a canonical normal form"""
    if a < 0:
        a = -a
        b = -b
    if c < 0:
        c = -c
        b = -b

    return a, b % a, c


def _combine_columns(u: list[int], v: list[int], row: int) -> tuple[list[int], list[int]]:
    """
    Apply a unimodular column operation that leaves gcd(u[row], v[row]) in u and 0 in v at `row`.

    The two new columns span the same lattice as the old ones.

    Returns:
        tuple[list[int], list[int]]: The new (u, v).
    """
    a = u[row]
    b = v[row]
    if b == 0:
        return u, v

    if a == 0:
        return v, u

    s_raw, t_raw, g_raw = gcdex(a, b)
    s, t, g = int(s_raw), int(t_raw), int(g_raw)  # s*a + t*b == g
    a_g = a // g
    b_g = b // g

    # det [[s, t], [b/g, -a/g]] == -(s*a + t*b)/g == -1
    return (
        [s * x + t * y for x, y in zip(u, v, strict=True)],
        [b_g * x - a_g * y for x, y in zip(u, v, strict=True)],
    )


def _lattice_coefficients(vectors: list[tuple[int, int]], target: tuple[int, int]) -> list[int] | None:
    """
    Return integers c with sum(c[i] * vectors[i]) == target, or None if target is not in the lattice they span.

    This is the column reduction behind a Hermite normal form, except each column also carries
        its coefficients in terms of the original vectors, so the solution can be read off at the end.
        Needs at least 2 vectors.

    Returns:
        list[int] | None: One coefficient per vector, or None if there is no integer solution.
    """
    n = len(vectors)
    columns: list[list[int]] = []
    for i, (x, y) in enumerate(vectors):
        column = [0] * (n + 2)
        column[0] = x
        column[1] = y
        column[2 + i] = 1
        columns.append(column)

    # Leave the gcd of the first coordinates in `first`, and 0 in the first coordinate of every other column...
    first = columns[0]
    rest = columns[1:]
    for i in range(len(rest)):
        first, rest[i] = _combine_columns(first, rest[i], 0)

    # ...then do the same with the second coordinate among the rest. Now the lattice is spanned by
    #   first = (A, B) and second = (0, C), and anything left in `rest` is a zero column (a relation).
    second = rest[0]
    for i in range(1, len(rest)):
        second, rest[i] = _combine_columns(second, rest[i], 1)

    A, B, C = first[0], first[1], second[1]
    tx, ty = target

    m = 0
    if A:
        m, r = divmod(tx, A)
        if r:
            return None
    elif tx:
        return None

    k = 0
    remaining = ty - m * B
    if C:
        k, r = divmod(remaining, C)
        if r:
            return None
    elif remaining:
        return None

    return [m * f + k * s for f, s in zip(first[2:], second[2:], strict=True)]


def _bezout_coefficients(ring: QuadraticRing, a: QuadInt, b: QuadInt, g: QuadInt) -> tuple[QuadInt, QuadInt] | None:
    """
    Return (s, t) with s*a + t*b == g, or None if g is not in the ideal (a, b).

    As a lattice, (a, b) is spanned by a, a*w, b, b*w (for the ring's integral basis 1, w),
        so this solves g == c0*a + c1*a*w + c2*b + c3*b*w over the integers, and then s = c0 + c1*w, t = c2 + c3*w.

    Returns:
        tuple[QuadInt, QuadInt] | None: The Bezout coefficients, or None if there are none.
    """
    w = _from_coords(ring, 0, 1)
    coeffs = _lattice_coefficients([_coords(a), _coords(a * w), _coords(b), _coords(b * w)], _coords(g))
    if coeffs is None:
        return None

    return _from_coords(ring, coeffs[0], coeffs[1]), _from_coords(ring, coeffs[2], coeffs[3])


class Ideal:
    """
    Integral ideal in a quadratic order.

    Ideals are stored as normalized rank-2 Z-lattices in the ring's integral basis.
        Iterating over a nonzero ideal yields its infinitely many elements in
            a deterministic expanding square-shell order.
    """

    __slots__ = ("ring", "hnf", "basis", "norm")

    ring: QuadraticRing
    hnf: tuple[int, int, int]
    basis: tuple[QuadInt, QuadInt]
    norm: int

    def __init__(
        self,
        ring: QuadraticRing,
        *generators: complex | int | float | QuadInt,
        _hnf: tuple[int, int, int] | None = None,
    ) -> None:
        """Create the ideal generated by elements, or from an internal HNF tuple."""
        self.ring = ring

        # TODO: Once mypyc handles __new__-based alternate constructors reliably,
        #   the _hnf path can move to a private constructor.
        if _hnf is not None:
            if generators:
                raise TypeError("cannot pass both generators and _hnf")

            a, b, c = int(_hnf[0]), int(_hnf[1]), int(_hnf[2])
            if a == 0 and b == 0 and c == 0:
                self.hnf = (0, 0, 0)
            else:
                if a == 0 or c == 0:
                    raise ValueError("Ideal HNF must be rank 2 or the zero ideal")
                self.hnf = _canonical_hnf(a, b, c)
        else:
            if not generators:
                raise TypeError("expected at least one generator or _hnf")

            vectors: list[tuple[int, int]] = []
            w = _from_coords(ring, 0, 1)
            for g in generators:
                x = _coerce(ring, g)
                vectors.extend((_coords(x), _coords(x * w)))

            matrix = Matrix([[x for x, _ in vectors], [y for _, y in vectors]])
            hnf = hermite_normal_form(matrix)

            if hnf.shape[1] == 0:
                self.hnf = (0, 0, 0)
            elif hnf.shape[1] == 2:
                a = int(hnf[0, 0])
                b = int(hnf[0, 1])
                c = int(hnf[1, 1])
                self.hnf = _canonical_hnf(a, b, c)
            else:
                raise ValueError("ideal generators must span a rank-2 lattice")

        a, b, c = self.hnf
        self.basis = (_from_coords(ring, a, 0), _from_coords(ring, b, c))
        self.norm = abs(a * c)

    def is_prime(self) -> bool:
        """Return True iff this ideal is prime."""
        if self.norm <= 1:
            return False

        factors = factorint(self.norm)
        if len(factors) != 1:
            return False

        p = next(iter(factors))
        return any(self == prime_ideal for prime_ideal in self.ring.prime_ideals_over(p))

    @cache
    def principal_generator(self) -> QuadInt | None:
        """Return a generator of this ideal, or None if it is not principal."""
        if self.norm == 0:
            return self.ring.zero

        if self.norm == 1:
            return self.ring.one

        ring = self.ring
        D = ring.D
        if D > 0:
            return self._principal_generator_real()

        # For D < 0 the norm is positive definite, so this ideal is a lattice with a shortest nonzero vector.
        #   Any nonzero x in the ideal has (x) inside it, so N(x) is a multiple of self.norm, and x generates
        #   the whole ideal exactly when N(x) == self.norm. So the ideal is principal iff its shortest vector has
        #   that norm, and Lagrange-Gauss reduction (the imaginary counterpart of the continued fraction) finds it.
        #   This works on the numerators (a, b) of (a + b*sqrt(D))/den, scaling every norm and dot product by den**2.
        u, v = self.basis
        ua, ub, va, vb = u.a, u.b, v.a, v.b
        nu, nv = ua * ua - D * ub * ub, va * va - D * vb * vb
        while True:
            if nv < nu:
                ua, ub, nu, va, vb, nv = va, vb, nv, ua, ub, nu  # keep u the shorter one

            k = (2 * (ua * va - D * ub * vb) + nu) // (2 * nu)  # the nearest integer to dot(u, v) / dot(u, u)
            if not k:
                break  # v is reduced against u, which makes u a shortest vector

            va, vb = va - k * ua, vb - k * ub
            nv = va * va - D * vb * vb

        if nu != self.norm * ring.den * ring.den:
            return None

        return ring.DEFAULT_KLASS(ua, ub, ring, skip_basis=True)._canonical_associate()

    def _principal_generator_real(self) -> QuadInt | None:
        """
        Return the most compact generator of this real quadratic ideal, found without factoring anything.

        Write I = c*J with J = [m, z + w] primitive (w = sqrt(D) for den=1, or (1 + sqrt(D))/2 for den=2).
            Then J = m*[1, theta] with theta = (z + w) / m, and J is principal exactly when theta is equivalent to w.
            When it is, the continued fraction of theta eventually reaches the cycle of w, where Q == +/-den,
            and the convergents at that point give an element of J with norm +/-m. That element generates J,
            since (alpha) is inside J with the same index.

        Returns:
            A generator, or None if this ideal is not principal.
        """
        ring = self.ring
        D = ring.D
        den = ring.den
        sqrt_d = isqrt(D)
        if sqrt_d * sqrt_d == D:
            raise NotImplementedError("principal generators need a nonsquare D, otherwise theta is rational")

        a, b, c = self.hnf
        m, z = a // c, b // c

        # This is the PQa algorithm, in the notation of Mollin (and Robertson). With A_i/B_i the convergents of
        #   theta = (P0 + sqrt(D)) / Q0, and G_i = Q0*A_i - P0*B_i, each step keeps
        #
        #     G_{i-1}**2 - D*B_{i-1}**2 == (-1)**i * Q0 * Q_i
        #
        # so Q_i == +/-den means (G + B*sqrt(D)) / den has norm +/-m. Those elements Q0*A + B*(sqrt(D) - P0) lie in
        #   [Q0, sqrt(D) - P0], which is why P0 is negated here, so that lattice is J itself (doubled when den=2).
        P, Q = (-z, m) if den == 1 else (-2 * z - 1, 2 * m)
        g_prev, g, b_prev, b_cur = -P, Q, 1, 0  # G_{-2}, G_{-1}, B_{-2}, B_{-1}
        seen: set[tuple[int, int]] = set()

        while abs(Q) != den:
            if (P, Q) in seen:
                return None  # went all the way around theta's cycle without meeting w's, so J is not principal

            seen.add((P, Q))
            q = (P + sqrt_d + (1 if Q < 0 else 0)) // Q  # floor((P + sqrt(D)) / Q)
            P = q * Q - P
            Q = (D - P * P) // Q
            g_prev, g = g, q * g + g_prev
            b_prev, b_cur = b_cur, q * b_cur + b_prev

        # Any associate is a generator, and the canonical one is the most compact
        return ring.DEFAULT_KLASS(c * g, c * b_cur, ring, skip_basis=True)._canonical_associate()

    def is_principal(self) -> bool:
        """Return True iff this ideal is principal."""
        return self.principal_generator() is not None

    def conjugate(self) -> Ideal:
        """Return the conjugate ideal."""
        return Ideal(self.ring, *(x.conjugate() for x in self.basis))

    def factor(self) -> tuple[Ideal, ...]:
        """Return the prime-ideal factorization as a tuple with repeated factors."""
        if self.norm == 0:
            raise ValueError("The zero ideal does not have a prime-ideal factorization")

        if self.norm == 1:
            return ()

        out: list[Ideal] = []
        for p, exponent in factorint(self.norm).items():
            prime_ideals = self.ring.prime_ideals_over(p)

            if len(prime_ideals) == 1:
                prime_ideal = prime_ideals[0]
                prime_norm_exp = 2 if prime_ideal.norm == p * p else 1
                if exponent % prime_norm_exp:
                    raise ArithmeticError("Ideal norm is incompatible with prime-ideal factorization")
                out.extend([prime_ideal] * (exponent // prime_norm_exp))
                continue

            for prime_ideal in prime_ideals:
                power = self.ring.unit_ideal()
                multiplicity = 0
                for _ in range(exponent):
                    power *= prime_ideal
                    if power.divides(self):
                        multiplicity += 1
                    else:
                        break
                out.extend([prime_ideal] * multiplicity)

        check = prod(out, start=self.ring.unit_ideal())
        if check != self:
            raise ArithmeticError("Prime-ideal factorization did not reconstruct the ideal")

        return tuple(out)

    def divides(self, other: Ideal) -> bool:
        """Return True iff this ideal divides other."""
        if self.ring is not other.ring:
            raise TypeError("Cannot compare ideals from different rings")

        return all(x in self for x in other.basis)

    def colon(self, other: Ideal) -> Ideal:
        """Return the colon ideal (self : other)."""
        if self.ring is not other.ring:
            raise TypeError("Cannot divide ideals from different rings")

        if other.norm == 0:
            # (I : 0) is conventionally the whole ring, since x*0 is in I for all x.
            return self.ring.unit_ideal()

        if self.norm == 0:
            # (0 : J) is zero for nonzero integral ideals in a domain.
            return self.ring.zero_ideal()

        a, b, c = self.hnf
        modulus = a * c

        # This matrix stores the current candidate lattice for x.
        #
        # Initially, every ring element x = u + v*w is allowed, so the coordinate
        # lattice is just Z^2 with basis columns (1, 0), (0, 1).
        #
        # Each condition "x * y is in self" cuts this lattice down by two modular
        # linear congruences. After processing both basis elements of other, the
        # remaining lattice is exactly (self : other).
        basis_matrix = Matrix([[1, 0], [0, 1]])

        for y in other.basis:
            y0, y1 = _coords(y)

            if self.ring.den == 1:
                m00 = y0
                m01 = self.ring.D * y1
                m10 = y1
                m11 = y0
            else:
                k = (self.ring.D - 1) // 4
                m00 = y0
                m01 = k * y1
                m10 = y1
                m11 = y0 + y1

            congruences = (
                (m10, m11, c),
                (c * m00 - b * m10, c * m01 - b * m11, modulus),
            )

            for r0, r1, mod in congruences:
                if mod == 1:
                    continue

                s0 = int(r0 * basis_matrix[0, 0] + r1 * basis_matrix[1, 0])
                s1 = int(r0 * basis_matrix[0, 1] + r1 * basis_matrix[1, 1])

                if s0 == 0 and s1 == 0:
                    continue

                g = gcd(abs(s0), abs(s1))
                h = gcd(g, mod)
                q = mod // h

                s0 //= g
                s1 //= g

                u, v, d = gcdex(s0, s1)
                if int(d) != 1:
                    raise ArithmeticError("Failed to solve ideal quotient congruence")

                solution_matrix = Matrix(
                    [
                        [-s1, q * int(u)],
                        [s0, q * int(v)],
                    ],
                )

                basis_matrix = hermite_normal_form(basis_matrix * solution_matrix)

        hnf = (
            int(basis_matrix[0, 0]),
            int(basis_matrix[0, 1]),
            int(basis_matrix[1, 1]),
        )
        return Ideal(self.ring, _hnf=hnf)

    def exact_div(self, other: Ideal) -> Ideal:
        """Return the integral ideal q such that self == other * q."""
        if self.ring is not other.ring:
            raise TypeError("Cannot divide ideals from different rings")

        if other.norm == 0:
            raise ZeroDivisionError("Cannot divide by the zero ideal")

        if self.norm == 0:
            return self.ring.zero_ideal()

        if self.norm % other.norm:
            raise ValueError("Ideal division is not exact")

        if not other.divides(self):
            raise ValueError("Ideal division is not exact")

        quotient = self.colon(other)
        if other * quotient != self:
            raise ValueError("Ideal division is not exact")

        return quotient

    def __floordiv__(self, other: object) -> Ideal:
        """Return the exact integral ideal quotient."""
        if isinstance(other, Ideal):
            return self.exact_div(other)

        return NotImplemented

    def __contains__(self, x: object) -> bool:
        if not isinstance(x, _IDEAL_OP_TYPES):
            return False

        try:
            element = _coerce(self.ring, x)
        except TypeError:
            return False

        if self.norm == 0:
            return not element

        a, b, c = self.hnf
        x0, y0 = _coords(element)
        n, yr = divmod(y0, c)
        if yr:
            return False

        return (x0 - b * n) % a == 0

    def __iter__(self) -> Iterator[QuadInt]:
        """Iterate over all elements of this ideal; nonzero ideals iterate forever."""
        if self.norm == 0:
            yield self.ring.zero
            return

        b0, b1 = self.basis
        yield self.ring.zero

        radius = 1
        while True:
            for m in range(-radius, radius + 1):
                yield m * b0 - radius * b1
                yield m * b0 + radius * b1

            for n in range(-radius + 1, radius):
                yield -radius * b0 + n * b1
                yield radius * b0 + n * b1

            radius += 1

    def __mul__(self, other: object) -> Ideal:
        if isinstance(other, Ideal):
            if self.ring is not other.ring:
                raise TypeError("Cannot multiply ideals from different rings")

            return Ideal(self.ring, *(x * y for x, y in product(self.basis, other.basis)))

        if isinstance(other, _IDEAL_OP_TYPES):
            scalar = _coerce(self.ring, other)
            return Ideal(self.ring, *(scalar * x for x in self.basis))

        return NotImplemented

    def __rmul__(self, other: object) -> Ideal:
        if isinstance(other, _IDEAL_OP_TYPES):
            scalar = _coerce(self.ring, other)
            return Ideal(self.ring, *(scalar * x for x in self.basis))

        return NotImplemented

    def __pow__(self, exp: int) -> Ideal:
        if exp < 0:
            raise ValueError("Negative ideal powers require fractional ideals")

        result = self.ring.unit_ideal()
        base = self
        e = int(exp)
        while e:
            if e & 1:
                result *= base

            e >>= 1
            if e:
                base *= base

        return result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Ideal):
            return False

        return self.ring == other.ring and self.hnf == other.hnf

    def __ne__(self, other: object) -> bool:
        # This shouldn't be required but mypyc is really messing this up...
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.ring, self.hnf))

    def __repr__(self) -> str:
        if self.norm == 0:
            return f"Ideal({self.ring!r}, 0)"

        return f"Ideal({self.ring!r}, {self.basis[0]!r}, {self.basis[1]!r})"

    def __str__(self) -> str:
        if self.norm == 0:
            return "(0)"

        return f"({self.basis[0]}, {self.basis[1]})"


class IdealClass:
    """Ideal class represented by a nonzero integral ideal."""

    __slots__ = ("representative", "_order")

    representative: Ideal
    _order: int | None

    def __init__(self, representative: Ideal) -> None:
        """Create the ideal class represented by a nonzero, invertible integral ideal."""
        ring = representative.ring
        norm = representative.norm
        if norm == 0:
            raise ValueError("The zero ideal does not define an ideal class")

        # Only invertible ideals have a class. Every nonzero ideal of a maximal order is invertible, but other orders
        #   have some that are not, like (2, 1 + sqrt(-3)) in Z[sqrt(-3)], where P*P == 2*P so no power of P is ever
        #   principal (and order would never return). An ideal I is invertible exactly when I * conj(I) == (N(I)),
        #   and it always is when N(I) is coprime to the order's conductor, which divides 2*D.
        if gcd(norm, 2 * ring.D) > 1 and representative * representative.conjugate() != ring.ideal(norm):
            raise ValueError(f"{representative} is not invertible, so it does not define an ideal class")

        self.representative = representative
        self._order = None

    @property
    def ring(self) -> QuadraticRing:
        """Return the underlying quadratic ring."""
        return self.representative.ring

    @property
    def order(self) -> int:
        """Return the multiplicative order of this ideal class."""
        if self._order is not None:
            return self._order

        power = self.representative
        order = 1
        while not power.is_principal():
            power *= self.representative
            order += 1

        self._order = order
        return order

    def is_trivial(self) -> bool:
        """Return True iff this is the principal ideal class."""
        return self.representative.is_principal()

    def __invert__(self) -> IdealClass:
        """Return the inverse ideal class."""
        return IdealClass(self.representative.conjugate())

    def __mul__(self, other: object) -> IdealClass:
        if not isinstance(other, IdealClass):
            return NotImplemented

        if self.ring is not other.ring:
            raise TypeError("Cannot multiply ideal classes from different rings")

        return IdealClass(self.representative * other.representative)

    def __pow__(self, exp: int) -> IdealClass:
        e = int(exp)
        if e == 0:
            return IdealClass(self.ring.unit_ideal())

        if e < 0:
            return (~self) ** -e

        return IdealClass(self.representative**e)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IdealClass):
            return False

        if self.ring is not other.ring:
            return False

        return (self.representative * other.representative.conjugate()).is_principal()

    def __ne__(self, other: object) -> bool:
        # This shouldn't be required but mypyc is really messing this up...
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self.representative.ring)

    def __repr__(self) -> str:
        return f"IdealClass({self.representative!r})"

    def __str__(self) -> str:
        return f"[{self.representative}]"


# @cache
# This is a trick I learned recently to make a singleton class and would work here, and does in pure python
# But boy does mypyc hate it, it completely crashes everything...
# we'll have to make singleton classes the old-fashioned way...
class ClassGroup:
    """Ideal class group of a quadratic order."""

    __slots__ = ("ring", "_classes", "_generators", "_initialized")

    _CACHE: ClassVar[dict[tuple[type, int, int], ClassGroup]] = {}

    ring: QuadraticRing
    _classes: tuple[IdealClass, ...] | None
    _generators: tuple[IdealClass, ...] | None
    _initialized: bool

    def __new__(cls, ring: QuadraticRing):
        """Return the cached class group for this exact kind of quadratic ring."""
        if ring.D in (0, 1) or (ring.D != -1 and not _is_squarefree(ring.D)):
            raise NotImplementedError("Class groups require a quadratic field/order")

        # A non-maximal order (like Z[sqrt(-3)] inside Z[(1 + sqrt(-3))/2]) has prime ideals that are not invertible,
        #   so its class group (the Picard group) needs a different set of generators than the Minkowski-bound primes.
        if ring.den != (2 if ring.D % 4 == 1 else 1):
            raise NotImplementedError(
                f"Class groups are only implemented for maximal orders, and {ring!r} is not one "
                f"(QuadraticRing({ring.D}) is)",
            )

        key = (type(ring), ring.D, ring.den)
        inst = cls._CACHE.get(key)
        if inst is not None:
            return inst

        inst = super().__new__(cls)
        cls._CACHE[key] = inst
        return inst

    def __init__(self, ring: QuadraticRing) -> None:
        """Create the ideal class group of ring."""
        if getattr(self, "_initialized", False):
            return

        self.ring = ring
        self._classes = None
        self._generators = None
        self._initialized = True

    @property
    def minkowski_bound(self) -> int:
        """Return a norm bound for ideal-class representatives."""
        disc = self.ring.discriminant()
        if disc < 0:
            return int(2 * sqrt(abs(disc)) / pi) + 1

        return int(sqrt(disc) / 2) + 1

    @property
    def generators(self) -> tuple[IdealClass, ...]:
        """Return prime ideal classes that generate this class group."""
        if self._generators is not None:
            return self._generators

        out: list[IdealClass] = []

        mb = self.minkowski_bound
        for p in primerange(2, mb + 1):
            for ideal in self.ring.prime_ideals_over(p):
                if ideal.norm > mb:
                    continue

                cls = IdealClass(ideal)
                if not cls.is_trivial() and not self._contains_class(out, cls):
                    out.append(cls)

        self._generators = tuple(out)
        return self._generators

    @property
    def classes(self) -> tuple[IdealClass, ...]:
        """Return all ideal classes in this class group."""
        if self._classes is not None:
            return self._classes

        out = [IdealClass(self.ring.unit_ideal())]

        for generator in self.generators:
            self._adjoin(out, generator)

        self._classes = tuple(out)
        return self._classes

    @property
    def order(self) -> int:
        """Return the class number of the underlying quadratic order."""
        return len(self.classes)

    def class_number(self) -> int:
        """Return the class number of the underlying quadratic order."""
        return self.order

    def __len__(self) -> int:
        return self.order

    def __iter__(self) -> Iterator[IdealClass]:
        return iter(self.classes)

    def __contains__(self, cls: object) -> bool:
        if not isinstance(cls, IdealClass):
            return False

        if cls.ring is not self.ring:
            return False

        return self._contains_class(self.classes, cls)

    def __repr__(self) -> str:
        return f"ClassGroup({self.ring!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ClassGroup) and self.ring == other.ring

    def __ne__(self, other: object) -> bool:
        # This shouldn't be required but mypyc is really messing this up...
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.ring)

    def _contains_class(self, classes: list[IdealClass] | tuple[IdealClass, ...], cls: IdealClass) -> bool:
        """Return True iff classes already contains cls."""
        return any(cls == known for known in classes)

    def _adjoin(self, classes: list[IdealClass], generator: IdealClass) -> None:
        """Expand classes in-place until it is closed under multiplication by generator."""
        todo = [generator]

        while todo:
            cls = todo.pop()
            if self._contains_class(classes, cls):
                continue

            classes.append(cls)

            for known in classes:
                product = cls * known
                if not self._contains_class(classes, product) and not self._contains_class(todo, product):
                    todo.append(product)
