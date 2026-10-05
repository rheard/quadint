from __future__ import annotations

from functools import cache
from itertools import product
from math import gcd, isqrt, pi, prod, sqrt
from typing import TYPE_CHECKING, ClassVar, NoReturn, overload

from sympy import factorint, primerange
from sympy.polys.domains import ZZ

from quadint.quad.int import QuadInt
from quadint.utils import _is_squarefree

if TYPE_CHECKING:
    from collections.abc import Iterator

    from quadint.quad.rings import QuadraticRing

_IDEAL_OP_TYPES: tuple[type[complex], type[int], type[float], type[QuadInt]] = (complex, int, float, QuadInt)


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
    """Normalize the lattice basis (a, 0), (b, c) to a > 0, c > 0 and 0 <= b < a, without changing the lattice."""
    # Each step trades the basis for another basis of the same lattice: negating (a, 0), negating (b, c) (where b has
    #   to flip along with c, as they are one vector), and subtracting multiples of (a, 0) from (b, c) (the b % a)
    a = abs(a)
    if c < 0:
        b, c = -b, -c

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

    # ZZ.gcdex is sympy's integer xgcd. (The top-level gcdex gives the same answer, but builds polynomials first,
    #   which makes it ~30x slower, and this runs a few times for every lattice solve.)
    s_raw, t_raw, g_raw = ZZ.gcdex(a, b)
    s, t, g = int(s_raw), int(t_raw), int(g_raw)  # s*a + t*b == g
    a_g = a // g
    b_g = b // g

    # det [[s, t], [b/g, -a/g]] == -(s*a + t*b)/g == -1
    return (
        [s * x + t * y for x, y in zip(u, v, strict=True)],
        [b_g * x - a_g * y for x, y in zip(u, v, strict=True)],
    )


def _lattice_hnf(vectors: list[tuple[int, int]]) -> tuple[int, int, int]:
    """
    Return the Hermite normal form (a, b, c) of the lattice the vectors span, the canonical basis (a, 0), (b, c).

    This is the same column reduction as in _lattice_coefficients, only without tracking coefficients, and with the
        rows the other way around, so the zero ends up below the diagonal.

    Returns:
        tuple[int, int, int]: The normalized (a, b, c), or (0, 0, 0) if every vector is zero.

    Raises:
        ValueError: If the vectors span a nonzero lattice of rank below 2.
    """
    columns = [[x, y] for x, y in vectors]
    columns.append([0, 0])  # so there are always at least two columns to combine

    # Leave the gcd of the second coordinates in `second`, and 0 in the second coordinate of every other column...
    second = columns[0]
    rest = columns[1:]
    for i in range(len(rest)):
        second, rest[i] = _combine_columns(second, rest[i], 1)

    # ...then the gcd of the first coordinates in `first`, among the rest. Those span the lattice with second.
    first = rest[0]
    for i in range(1, len(rest)):
        first, rest[i] = _combine_columns(first, rest[i], 0)

    a, b, c = first[0], second[0], second[1]
    if a == 0 and b == 0 and c == 0:
        return 0, 0, 0

    if a == 0 or c == 0:
        raise ValueError("ideal generators must span a rank-2 lattice")

    return _canonical_hnf(a, b, c)


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

            self.hnf = _lattice_hnf(vectors)

        a, b, c = self.hnf
        self.basis = (_from_coords(ring, a, 0), _from_coords(ring, b, c))
        self.norm = abs(a * c)

    @property
    def is_prime(self) -> bool:
        """Is this a prime ideal?"""
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
        return self._generator()

    def _generator(self) -> QuadInt | None:
        """
        Return principal_generator without its cache, for callers like xgcd that only ever ask once per ideal.

        Returns:
            QuadInt | None: The canonical generator, or None if this ideal is not principal.
        """
        if self.norm == 0:
            return self.ring.zero

        if self.norm == 1:
            return self.ring.one

        ring = self.ring
        D = ring.D

        # The dual numbers (D == 0) go there too, to be refused like every other square D, since their norm a**2 is not
        #   positive definite, which the reduction below needs
        if D >= 0:
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

    @property
    def is_principal(self) -> bool:
        """Is this a principal ideal, one that a single element generates?"""
        return self.principal_generator() is not None

    def conjugate(self) -> Ideal:
        """Return the conjugate ideal."""
        # Conjugating the basis a, b + c*w gives a basis of the conjugate. conj(w) is -w when den == 1 (w = sqrt(D)),
        #   and 1 - w when den == 2, so conj(b + c*w) is b + c*(den - 1) - c*w, which the _hnf path normalizes
        a, b, c = self.hnf
        return Ideal(self.ring, _hnf=(a, b + c * (self.ring.den - 1), -c))

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

        # (la, lb, lc) is the current candidate lattice for x, as the basis (la, 0), (lb, lc) (like an ideal's hnf).
        #
        # Initially, every ring element x = u + v*w is allowed, so the coordinate
        # lattice is just Z^2 with basis (1, 0), (0, 1).
        #
        # Each condition "x * y is in self" cuts this lattice down by two modular
        # linear congruences. After processing both basis elements of other, the
        # remaining lattice is exactly (self : other).
        la, lb, lc = 1, 0, 1

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

                # The congruence is r0*x0 + r1*x1 == 0 (mod mod) for x = x0 + x1*w, and s0, s1 are its left side on the
                #   two basis vectors, so x = alpha*(la, 0) + beta*(lb, lc) passes iff s0*alpha + s1*beta == 0 (mod mod)
                s0 = r0 * la
                s1 = r0 * lb + r1 * lc

                if s0 == 0 and s1 == 0:
                    continue

                g = gcd(abs(s0), abs(s1))
                h = gcd(g, mod)
                q = mod // h

                s0 //= g
                s1 //= g

                u, v, d = ZZ.gcdex(s0, s1)
                if int(d) != 1:
                    raise ArithmeticError("Failed to solve ideal quotient congruence")

                # The combinations of the basis vectors that solve it are spanned by (-s1, s0) and q*(u, v), so the new
                #   lattice is spanned by -s1*(la, 0) + s0*(lb, lc) and q*u*(la, 0) + q*v*(lb, lc)
                qu, qv = q * int(u), q * int(v)
                la, lb, lc = _lattice_hnf([(s0 * lb - s1 * la, s0 * lc), (qu * la + qv * lb, qv * lc)])

        return Ideal(self.ring, _hnf=(la, lb, lc))

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

    # other is an Ideal rather than any object, so that the compiled build turns anything else away itself. Compiled,
    #   returning NotImplemented from here would raise TypeError instead, converting it to the Ideal it has to return
    def __floordiv__(self, other: Ideal) -> Ideal:
        """Return the exact integral ideal quotient."""
        if isinstance(other, Ideal):
            return self.exact_div(other)

        return NotImplemented

    def __rfloordiv__(self, other: NoReturn) -> object:
        # Only an ideal divides an ideal. Without this, 5 // I recursed when compiled (see QuadInt.__rpow__)
        return NotImplemented

    def __contains__(self, x: object) -> bool:
        # A Python number only counts if it equals a ring element (QuadraticRing.__contains__), instead of being
        #   truncated the way arithmetic truncates it: 2.5 is not in (2), and 1.5 is in no ideal at all
        if not isinstance(x, _IDEAL_OP_TYPES) or x not in self.ring:
            return False

        if isinstance(x, complex) and not x.imag:
            x = x.real  # from_obj only takes complex numbers in the Gaussian integers, but 3+0j is plain 3 in any ring

        element = _coerce(self.ring, x)
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

    # Overloads over implementations that take and return any object, for the reason at __mul__. A number stands for its
    #   principal ideal, as it does in I * x, which also lets sum() add up ideals (it starts from 0, the zero ideal).
    @overload
    def __add__(self, other: Ideal) -> Ideal: ...

    @overload
    def __add__(self, other: complex | int | float | QuadInt) -> Ideal: ...

    def __add__(self, other: object) -> object:
        """Return the sum I + J, the smallest ideal that holds both, which is their gcd in a maximal order."""
        if isinstance(other, _IDEAL_OP_TYPES):
            other = Ideal(self.ring, other)

        if not isinstance(other, Ideal):
            return NotImplemented

        if self.ring is not other.ring:
            raise TypeError("Cannot add ideals from different rings")

        # Each basis spans its ideal as a lattice, so the two together span the sum
        return Ideal(self.ring, _hnf=_lattice_hnf([_coords(x) for x in (*self.basis, *other.basis)]))

    @overload
    def __radd__(self, other: QuadInt) -> Ideal: ...

    @overload
    def __radd__(self, other: complex | int | float) -> Ideal: ...

    def __radd__(self, other: object) -> object:
        if isinstance(other, _IDEAL_OP_TYPES):
            return Ideal(self.ring, other) + self

        return NotImplemented

    # The overloads are what type checkers see. The implementations take and return any object, since compiled, a
    #   NotImplemented returned as an Ideal would fail mypyc's conversion to Ideal, and raise its own TypeError before
    #   the other operand's __rmul__ (or Python's usual message) got a turn. Unlike // and IdealClass's *, these can't
    #   have mypyc turn the other operand away first: they take Python complex numbers, which it only takes as objects.
    @overload
    def __mul__(self, other: Ideal) -> Ideal: ...

    @overload
    def __mul__(self, other: complex | int | float | QuadInt) -> Ideal: ...

    def __mul__(self, other: object) -> object:
        if isinstance(other, Ideal):
            if self.ring is not other.ring:
                raise TypeError("Cannot multiply ideals from different rings")

            # The unit ideal is the only one with norm 1
            if self.norm == 1:
                return other

            if other.norm == 1:
                return self

            # Each element of I*J is a sum of products x*y with x in I and y in J, and each x*y is an integer
            #   combination of the products of their basis elements. So those four span I*J as a lattice already,
            #   without also multiplying them by w the way generators of an ideal need.
            return Ideal(self.ring, _hnf=_lattice_hnf([_coords(x * y) for x, y in product(self.basis, other.basis)]))

        if isinstance(other, _IDEAL_OP_TYPES):
            scalar = _coerce(self.ring, other)
            return Ideal(self.ring, *(scalar * x for x in self.basis))

        return NotImplemented

    @overload
    def __rmul__(self, other: QuadInt) -> Ideal: ...

    @overload
    def __rmul__(self, other: complex | int | float) -> Ideal: ...

    def __rmul__(self, other: object) -> object:
        if isinstance(other, _IDEAL_OP_TYPES):
            scalar = _coerce(self.ring, other)
            return Ideal(self.ring, *(scalar * x for x in self.basis))

        return NotImplemented

    def __pow__(self, exp: int) -> Ideal:
        if not isinstance(exp, int):
            return NotImplemented  # what the compiled build does with any other exponent, a float included

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

    def __rpow__(self, other: NoReturn) -> object:
        # Nothing takes an ideal as an exponent. This only works around the mypyc bug at QuadInt.__rpow__, which had
        #   2 ** I and I ** I recurse until RecursionError when compiled
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Ideal):
            return False

        return self.ring == other.ring and self.hnf == other.hnf

    def __ne__(self, other: object) -> bool:
        # This shouldn't be required but mypyc is really messing this up...
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.ring, self.hnf))

    def __reduce__(self) -> tuple:
        # The basis elements generate the ideal too (mypyc would rebuild it with no arguments)
        return Ideal, (self.ring, *self.basis)

    def __repr__(self) -> str:
        if self.norm == 0:
            return f"Ideal({self.ring!r}, 0)"

        return f"Ideal({self.ring!r}, {self.basis[0]!r}, {self.basis[1]!r})"

    def __str__(self) -> str:
        if self.norm == 0:
            return "(0)"

        return f"({self.basis[0]}, {self.basis[1]})"


# region Binary quadratic forms
#   In an imaginary quadratic order, an ideal's class can be pinned down by a binary quadratic form, which turns class
#   comparisons into comparing three integers. A form a*x**2 + b*x*y + c*y**2 with b**2 - 4*a*c equal to the
#   discriminant goes with the ideal [a, (-b + sqrt(disc))/2], and two ideals are in the same class exactly when their
#   forms are properly equivalent: one becomes the other under a change of variables (x, y) -> (p*x + q*y, r*x + s*y)
#   with p*s - q*r == 1. When the discriminant is negative, every equivalence class has exactly one reduced form.
def _reduce_form(a: int, b: int, c: int) -> tuple[int, int, int]:
    """
    Return the reduced form properly equivalent to the positive definite form a*x**2 + b*x*y + c*y**2.

    Reduced means |b| <= a <= c, with b >= 0 when |b| == a or a == c. This is Gauss's reduction, alternating two
        changes of variables: x -> x + k*y, which adds 2*a*k to b (to bring it into (-a, a]), and (x, y) -> (-y, x),
        which swaps a and c and negates b (when c < a). Each swap makes a smaller, so this stops.

    Returns:
        tuple[int, int, int]: The reduced (a, b, c).
    """
    while True:
        if not -a < b <= a:
            k = (a - b) // (2 * a)
            b, c = b + 2 * a * k, c + k * (b + a * k)

        if a <= c:
            break

        a, b, c = c, -b, a

    if a == c and b < 0:
        b = -b  # (x, y) -> (-y, x) again, which only negates b when a == c

    return a, b, c


def _class_form(ideal: Ideal) -> tuple[int, int, int]:
    """
    Return the reduced form of a nonzero invertible ideal's class, in an imaginary quadratic order.

    The ideal is k*J for J = [m, z + w], which is in the same class. Writing z + w as (B + sqrt(disc))/2, the form that
        goes with J is (m, -B, (B**2 - disc)/(4*m)), so J is the ideal [a, (-b + sqrt(disc))/2] of that form.

    Returns:
        tuple[int, int, int]: The reduced (a, b, c), the same one for every ideal in the class.
    """
    ring = ideal.ring
    a, b, k = ideal.hnf
    m, z = a // k, b // k
    B = 2 * z + ring.den - 1
    disc = ring.discriminant()
    return _reduce_form(m, -B, (B * B - disc) // (4 * m))


def _form_ideal(ring: QuadraticRing, a: int, b: int) -> Ideal:
    """Return the ideal [a, (-b + sqrt(disc))/2] of a form (a, b, c) of the ring's discriminant (see _class_form)."""
    return Ideal(ring, _hnf=(a, -(b + ring.den - 1) // 2, 1))


def _reduced_forms(disc: int) -> Iterator[tuple[int, int, int]]:
    """
    Yield every primitive reduced form (a, b, c) of the negative discriminant disc, by increasing a.

    These are one per ideal class of the order with that discriminant, starting with the principal form (1, b, c).
        A reduced form has |b| <= a <= c, so -disc == 4*a*c - b**2 >= 3*a**2, which bounds a.

    Yields:
        tuple[int, int, int]: The forms (a, b, c).
    """
    a = 1
    while 3 * a * a <= -disc:
        # b**2 - disc must be a multiple of 4*a, so b has the parity of disc, starting from the first such b > -a
        for b in range(-a + 1 + (a - 1 - disc) % 2, a + 1, 2):
            c, r = divmod(b * b - disc, 4 * a)
            if r == 0 and (a < c or (a == c and b >= 0)) and gcd(gcd(a, b), c) == 1:
                yield a, b, c

        a += 1


# endregion


class IdealClass:
    """
    Ideal class represented by a nonzero integral ideal.

    In imaginary orders, the classes that products and powers give back are represented by the ideal of their reduced
        form (the one with the smallest norm in the class), instead of the product of the representatives.
    """

    __slots__ = ("representative", "_order", "_form")

    representative: Ideal
    _order: int | None
    _form: tuple[int, int, int] | None

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
        # Real orders have no reduced form that is unique to the class (they come in cycles), so those still compare
        #   classes by testing whether I * conj(J) is principal
        self._form = _class_form(representative) if ring.D < 0 else None

    @property
    def ring(self) -> QuadraticRing:
        """Return the underlying quadratic ring."""
        return self.representative.ring

    @property
    def order(self) -> int:
        """Return the multiplicative order of this ideal class."""
        if self._order is not None:
            return self._order

        # Multiplying classes rather than ideals keeps the powers small in imaginary orders (see __mul__)
        power = self
        order = 1
        while not power.is_trivial:
            power *= self
            order += 1

        self._order = order
        return order

    @property
    def is_trivial(self) -> bool:
        """Is this the principal ideal class, the identity of the class group?"""
        form = self._form
        if form is not None:
            return form[0] == 1  # the principal form (1, b, c) is the only reduced form with a == 1

        return self.representative.is_principal

    def __invert__(self) -> IdealClass:
        """Return the inverse ideal class."""
        return IdealClass(self.representative.conjugate())

    def __mul__(self, other: IdealClass) -> IdealClass:  # not any object, for the reason at Ideal.__floordiv__
        if not isinstance(other, IdealClass):
            return NotImplemented

        if self.ring is not other.ring:
            raise TypeError("Cannot multiply ideal classes from different rings")

        product = self.representative * other.representative
        if self._form is None:
            return IdealClass(product)

        # The ideal of the reduced form is in the same class, and its norm is at most sqrt(|disc|/3), while the norm of
        #   a product is the product of the norms: through ** or order, those would keep growing without end
        a, b, _ = _class_form(product)
        return IdealClass(_form_ideal(self.ring, a, b))

    def __rmul__(self, other: NoReturn) -> object:
        # Only a class multiplies a class. Without this, 5 * C recursed when compiled (see QuadInt.__rpow__)
        return NotImplemented

    def __pow__(self, exp: int) -> IdealClass:
        if not isinstance(exp, int):
            return NotImplemented  # what the compiled build does with any other exponent (see Ideal.__pow__)

        e = int(exp)
        if e < 0:
            return (~self) ** -e

        if self._form is None:
            return IdealClass(self.representative**e)

        # Square and multiply on the classes rather than the ideals, so every step stays reduced (see __mul__)
        result = IdealClass(self.ring.unit_ideal())
        base = self
        while e:
            if e & 1:
                result *= base

            e >>= 1
            if e:
                base *= base

        return result

    def __rpow__(self, other: NoReturn) -> object:
        # Nothing takes an ideal class as an exponent (this only works around the mypyc bug at QuadInt.__rpow__)
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IdealClass):
            return False

        if self.ring is not other.ring:
            return False

        if self._form is not None:
            return self._form == other._form

        return (self.representative * other.representative.conjugate()).is_principal

    def __ne__(self, other: object) -> bool:
        # This shouldn't be required but mypyc is really messing this up...
        return not self.__eq__(other)

    def __hash__(self) -> int:
        # Without a reduced form (in real orders), every class of the ring hashes the same
        return hash((self.representative.ring, self._form))

    def __reduce__(self) -> tuple:
        return IdealClass, (self.representative,)

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
                if not cls.is_trivial and not self._contains_class(out, cls):
                    out.append(cls)

        self._generators = tuple(out)
        return self._generators

    @property
    def classes(self) -> tuple[IdealClass, ...]:
        """Return all ideal classes in this class group, starting with the principal class."""
        if self._classes is not None:
            return self._classes

        ring = self.ring
        if ring.D < 0:
            # Every class has exactly one reduced form, so listing those lists the classes without multiplying any
            #   ideals. Each is represented by the ideal [a, (-b + sqrt(disc))/2] of its form, which has the smallest
            #   norm in its class.
            forms = _reduced_forms(ring.discriminant())
            self._classes = tuple(IdealClass(_form_ideal(ring, a, b)) for a, b, _ in forms)
            return self._classes

        out = [IdealClass(ring.unit_ideal())]

        for generator in self.generators:
            self._adjoin(out, generator)

        self._classes = tuple(out)
        return self._classes

    @property
    def order(self) -> int:
        """Return the class number of the underlying quadratic order."""
        return len(self.classes)

    @property
    def is_trivial(self) -> bool:
        """
        Is this the trivial group, so the class number is one and every ideal is principal?

        The classes of the prime ideals up to the Minkowski bound generate the group (see generators), so it is trivial
            exactly when those are all principal. This stops at the first one that is not, instead of finding every
            class like order does, so it takes next to no time for most rings with a bigger class group.

        Returns:
            bool: Whether the class number is one.
        """
        if self._classes is not None:
            return len(self._classes) == 1

        mb = self.minkowski_bound
        for p in primerange(2, mb + 1):
            for ideal in self.ring.prime_ideals_over(p):
                if ideal.norm <= mb and not IdealClass(ideal).is_trivial:
                    return False

        return True

    @property
    def class_number(self) -> int:
        """Return the class number of the underlying quadratic order (the same as order and QuadraticRing's)."""
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

    def __reduce__(self) -> tuple:
        # Hands back the cached class group, like QuadraticRing.__reduce__ does for rings
        return ClassGroup, (self.ring,)

    def __repr__(self) -> str:
        return f"ClassGroup({self.ring!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ClassGroup) and self.ring == other.ring

    def __ne__(self, other: object) -> bool:
        # This shouldn't be required but mypyc is really messing this up...
        return not self.__eq__(other)

    def __hash__(self) -> int:
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
