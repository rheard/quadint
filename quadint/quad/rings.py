from __future__ import annotations

import warnings

from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from math import isqrt, prod
from typing import Callable, ClassVar

from sympy import factorint, sqrt_mod

from quadint.quad.int import QuadInt
from quadint.utils import requires_modules

try:
    import cypari  # noqa: F401
except ImportError:
    warnings.warn(
        "cypari is not installed. "
        "Without it, Harper-like division is only available with rings that have a D value below 100 "
        "(because these are hard-coded).",
        ImportWarning,
        stacklevel=2,
    )

NORM_EUCLID_D: set[int] = {-11, -7, -3, -2, -1, 2, 3, 5, 6, 7, 11, 13, 17, 19, 21, 29, 33, 37, 41, 57, 73}


# TODO: Once Py3.9 support has been dropped, add slots=True
# @dataclass(frozen=True, slots=True)
@dataclass(frozen=True)
class Factorization:
    """x = unit * P1 * P2 * ... * Pk"""

    unit: QuadInt
    primes: dict[QuadInt, int]

    def prod(self):
        """Recreate the number using prod"""
        return prod((p**k for p, k in self.primes.items()), start=self.unit)


def _check_den(den: int) -> int:
    """Validate and return ring denominator."""
    den0 = int(den)
    if den0 not in (1, 2):
        raise ValueError(f"den must be 1 or 2, got {den0!r}")
    return den0


def _round_div_ties_away_from_zero(n: int, d: int) -> int:
    """Round n/d to nearest int; ties go away from 0. d must be > 0."""
    if d == 0:
        raise ZeroDivisionError
    if d < 0:
        n, d = -n, -d
    if n >= 0:
        return (n + d // 2) // d
    return -((-n + d // 2) // d)


def _split_uv(x: QuadInt) -> tuple[int, int]:
    """Return (u,v) for D=1 split-complex where u=(a+b)/den, v=(a-b)/den."""
    den = x.ring.den
    apb = x.a + x.b
    amb = x.a - x.b
    if apb % den or amb % den:
        # should be impossible if ring invariants hold
        raise ArithmeticError("Non-integral split-complex coordinates; check ring parameters/parity")
    return apb // den, amb // den


class _NeighborhoodSearch:
    """
    Incremental local lattice search around (A0, B0_for_A(A)).

    Expanding from radius r to R only evaluates the newly-added Chebyshev shells
    (r+1, r+2, ..., R), so repeated calls do not rescan the interior.
    """

    __slots__ = (
        "A0",
        "B0_for_A",
        "score_for_AB",
        "den",
        "_expanded_radius",
        "_best_score",
        "_best_a",
        "_best_b",
    )

    def __init__(
        self,
        *,
        A0: int,
        B0_for_A: Callable[[int], int],
        score_for_AB: Callable[[int, int], tuple[int, ...]],
        den: int,
    ) -> None:
        self.A0 = int(A0)
        self.B0_for_A = B0_for_A
        self.score_for_AB = score_for_AB
        self.den = int(den)

        # Highest radius already fully scanned. Start at -1 = nothing scanned yet.
        self._expanded_radius = -1

        self._best_score: tuple[int, ...] | None = None
        self._best_a = 0
        self._best_b = 0

    def _consider(self, A: int, B: int) -> None:
        if self.den == 2 and ((A ^ B) & 1):
            return

        s = self.score_for_AB(A, B)
        if self._best_score is None or s < self._best_score:
            self._best_score = s
            self._best_a = A
            self._best_b = B

    def _scan_shell(self, radius: int) -> None:
        """Scan exactly the Chebyshev shell at the given radius (new points only)."""
        if radius < 0:
            return

        if radius == 0:
            A = self.A0
            B = self.B0_for_A(A)
            self._consider(A, B)
            return

        left = self.A0 - radius
        right = self.A0 + radius

        for A in range(left, right + 1):
            B0 = self.B0_for_A(A)

            if A in (left, right):
                # New vertical edges: full range
                for B in range(B0 - radius, B0 + radius + 1):
                    self._consider(A, B)
            else:
                # New top/bottom edge only
                self._consider(A, B0 - radius)
                self._consider(A, B0 + radius)

    def expand_to(self, radius: int) -> tuple[int, int]:
        """Expand the search incrementally up to `radius` and return current best (A,B)."""
        r = int(radius)
        if r < 0:
            raise ValueError("radius must be >= 0")

        if r > self._expanded_radius:
            for shell in range(self._expanded_radius + 1, r + 1):
                self._scan_shell(shell)
            self._expanded_radius = r

        return self._best_a, self._best_b

    @property
    def best_score(self) -> tuple[int, ...] | None:
        return self._best_score

    @property
    def best_ab(self) -> tuple[int, int]:
        return self._best_a, self._best_b


def _choose_best_in_neighborhood(
    *,
    A0: int,
    B0_for_A: Callable,
    score_for_AB: Callable,
    den: int,
    radius: int = 1,
) -> tuple[int, int]:
    """
    Search (A0±radius) * (B0(A)±radius) and return best (A,B).

    This is a tiny local lattice search used by all our divmod implementations.

    Args:
        A0: Initial guess for A.
        B0_for_A: Given A, return an initial guess for B (may depend on A).
        score_for_AB: Lexicographic score; smaller is better.
        den: Ring denominator (1 or 2). If den==2, enforce parity constraint A ≡ B (mod 2).
        radius: Search radius around the initial guess(es). radius=1 reproduces the old behavior.

    Returns:
        (bestA, bestB): Best candidate found.
    """
    search = _NeighborhoodSearch(
        A0=A0,
        B0_for_A=B0_for_A,
        score_for_AB=score_for_AB,
        den=den,
    )
    return search.expand_to(radius)


class QuadraticRing:
    """
    The quadratic integer ring (order) with basis (1, sqrt(D)) and fixed denominator den in {1,2}.

    Elements are represented as:
        (a + b*sqrt(D)) / den

    where a,b are integers stored as numerators.
    When den==2, integrality requires a ≡ b (mod 2).

    This object is *not* a type factory (no nested classes) — it just carries parameters.
    """

    __slots__ = ("D", "den")

    SUPPORTS_DIVISION: ClassVar[bool] = False
    SUPPORTS_FACTORIZATION: ClassVar[bool] = False
    _CACHE: ClassVar[dict[tuple[int, int], object]] = {}

    D: int
    den: int

    def __new__(cls, D: int, den: int | None = None):
        """Handle singleton logic"""
        if cls is not QuadraticRing:
            return super().__new__(cls)

        D0 = int(D)
        default_den = 2 if (D0 % 4) == 1 else 1
        den0 = default_den if den is None else _check_den(den)

        key = (D0, den0)
        inst = cls._CACHE.get(key)
        if inst is not None:
            return inst

        # choose subclass
        new_inst: QuadraticRing
        for subcls in cls._subclasses():
            if subcls.accept_override(D0, den0, default_den):
                new_inst = subcls(D0, den0)
                break
        else:
            new_inst = super().__new__(cls)

        cls._CACHE[key] = new_inst
        return new_inst

    def __init__(self, D: int, den: int | None = None) -> None:
        """Initialize the ring settings"""
        self.D = int(D)
        self.den = (2 if (self.D % 4) == 1 else 1) if den is None else _check_den(den)

    @classmethod
    def _subclasses(cls):
        """Recursively yield all subclasses of cls (depth-first)."""
        for sub in cls.__subclasses__():
            yield from sub._subclasses()
            yield sub

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(D={self.D}, den={self.den})"

    def __call__(self, a: int = 0, b: int = 0) -> QuadInt:
        """Create element (a + b*sqrt(D))/den with numerator coefficients a,b."""
        return QuadInt(int(a), int(b), self)

    def __contains__(self, x: object) -> bool:
        """Return True iff x is a QuadInt element of this ring (by parameters)."""
        if isinstance(x, int):
            return True

        if isinstance(x, float):
            return x.is_integer()

        if isinstance(x, complex):
            return self.D == -1 and self.den == 1

        if not isinstance(x, QuadInt):
            return False

        other = x.ring
        return (other.D == self.D) and (other.den == self.den)

    def __eq__(self, other: object):
        if not isinstance(other, QuadraticRing):
            return False

        return self.D == other.D and self.den == other.den

    @property
    def zero(self) -> QuadInt:
        """Additive identity (0)."""
        return QuadInt(0, 0, self)

    @property
    def one(self) -> QuadInt:
        """Multiplicative identity (1)."""
        return QuadInt(self.den, 0, self)

    def from_obj(self, n: complex | int | float | QuadInt) -> QuadInt:
        """Embed integer (or float) n as (n*den + 0*sqrt(D))/den. Also supports complex if D==-1"""
        if isinstance(n, (int, float)):
            a = int(n)
            b = 0
        elif isinstance(n, complex):
            if self.D != -1 or self.den != 1:
                raise TypeError("Cannot mix QuadInt from different rings")

            a = int(n.real)
            b = int(n.imag)
        elif isinstance(n, QuadInt):
            if n.ring is not self:
                raise TypeError("Cannot mix QuadInt from different rings")

            return n
        else:
            return NotImplemented

        # The only time b is not 0 is if self.den is 1 anyway... No need to multiply
        return QuadInt(a * self.den, b, self)

    def from_ab(self, a: int, b: int) -> QuadInt:
        """Create an integer keeping in mind the denominator"""
        da = int(a) * self.den
        db = int(b) * self.den
        return QuadInt(da, db, self)

    def supports_division(self) -> bool:
        """
        Return whether this ring advertises Euclidean-style ``divmod`` support.

        Returns:
            bool: Whether this ring class sets ``SUPPORTS_DIVISION``.
        """
        return self.SUPPORTS_DIVISION

    def supports_factorization(self) -> bool:
        """
        Return whether this ring advertises prime-factorization support.

        Returns:
            bool: Whether this ring class sets ``SUPPORTS_FACTORIZATION``.
        """
        return self.SUPPORTS_FACTORIZATION

    def phi(self, x: QuadInt) -> int:
        """Return the default Euclidean size `|N(x)|` used for division heuristics."""
        return abs(abs(x))

    def discriminant(self) -> int:
        """Return the discriminant of the order represented by this ring."""
        return self.D if self.den == 2 else 4 * self.D

    @requires_modules(["cypari"])
    def class_number(self) -> int:
        """Return the ideal class number of this order using PARI/GP."""
        from cypari import pari  # noqa: PLC0415

        disc: int = self.discriminant()
        h = pari(f"bnfinit(quadpoly({disc})).clgp.no")
        return int(h)

    def divmod(self, x: QuadInt, y: QuadInt) -> tuple[QuadInt, QuadInt]:
        """An override for defining division algorithms in subclasses for different D values"""
        raise NotImplementedError

    def factor_detail(self, x: QuadInt) -> Factorization:
        """Factor `x` and return structured details when supported by this ring."""
        raise NotImplementedError("Factorization is not implemented for this ring")

    def factor(self, x: QuadInt) -> dict[QuadInt, int]:
        """
        Factor x and return a dict of factors whose product is exactly `x`.

        The unit part is folded into the first factor so users get a simple factor list
            (matching familiar integer-factorization APIs).

        Returns:
            dict: A simple dictionary containing the factorization similar to sympy's factorint.
        """
        factorization = self.factor_detail(x)
        unit = factorization.unit
        factors = factorization.primes

        if not factors:
            return {x: 1}

        # Pick any factor where factor * unit is already in factors,
        #   otherwise pick the minimum exponent...
        first_prime, first_k, first_prime_normed = min(
            ((x, y, x * unit) for x, y in factors.items()),
            key=lambda z: (z[2] not in factors, z[1], abs(z[0].b), abs(z[0].a)),
        )

        if first_k == 1:
            del factors[first_prime]
        else:
            factors[first_prime] -= 1

        folded = first_prime_normed
        factors[folded] = factors.get(folded, 0) + 1
        return factors

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:  # noqa: ARG003
        """Should this class be used for the given values?"""
        return True


class DualRing(QuadraticRing):
    """
    Handle overrides for D=0, dual integer solutions

    While the general algorith in RealNormEuclidRing will find a solution for D=0,
        it does not take into account that the ε part is not part of the norm,
        so is not relevant in the division algorithm.

    This class will solve division for the real part while trying to minimize the ε part.
    """

    SUPPORTS_DIVISION: ClassVar[bool] = True

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:  # noqa: ARG003
        """Should this class be used for the given values?"""
        return D == 0

    def divmod(self, x: QuadInt, y: QuadInt) -> tuple[QuadInt, QuadInt]:
        """Division with D=0"""
        # In dual numbers, (c + dε) is invertible iff c != 0.
        n = y.a
        num = x

        if n == 0:
            raise ZeroDivisionError

        A0 = _round_div_ties_away_from_zero(num.a, n)

        c, d = y.a, y.b

        def B0_for_A(A: int) -> int:
            return _round_div_ties_away_from_zero(x.b - A * d, c)

        # Lexicographic “small remainder”: minimize real remainder first, then ε remainder.
        def score_for_AB(A: int, B: int) -> tuple[int, ...]:
            r0 = x.a - A * c
            r1 = x.b - A * d - B * c
            return r0 * r0, r1 * r1

        best_a, best_b = _choose_best_in_neighborhood(
            A0=A0,
            B0_for_A=B0_for_A,
            score_for_AB=score_for_AB,
            den=self.den,
        )

        q = x._make(best_a, best_b)
        r = x - q * y
        return q, r


class SplitRing(QuadraticRing):
    """
    Handle overrides for D=1, split integer solutions

    This class performs division in the split (u, v) coordinates (where multiplication
        is component-wise), choosing a quotient that reduces both components and yields a
        more stable, integer-like remainder.

    This structural shortcut is only possible with D=1 (because Z[sqrt(1)]... well, splits.)
    """

    SUPPORTS_DIVISION: ClassVar[bool] = True

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:  # noqa: ARG003
        """Should this class be used for the given values?"""
        return D == 1

    def divmod(self, x: QuadInt, y: QuadInt) -> tuple[QuadInt, QuadInt]:
        """Division with D=1"""
        u1, v1 = _split_uv(x)
        u2, v2 = _split_uv(y)

        # Division by zero divisor (u2==0 or v2==0) is not well-defined.
        if u2 == 0 or v2 == 0:
            raise ZeroDivisionError("division by zero divisor in split-complex integers (a=±b)")

        qu0 = _round_div_ties_away_from_zero(u1, abs(u2))
        if u2 < 0:
            qu0 = -qu0

        qv0 = _round_div_ties_away_from_zero(v1, abs(v2))
        if v2 < 0:
            qv0 = -qv0

        def B0_for_A(A: int) -> int:  # noqa: ARG001
            return qv0

        def score_for_AB(A: int, B: int) -> tuple[int, ...]:
            # remainder in (u,v)
            ru = u1 - A * u2
            rv = v1 - B * v2
            return (ru * ru + rv * rv,)

        # We only need qu ≡ qv (mod 2) when self.den is odd (in practice: self.den==1),
        # because we later divide (qu±qv)*self.den by 2.
        parity_den = 2 if self.den == 1 else 1

        # Enforce qu ≡ qv (mod 2) so (qu+qv)/2 and (qu-qv)/2 are integers.
        best_qu, best_qv = _choose_best_in_neighborhood(
            A0=qu0,
            B0_for_A=B0_for_A,
            score_for_AB=score_for_AB,
            den=parity_den,
        )

        # Convert back: a = den*(qu+qv)/2, b = den*(qu-qv)/2
        s = best_qu + best_qv
        t = best_qu - best_qv
        qa = (s * self.den) // 2
        qb = (t * self.den) // 2

        q = x._make(qa, qb)
        r = x - q * y
        return q, r


class RealNormEuclidRing(QuadraticRing):
    """
    Handle overrides where the ring of integers is norm-Euclidean.

    This class provides the general division algorithm used for both positive and negative
        discriminants in NORM_EUCLID_D (excluding D=0 and D=1 special cases).
    """

    SUPPORTS_DIVISION: ClassVar[bool] = True

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:
        """Should this class be used for the given values?"""
        return D in NORM_EUCLID_D and den == default_den

    def divmod(self, x: QuadInt, y: QuadInt):
        """
        Division in a norm-Euclidean real quadratic ring.

        We use the absolute norm as the Euclidean function:
            f(z) = |N(z)|.

        For the known finite list of D where the ring of integers is norm-Euclidean,
        there exists q such that |N(x - qy)| < |N(y)|.

        Returns:
            q, r: The quotient and remainder

        Raises:
            ZeroDivisionError: If the magnitude of the divisor is 0.
            ArithmeticError: In the event of a parity mismatch.
        """
        y_norm = abs(y)  # signed norm (may be negative for D>0)
        abs_y_norm = abs(y_norm)
        if abs_y_norm == 0:
            raise ZeroDivisionError

        # Candidate center from x/y ≈ (x * conj(y)) / N(y)
        a1, b1 = x.a, x.b
        a2, b2 = y.a, y.b

        # num = x * conj(y), but computed in numerators directly
        # (a1+b1√D)(a2-b2√D) = (a1*a2 - b1*b2*D) + (a2*b1 - a1*b2)√D
        num_a = a1 * a2 - b1 * b2 * self.D
        num_b = a2 * b1 - a1 * b2

        if self.den != 1:
            if (num_a % self.den) != 0 or (num_b % self.den) != 0:
                raise ArithmeticError("Non-integral product; check ring parameters / parity")
            num_a //= self.den
            num_b //= self.den

        A0 = _round_div_ties_away_from_zero(num_a, y_norm)
        B0 = _round_div_ties_away_from_zero(num_b, y_norm)
        dd = self.den**2
        threshold = abs_y_norm * abs_y_norm * dd

        def B0_for_A(A: int) -> int:  # noqa: ARG001
            return B0

        # Prefer any norm-reducing remainder; among those, minimize |N(r)| then distance to (A0,B0).
        def score_for_AB(A: int, B: int) -> tuple[int, ...]:
            da = A * y_norm - num_a
            db = B * y_norm - num_b

            # numerator of N(w) where w=(da + db*sqrt(D))/den
            nw_num = da * da - self.D * (db * db)
            abs_nw_num = abs(nw_num)

            # norm-reducing condition: |N(w)| < |Ny|^2  <=>  |nw_num| < |Ny|^2 * den^2
            flag = 0 if abs_nw_num < threshold else 1

            dist2 = (A - A0) * (A - A0) + (B - B0) * (B - B0)
            return flag, abs_nw_num, dist2

        # Expand search radius until we find a norm-reducing remainder.
        search = _NeighborhoodSearch(
            A0=A0,
            B0_for_A=B0_for_A,
            score_for_AB=score_for_AB,
            den=self.den,
        )

        for rad in (1, 2, 3, 4, 6, 8):
            best_a, best_b = search.expand_to(rad)

            # score_for_AB returns (flag, abs_nw_num, dist2)
            best_score = search.best_score
            if best_score is not None and best_score[0] == 0:
                q = x._make(best_a, best_b)
                r = x - q * y
                return q, r

        raise NotImplementedError(
            f"No norm-reducing quotient found for D={self.D}, den={self.den} within search radii",
        )


# region Cornacchia rings (for factorization)
class CornacchiaRing(RealNormEuclidRing):
    """Shared split-prime factorization flow for rings with norm form x**2 + k*y**2."""

    SUPPORTS_FACTORIZATION: ClassVar[bool] = True

    RAMIFIED_PRIME: ClassVar[int]
    SPLIT_K: ClassVar[int]

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:  # noqa: ARG003
        """Should this class be used for the given values?"""
        # No, this is purely a sub-abstract base class that needs to be subclassed
        return False

    @classmethod
    def _is_split_prime(cls, p: int) -> bool:
        raise NotImplementedError

    @classmethod
    def _is_inert_prime(cls, p: int) -> bool:
        raise NotImplementedError

    @classmethod
    def _ramified_generator(cls, x: QuadInt) -> QuadInt:
        raise NotImplementedError

    @classmethod
    def _inert_generator(cls, x: QuadInt, p: int) -> QuadInt:
        return x._make(p, 0)

    @classmethod
    def _split_generator(cls, x: QuadInt, p: int, x0: int, y0: int) -> QuadInt:
        raise NotImplementedError

    @classmethod
    def _decompose_prime(cls, p: int) -> tuple[int, int]:
        """Find x,y with p = x**2 + k*y**2 for split primes, where k=SPLIT_K."""
        if not cls._is_split_prime(p):
            raise ValueError(f"Could not decompose {p!r}")

        root = sqrt_mod(-cls.SPLIT_K, p, all_roots=False)
        if root is None:
            raise ValueError(f"Could not decompose {p!r}")

        a = p
        b = min(root, p - root)
        while b * b > p:
            a, b = b, a % b

        y2_num = p - b * b
        if y2_num % cls.SPLIT_K:
            raise ValueError(f"Could not decompose {p!r}")

        y2 = y2_num // cls.SPLIT_K
        y = isqrt(y2)
        if y * y != y2:
            raise ValueError(f"Could not decompose {p!r}")

        return b, y

    def factor_detail(self, x: QuadInt) -> Factorization:
        """
        Return a structured factorization for Cornacchia-style imaginary quadratic rings.

        The result is returned as `Factorization(unit, primes)` where `primes` is a
            mapping ``{prime_element: exponent}`` and:

        * `unit * prod(p**e for p, e in primes.items()) == x`
        * each listed `prime_element` is a non-unit irreducible in this ring

        Strategy:

        1. Normalize by extracting a canonical unit associate.
        2. Remove powers of the ramified prime generator.
        3. Factor the remaining integer norm with ``sympy.factorint``.
        4. For each rational prime factor, use split/inert classification:
           * inert primes stay prime in the ring,
           * split primes are decomposed via Cornacchia's method and tested (with conjugates)
             as divisors.

        Args:
            x: A non-zero element of this ring.

        Returns:
            Factorization: The unit and prime-power data for ``x``.

        Raises:
            ValueError: If `x` is zero (zero has no finite prime factorization).
        """
        if not x:
            raise ValueError("0 does not have a finite factorization")

        rem = x
        unit = x.one

        for u in x.units:
            q, r = divmod(rem, u)
            if not r and (q.a, q.b) < (rem.a, rem.b):
                rem = q
                unit *= u

        factors: dict[QuadInt, int] = defaultdict(int)

        ramified = self._ramified_generator(x)
        while True:
            q, r = divmod(rem, ramified)
            if r:
                break
            factors[ramified] += 1
            rem = q

        n = abs(rem)
        int_factors = factorint(n)

        for p in sorted(int_factors):
            if p == self.RAMIFIED_PRIME:
                continue

            if self._is_inert_prime(p):
                cand = self._inert_generator(x, p)
                while True:
                    q, r = divmod(rem, cand)
                    if r:
                        break
                    factors[cand] += 1
                    rem = q
                continue

            if self._is_split_prime(p):
                x0, y0 = self._decompose_prime(p)
                cand_base = self._split_generator(x, p, x0, y0)
                for cand in (cand_base, cand_base.conjugate()):
                    if abs(cand) <= 1:
                        continue
                    while rem != self.den:
                        q, r = divmod(rem, cand)
                        if r:
                            break
                        factors[cand] += 1
                        rem = q

        if abs(rem) != 1:
            factors[rem] += 1
        else:
            unit *= rem

        return Factorization(unit=unit, primes=dict(factors))


class GaussianRing(CornacchiaRing):
    """Specialized factorization strategy for Gaussian integers Z[i]."""

    RAMIFIED_PRIME = 2
    SPLIT_K = 1

    @classmethod
    def _is_split_prime(cls, p: int) -> bool:
        return p % 4 == 1

    @classmethod
    def _is_inert_prime(cls, p: int) -> bool:
        return p % 4 == 3

    @classmethod
    def _ramified_generator(cls, x: QuadInt) -> QuadInt:
        return x._make(1, 1)

    @classmethod
    def _split_generator(cls, x: QuadInt, p: int, x0: int, y0: int) -> QuadInt:  # noqa: ARG003
        return x._make(x0, y0)

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:  # noqa: ARG003
        """Should this class be used for the given values?"""
        return D == -1 and den == 1


class SqrtMinusTwoRing(CornacchiaRing):
    """Specialized factorization strategy for Z[sqrt(-2)]."""

    RAMIFIED_PRIME = 2
    SPLIT_K = 2

    @classmethod
    def _is_split_prime(cls, p: int) -> bool:
        return p % 8 in (1, 3)

    @classmethod
    def _is_inert_prime(cls, p: int) -> bool:
        return p % 8 in (5, 7)

    @classmethod
    def _ramified_generator(cls, x: QuadInt) -> QuadInt:
        return x._make(0, 1)

    @classmethod
    def _split_generator(cls, x: QuadInt, p: int, x0: int, y0: int) -> QuadInt:  # noqa: ARG003
        return x._make(x0, y0)

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:  # noqa: ARG003
        """Should this class be used for the given values?"""
        return D == -2 and den == 1


class EisensteinRing(CornacchiaRing):
    """Specialized factorization strategy for Eisenstein integers Z[ω]."""

    RAMIFIED_PRIME = 3
    SPLIT_K = 3

    @classmethod
    def _is_split_prime(cls, p: int) -> bool:
        return p % 3 == 1

    @classmethod
    def _is_inert_prime(cls, p: int) -> bool:
        return p % 3 == 2

    @classmethod
    def _ramified_generator(cls, x: QuadInt) -> QuadInt:
        return x._make(3, 1)

    @classmethod
    def _inert_generator(cls, x: QuadInt, p: int) -> QuadInt:
        return x._make(2 * p, 0)

    @classmethod
    def _split_generator(cls, x: QuadInt, p: int, x0: int, y0: int) -> QuadInt:  # noqa: ARG003
        # Convert x0**2 + 3*y0**2 = p into internal numerator basis (A + B*sqrt(-3))/2.
        return x._make(2 * x0, 2 * y0)

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:  # noqa: ARG003
        """Should this class be used for the given values?"""
        return D == -3 and den == 2


# region Heegner rings
class HeegnerDen2Ring(EisensteinRing):
    """Shared split-prime factorization helper for D=-7 and D=-11 (den=2)."""

    SPLIT_K = 1  # unused by this strategy

    @staticmethod
    def _gcd_ring(a: QuadInt, b: QuadInt) -> QuadInt:
        """Compute a gcd using Euclidean division in norm-Euclidean rings."""
        x, y = a, b
        while y:
            x, y = y, divmod(x, y)[1]

        return x._canonical_associate()

    @classmethod
    def _is_split_prime(cls, p: int) -> bool:
        if p == cls.RAMIFIED_PRIME:
            return False
        if p == 2:
            return -cls.RAMIFIED_PRIME % 8 == 1
        return sqrt_mod(-cls.RAMIFIED_PRIME, p, all_roots=False) is not None

    @classmethod
    def _is_inert_prime(cls, p: int) -> bool:
        if p == cls.RAMIFIED_PRIME:
            return False
        if p == 2:
            return -cls.RAMIFIED_PRIME % 8 == 5
        return sqrt_mod(-cls.RAMIFIED_PRIME, p, all_roots=False) is None

    @classmethod
    def _ramified_generator(cls, x: QuadInt) -> QuadInt:
        return x._make(0, 2)

    @classmethod
    def _decompose_prime(cls, p: int) -> tuple[int, int]:
        """Return odd representatives of the two sqrt(D) roots modulo p."""
        root = sqrt_mod(-cls.RAMIFIED_PRIME, p, all_roots=False)
        if root is None:
            raise ValueError(f"Could not decompose {p!r}")

        t = int(root)
        if (t ^ 1) & 1:
            t += p

        t_alt = p - int(root)
        if (t_alt ^ 1) & 1:
            t_alt += p

        return t, t_alt

    @classmethod
    def _split_generator(cls, x: QuadInt, p: int, x0: int, y0: int) -> QuadInt:
        p_elem = x._make(2 * p, 0)
        cand = cls._gcd_ring(p_elem, x._make(x0, 1))
        if abs(cand) in (1, p * p):
            cand = cls._gcd_ring(p_elem, x._make(y0, 1))

        return cand

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:  # noqa: ARG003
        """Should this class be used for the given values?"""
        # No, this is purely a sub-abstract base class that needs to be subclassed
        return False


class HeegnerSevenRing(HeegnerDen2Ring):
    """Specialized factorization strategy for the maximal order with D=-7."""

    RAMIFIED_PRIME = 7

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:  # noqa: ARG003
        """Should this class be used for the given values?"""
        return D == -7 and den == 2


class HeegnerElevenRing(HeegnerDen2Ring):
    """Specialized factorization strategy for the maximal order with D=-11."""

    RAMIFIED_PRIME = 11

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:  # noqa: ARG003
        """Should this class be used for the given values?"""
        return D == -11 and den == 2


# endregion
# endregion


class Clark69Ring(QuadraticRing):
    """
    Euclidean division for the maximal order of Q(sqrt(69)), i.e. Z[(1+sqrt(69))/2].

    This ring is Euclidean but not norm-Euclidean (Clark, 1994).

    A working Euclidean function can be taken as: |N(x)| with a single tweak that replaces each prime
        factor "23" in |N(x)| by "26". Equivalently: if v23(|N(x)|)=e then
        phi(x) = (|N(x)| / 23**e) * 26**e

    The obstruction to norm-Euclidean-ness lives entirely at the primes above 23;
        inflating 23->26 fixes Euclidean descent.
    """

    SUPPORTS_DIVISION: ClassVar[bool] = True

    _BAD_P: ClassVar[int] = 23
    _BAD_REPL: ClassVar[int] = 26

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:
        """Supported for the maximal order for D=69 only"""
        return D == 69 and den == default_den

    @classmethod
    def _phi_from_abs_norm(cls, abs_norm: int) -> int:
        """Compute Clark's adjusted Euclidean function value from the integer |N(x)|."""
        n = int(abs_norm)
        if n < 0:
            n = -n
        if n == 0:
            return 0

        p = cls._BAD_P
        e = 0
        while (n % p) == 0:
            n //= p
            e += 1

        if e:
            n *= pow(cls._BAD_REPL, e)

        return n

    def phi(self, x: QuadInt) -> int:
        """Return Clark's adjusted Euclidean function for the `D=69` maximal order."""
        return self._phi_from_abs_norm(super().phi(x))

    def divmod(self, x: QuadInt, y: QuadInt) -> tuple[QuadInt, QuadInt]:
        """Divide `x` by `y` and return a Clark-admissible quotient and remainder."""
        # Same scaffolding as RealNormEuclidRing.divmod, but accept via phi() instead of |N|.
        y_norm = abs(y)  # signed norm (may be negative for D>0)
        abs_y_norm = abs(y_norm)
        if abs_y_norm == 0:
            raise ZeroDivisionError

        phi_y = self._phi_from_abs_norm(abs_y_norm)
        phi_y2 = phi_y * phi_y

        # Candidate center from x/y ≈ (x * conj(y)) / N(y)
        a1, b1 = x.a, x.b
        a2, b2 = y.a, y.b

        # num = x * conj(y), computed in numerators directly:
        # (a1+b1√D)(a2-b2√D) = (a1*a2 - b1*b2*D) + (a2*b1 - a1*b2)√D
        num_a = a1 * a2 - b1 * b2 * self.D
        num_b = a2 * b1 - a1 * b2

        if self.den != 1:
            if (num_a % self.den) != 0 or (num_b % self.den) != 0:
                raise ArithmeticError("Non-integral product; check ring parameters / parity")
            num_a //= self.den
            num_b //= self.den

        A0 = _round_div_ties_away_from_zero(num_a, y_norm)
        B0 = _round_div_ties_away_from_zero(num_b, y_norm)

        dd = self.den * self.den  # here dd=4

        def B0_for_A(A: int) -> int:  # noqa: ARG001
            return B0

        # Prefer any phi-reducing remainder; among those, minimize phi(w), then distance to (A0,B0).
        #
        # We work with w = q*N(y) - x*conj(y) = -(x-qy)*conj(y).
        # Since phi is multiplicative (by construction), phi(w) = phi(x-qy)*phi(y),
        # so phi(x-qy) < phi(y)  <=>  phi(w) < phi(y)^2.
        def score_for_AB(A: int, B: int) -> tuple[int, ...]:
            da = A * y_norm - num_a
            db = B * y_norm - num_b

            # numerator of N(w) where w=(da + db*sqrt(D))/den
            nw_num = da * da - self.D * (db * db)
            abs_nw_num = abs(nw_num)

            if abs_nw_num % dd:
                # Should not happen if parity/integrality is consistent, but be safe.
                return (1, abs_nw_num, (A - A0) * (A - A0) + (B - B0) * (B - B0))

            abs_nw = abs_nw_num // dd
            phi_w = self._phi_from_abs_norm(abs_nw)

            flag = 0 if phi_w < phi_y2 else 1
            dist2 = (A - A0) * (A - A0) + (B - B0) * (B - B0)
            return flag, phi_w, dist2

        # A slightly bigger radius schedule than the norm-euclid case, just in case.
        search = _NeighborhoodSearch(
            A0=A0,
            B0_for_A=B0_for_A,
            score_for_AB=score_for_AB,
            den=self.den,
        )

        for rad in (1, 2, 3, 4, 6, 8, 12, 16):
            best_a, best_b = search.expand_to(rad)

            best_score = search.best_score
            if best_score is not None and best_score[0] == 0:
                q = x._make(best_a, best_b)
                r = x - q * y
                return q, r

        raise NotImplementedError(
            f"No phi-reducing quotient found for D={self.D}, den={self.den} within search radii",
        )


def _is_squarefree(n: int) -> bool:
    n = abs(n)
    if n <= 1:
        return False

    facts = factorint(n)
    return all(i < 2 for i in facts.values())


class HarperRing(QuadraticRing):
    """
    Harper-style Euclidean division for selected real quadratic maximal orders.

    This implementation uses admissible prime-pair witnesses to define a weighted
    Euclidean score `phi` and then performs a nearest-lattice quotient search.

    We can really only do the Harper-like method with cypari, and even then it requires care to
        test the witness speedup.

    Therefor without cypari, this will only work for the D values that have been hard-coded and validated.
    Even with cypari, the default behavior will be to tend towards accuracy, so division algorithms may be slow
        (or heck, untested or incorrect).
    """

    SUPPORTS_DIVISION = True  # once divmod is implemented

    # According to the rules, any D value added here (with default den):
    #   * Must be square free (no prime factors with an exponent 2 or greater).
    #   * Must have class number 1 (_class_number_is_one is True).
    #   * Must have an admissible prime pair.
    #
    # This is a list of witness primes OR principal generators (which are defined in the _POST_HARDCODED list below)
    _HARDCODED: ClassVar[dict[tuple[int, int], tuple[int, int, int, int] | tuple[QuadInt, QuadInt]]] = {
        (14, 1): (5, 1, 43, 1),
        (22, 1): (3, 1, 29, 1),
        (23, 1): (11, 1, 13, 1),
        (31, 1): (3, 1, 5, 1),
        (43, 1): (7, 1, 53, 1),
        (46, 1): (3, 1, 5, 1),
        (47, 1): (11, 1, 53, 1),
        (53, 2): (11, 1, 29, 1),
        (59, 1): (5, 1, 47, 1),
        (61, 2): (3, 1, 5, 1),
        (62, 1): (13, 1, 23, 1),
        (67, 1): (7, 1, 149, 1),
        # While this algorithm can apply to D=69 (with den=2), it is less efficient.
        #   Its best to keep Clark69Ring for D=69 for efficiency
        #       It can also technically apply to norm-Euclidean rings, but we don't use it for them either...
        # (69, 2): (11, 1, 53, 1),
        #
        # D=71 was fun and required further investigation and expansion of this algorithm.
        #   Essentially for this ring, witnesses are simply not good enough for division.
        #   Instead they need to be converted to principal generators (which are quadratic integers),
        #       and use a slightly different (read slower) algorithm.
        #   See _POST_HARDCODED at end of file.
        # (71, 1): (5, 23),
        (77, 2): (13, 1, 23, 1),
        # D=83 was also quite fun, and required expansion of the heuristic search area to beyond 60,000.
        #   I worry this means the heuristic search area will need to increase for larger D values...
        #   But frankly I don't know what the theoretical max should be?
        (83, 1): (19, 1, 29, 1),
        (86, 1): (5, 1, 7, 1),
        (89, 2): (11, 1, 17, 1),
        (93, 2): (7, 1, 11, 1),
        (94, 1): (3, 1, 5, 1),
        (97, 2): (3, 1, 11, 1),
    }

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:
        """Read the above docstring and comments to understand the logic in this method."""
        # Harper-style only relevant for real quadratic maximal orders
        if D <= 0 or den != default_den:
            return False

        if (D, den) in cls._HARDCODED:
            return True

        try:
            import cypari  # noqa: PLC0415, F401
        except ImportError:
            warnings.warn(
                "This may be a Harper-like ring, however without cypari installed we cannot use it as such.",
                ImportWarning,
                stacklevel=3,
            )
            return False

        if not _is_squarefree(D):
            warnings.warn(
                "D is not squarefree, and may be an alias for a more primitive ring.",
                RuntimeWarning,
                stacklevel=3,
            )
            return False

        temp_ring = cls(D, den)
        disc = temp_ring.discriminant()

        # if isqrt(abs(disc)) ** 2 == abs(disc):
        #     return False

        # Must be a PID
        if temp_ring.class_number() != 1:
            return False

        # Murty-Srinivas-Subramani state that Harper's thesis established
        #   all real quadratic fields with discriminant ≤ 500
        #   and class number one are Euclidean.
        if disc <= 500:
            return True

        return temp_ring._find_admissible_witness_primes() is not None

    # region Finding admissible primes
    #   The code in this region is the code needed to add new entries to the HARDCODED caches above and below.
    #   The process for that should be to find admissible witness primes, then add them to _HARDCODED above.
    #       Then run the test suite. If it does not pass then the witness speed-up trick will not work.
    #   You will need to turn the witness primess into the more reliable principal generators,
    #       by passing the witness primes set to _principal_generators_from_witness, and then add these
    #       to _POST_HARDCODED at the end of the file.
    @requires_modules(["cypari"])
    def _find_admissible_witness_primes(
        self,
        *,
        prime_bound: int = 200,
    ) -> tuple[int, int, int, int] | None:
        """
        Search for a Harper-style admissible prime-ideal pair for the real quadratic field
        with discriminant `disc`, and return a *Harper-like* witness (small generators).

        Returns:
            None if not found within prime_bound.
            Otherwise (p1, i1, p2, i2) where:
              - p1, p2 are rational primes
              - i1 is the 1-based index of the chosen prime ideal in idealprimedec(nf,p1)
              - i2 is the 1-based index of the chosen prime ideal in idealprimedec(nf,p2)
        """
        from cypari import pari  # noqa: PLC0415

        disc0 = int(self.discriminant())
        B = int(prime_bound)

        # We do the heavy lifting in GP to avoid depending on cypari object APIs.
        # Strategy:
        #  - Build bnf/nf and fundamental unit eps.
        #  - Loop over rational primes p1<p2 up to B.
        #  - For each split prime ideal P with Norm(P)=p, try pairs (P1,P2).
        #  - Let I=P1^2*P2^2, compute bid=idealstar(nf,I,2) with invariants cyc.
        #  - Compute v(-1), v(eps) = ideallog(...) vectors mod cyc.
        #  - Decide if these generate the full group via Smith normal form index test.
        #
        # Return [] if none found, or [p1,i1,p2,i2] if found.

        gp = f"""
    {{
      my(disc={disc0}, B={B});
      my(bnf = bnfinit(quadpoly(disc)));
      my(nf  = bnf.nf);
      my(eps = Vec(bnf.fu)[1]);   /* normalize */

      my(res = []);
      my(found = 0);

      forprime(p1=3, B,
        if(found, break);
        if(p1==2 || (disc % p1)==0, next);

        my(dec1 = Vec(idealprimedec(nf, p1)));
        for(i1=1, #dec1,
          if(found, break);
          my(P1 = dec1[i1]);
          if(idealnorm(nf, P1) != p1, next);  /* split prime ideal of norm p1 */

          forprime(p2=p1+1, B,
            if(found, break);
            if(p2==2 || (disc % p2)==0, next);

            my(dec2 = Vec(idealprimedec(nf, p2)));
            for(i2=1, #dec2,
              if(found, break);
              my(P2 = dec2[i2]);
              if(idealnorm(nf, P2) != p2, next);

              my(I = idealmul(nf, idealpow(nf, P1, 2), idealpow(nf, P2, 2)));
              my(bid = idealstar(nf, I, 2));

              my(cyc = Vec(bid.cyc));
              my(k = #cyc);

              /* trivial group => surjective */
              if(k==0, res = [p1,i1,p2,i2]; found=1; break);

              my(v1 = Vec(ideallog(nf, -1,  bid)));
              my(v2 = Vec(ideallog(nf, eps, bid)));

              /* M = [diag(cyc) | v1 | v2] is k x (k+2) over Z */
              my(M = matrix(k, k+2, i,j, 0));
              for(i=1,k,
                M[i,i]   = cyc[i];
                M[i,k+1] = v1[i];
                M[i,k+2] = v2[i];
              );

              /* Index of lattice generated by columns:
                 idx = abs(det(mathnf(M))) when rank = k */
              my(H = mathnf(M));
              my(idx = abs(matdet(H)));

              if(idx==1, res = [p1,i1,p2,i2]; found=1; break);
            );
          );
        );
      );

      res
    }}
    """
        out = pari(gp)

        # `out` is either [] (empty GP vector) or [p1,i1,p2,i2].
        try:
            if len(out) == 0:
                return None
            # PARI vectors are 1-indexed; cypari wrappers usually expose 0-indexed python access.
            # We therefore read by python indexing first, and fall back to 1-index style if needed.
            try:
                p1, i1, p2, i2 = (int(out[0]), int(out[1]), int(out[2]), int(out[3]))
            except (ValueError, TypeError):
                p1, i1, p2, i2 = (int(out[1]), int(out[2]), int(out[3]), int(out[4]))
        except Exception:
            # Extremely defensive: if wrapper doesn't support len()/indexing cleanly
            s = str(out)
            if s in ("[]", "Vecsmall([])", "vector([])"):
                return None
            raise

        return p1, i1, p2, i2

    @requires_modules(["cypari"])
    def _principal_generator_from_witness_prime(self, p: int, i: int) -> QuadInt:
        """
        Convert one PARI witness component (p, i) into a principal generator π of the
            chosen prime ideal P = idealprimedec(nf,p)[i], returned as a QuadInt.

        The result is only defined up to multiplication by a unit.

        Returns:
            QuadInt: The principal generator π.
        """
        from cypari import pari  # noqa: PLC0415

        disc = int(self.discriminant())
        p0 = int(p)
        i0 = int(i)
        den0 = int(self.den)

        # We use bnfinit(...,1) so PARI has exact algebraic data for generators/units.
        # Then:
        #   P = idealprimedec(nf,p)[i]
        #   [e,t] = bnfisprincipal(bnf,P)
        # In class number 1, e is empty (or zero vector), and t generates P up to units.
        #
        # Convert t to integral-basis coordinates c = [c0,c1] relative to nf.zk = [1,w]
        # where w = sqrt(D) if den=1, and w = (1+sqrt(D))/2 if den=2.
        # Then map to internal numerators (a + b*sqrt(D))/den:
        #   den=1: a=c0, b=c1
        #   den=2: c0 + c1*w = (2*c0 + c1 + c1*sqrt(D))/2
        #          so internal (a,b) = (2*c0 + c1, c1)
        gp = f"""
    {{
      my(disc={disc}, p={p0}, idx={i0}, den={den0});
      my(bnf = bnfinit(quadpoly(disc), 1));
      my(nf  = bnf.nf);

      my(dec = Vec(idealprimedec(nf, p)));
      if(idx < 1 || idx > #dec, error("witness index out of range"));

      my(P = dec[idx]);

      /* Must be a prime ideal of norm p (split case in your search). */
      if(idealnorm(nf, P) != p, error("idealprimedec witness is not norm-p prime ideal"));

      my(v = bnfisprincipal(bnf, P));  /* [e, t] */
      my(e = v[1], t = v[2]);

      /* In class number 1, PARI returns empty e-vector; in general, require trivial class. */
      if(#e > 0,
        for(k=1, #e, if(e[k] != 0, error("prime ideal is not principal in this field")))
      );

      my(c = Vec(nfalgtobasis(nf, t)));
      if(#c != 2, error("expected quadratic field basis coordinates"));

      /* c = [c0,c1] in integral basis [1,w]. Convert to your QuadInt numerators. */
      if(den == 1,
        [c[1], c[2]],
        [2*c[1] + c[2], c[2]]
      )
    }}
    """
        out = pari(gp)

        try:
            a = int(out[0])
            b = int(out[1])
        except (ValueError, TypeError):
            a = int(out[1])
            b = int(out[2])

        x = self(a, b)

        # Normalize only by ± and conjugation so literature comparison is obvious.
        # (Real quadratic torsion units are usually ±1, so this is the relevant ambiguity.)
        candidates = (x, -x, x.conjugate(), (-x).conjugate())

        def key(z: QuadInt) -> tuple[int, int, int, int]:
            # prefer smaller |a|, then |b|, then sign-tie-breakers
            return abs(z.a), abs(z.b), z.a, z.b

        return min(candidates, key=key)

    def _principal_generators_from_witness(
        self,
        witness: tuple[int, int, int, int],
    ) -> tuple[QuadInt, QuadInt]:
        """Convert a full admissible witness (p1,i1,p2,i2) into two prime generators."""
        p1, i1, p2, i2 = witness
        pi1 = self._principal_generator_from_witness_prime(p1, i1)
        pi2 = self._principal_generator_from_witness_prime(p2, i2)

        # Order deterministically for easy comparison
        if (abs(pi2.a), abs(pi2.b), abs(abs(pi2))) < (abs(pi1.a), abs(pi1.b), abs(abs(pi1))):
            pi1, pi2 = pi2, pi1
        return pi1, pi2

    # endregion

    def _phi_from_abs_norm(self, abs_norm: int, witness: tuple) -> int:
        """
        Weighted-norm score used as a Harper-style search heuristic.

        Start from |N(x)| and replace selected rational-prime factors p by p+1
            (based on an admissible-pair witness), analogous in spirit to Clark69's 23->26 trick.

        Returns:
            int: Phi
        """
        n = int(abs_norm)
        if n < 0:
            n = -n
        if n == 0:
            return 0

        p1, _, p2, _ = witness
        replacements = {p1: p1 + 1, p2: p2 + 1}

        for p, p_new in replacements.items():
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            if e:
                n *= pow(p_new, e)

        return n

    def phi(self, x: QuadInt) -> int:
        """Return Harper's weighted Euclidean size for `x`."""
        n = super().phi(x)  # |N(x)| in your QuadraticRing base
        if n == 0:
            return 0

        cached_pair: tuple = self._HARDCODED.get((self.D, self.den), ())
        if len(cached_pair) == 4:
            return self._phi_from_abs_norm(n, cached_pair)  # faster fallback for witness cache entries

        if len(cached_pair) == 0:
            witness = self._find_admissible_witness_primes()
            if witness is None:
                raise RuntimeError("This should never happen but mypyc needs it.")

            cached_pair = self._principal_generators_from_witness(witness)

        out = n
        for pi in cached_pair:
            p = abs(abs(pi))  # |N(pi)| = rational prime p
            e = self._valuation_at_generator(x, pi)
            if e:
                out //= p**e
                out *= (p + 1) ** e
        return out

    # region Fallback helpers
    #   The methods in this region are only used if the witness primes speed-up trick fails,
    #       or obviously if that has not been manually validated.
    def _try_exact_quotient(self, x: QuadInt, y: QuadInt) -> QuadInt | None:
        """
        Return q if x == q*y exactly in this ring, else None.

        This avoids calling divmod() (and therefore avoids phi()/divmod recursion)
        when phi() wants valuations at selected Harper generators.

        Returns:
            QuadInt: Solely the quotient if the remainder is 0, else None.
        """
        y_norm = abs(y)  # signed norm
        if y_norm == 0:
            return None

        # num = x * conj(y), in numerator coordinates (same convention as divmod code)
        num_a = x.a * y.a - x.b * y.b * self.D
        num_b = y.a * x.b - x.a * y.b

        if self.den != 1:
            if (num_a % self.den) != 0 or (num_b % self.den) != 0:
                return None
            num_a //= self.den
            num_b //= self.den

        if (num_a % y_norm) != 0 or (num_b % y_norm) != 0:
            return None

        A = num_a // y_norm
        B = num_b // y_norm

        if self.den == 2 and ((A ^ B) & 1):
            return None

        return x._make(A, B)

    def _valuation_at_generator(self, x: QuadInt, pi: QuadInt) -> int:
        """v_pi(x) for a fixed chosen principal prime generator pi (ideal-specific)."""
        e = 0
        rem = x
        while rem:
            q = self._try_exact_quotient(rem, pi)
            if q is None:
                break
            rem = q
            e += 1
        return e

    # endregion

    def divmod(self, x: QuadInt, y: QuadInt) -> tuple[QuadInt, QuadInt]:
        """
        Practical Harper-style division search.

        This reuses the local lattice search used in RealNormEuclidRing / Clark69Ring,
        but scores candidates with a weighted norm heuristic based on an admissible-pair
        witness (when available). Empirical validation via tests is essential.

        Returns:
            tuple: The quotient and remainder.

        Raises:
            ZeroDivisionError: If y has an absolute norm of 0.
            ArithmeticError: TODO: Remove?
            NotImplementedError: If we were unable to find a quotient and remainder.
                Shouldn't happen. If it does, please contact a developer. Preferably one smarter than me.
        """
        y_norm = abs(y)  # signed norm (may be negative for D>0)
        abs_y_norm = abs(y_norm)
        if abs_y_norm == 0:
            raise ZeroDivisionError

        # Use the actual Harper phi on the actual divisor.
        # (Do NOT use the old phi(w) < phi(y)^2 shortcut once phi is ideal-sensitive.)
        phi_y = self.phi(y)

        # Candidate center from x/y ≈ (x * conj(y)) / N(y)
        a1, b1 = x.a, x.b
        a2, b2 = y.a, y.b

        # num = x * conj(y), in numerator coordinates:
        # (a1+b1√D)(a2-b2√D) = (a1*a2 - b1*b2*D) + (a2*b1 - a1*b2)√D
        num_a = a1 * a2 - b1 * b2 * self.D
        num_b = a2 * b1 - a1 * b2

        if self.den != 1:
            if (num_a % self.den) != 0 or (num_b % self.den) != 0:
                raise ArithmeticError("Non-integral product; check ring parameters / parity")
            num_a //= self.den
            num_b //= self.den

        A0 = _round_div_ties_away_from_zero(num_a, y_norm)
        B0 = _round_div_ties_away_from_zero(num_b, y_norm)

        def B0_for_A(A: int) -> int:  # noqa: ARG001
            return B0

        def score_for_AB(A: int, B: int) -> tuple[int, ...]:
            q = x._make(A, B)
            r = x - q * y
            pr = self.phi(r)
            dist2 = (A - A0) * (A - A0) + (B - B0) * (B - B0)
            flag = 0 if pr < phi_y else 1
            return flag, pr, dist2

        # Wider schedule than norm-euclid / Clark69; these cases are trickier.
        search = _NeighborhoodSearch(
            A0=A0,
            B0_for_A=B0_for_A,
            score_for_AB=score_for_AB,
            den=self.den,
        )

        for rad in (1, 2, 4, 8, 16, 32):
            best_a, best_b = search.expand_to(rad)

            best_score = search.best_score
            if best_score is not None and best_score[0] == 0:
                q = x._make(best_a, best_b)
                r = x - q * y
                return q, r

        # Branch-aware fallback for real quadratic indefinite norm.
        # For fixed A, small |da^2 - D*db^2| tends to occur near db ~= +/- |da|/sqrt(D),
        # which may correspond to B far away from the naive center B0.
        sqrtD = self.D**0.5
        best_q: QuadInt | None = None
        best_r: QuadInt | None = None
        seen: set[tuple[int, int]] = set()

        phi = self.phi
        make = x._make
        den = self.den

        def consider(A: int, B: int) -> None:
            """Score one lattice candidate (A,B) exactly once."""
            nonlocal best_score, best_q, best_r

            if den == 2 and ((A ^ B) & 1):
                return

            key = (A, B)
            if key in seen:
                return
            seen.add(key)

            q = make(A, B)
            r = x - q * y
            pr = phi(r)
            dist2 = (A - A0) * (A - A0) + (B - B0) * (B - B0)
            s = (0 if pr < phi_y else 1, pr, dist2)

            if best_score is None or s < best_score:
                best_score = s
                best_q = q
                best_r = r

        def has_reducing_best() -> bool:
            return best_score is not None and best_score[0] == 0

        @cache
        def _candidate_Bs_for_A(A: int) -> tuple[int, ...]:
            """Generate a small cached set of promising B values for this A."""
            da = A * y_norm - num_a
            cands: set[int] = set()

            def add_with_parity(Bcand: int):
                if den == 1:
                    cands.add(Bcand)
                    return

                # den==2 parity constraint: A ≡ B (mod 2)
                if ((A ^ Bcand) & 1) == 0:
                    cands.add(Bcand)
                else:
                    cands.add(Bcand - 1)
                    cands.add(Bcand + 1)

            # Center-ish values
            for dB in (-2, -1, 0, 1, 2):
                add_with_parity(B0 + dB)

            # Hyperbola branch targets: db ~= +/- |da| / sqrt(D)
            # where db = B*y_norm - num_b
            t = abs(da) / sqrtD

            for sgn in (-1.0, 1.0):
                db_target = sgn * t
                Bf = (num_b + db_target) / y_norm
                Bc = round(Bf)

                for dB in range(-4, 5):
                    add_with_parity(int(Bc) + dB)

            # Midpoint spread
            mid = round(num_b / y_norm)
            for dB in (-3, -2, -1, 0, 1, 2, 3):
                add_with_parity(int(mid) + dB)

            return tuple(sorted(cands))

        prev_branch_rad = -1

        for rad in (64, 128, 256, 512, 1024, 2048, 4096, 65536):
            a_ranges: tuple[range, ...]
            if prev_branch_rad < 0:
                # First pass: scan full A range once
                a_ranges = (range(A0 - rad, A0 + rad + 1),)
            else:
                # Later passes: scan only the new A annulus
                a_ranges = (
                    range(A0 - rad, A0 - prev_branch_rad),
                    range(A0 + prev_branch_rad + 1, A0 + rad + 1),
                )

            for a_range in a_ranges:
                for A in a_range:
                    for B in _candidate_Bs_for_A(A):
                        consider(A, B)

            prev_branch_rad = rad

            if has_reducing_best() and best_q is not None and best_r is not None:
                return best_q, best_r

        raise NotImplementedError(
            f"No Harper-style phi-reducing quotient found for D={self.D}, den={self.den} "
            "within search radii; expand radius or refine weighted-phi construction",
        )


# While I've defined principal generators for all not-norm-Euclidean Euclidean fields with D<100 here,
#   the witness primes defined in _HARDCODED work for most D and the algorithm with them is faster.
_POST_HARDCODED = {
    # The following admissible primes are the ones Harper originally found:
    (14, 1): ((5, -1), (3, -2)),
    #   I've done my best to re-create the method Harper used to find these, and while I do find them as candidates,
    #       my best attempt at canonicalization of the candidates (using the code above) will find these:
    # (14, 1) = ((-1, -1), (-3, -2))
    #   While this works, I'm going to stick with Harper's originals here to pay homage.
    #       This is only used for validation of the above algorithm with the literature though anyway...
    (22, 1): ((-5, -1), (-13, -3)),
    (23, 1): ((-6, -1), (-9, -2)),
    (31, 1): ((-6, -1), (-11, -2)),
    (43, 1): ((-6, -1), (-15, -2)),
    (46, 1): ((-7, -1), (-61, -9)),
    (47, 1): ((-6, -1), (-10, -1)),
    (53, 2): ((-3, -1), (-13, -1)),
    (59, 1): ((-8, -1), (-22, -3)),
    (61, 2): ((-7, -1), (-9, -1)),
    (62, 1): ((-7, -1), (-15, -2)),
    (67, 1): ((-90, -11), (-156, -19)),
    # (69, 2): ((-5, -1), (-8, -2)),
    (71, 1): ((-17, -2), (-101, -12)),
    (77, 2): ((-5, -1), (-13, -1)),
    (83, 1): ((-8, -1), (-19, 23)),
    (86, 1): ((-9, -1), (-37, -4)),
    (89, 2): ((-20, -2), (-56, -6)),
    (93, 2): ((-7, -1), (-11, -1)),
    (94, 1): ((-29, -3), (-223, -23)),
    (97, 2): ((-20, -2), (-118, -12)),
}


for (D_, den_), ((p1_a_, p1_b_), (p2_a_, p2_b_)) in _POST_HARDCODED.items():
    if (D_, den_) in HarperRing._HARDCODED:
        continue

    _ring = HarperRing(D_, den_)
    HarperRing._HARDCODED[D_, den_] = (_ring(p1_a_, p1_b_), _ring(p2_a_, p2_b_))
