from collections import defaultdict
from dataclasses import dataclass
from math import isqrt, prod
from typing import Callable, ClassVar, Union

from sympy import factorint, sqrt_mod

from quadint.quad.int import OP_TYPES, QuadInt

NORM_EUCLID_D: set[int] = {-11, -7, -3, -2, -1, 2, 3, 5, 6, 7, 11, 13, 17, 19, 21, 29, 33, 37, 41, 57, 73}


# TODO: Once Py3.9 support has been dropped, add slots=True
# @dataclass(frozen=True, slots=True)
@dataclass(frozen=True)
class Factorization:
    """x = unit * P1 * P2 * ... * Pk"""

    unit: "QuadInt"
    primes: dict["QuadInt", int]

    def prod(self):
        """Recreate the number using prod"""
        return prod((p**k for p, k in self.primes.items()), start=self.unit)


def _round_div_ties_away_from_zero(n: int, d: int) -> int:
    """Round n/d to nearest int; ties go away from 0. d must be > 0."""
    if d == 0:
        raise ZeroDivisionError
    if d < 0:
        n, d = -n, -d
    if n >= 0:
        return (n + d // 2) // d
    return -((-n + d // 2) // d)


def _split_uv(x: "QuadInt") -> tuple[int, int]:
    """Return (u,v) for D=1 split-complex where u=(a+b)/den, v=(a-b)/den."""
    den = x.ring.den
    apb = x.a + x.b
    amb = x.a - x.b
    if apb % den or amb % den:
        # should be impossible if ring invariants hold
        raise ArithmeticError("Non-integral split-complex coordinates; check ring parameters/parity")
    return apb // den, amb // den


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
    best_score: Union[tuple[int, ...], None] = None
    bestA = bestB = 0

    for A in range(A0 - radius, A0 + radius + 1):
        B0 = B0_for_A(A)
        for B in range(B0 - radius, B0 + radius + 1):
            if den == 2 and ((A ^ B) & 1):
                continue

            s = score_for_AB(A, B)
            if best_score is None or s < best_score:
                best_score = s
                bestA, bestB = A, B

    return bestA, bestB


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

    _CACHE: ClassVar[dict[tuple[int, int], object]] = {}

    D: int
    den: int

    def __new__(cls, D: int, den: Union[int, None] = None):
        """Handle singleton logic"""
        D0 = int(D)
        default_den = 2 if (D0 % 4) == 1 else 1
        den0 = default_den if den is None else int(den)

        key = (D0, den0)
        inst = cls._CACHE.get(key)
        if inst is not None:
            return inst

        # choose subclass
        new_inst: QuadraticRing
        if D0 == 0:
            new_inst = DualRing(D0, den0)
        elif D0 == 1:
            new_inst = SplitRing(D0, den0)
        elif D0 == -1 and den0 == 1:
            new_inst = GaussianRing(D0, den0)
        elif D0 == -2 and den0 == 1:
            new_inst = SqrtMinusTwoRing(D0, den0)
        elif D0 == -3 and den0 == 2:
            new_inst = EisensteinRing(D0, den0)
        elif D0 == -7 and den0 == 2:
            new_inst = HeegnerSevenRing(D0, den0)
        elif D0 == -11 and den0 == 2:
            new_inst = HeegnerElevenRing(D0, den0)
        elif D0 in NORM_EUCLID_D and den0 == default_den:
            new_inst = RealNormEuclidRing(D0, den0)
        else:
            new_inst = super().__new__(cls)

        cls._CACHE[key] = new_inst
        return new_inst

    def __init__(self, D: int, den: Union[int, None] = None) -> None:
        """Initialize the ring settings"""
        self.D = int(D)
        self.den = (2 if (self.D % 4) == 1 else 1) if den is None else int(den)

    def __repr__(self) -> str:
        return f"QuadraticRing(D={self.D}, den={self.den})"

    def __call__(self, a: int = 0, b: int = 0) -> "QuadInt":
        """Create element (a + b*sqrt(D))/den with numerator coefficients a,b."""
        return QuadInt(self, int(a), int(b))

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
    def zero(self) -> "QuadInt":
        """Additive identity (0)."""
        return QuadInt(self, 0, 0)

    @property
    def one(self) -> "QuadInt":
        """Multiplicative identity (1)."""
        return QuadInt(self, self.den, 0)

    def from_obj(self, n: OP_TYPES) -> "QuadInt":
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
            return n
        else:
            return NotImplemented

        # The only time b is not 0 is if self.den is 1 anyway... No need to multiply
        return QuadInt(self, a * self.den, b)

    def from_ab(self, a: int, b: int) -> "QuadInt":
        """Create an integer keeping in mind the denominator"""
        da = int(a) * self.den
        db = int(b) * self.den
        return QuadInt(self, da, db)

    def divmod(self, x: "QuadInt", y: "QuadInt"):
        """An override for defining division algorithms in subclasses for different D values"""
        raise NotImplementedError

    def factor_detail(self, x: "QuadInt") -> "Factorization":
        """Factor `x` and return structured details when supported by this ring."""
        raise NotImplementedError("Factorization is not implemented for this ring")

    def factor(self, x: "QuadInt") -> dict["QuadInt", int]:
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


class DualRing(QuadraticRing):
    """
    Handle overrides for D=0, dual integer solutions

    While the general algorith in RealNormEuclidRing will find a solution for D=0,
        it does not take into account that the ε part is not part of the norm,
        so is not relevant in the division algorithm.

    This class will solve division for the real part while trying to minimize the ε part.
    """

    def __new__(cls, D: int, den: Union[int, None] = None):  # noqa: ARG004
        """Don't go to superclass logic, just create the object. Needed for mypyc"""
        return object.__new__(cls)

    def divmod(self, x: "QuadInt", y: "QuadInt"):
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

        bestA, bestB = _choose_best_in_neighborhood(
            A0=A0,
            B0_for_A=B0_for_A,
            score_for_AB=score_for_AB,
            den=self.den,
        )

        q = x._make(bestA, bestB)
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

    def __new__(cls, D: int, den: Union[int, None] = None):  # noqa: ARG004
        """Don't go to superclass logic, just create the object. Needed for mypyc"""
        return object.__new__(cls)

    def divmod(self, x: "QuadInt", y: "QuadInt"):
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

    def __new__(cls, D: int, den: Union[int, None] = None):  # noqa: ARG004
        """Don't go to superclass logic, just create the object. Needed for mypyc."""
        return object.__new__(cls)

    def divmod(self, x: "QuadInt", y: "QuadInt"):
        """
        Division in a norm-Euclidean real quadratic ring.

        We use the absolute norm as the Euclidean function:
            f(z) = |N(z)|.

        For the known finite list of D where the ring of integers is norm-Euclidean,
        there exists q such that |N(x - qy)| < |N(y)|.

        Raises:
            ZeroDivisionError: If the magnitude of the divisor is 0.
            ArithmeticError: In the event of a parity mismatch.

        Returns:
            q, r: The quotient and remainder
        """
        Ny = abs(y)  # signed norm (may be negative for D>0)
        absNy = abs(Ny)
        if absNy == 0:
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

        A0 = _round_div_ties_away_from_zero(num_a, Ny)
        B0 = _round_div_ties_away_from_zero(num_b, Ny)
        dd = self.den**2
        threshold = absNy * absNy * dd

        def B0_for_A(A: int) -> int:  # noqa: ARG001
            return B0

        # Prefer any norm-reducing remainder; among those, minimize |N(r)| then distance to (A0,B0).
        def score_for_AB(A: int, B: int) -> tuple[int, ...]:
            da = A * Ny - num_a
            db = B * Ny - num_b

            # numerator of N(w) where w=(da + db*sqrt(D))/den
            nw_num = da * da - self.D * (db * db)
            abs_nw_num = abs(nw_num)

            # norm-reducing condition: |N(w)| < |Ny|^2  <=>  |nw_num| < |Ny|^2 * den^2
            flag = 0 if abs_nw_num < threshold else 1

            dist2 = (A - A0) * (A - A0) + (B - B0) * (B - B0)
            return flag, abs_nw_num, dist2

        # Expand search radius until we find a norm-reducing remainder.
        for rad in (1, 2, 3, 4, 6, 8):
            bestA, bestB = _choose_best_in_neighborhood(
                A0=A0,
                B0_for_A=B0_for_A,
                score_for_AB=score_for_AB,
                den=self.den,
                radius=rad,
            )

            da = bestA * Ny - num_a
            db = bestB * Ny - num_b
            nw_num = da * da - self.D * (db * db)

            if abs(nw_num) < threshold:
                q = x._make(bestA, bestB)
                r = x - q * y
                return q, r

        raise NotImplementedError(
            f"No norm-reducing quotient found for D={self.D}, den={self.den} within search radii",
        )


class CornacchiaRing(RealNormEuclidRing):
    """Shared split-prime factorization flow for rings with norm form x**2 + k*y**2."""

    RAMIFIED_PRIME: ClassVar[int]
    SPLIT_K: ClassVar[int]

    @classmethod
    def _is_split_prime(cls, p: int) -> bool:
        raise NotImplementedError

    @classmethod
    def _is_inert_prime(cls, p: int) -> bool:
        raise NotImplementedError

    @classmethod
    def _ramified_generator(cls, x: "QuadInt") -> "QuadInt":
        raise NotImplementedError

    @classmethod
    def _inert_generator(cls, x: "QuadInt", p: int) -> "QuadInt":
        return x._make(p, 0)

    @classmethod
    def _split_generator(cls, x: "QuadInt", p: int, x0: int, y0: int) -> "QuadInt":
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

    def factor_detail(self, x: "QuadInt") -> "Factorization":
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

        Raises:
            ValueError: If `x` is zero (zero has no finite prime factorization).

        Returns:
            Factorization: The unit and prime-power data for ``x``.
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
    def _ramified_generator(cls, x: "QuadInt") -> "QuadInt":
        return x._make(1, 1)

    @classmethod
    def _split_generator(cls, x: "QuadInt", p: int, x0: int, y0: int) -> "QuadInt":  # noqa: ARG003
        return x._make(x0, y0)


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
    def _ramified_generator(cls, x: "QuadInt") -> "QuadInt":
        return x._make(0, 1)

    @classmethod
    def _split_generator(cls, x: "QuadInt", p: int, x0: int, y0: int) -> "QuadInt":  # noqa: ARG003
        return x._make(x0, y0)


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
    def _ramified_generator(cls, x: "QuadInt") -> "QuadInt":
        return x._make(3, 1)

    @classmethod
    def _inert_generator(cls, x: "QuadInt", p: int) -> "QuadInt":
        return x._make(2 * p, 0)

    @classmethod
    def _split_generator(cls, x: "QuadInt", p: int, x0: int, y0: int) -> "QuadInt":  # noqa: ARG003
        # Convert x0**2 + 3*y0**2 = p into internal numerator basis (A + B*sqrt(-3))/2.
        return x._make(2 * x0, 2 * y0)


class HeegnerDen2Ring(EisensteinRing):
    """Shared split-prime factorization helper for D=-7 and D=-11 (den=2)."""

    SPLIT_K = 1  # unused by this strategy

    @staticmethod
    def _gcd_ring(a: "QuadInt", b: "QuadInt") -> "QuadInt":
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
    def _ramified_generator(cls, x: "QuadInt") -> "QuadInt":
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
    def _split_generator(cls, x: "QuadInt", p: int, x0: int, y0: int) -> "QuadInt":
        p_elem = x._make(2 * p, 0)
        cand = cls._gcd_ring(p_elem, x._make(x0, 1))
        if abs(cand) in (1, p * p):
            cand = cls._gcd_ring(p_elem, x._make(y0, 1))

        return cand


class HeegnerSevenRing(HeegnerDen2Ring):
    """Specialized factorization strategy for the maximal order with D=-7."""

    RAMIFIED_PRIME = 7


class HeegnerElevenRing(HeegnerDen2Ring):
    """Specialized factorization strategy for the maximal order with D=-11."""

    RAMIFIED_PRIME = 11
