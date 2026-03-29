from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Callable, ClassVar

from quadint.quad.int import QuadInt
from quadint.utils import requires_modules


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

    __slots__ = ("D", "den", "DEFAULT_KLASS")

    SUPPORTS_DIVISION: ClassVar[bool] = False
    SUPPORTS_FACTORIZATION: ClassVar[bool] = False
    _CACHE: ClassVar[dict[tuple[int, int], object]] = {}

    D: int
    den: int
    DEFAULT_KLASS: type[QuadInt]

    def __new__(cls, D: int, den: int | None = None):
        """Handle singleton logic"""
        D0 = int(D)
        default_den = 2 if (D0 % 4) == 1 else 1
        den0 = default_den if den is None else _check_den(den)

        key = (D0, den0)
        inst = cls._CACHE.get(key)
        if inst is not None:
            return inst

        if cls is not QuadraticRing:
            new_inst = super().__new__(cls)
        else:
            # choose subclass
            new_inst: QuadraticRing
            for subcls in cls._subclasses():
                if subcls.accept_override(D0, den0, default_den):
                    new_inst = subcls(D0, den0)
                    break
            else:
                new_inst = super().__new__(cls)

        new_inst.DEFAULT_KLASS = QuadInt
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
        cls = self.DEFAULT_KLASS
        return cls(int(a), int(b), self)

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
        cls = self.DEFAULT_KLASS
        return cls(0, 0, self)

    @property
    def one(self) -> QuadInt:
        """Multiplicative identity (1)."""
        cls = self.DEFAULT_KLASS
        return cls(self.den, 0, self)

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
        cls = self.DEFAULT_KLASS
        return cls(a * self.den, b, self)

    def from_ab(self, a: int, b: int) -> QuadInt:
        """Create an integer keeping in mind the denominator"""
        da = int(a) * self.den
        db = int(b) * self.den
        cls = self.DEFAULT_KLASS
        return cls(da, db, self)

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

    def exact_div(self, x: QuadInt, y: QuadInt) -> QuadInt | None:
        """
        Return q if x == q*y exactly in this ring, else None.

        Uses the identity q = self*conj(divisor)/N(divisor) and then checks integrality
            in the order (including den=2 parity constraint).

        Returns:
            QuadInt: Solely the quotient if the remainder is 0, else None.
        """
        N = abs(y)  # signed norm
        if N == 0:
            # This happens in zero-divisor rings (D=0 dual, D=1 split) and for zero divisors.
            # TODO: Divisibility is still meaningful there, but needs a different solver.
            raise NotImplementedError

        # num = x * conj(y), in numerator coordinates (same convention as divmod code)
        num_a = x.a * y.a - x.b * y.b * self.D
        num_b = y.a * x.b - x.a * y.b

        # IMPORTANT: keep representation /den (same rule as __mul__ and divmod)
        if self.den != 1:
            if (num_a % self.den) != 0 or (num_b % self.den) != 0:
                return None
            num_a //= self.den
            num_b //= self.den

        # integrality test: coordinates divisible by |N|
        if (num_a % N) != 0 or (num_b % N) != 0:
            return None

        qa = num_a // N
        qb = num_b // N

        if self.den == 2 and ((qa ^ qb) & 1):
            return None

        return x._make(qa, qb)

    def divides(self, x: QuadInt, y: QuadInt) -> bool:
        """Return True iff x | y in this ring."""
        return self.exact_div(x, y) is not None

    def _canonicalize_bezout_result(self, g: QuadInt, s: QuadInt, t: QuadInt) -> tuple[QuadInt, QuadInt, QuadInt]:
        """
        Normalize a Bézout triple to the ring's canonical gcd representative.

        Given a triple (g, s, t) produced for some fixed inputs a and b
        with s*a + t*b == g, replace g by its canonical associate and
        multiply s and t by the same torsion unit so that the Bézout
        identity remains exactly true.

        This is a post-processing helper for xgcd-style routines. It makes gcd
        output deterministic across associate choices while preserving the original
        linear relation.

        Args:
            g: A gcd candidate, defined only up to multiplication by a unit.
            s: Bézout coefficient for the first input.
            t: Bézout coefficient for the second input.

        Returns:
            tuple[QuadInt, QuadInt, QuadInt]: A triple (g_can, s_can, t_can)
            where g_can == g._canonical_associate() and the same Bézout identity
            still holds with the adjusted coefficients.

        Notes:
            - Only torsion units are used to transport the identity.
            - If g is already canonical, the input triple is returned unchanged.
        """
        g_can = g._canonical_associate()
        if g_can != g:
            for u in g.units:
                if g * u == g_can:
                    g = g_can
                    s = s * u
                    t = t * u
                    break

        return g, s, t

    def xgcd(self, a: QuadInt, b: QuadInt) -> tuple[QuadInt, QuadInt, QuadInt]:
        """
        Extended gcd in Euclidean quadratic rings.

        Notes:
            - This is only implemented for rings with divmod support (Euclidean-style division).
            - The gcd is only defined up to multiplication by a unit; this returns a stable
              associate using QuadInt._canonical_associate() and adjusts (s,t) accordingly
              using the (finite) torsion unit list.

        Returns:
            (g, s, t): such that s*a + t*b == g
        """
        # TODO: For now: avoid the zero-divisor rings (dual), where "gcd" semantics differ.
        if self.D == 0:
            raise NotImplementedError("xgcd not implemented for D=0 (non-domains)")

        if not self.supports_division():
            raise NotImplementedError("xgcd requires Euclidean-style division (supports_division()==True)")

        # region Handle trivial cases
        if not a:
            g = b._canonical_associate()
            # Find unit u with u*b == g so that 0*a + u*b == g
            t = self.one
            if g != b:
                for u in b.units:
                    if u * b == g:
                        t = u
                        break
            return g, self.zero, t

        if not b:
            g = a._canonical_associate()
            # Find unit u with u*a == g so that u*a + 0*b == g
            s = self.one
            if g != a:
                for u in a.units:
                    if u * a == g:
                        s = u
                        break
            return g, s, self.zero
        # endregion

        r0, r1 = a, b
        s0, s1 = self.one, self.zero
        t0, t1 = self.zero, self.one

        while r1:
            q, r = self.divmod(
                r0,
                r1,
            )
            r0, r1 = r1, r
            s0, s1 = s1, s0 - q * s1
            t0, t1 = t1, t0 - q * t1

        # r0 is a gcd up to a unit. Normalize it for stable output, and adjust (s,t).
        return self._canonicalize_bezout_result(r0, s0, t0)

    def gcd(self, a: QuadInt, b: QuadInt) -> QuadInt:
        """
        Greatest common divisor in Euclidean quadratic rings.

        The result is only defined up to multiplication by a unit; this method returns the
        same stable representative as `xgcd()` (via `_canonical_associate()`), so callers
        get deterministic output.

        Notes:
            - Implemented via `xgcd()`, so it is available exactly when `xgcd()` is available.
            - Raises `NotImplementedError` for non-Euclidean rings (supports_division()==False)
              and for D==0 where the ring is not a domain.

        Returns:
            g: A *canonical associate* such that g divides both `a` and `b`,
                and for any `d` dividing both `a` and `b`, `d` also divides `g`.
        """
        g, _, _ = self.xgcd(a, b)
        return g

    def inv_mod(self, a: QuadInt, m: QuadInt) -> QuadInt:
        """
        Modular inverse in Euclidean quadratic rings.

        Returns:
            inv: such that (a * inv) % m == 1 % m

        Raises:
            ZeroDivisionError: if m == 0.
            ValueError: if a is not invertible modulo m (i.e. gcd(a, m) is not a unit).
        """
        if not m:
            raise ZeroDivisionError("modulus cannot be 0")

        # xgcd gives s*a + t*m == g
        g, s, _t = self.xgcd(a, m)

        # Invertible mod m  <=>  gcd(a,m) is a unit.
        # In these quadratic integer rings, unit <=> |N(g)| == 1.
        if abs(abs(g)) != 1:
            raise ValueError(f"{a} is not invertible mod {m} (gcd={g})")

        # g^{-1} = conjugate(g) / N(g), and N(g) is ±1 here.
        Ng = abs(g)  # signed norm
        g_inv = g.conjugate() if Ng == 1 else -g.conjugate()

        return (s * g_inv) % m

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
