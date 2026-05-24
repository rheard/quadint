from __future__ import annotations

import functools

from dataclasses import dataclass
from itertools import count
from math import isqrt, prod
from typing import TYPE_CHECKING, ClassVar

from sympy import isprime, sqrt_mod
from sympy.solvers.diophantine.diophantine import diop_DN

from quadint.quad.ideal import ClassGroup, Ideal
from quadint.quad.int import QuadInt
from quadint.utils import _is_squarefree

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


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


class PrimeIdealData:
    """
    A prime ideal over a rational prime, plus defining root data when available.

    If root is not None, then the ideal is represented as:

        P = (p, w - root)

    where w is the integral-basis generator:

        den == 1: w = sqrt(D)
        den == 2: w = (1 + sqrt(D)) / 2

    For odd unramified primes, root_p2 lazily lifts root from modulo p to modulo p**2.
    """

    __slots__ = ("ring", "p", "index", "ideal", "root")

    ring: QuadraticRing
    p: int
    index: int
    ideal: Ideal
    root: int | None

    def __init__(
        self,
        *,
        ring: QuadraticRing,
        p: int,
        index: int,
        ideal: Ideal,
        root: int | None = None,
    ) -> None:
        """Create prime-ideal data."""
        self.ring = ring
        self.p = int(p)
        self.index = int(index)
        self.ideal = ideal
        self.root = None if root is None else int(root)

    @functools.cache
    def root_p2(self) -> int | None:
        """Return root lifted modulo p**2 when this prime ideal has such a lift."""
        if self.root is None:
            return None

        # Harper will only use odd, unramified split primes here.
        # For p == 2 or ramified primes, the simple Hensel lift is not valid/useful.
        # TODO: Will this be a problem? I think it will be?
        if self.p == 2 or self.ring.discriminant() % self.p == 0:
            return None

        root = self.root
        p = self.p

        if self.ring.den == 1:
            # f(x) = x**2 - D
            f_root = root * root - self.ring.D
            f_prime_root = 2 * root
        else:
            # f(x) = x**2 - x + (1-D)//4
            c = (1 - self.ring.D) // 4
            f_root = root * root - root + c
            f_prime_root = 2 * root - 1

        # We want root_p2 = root + t*p, with f(root_p2) == 0 mod p**2.
        #
        # Since f(root + t*p) == f(root) + t*p*f'(root) mod p**2,
        # we need:
        #
        #     f(root)//p + t*f'(root) == 0 mod p
        error = f_root // p
        correction = -error * pow(f_prime_root % p, -1, p)
        return root + (correction % p) * p

    def residue_mod_p2(self, x: QuadInt) -> int:
        """Compute the residue mod self.root_p2"""
        root = self.root_p2()
        if root is None:
            raise ValueError("residue_mod_p2 requires an odd unramified split prime ideal")

        element = self.ring.from_obj(x)
        if element is NotImplemented:
            raise TypeError(f"Cannot coerce {type(x).__name__} into {self.ring!r}")

        if self.ring.den == 1:
            c0, c1 = element.a, element.b
        else:
            c0, c1 = (element.a - element.b) // 2, element.b

        mod = self.p * self.p
        return (c0 + c1 * root) % mod

    def __repr__(self) -> str:
        args = [
            f"ring={self.ring!r}",
            f"p={self.p!r}",
            f"index={self.index!r}",
            f"ideal={self.ideal!r}",
        ]

        if self.root is not None:
            args.append(f"root={self.root!r}")

        return f"PrimeIdealData({', '.join(args)})"

    def __eq__(self, other: object) -> bool:
        """Return True iff two prime-ideal records describe the same indexed prime ideal."""
        if not isinstance(other, PrimeIdealData):
            return False

        return (
            self.ring is other.ring
            and self.p == other.p
            and self.index == other.index
            and self.ideal == other.ideal
            and self.root == other.root
        )

    def __hash__(self) -> int:
        """Hash the immutable identity of this prime-ideal record."""
        return hash(
            (
                self.ring,
                self.p,
                self.index,
                self.ideal.hnf,
                self.root,
            ),
        )


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
    qp, rp = divmod(apb, den)
    qm, rm = divmod(amb, den)
    if rp or rm:
        # should be impossible if ring invariants hold
        raise ArithmeticError("Non-integral split-complex coordinates; check ring parameters/parity")
    return qp, qm


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


def _key(z: tuple, factors: dict[QuadInt, int]):
    """This is required (for now) as it appears that mypyc is having problems with sub-functions/lambdas"""
    return z[2] not in factors, z[1], abs(z[0].b), abs(z[0].a)


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
    _CACHE: ClassVar[dict[tuple[int, int], QuadraticRing]] = {}

    D: int
    den: int
    DEFAULT_KLASS: type[QuadInt]

    def __new__(cls, D: int, den: int | None = None):
        """Handle singleton logic"""
        D0 = int(D)
        default_den = 2 if (D0 % 4) == 1 else 1
        den0 = default_den if den is None else _check_den(den)

        key = (D0, den0)
        inst = QuadraticRing._CACHE.get(key)
        if inst is not None:
            return inst

        new_inst: QuadraticRing
        if cls is not QuadraticRing:
            new_inst = super().__new__(cls)
        else:
            # choose subclass
            for subcls in cls._subclasses():
                if subcls.accept_override(D0, den0, default_den):
                    new_inst = subcls(D0, den0)
                    break
            else:
                new_inst = super().__new__(cls)

        new_inst.DEFAULT_KLASS = QuadInt
        QuadraticRing._CACHE[key] = new_inst
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

    def __eq__(self, other: object) -> bool:
        # Type is checked here (and in hash) because it can cause problems with cached methods during setup
        #   when determining which quadratic ring class to use. Then when actually using the ring, the `self` variable
        #   is the old cached value we determined would not work.
        # Adding type here (and in hash) makes sure that we can still easily and simply cache those methods, while
        #   still making sure each QuadraticRing type's cache is separate from each other
        return type(self) is type(other) and self.D == other.D and self.den == other.den

    def __ne__(self, other: object) -> bool:
        # Work around occasional mypyc glue-generation assertions for __ne__.
        return not self.__eq__(other)

    def __hash__(self):
        return hash((type(self), self.D, self.den))

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

    def ideal(self, *generators: complex | int | float | QuadInt, _hnf: tuple[int, int, int] | None = None):
        """Return the integral ideal generated by the given elements."""
        return Ideal(self, *generators, _hnf=_hnf)

    def unit_ideal(self):
        """Return the unit ideal of this ring."""
        return self.ideal(1)

    def zero_ideal(self):
        """Return the zero ideal of this ring."""
        return self.ideal(0)

    @functools.cache
    def prime_ideals_data_over(self, p: int) -> tuple[PrimeIdealData, ...]:
        """Return prime ideals over p together with their defining root data."""
        p = int(p)
        if not isprime(p):
            raise ValueError(f"p must be prime, got {p!r}")

        cls = self.DEFAULT_KLASS

        if self.den == 1:
            # O = Z[sqrt(D)]
            #
            # The defining polynomial is:
            #     f(x) = x**2 - D
            #
            # A root r modulo p gives the prime ideal:
            #     P = (p, sqrt(D) - r)
            w = cls(0, 1, self, skip_basis=True)
            roots = sqrt_mod(self.D % p, p, all_roots=True)

        else:
            if self.D % 4 != 1:
                raise NotImplementedError("den=2 prime ideals require D ≡ 1 mod 4")

            # O = Z[w], where w = (1 + sqrt(D)) / 2
            #
            # The defining polynomial is:
            #     f(x) = x**2 - x + (1-D)//4
            #
            # For odd p, we can find roots by first finding sqrt(D) mod p:
            #     w = (1 + sqrt(D)) / 2
            w = cls(1, 1, self, skip_basis=True)

            if p == 2:
                c = (1 - self.D) // 4
                roots = [r for r in range(2) if (r * r - r + c) % 2 == 0]
            else:
                inv2 = pow(2, -1, p)
                roots = [((1 + int(s)) * inv2) % p for s in sqrt_mod(self.D % p, p, all_roots=True)]

        roots = tuple(sorted({int(root) for root in roots}))

        # No roots means p is inert, so (p) itself is prime.
        if not roots:
            return (
                PrimeIdealData(
                    ring=self,
                    p=p,
                    index=1,
                    ideal=self.ideal(p),
                ),
            )

        out = []
        for index, root in enumerate(roots, start=1):
            out.append(
                PrimeIdealData(
                    ring=self,
                    p=p,
                    index=index,
                    ideal=self.ideal(p, w - root),
                    root=root,
                ),
            )

        return tuple(out)

    def prime_ideals_over(self, p: int) -> tuple[Ideal, ...]:
        """Return the prime ideals lying over the rational prime p."""
        return tuple(data.ideal for data in self.prime_ideals_data_over(p))

    def elements_with_norm(self, n: int) -> Iterator[QuadInt]:
        """
        Yield quadratic integers in this order with norm `n`.

        Elements of this ring are stored as `(a + b*sqrt(D)) / den`, so this method solves the integer norm equation:
            a**2 - D*b**2 == den**2 * n

        For imaginary quadratic orders, the norm form is positive definite.
            The solution set is finite, so this iterator yields every solution and then stops.

        For real quadratic orders, a nonzero norm equation may have infinitely many solutions.
            In that case, this method first finds finitely many Pell seed solutions and then yields their unit orbits.
            If there are no seed solutions, the iterator stops immediately.

        Args:
            n: The signed norm to solve for.

        Yields:
            QuadInt: Elements of this order with norm `n`.

        Raises:
            NotImplementedError: If called for a degenerate quadratic order where
                this norm-equation interpretation is not appropriate.
        """
        n = int(n)

        if self.D in (0, 1):
            raise NotImplementedError("elements_with_norm requires a quadratic field order")

        if self.D != -1 and not _is_squarefree(self.D):
            raise NotImplementedError("elements_with_norm requires nonsquare D")

        target = n * self.den * self.den
        cls = self.DEFAULT_KLASS

        if target == 0:
            yield self.zero
            return

        seen: set[tuple[int, int]] = set()

        if self.D < 0:
            if target < 0:
                return

            max_b = isqrt(target // abs(self.D)) + 1

            for b in range(-max_b, max_b + 1):
                a_squared = target + self.D * b * b
                if a_squared < 0:
                    continue

                a_abs = isqrt(a_squared)
                if a_abs * a_abs != a_squared:
                    continue

                a_values = (0,) if a_abs == 0 else (-a_abs, a_abs)

                for a in a_values:
                    if self.den == 2 and ((a ^ b) & 1):
                        continue

                    key = (a, b)
                    if key in seen:
                        continue

                    seen.add(key)
                    yield cls(a, b, self, skip_basis=True)

            return

        seed_values: list[QuadInt] = []
        for a_raw, b_raw in diop_DN(self.D, target):
            a = int(a_raw)
            b = int(b_raw)

            if self.den == 2 and ((a ^ b) & 1):
                continue

            for candidate in (
                cls(a, b, self, skip_basis=True),
                cls(-a, -b, self, skip_basis=True),
                cls(a, -b, self, skip_basis=True),
                cls(-a, b, self, skip_basis=True),
            ):
                key = (candidate.a, candidate.b)
                if key in seen:
                    continue

                seen.add(key)
                seed_values.append(candidate)

        if not seed_values:
            return

        unit = self.fundamental_unit()
        if abs(unit) == -1:
            unit *= unit

        unit_inverse = unit.conjugate()

        yielded: set[tuple[int, int]] = set()

        for seed in seed_values:
            key = (seed.a, seed.b)
            if key not in yielded:
                yielded.add(key)
                yield seed

        positive_power = self.one
        negative_power = self.one

        for _ in count(1):
            positive_power *= unit
            negative_power *= unit_inverse

            for seed in seed_values:
                positive_candidate = seed * positive_power
                positive_key = (positive_candidate.a, positive_candidate.b)
                if positive_key not in yielded:
                    yielded.add(positive_key)
                    yield positive_candidate

                negative_candidate = seed * negative_power
                negative_key = (negative_candidate.a, negative_candidate.b)
                if negative_key not in yielded:
                    yielded.add(negative_key)
                    yield negative_candidate

    def has_element_with_norm(self, n: int) -> bool:
        """Return True iff this order has at least one element with norm `n`."""
        return next(self.elements_with_norm(n), None) is not None

    @property
    def class_group(self) -> ClassGroup:
        """Return the ideal class group of this quadratic order."""
        return ClassGroup(self)

    @property
    def class_number(self) -> int:
        """Return the class number of this quadratic order."""
        return self.class_group.order

    @functools.cache
    def fundamental_unit(self) -> QuadInt:
        """
        Return the smallest unit greater than 1 in this real quadratic order.

        This uses the continued fraction for the positive generator of the order:

            den == 1:  sqrt(D)
            den == 2:  w = (1 + sqrt(D)) / 2

        If p/q is a convergent to that generator, then the corresponding quadratic integer eventually has norm ±1.
            The first such unit found is the fundamental unit.

        Returns:
            QuadInt: The fundamental unit.

        Raises:
            ArithmeticError: If a converging continued fraction is impossible to find.
        """
        D = self.D
        sqrt_d = isqrt(D)

        if D <= 0 or sqrt_d * sqrt_d == D:
            raise NotImplementedError("fundamental units require a nonsquare real quadratic order")

        if self.den == 1:
            # We are expanding sqrt(D), written as (P + sqrt(D)) / Q.
            P = 0
            Q = 1
        elif self.den == 2 and D % 4 == 1:
            # We are expanding w = (1 + sqrt(D)) / 2.
            P = 1
            Q = 2
        else:
            raise NotImplementedError("fundamental units only support den=1 or den=2 with D ≡ 1 mod 4")

        cls = self.DEFAULT_KLASS

        def unit_candidate(p: int, q: int) -> QuadInt:
            """
            Convert a continued-fraction convergent p/q into a positive element.

            For den == 1, p/q approximates sqrt(D), so p + q*sqrt(D) is the natural positive candidate.

            For den == 2, p/q approximates w = (1 + sqrt(D)) / 2.
                The small element is p - q*w, so the positive conjugate is:

                p - q*w' = p - q*(1 - sqrt(D))/2
                         = (2*p - q + q*sqrt(D)) / 2

            Returns:
                QuadInt: The unit candidate.
            """
            if self.den == 1:
                return cls(p, q, self, skip_basis=True)

            return cls(2 * p - q, q, self, skip_basis=True)

        # Standard convergent recurrence:
        #     p_n = a_n*p_{n-1} + p_{n-2}
        #     q_n = a_n*q_{n-1} + q_{n-2}
        #
        # The starting values encode p[-2]/q[-2] = 0/1 and p[-1]/q[-1] = 1/0.
        prev_p, curr_p = 0, 1
        prev_q, curr_q = 1, 0

        while True:
            # Current continued-fraction term for (P + sqrt(D)) / Q.
            a = (sqrt_d + P) // Q

            next_p = a * curr_p + prev_p
            next_q = a * curr_q + prev_q

            candidate = unit_candidate(next_p, next_q)
            if candidate.is_unit():
                return candidate

            prev_p, curr_p = curr_p, next_p
            prev_q, curr_q = curr_q, next_q

            # Move to the next complete quotient:
            #     (P + sqrt(D)) / Q
            #
            # where:
            #     P_next = a*Q - P
            #     Q_next = (D - P_next**2) / Q
            next_P = a * Q - P
            next_Q_num = D - next_P * next_P

            if next_Q_num % Q:
                raise ArithmeticError("continued-fraction state became non-integral")

            next_Q = next_Q_num // Q
            if next_Q <= 0:
                raise ArithmeticError("continued-fraction search left the real quadratic cycle")

            P = next_P
            Q = next_Q

    def supports_division(self) -> bool:
        """
        Return whether this ring advertises Euclidean-style `divmod` support.

        Returns:
            bool: Whether this ring class sets `SUPPORTS_DIVISION`.
        """
        return self.SUPPORTS_DIVISION

    def supports_factorization(self) -> bool:
        """
        Return whether this ring advertises prime-factorization support.

        Returns:
            bool: Whether this ring class sets `SUPPORTS_FACTORIZATION`.
        """
        return self.SUPPORTS_FACTORIZATION

    def phi(self, x: QuadInt) -> int:
        """Return the default Euclidean size `|N(x)|` used for division heuristics."""
        return abs(abs(x))

    def discriminant(self) -> int:
        """Return the discriminant of the order represented by this ring."""
        return self.D if self.den == 2 else 4 * self.D

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
            num_a, rA = divmod(num_a, self.den)
            num_b, rB = divmod(num_b, self.den)

            if rA != 0 or rB != 0:
                return None

        # integrality test: coordinates divisible by |N|
        qa, ra = divmod(num_a, N)
        qb, rb = divmod(num_b, N)
        if ra != 0 or rb != 0:
            return None

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
            key=functools.partial(_key, factors=factors),
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
