from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar

from sympy import factorint

from quadint.quad.rings.base import Factorization
from quadint.quad.rings.norm_euclid import RealNormEuclidRing
from quadint.sums import decompose_prime

if TYPE_CHECKING:
    from quadint.quad.int import QuadInt


# region Cornacchia rings (for factorization)
class CornacchiaRing(RealNormEuclidRing):
    """Shared split-prime factorization flow for rings with norm form x**2 + k*y**2."""

    SUPPORTS_FACTORIZATION: ClassVar[bool] = True

    RAMIFIED_PRIME: ClassVar[int]

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:  # noqa: ARG003
        """Should this class be used for the given values?"""
        # No, this is purely a sub-abstract base class that needs to be subclassed
        return False

    def _is_split_prime(self, p: int) -> bool:
        raise NotImplementedError

    def _is_inert_prime(self, p: int) -> bool:
        raise NotImplementedError

    def _ramified_generator(self, x: QuadInt) -> QuadInt:
        raise NotImplementedError

    def _ramified_prime(self) -> int:
        return self.RAMIFIED_PRIME

    def _inert_generator(self, x: QuadInt, p: int) -> QuadInt:
        # Integer p in numerator coordinates is (2p + 0*sqrt(D)) / 2
        return x._make(self.den * p, 0)

    def _split_generator(self, x: QuadInt, p: int) -> QuadInt:
        x0, y0 = decompose_prime(p, -self.D, self.den)
        return x._make(x0, y0)

    def factor_detail(self, x: QuadInt) -> Factorization:
        """
        Return a structured factorization for Cornacchia-style imaginary quadratic rings.

        The result is returned as `Factorization(unit, primes)` where `primes` is a
            mapping `{prime_element: exponent}` and:

        * `unit * prod(p**e for p, e in primes.items()) == x`
        * each listed `prime_element` is a non-unit irreducible in this ring

        Strategy:

        1. Normalize by extracting a canonical unit associate.
        2. Remove powers of the ramified prime generator.
        3. Factor the remaining integer norm with `sympy.factorint`.
        4. For each rational prime factor, use split/inert classification:
           * inert primes stay prime in the ring,
           * split primes are decomposed via Cornacchia's method and tested (with conjugates)
             as divisors.

        Args:
            x: A non-zero element of this ring.

        Returns:
            Factorization: The unit and prime-power data for `x`.

        Raises:
            ValueError: If `x` is zero (zero has no finite prime factorization).
        """
        if not x:
            raise ValueError("0 does not have a finite factorization")

        rem = x
        unit = x.one

        for u in x.units:
            q = self.exact_div(rem, u)
            if q is not None and (q.a, q.b) < (rem.a, rem.b):
                rem = q
                unit *= u

        factors: dict[QuadInt, int] = defaultdict(int)

        ramified = self._ramified_generator(x)
        while True:
            q = self.exact_div(rem, ramified)
            if q is None:
                break
            factors[ramified] += 1
            rem = q

        n = abs(rem)
        int_factors = factorint(n)
        ram_p = self._ramified_prime()

        for p in sorted(int_factors):
            if p == ram_p:
                continue

            if self._is_inert_prime(p):
                cand = self._inert_generator(x, p)
                while True:
                    q = self.exact_div(rem, cand)
                    if q is None:
                        break
                    factors[cand] += 1
                    rem = q
                continue

            if self._is_split_prime(p):
                cand_base = self._split_generator(x, p)
                for cand in (cand_base, cand_base.conjugate()):
                    if abs(cand) <= 1:
                        continue
                    while True:
                        q = self.exact_div(rem, cand)
                        if q is None:
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

    def _is_split_prime(self, p: int) -> bool:
        return p % 4 == 1

    def _is_inert_prime(self, p: int) -> bool:
        return p % 4 == 3

    def _ramified_generator(self, x: QuadInt) -> QuadInt:
        return x._make(1, 1)

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:  # noqa: ARG003
        """Should this class be used for the given values?"""
        return D == -1 and den == 1


class SqrtMinusTwoRing(CornacchiaRing):
    """Specialized factorization strategy for Z[sqrt(-2)]."""

    RAMIFIED_PRIME = 2

    def _is_split_prime(self, p: int) -> bool:
        return p % 8 in (1, 3)

    def _is_inert_prime(self, p: int) -> bool:
        return p % 8 in (5, 7)

    def _ramified_generator(self, x: QuadInt) -> QuadInt:
        return x._make(0, 1)

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:  # noqa: ARG003
        """Should this class be used for the given values?"""
        return D == -2 and den == 1


class EisensteinRing(CornacchiaRing):
    """Specialized factorization strategy for Eisenstein integers Z[ω]."""

    RAMIFIED_PRIME = 3

    def _is_split_prime(self, p: int) -> bool:
        return p % 3 == 1

    def _is_inert_prime(self, p: int) -> bool:
        return p % 3 == 2

    def _ramified_generator(self, x: QuadInt) -> QuadInt:
        return x._make(3, 1)

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:  # noqa: ARG003
        """Should this class be used for the given values?"""
        return D == -3 and den == 2


# endregion
