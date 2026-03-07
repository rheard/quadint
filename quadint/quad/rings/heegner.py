from __future__ import annotations

from math import isqrt
from typing import TYPE_CHECKING, ClassVar

from sympy import sqrt_mod

from quadint.quad.rings.cornacchia import CornacchiaRing, EisensteinRing

if TYPE_CHECKING:
    from quadint.quad.int import QuadInt


# region Heegner rings
class HeegnerDen2Ring(EisensteinRing):
    """Shared split-prime factorization helper for D=-7 and D=-11 (den=2)."""

    SPLIT_K = 1  # unused by this strategy

    def _is_split_prime(self, p: int) -> bool:
        if p == self.RAMIFIED_PRIME:
            return False
        if p == 2:
            return -self.RAMIFIED_PRIME % 8 == 1
        return sqrt_mod(-self.RAMIFIED_PRIME, p, all_roots=False) is not None

    def _is_inert_prime(self, p: int) -> bool:
        if p == self.RAMIFIED_PRIME:
            return False
        if p == 2:
            return -self.RAMIFIED_PRIME % 8 == 5
        return sqrt_mod(-self.RAMIFIED_PRIME, p, all_roots=False) is None

    def _ramified_generator(self, x: QuadInt) -> QuadInt:
        return x._make(0, 2)

    def _decompose_prime(self, p: int) -> tuple[int, int]:
        """Return odd representatives of the two sqrt(D) roots modulo p."""
        root = sqrt_mod(-self.RAMIFIED_PRIME, p, all_roots=False)
        if root is None:
            raise ValueError(f"Could not decompose {p!r}")

        t = int(root)
        if (t ^ 1) & 1:
            t += p

        t_alt = p - int(root)
        if (t_alt ^ 1) & 1:
            t_alt += p

        return t, t_alt

    def _split_generator(self, x: QuadInt, p: int) -> QuadInt:
        x0, y0 = self._decompose_prime(p)

        p_elem = x._make(2 * p, 0)
        cand = self.gcd(p_elem, x._make(x0, 1))
        if abs(cand) in (1, p * p):
            cand = self.gcd(p_elem, x._make(y0, 1))

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


class HeegnerNonEuclidUfdRing(CornacchiaRing):
    """
    Factorization for imaginary quadratic maximal orders with class number 1 but *not* Euclidean.

    This covers the remaining Heegner (class number 1) fields beyond the norm-Euclidean ones:
        D in {-19, -43, -67, -163}   (all have default den=2)

    Key point:
      - We do NOT rely on divmod()/Euclidean division.
      - We generate split prime elements π with N(π)=p using Cornacchia-style integer work.
      - Then we strip valuations using exact_div (fast, and works even when supports_division()==False).
    """

    SUPPORTS_DIVISION: ClassVar[bool] = False
    SUPPORTS_FACTORIZATION: ClassVar[bool] = True

    HEEGNER_NON_EUCLID_D: ClassVar[set[int]] = {-19, -43, -67, -163}

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:
        """Return True iff this ring override should be selected for (D, den)."""
        # Only maximal orders (den=default_den=2 for these D).
        return D in cls.HEEGNER_NON_EUCLID_D and den == default_den

    def divmod(self, x: QuadInt, y: QuadInt) -> tuple[QuadInt, QuadInt]:
        """This ring is not Euclidean; divmod is intentionally unavailable."""
        raise NotImplementedError("HeegnerNonEuclidUfdRing does not support Euclidean division")

    def _ramified_prime(self) -> int:
        # For these D, the discriminant is D itself (odd prime), so the unique ramified prime is |D|.
        return -self.D

    def _is_split_prime(self, p: int) -> bool:
        if p == self._ramified_prime():
            return False
        if p == 2:
            # For odd discriminant D ≡ 1 (mod 4), the Kronecker symbol (D/2) depends on D mod 8:
            # split if D ≡ 1 (mod 8), inert if D ≡ 5 (mod 8).
            return (self.D % 8) == 1
        return sqrt_mod(self.D, p, all_roots=False) is not None

    def _is_inert_prime(self, p: int) -> bool:
        if p == self._ramified_prime():
            return False
        if p == 2:
            return (self.D % 8) == 5
        return sqrt_mod(self.D, p, all_roots=False) is None

    # ---------- prime element generators ----------

    def _ramified_generator(self, x: QuadInt) -> QuadInt:
        # sqrt(D) = (0 + 2*sqrt(D))/2 in numerator coords
        return x._make(0, self.den)

    def _inert_generator(self, x: QuadInt, p: int) -> QuadInt:
        # inert primes remain prime as elements: just the integer p
        return x._make(p * self.den, 0)

    def _decompose_prime(self, p: int) -> tuple[int, int]:
        """
        Return (A,B) such that:
            A^2 - D*B^2 = 4p
        with A ≡ B (mod 2), so that π = (A + B*sqrt(D))/2 is in this maximal order and N(π)=p.

        We do this in two passes:

        1) "even-even" case: find x,y with p = x^2 + (-D)*y^2, then set (A,B)=(2x,2y).
           (This corresponds to elements that actually lie in the suborder Z[sqrt(D)].)

        2) "odd-odd" case: scan Euclidean remainders from (p, r) where r^2 ≡ D (mod p),
           and test b as a candidate A for 4p = b^2 + (-D)*B^2. This finds the genuinely den=2 generators.

        Returns:
            tuple: The decomposed prime.

        Raises:
            ValueError: If it is not possible to split the prime.
        """
        if not self._is_split_prime(p):
            raise ValueError(f"{p!r} is not split for D={self.D}")

        k = -self.D  # positive

        # ---- pass 1: solve p = x^2 + k*y^2 (standard Cornacchia on a prime modulus)
        roots = sqrt_mod(self.D, p, all_roots=True)
        if roots is not None:
            for root in roots:
                a = p
                b = min(int(root), p - int(root))
                while b * b > p:
                    a, b = b, a % b

                y2_num = p - b * b
                if y2_num % k:
                    continue
                y2 = y2_num // k
                y = isqrt(y2)
                if y * y != y2:
                    continue

                # Lift to den=2 coordinates: (2x,2y)
                A = 2 * b
                B = 2 * y
                return A, B

        # ---- pass 2: solve 4p = A^2 + k*B^2 with A,B same parity (odd-odd typically)
        roots = sqrt_mod(self.D, p, all_roots=True)
        if roots is None:
            raise ValueError(f"Could not decompose split prime {p!r} for D={self.D}")

        for root in roots:
            a = p
            b = min(int(root), p - int(root))

            # Important: we don't stop at the first b with b^2 <= 4p.
            # Some primes require continuing the remainder chain (this matters e.g. for D=-19).
            while b:
                if b * b <= 4 * p:
                    y2_num = 4 * p - b * b
                    if y2_num % k == 0:
                        y2 = y2_num // k
                        y = isqrt(y2)
                        if y * y == y2:
                            A = b
                            B = y
                            if self.den == 2 and ((A ^ B) & 1):
                                # parity mismatch: not an algebraic integer in this order
                                pass
                            else:
                                return A, B
                a, b = b, a % b

        raise ValueError(f"Could not decompose split prime {p!r} for D={self.D}")


# endregion
