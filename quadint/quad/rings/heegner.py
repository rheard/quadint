from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from sympy import sqrt_mod

from quadint.quad.rings.cornacchia import CornacchiaRing

if TYPE_CHECKING:
    from quadint.quad.int import QuadInt


class HeegnerDen2Ring(CornacchiaRing):
    """Shared split-prime factorization helper for D=-7 and D=-11 (den=2)."""

    def _is_split_prime(self, p: int) -> bool:
        if p == self._ramified_prime():
            return False
        if p == 2:
            return self.D % 8 == 1
        return sqrt_mod(self.D, p, all_roots=False) is not None

    def _is_inert_prime(self, p: int) -> bool:
        if p == self._ramified_prime():
            return False
        if p == 2:
            return self.D % 8 == 5
        return sqrt_mod(self.D, p, all_roots=False) is None

    def _ramified_generator(self, x: QuadInt) -> QuadInt:
        # sqrt(D) = (0 + 2*sqrt(D)) / 2
        return x._make(0, self.den)

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


class HeegnerNonEuclidUfdRing(HeegnerDen2Ring):
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
