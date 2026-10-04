from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from quadint.quad.rings.cornacchia import CornacchiaRing

if TYPE_CHECKING:
    from quadint.quad.int import QuadInt


class HeegnerDen2Ring(CornacchiaRing):
    """The imaginary PIDs with den=2 other than Z[w]: D=-7 and D=-11, and the ones that aren't Euclidean."""

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:  # ruff: ignore[unused-class-method-argument]
        """Should this class be used for the given values?"""
        # No, this is purely a sub-abstract base class that needs to be subclassed
        return False


class HeegnerSevenRing(HeegnerDen2Ring):
    """The maximal order with D=-7."""

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:  # ruff: ignore[unused-class-method-argument]
        """Should this class be used for the given values?"""
        return D == -7 and den == 2


class HeegnerElevenRing(HeegnerDen2Ring):
    """The maximal order with D=-11."""

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:  # ruff: ignore[unused-class-method-argument]
        """Should this class be used for the given values?"""
        return D == -11 and den == 2


class HeegnerNonEuclidUfdRing(HeegnerDen2Ring):
    """
    The imaginary quadratic maximal orders with class number 1 that are *not* Euclidean.

    This covers the remaining Heegner (class number 1) fields beyond the norm-Euclidean ones:
        D in {-19, -43, -67, -163}   (all have default den=2)

    There is no divmod here. But these are still PIDs, so they factor with QuadraticRing.factor_detail, and gcd, xgcd,
        inv_mod and pow(x, e, m) work from a generator of the ideal (a, b) (see QuadraticRing.xgcd).
    """

    SUPPORTS_DIVISION: ClassVar[bool] = False

    HEEGNER_NON_EUCLID_D: ClassVar[set[int]] = {-19, -43, -67, -163}

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:
        """Return True iff this ring override should be selected for (D, den)."""
        # Only maximal orders (den=default_den=2 for these D).
        return D in cls.HEEGNER_NON_EUCLID_D and den == default_den

    def divmod(self, x: QuadInt, y: QuadInt) -> tuple[QuadInt, QuadInt]:
        """This ring is not Euclidean; divmod is intentionally unavailable."""
        raise NotImplementedError("HeegnerNonEuclidUfdRing does not support Euclidean division")
