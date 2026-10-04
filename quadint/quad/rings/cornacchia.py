from __future__ import annotations

from quadint.quad.rings.norm_euclid import RealNormEuclidRing


class CornacchiaRing(RealNormEuclidRing):
    """
    The imaginary norm-Euclidean rings Z[i], Z[sqrt(-2)] and Z[w], whose norm form is x**2 + k*y**2.

    These used to factor with Cornacchia's method, which is where the name comes from. They factor with
        QuadraticRing.factor_detail now, like every other PID, and divide with RealNormEuclidRing.divmod.
    """

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:  # ruff: ignore[unused-class-method-argument]
        """Should this class be used for the given values?"""
        # No, this is purely a sub-abstract base class that needs to be subclassed
        return False


class GaussianRing(CornacchiaRing):
    """The Gaussian integers Z[i]."""

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:  # ruff: ignore[unused-class-method-argument]
        """Should this class be used for the given values?"""
        return D == -1 and den == 1


class SqrtMinusTwoRing(CornacchiaRing):
    """The ring Z[sqrt(-2)]."""

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:  # ruff: ignore[unused-class-method-argument]
        """Should this class be used for the given values?"""
        return D == -2 and den == 1


class EisensteinRing(CornacchiaRing):
    """The Eisenstein integers Z[ω]."""

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:  # ruff: ignore[unused-class-method-argument]
        """Should this class be used for the given values?"""
        return D == -3 and den == 2
