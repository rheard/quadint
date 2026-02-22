from __future__ import annotations

from typing import ClassVar

from quadint.quad import QuadInt, QuadraticRing

# Split-complex (hyperbolic) integers: a + b*j with j^2 = +1.
# IMPORTANT: force den=1, otherwise D=1 would default to den=2 under the
#   quadratic-field "maximal order" convention (D % 4 == 1), which is not what
#   we want for the split-complex algebra.
_ZJ = QuadraticRing(1, den=1)


class splitint(QuadInt):
    """
    Represents a split-complex (hyperbolic) number with integer real and j parts, where j**2 = 1.

    Properties:
        real (int): The real component.
        hyper (int): The hyperbolic (j) component.
    """

    __slots__ = ()

    SYMBOL: ClassVar[str] = "j"
    DEFAULT_RING = _ZJ

    @property
    def real(self) -> int:
        """Alias for splitint"""
        return self.a

    @property
    def hyper(self) -> int:
        """Alias for splitint"""
        return self.b

    @property
    def j(self) -> int:
        """Alias for splitint"""
        return self.b
