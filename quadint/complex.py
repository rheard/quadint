from __future__ import annotations

from typing import ClassVar

from quadint.quad import QuadInt, QuadraticRing

_ZI = QuadraticRing(-1)  # Gaussian integers


class complexint(QuadInt):
    """
    Represents a complex number with integer real and imaginary parts.

    Properties:
        real (int): The real component.
        imag (int): The imaginary component.
    """

    __slots__ = ()

    SYMBOL: ClassVar[str] = "j"
    DEFAULT_RING = _ZI

    @property
    def real(self) -> int:
        """Alias for complexint"""
        return self.a

    @property
    def imag(self) -> int:
        """Alias for complexint"""
        return self.b
