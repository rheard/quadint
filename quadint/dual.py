from __future__ import annotations

from typing import ClassVar

from quadint.quad import QuadInt, QuadraticRing

_ZE: QuadraticRing = QuadraticRing(0)


class dualint(QuadInt):
    """
    Represents a dual number with integer real and dual parts.

    Properties:
        real (int): The real component.
        dual (int): The dual or epsilon component.
    """

    __slots__ = ()

    SYMBOL: ClassVar[str] = "ε"
    DEFAULT_RING: ClassVar[QuadraticRing | None] = _ZE

    @property
    def real(self) -> int:
        """Alias for dualint"""
        return self.a

    @property
    def dual(self) -> int:
        """Alias for dualint"""
        return self.b

    @property
    def epsilon(self) -> int:
        """Alias for dualint"""
        return self.b
