from __future__ import annotations

from typing import ClassVar

from quadint.quad import OP_TYPES, QuadInt, QuadraticRing

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

    def _from_obj(self, n: OP_TYPES) -> complexint:
        """Make a QuadInt on the current ring from a given object"""
        if isinstance(n, (int, float)):
            a = int(n)
            b = 0
        elif isinstance(n, complex):
            a = int(n.real)
            b = int(n.imag)
        elif isinstance(n, complexint):
            return n
        else:
            return NotImplemented

        # The only time b is not 0 is if self.den is 1 anyway... No need to multiply
        return self._make(a * self.ring.den, b)

    @property
    def real(self) -> int:
        """Alias for complexint"""
        return self.a

    @property
    def imag(self) -> int:
        """Alias for complexint"""
        return self.b
