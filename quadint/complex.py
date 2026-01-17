from quadint.quad import OP_TYPES, QuadInt, QuadraticRing

_ZI = QuadraticRing(-1)   # Gaussian integers


class complexint(QuadInt):
    """
    Represents a complex number with integer real and imaginary parts.

    Properties:
        real (int): The real component.
        imag (int): The imaginary component.
    """
    __slots__ = ()

    def __init__(self, a: int = 0, b: int = 0) -> None:
        """Initialize a complexint instance (use the _ZI ring by default)."""
        super().__init__(_ZI, int(a), int(b))

    def _make(self, a: int, b: int) -> "complexint":
        # a,b are internal numerators; for D=-1, den=1 so these match user coords
        return complexint(a, b)

    def _from_obj(self, n: OP_TYPES) -> "complexint":
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

    def __repr__(self) -> str:
        parens = self.real != 0

        lead = "(" if parens else ""
        tail = ")" if parens else ""
        op = ("+" if self.imag >= 0 else "-") if parens else ""
        a = self.real or ""
        b = abs(self.imag)

        return f"{lead}{a}{op}{b}j{tail}"
