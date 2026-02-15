from quadint.quad import QuadInt, QuadraticRing

_ZW = QuadraticRing(-3)


class eisensteinint(QuadInt):
    """
    Represents an Eisenstein number with integer real and omega parts.

    Properties:
        real (int): The real component.
        omega (int): The omega component.
    """
    __slots__ = ()

    def __init__(self, a: int = 0, b: int = 0) -> None:
        """Initialize an eisensteinint instance."""
        # user basis: a + b*ω, where ω = (-1 + sqrt(-3))/2
        # internal numerator basis: (2a - b) + b*sqrt(-3) over den=2
        a, b = int(a), int(b)
        super().__init__(_ZW, 2 * a - b, b)

    def _make(self, A: int, B: int) -> "eisensteinint":
        # A,B are internal numerators for (A + B*sqrt(-3))/2
        # Convert back to ω-basis: a = (A + B)/2, b = B
        a = (A + B) // 2
        b = B
        return eisensteinint(a, b)

    @property
    def real(self) -> int:
        """Alias for eisensteinint"""
        return (self.a + self.b) // 2

    @property
    def omega(self) -> int:
        """Alias for eisensteinint"""
        return self.b

    def __repr__(self) -> str:
        parens = self.real != 0

        lead = "(" if parens else ""
        tail = ")" if parens else ""
        op = ("+" if parens else "") if self.omega >= 0 else "-"
        a = self.real or ""
        b = abs(self.omega)

        return f"{lead}{a}{op}{b}ω{tail}"
