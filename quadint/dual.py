from quadint.quad import QuadInt, QuadraticRing

_ZE = QuadraticRing(0)   # Gaussian integers


class dualint(QuadInt):
    """
    Represents a dual number with integer real and dual parts.

    Properties:
        real (int): The real component.
        dual (int): The dual or epsilon component.
    """
    __slots__ = ()

    def __init__(self, a: int = 0, b: int = 0) -> None:
        """Initialize a dualint instance (use the _ZE ring by default)."""
        super().__init__(_ZE, int(a), int(b))

    def _make(self, a: int, b: int) -> "dualint":
        # a,b are internal numerators; for D=0, den=1 so these match user coords
        return dualint(a, b)

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

    def __repr__(self) -> str:
        parens = self.real != 0

        lead = "(" if parens else ""
        tail = ")" if parens else ""
        op = ("+" if parens else "") if self.dual >= 0 else "-"
        a = self.real or ""
        b = abs(self.dual)

        return f"{lead}{a}{op}{b}ε{tail}"
