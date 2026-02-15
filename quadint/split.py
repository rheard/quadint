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

    def __init__(self, a: int = 0, b: int = 0) -> None:
        """Initialize a splitint instance (use the _ZJ ring by default)."""
        super().__init__(_ZJ, int(a), int(b))

    def _make(self, a: int, b: int) -> "splitint":
        # a,b are internal numerators; for D=+1 with den=1 these match user coords
        return splitint(a, b)

    @property
    def real(self) -> int:
        """Alias for splitint"""
        return self.a

    @property
    def hyper(self) -> int:
        """Alias for splitint."""
        return self.b

    @property
    def j(self) -> int:
        """Alias for splitint"""
        return self.b

    def __repr__(self) -> str:
        parens = self.real != 0

        lead = "(" if parens else ""
        tail = ")" if parens else ""
        op = ("+" if parens else "") if self.hyper >= 0 else "-"
        a = self.real or ""
        b = abs(self.hyper)

        # Python's complex uses j instead of i, so complexint also does for continuity
        #   This presents a problem here, as that is the convention for split-complex numbers.
        #   Thus complexint and splitint have the same string values, even though the `j` in them is different.
        return f"{lead}{a}{op}{b}j{tail}"
