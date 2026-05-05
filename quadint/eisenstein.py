from __future__ import annotations

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

    # user basis: x + y*ω, where ω = (-1 + sqrt(-3))/2
    # internal numerator basis: (2x - y) + y*sqrt(-3) over den=2
    BASIS_TO_INTERNAL = ((2, -1), (0, 1))
    INTERNAL_TO_BASIS = ((1, 1), (0, 2))
    INTERNAL_TO_BASIS_DEN = 2

    # Cannot use DEFAULT_RING here as that would register eisensteinint as the default for Z[-3], which it isn't.
    def __init__(self, a: int = 0, b: int = 0, ring: QuadraticRing | None = _ZW, *, skip_basis: bool = False):
        """Initialize an eisensteinint instance."""
        super().__init__(a, b, ring, skip_basis=skip_basis)

    @property
    def real(self) -> int:
        """Alias for eisensteinint"""
        return self.basis_a

    @property
    def omega(self) -> int:
        """Alias for eisensteinint"""
        return self.basis_b

    def __repr__(self) -> str:
        parens = self.real != 0

        lead = "(" if parens else ""
        tail = ")" if parens else ""
        op = ("+" if parens else "") if self.omega >= 0 else "-"
        a = self.real or ""
        b = abs(self.omega)

        return f"{lead}{a}{op}{b}ω{tail}"
