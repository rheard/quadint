from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from quadint.quad.rings.base import (
    QuadraticRing,
    _choose_best_in_neighborhood,
    _round_div_ties_away_from_zero,
    _split_uv,
)

if TYPE_CHECKING:
    from quadint.quad.int import QuadInt


class DualRing(QuadraticRing):
    """
    Handle overrides for D=0, dual integer solutions

    While the general algorith in RealNormEuclidRing will find a solution for D=0,
        it does not take into account that the ε part is not part of the norm,
        so is not relevant in the division algorithm.

    This class will solve division for the real part while trying to minimize the ε part.
    """

    SUPPORTS_DIVISION: ClassVar[bool] = True

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:  # noqa: ARG003
        """Should this class be used for the given values?"""
        return D == 0

    def divmod(self, x: QuadInt, y: QuadInt) -> tuple[QuadInt, QuadInt]:
        """Division with D=0"""
        # In dual numbers, (c + dε) is invertible iff c != 0.
        n = y.a
        num = x

        if n == 0:
            raise ZeroDivisionError

        A0 = _round_div_ties_away_from_zero(num.a, n)

        c, d = y.a, y.b

        def B0_for_A(A: int) -> int:
            return _round_div_ties_away_from_zero(x.b - A * d, c)

        # Lexicographic "small remainder": minimize real remainder first, then ε remainder.
        def score_for_AB(A: int, B: int) -> tuple[int, ...]:
            r0 = x.a - A * c
            r1 = x.b - A * d - B * c
            return r0 * r0, r1 * r1

        best_a, best_b = _choose_best_in_neighborhood(
            A0=A0,
            B0_for_A=B0_for_A,
            score_for_AB=score_for_AB,
            den=self.den,
        )

        q = x._make(best_a, best_b)
        r = x - q * y
        return q, r


class SplitRing(QuadraticRing):
    """
    Handle overrides for D=1, split integer solutions

    This class performs division in the split (u, v) coordinates (where multiplication
        is component-wise), choosing a quotient that reduces both components and yields a
        more stable, integer-like remainder.

    This structural shortcut is only possible with D=1 (because Z[sqrt(1)]... well, splits.)
    """

    SUPPORTS_DIVISION: ClassVar[bool] = True

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:  # noqa: ARG003
        """Should this class be used for the given values?"""
        return D == 1

    def divmod(self, x: QuadInt, y: QuadInt) -> tuple[QuadInt, QuadInt]:
        """Division with D=1"""
        u1, v1 = _split_uv(x)
        u2, v2 = _split_uv(y)

        # Division by zero divisor (u2==0 or v2==0) is not well-defined.
        if u2 == 0 or v2 == 0:
            raise ZeroDivisionError("division by zero divisor in split-complex integers (a=±b)")

        qu0 = _round_div_ties_away_from_zero(u1, abs(u2))
        if u2 < 0:
            qu0 = -qu0

        qv0 = _round_div_ties_away_from_zero(v1, abs(v2))
        if v2 < 0:
            qv0 = -qv0

        def B0_for_A(A: int) -> int:  # noqa: ARG001
            return qv0

        def score_for_AB(A: int, B: int) -> tuple[int, ...]:
            # remainder in (u,v)
            ru = u1 - A * u2
            rv = v1 - B * v2
            return (ru * ru + rv * rv,)

        # We only need qu ≡ qv (mod 2) when self.den is odd (in practice: self.den==1),
        # because we later divide (qu±qv)*self.den by 2.
        parity_den = 2 if self.den == 1 else 1

        # Enforce qu ≡ qv (mod 2) so (qu+qv)/2 and (qu-qv)/2 are integers.
        best_qu, best_qv = _choose_best_in_neighborhood(
            A0=qu0,
            B0_for_A=B0_for_A,
            score_for_AB=score_for_AB,
            den=parity_den,
        )

        # Convert back: a = den*(qu+qv)/2, b = den*(qu-qv)/2
        s = best_qu + best_qv
        t = best_qu - best_qv
        qa = (s * self.den) // 2
        qb = (t * self.den) // 2

        q = x._make(qa, qb)
        r = x - q * y
        return q, r
