from __future__ import annotations

from math import gcd as igcd
from typing import TYPE_CHECKING, ClassVar

from sympy import gcdex

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
        q = self._uv_to_ab(best_qu, best_qv)

        r = x - q * y
        return q, r

    def _uv_to_ab(self, u: int, v: int) -> QuadInt:
        """Convert split coordinates (u, v) back to stored numerator coordinates (a, b)."""
        # _split_uv: u = (a+b)/den, v = (a-b)/den
        # Inverse: a = den*(u+v)/2, b = den*(u-v)/2
        s = u + v
        t = u - v
        return self((s * self.den) // 2, (t * self.den) // 2)

    def exact_div(self, x: QuadInt, y: QuadInt) -> QuadInt | None:
        """Exact division in split coordinates (handles zero-norm divisors)."""
        u1, v1 = _split_uv(x)
        u2, v2 = _split_uv(y)

        if u2 == 0 and v2 == 0:
            raise NotImplementedError

        # Both components must divide exactly (or be 0/0 which we skip)
        if u2 == 0 or v2 == 0:
            # Zero divisor: cannot divide uniquely in general
            return None

        qu, ru = divmod(u1, u2)
        qv, rv = divmod(v1, v2)
        if ru != 0 or rv != 0:
            return None

        # Parity check for den=1
        if self.den == 1 and ((qu ^ qv) & 1):
            return None

        return self._uv_to_ab(qu, qv)

    def _split_gcd(self, u1: int, v1: int, u2: int, v2: int) -> tuple[int, int]:
        """Compute GCD in split coordinates, respecting den=1 parity constraints."""
        gu = igcd(abs(u1), abs(u2))
        gv = igcd(abs(v1), abs(v2))

        if self.den != 1 or gu == 0 or gv == 0:
            return gu, gv

        # For den=1: (gu, gv) must be in L (same parity) AND quotients must be in L.
        # If both gu, gv are odd, quotient parity is automatically satisfied.
        # If both even, quotients might fail; halve both until it works.
        inputs = [(u1, v1), (u2, v2)]

        while gu > 0 and gv > 0:
            # Ensure same parity
            while (gu ^ gv) & 1:
                if gu % 2 == 0:
                    gu //= 2
                else:
                    gv //= 2

            if gu == 0 or gv == 0:
                break

            # Both odd → guaranteed to work (proof: inputs have same parity,
            # dividing by same-parity odd divisor preserves parity of quotient)
            if gu & 1:
                break

            # Both even: check that all quotients have matching parity
            ok = True
            for ui, vi in inputs:
                if ui == 0 and vi == 0:
                    continue
                qu = ui // gu
                qv = vi // gv
                if (qu ^ qv) & 1:
                    ok = False
                    break

            if ok:
                break

            # Quotient parity mismatch: halve both
            gu //= 2
            gv //= 2

        return gu, gv

    def gcd(self, a: QuadInt, b: QuadInt) -> QuadInt:
        """GCD in split coordinates, respecting sublattice parity constraints."""
        u1, v1 = _split_uv(a)
        u2, v2 = _split_uv(b)

        gu, gv = self._split_gcd(u1, v1, u2, v2)

        g = self._uv_to_ab(gu, gv)
        return g._canonical_associate()

    def xgcd(self, a: QuadInt, b: QuadInt) -> tuple[QuadInt, QuadInt, QuadInt]:
        """Extended GCD in split coordinates (den=2 only; den=1 ring is not a PID)."""
        if self.den == 1:
            raise NotImplementedError(
                "xgcd not supported for D=1 den=1 (ring has zero divisors and is not a PID); use gcd() instead",
            )

        u1, v1 = _split_uv(a)
        u2, v2 = _split_uv(b)

        su, tu, gu = gcdex(u1, u2)
        sv, tv, gv = gcdex(v1, v2)

        # For den=2, no parity constraint on (u, v) — the ring IS Z*Z.
        return self._canonicalize_bezout_result(
            self._uv_to_ab(int(gu), int(gv)),
            self._uv_to_ab(int(su), int(sv)),
            self._uv_to_ab(int(tu), int(tv)),
        )
