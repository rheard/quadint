from __future__ import annotations

from math import gcd as igcd
from typing import TYPE_CHECKING, ClassVar

from sympy.polys.domains import ZZ

from quadint.quad.rings.base import (
    QuadraticRing,
    _NeighborhoodSearch,
    _round_div,
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
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:  # ruff: ignore[unused-class-method-argument]
        """Should this class be used for the given values?"""
        return D == 0

    def divmod(self, x: QuadInt, y: QuadInt) -> tuple[QuadInt, QuadInt]:
        """Division with D=0"""
        # In dual numbers, (c + dε) is invertible iff c != 0.
        n = y.a
        num = x

        if n == 0:
            # A multiple of ε only has multiples with no real part, so there is no remainder to make small
            raise ZeroDivisionError("division by a multiple of ε, which is a zero divisor (exact_div still works)")

        A0 = _round_div(num.a, n)  # so x % y only depends on x's class mod y (see _round_div)

        c, d = y.a, y.b

        def B0_for_A(A: int) -> int:
            return _round_div(x.b - A * d, c)

        # Lexicographic "small remainder": minimize real remainder first, then ε remainder.
        def score_for_AB(A: int, B: int) -> tuple[int, ...]:
            r0 = x.a - A * c
            r1 = x.b - A * d - B * c
            return r0 * r0, r1 * r1

        # The best of A0 +/- 1, each with B0_for_A(A) +/- 1
        best_a, best_b = _NeighborhoodSearch(
            A0=A0,
            B0_for_A=B0_for_A,
            score_for_AB=score_for_AB,
            den=self.den,
        ).expand_to(1)

        q = x._make(best_a, best_b)
        r = x - q * y
        return q, r

    def exact_div(self, x: QuadInt, y: QuadInt) -> QuadInt | None:
        """
        Exact division, where a multiple of ε (a zero divisor, since ε**2 == 0) can divide too.

        (e + f*ε) * (q0 + q1*ε) == e*q0 + (e*q1 + f*q0)*ε, so when e != 0 there is at most one quotient, which the
            general formula finds. But f*ε only has the multiples f*q0*ε: it divides x when x has no real part and
            f divides x's ε part, and then q1 could be anything, so this picks 0, which makes q the plain integer x/y
            (like 6ε / 2ε == 3).

        Returns:
            QuadInt | None: A quotient q with x == q*y, or None if there is none.
        """
        if y.a or not y.b:
            return super().exact_div(x, y)  # which also raises ZeroDivisionError for y == 0

        q, r = divmod(x.b, y.b)
        return x._make(q, 0) if x.a == 0 and r == 0 else None


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
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:  # ruff: ignore[unused-class-method-argument]
        """Should this class be used for the given values?"""
        return D == 1

    def divmod(self, x: QuadInt, y: QuadInt) -> tuple[QuadInt, QuadInt]:
        """Division with D=1"""
        u1, v1 = _split_uv(x)
        u2, v2 = _split_uv(y)

        # Division by zero divisor (u2==0 or v2==0) is not well-defined.
        if u2 == 0 or v2 == 0:
            raise ZeroDivisionError("division by zero divisor in split-complex integers (a=±b)")

        # Rounded so that x % y only depends on x's class mod y (see _round_div)
        qu0 = _round_div(u1, u2)
        qv0 = _round_div(v1, v2)

        def B0_for_A(A: int) -> int:  # ruff: ignore[unused-function-argument]
            return qv0

        def score_for_AB(A: int, B: int) -> tuple[int, ...]:
            # remainder in (u,v)
            ru = u1 - A * u2
            rv = v1 - B * v2
            return (ru * ru + rv * rv,)

        # We only need qu ≡ qv (mod 2) when self.den is odd (in practice: self.den==1),
        # because we later divide (qu±qv)*self.den by 2.
        parity_den = 2 if self.den == 1 else 1

        # The best of qu0 +/- 1 and qv0 +/- 1 (those with qu ≡ qv (mod 2), when den=1)
        best_qu, best_qv = _NeighborhoodSearch(
            A0=qu0,
            B0_for_A=B0_for_A,
            score_for_AB=score_for_AB,
            den=parity_den,
        ).expand_to(1)

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
        """
        Exact division in split coordinates, where it is componentwise (handles zero-norm divisors).

        A zero divisor y (u2 == 0 or v2 == 0) still divides x when x is 0 in that component too,
            but then that component of the quotient could be anything, so this picks one.

        Returns:
            QuadInt | None: A quotient q with x == q*y, or None if there is none.

        Raises:
            ZeroDivisionError: If y is 0.
        """
        u1, v1 = _split_uv(x)
        u2, v2 = _split_uv(y)

        if u2 == 0 and v2 == 0:
            raise ZeroDivisionError("division by zero")

        if u2 == 0 or v2 == 0:
            # Taking the free component equal to the other one keeps the den=1 parity rule (u == v mod 2),
            #   and makes q the plain integer x/y whenever there is one, e.g. (3 + 3j) / (1 + j) == 3.
            n, d, x_zero = (v1, v2, u1) if u2 == 0 else (u1, u2, v1)
            q, r = divmod(n, d)
            return self._uv_to_ab(q, q) if x_zero == 0 and r == 0 else None

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

        su, tu, gu = ZZ.gcdex(u1, u2)
        sv, tv, gv = ZZ.gcdex(v1, v2)

        # For den=2, no parity constraint on (u, v) — the ring IS Z*Z.
        return self._canonicalize_bezout_result(
            self._uv_to_ab(int(gu), int(gv)),
            self._uv_to_ab(int(su), int(sv)),
            self._uv_to_ab(int(tu), int(tv)),
        )
