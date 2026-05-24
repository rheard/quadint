from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from quadint.quad.rings.base import (
    QuadraticRing,
    _NeighborhoodSearch,
    _round_div_ties_away_from_zero,
)

if TYPE_CHECKING:
    from quadint.quad.int import QuadInt

NORM_EUCLID_D: set[int] = {-11, -7, -3, -2, -1, 2, 3, 5, 6, 7, 11, 13, 17, 19, 21, 29, 33, 37, 41, 57, 73}


class RealNormEuclidRing(QuadraticRing):
    """
    Handle overrides where the ring of integers is norm-Euclidean.

    This class provides the general division algorithm used for both positive and negative
        discriminants in NORM_EUCLID_D (excluding D=0 and D=1 special cases).
    """

    SUPPORTS_DIVISION: ClassVar[bool] = True

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:
        """Should this class be used for the given values?"""
        return D in NORM_EUCLID_D and den == default_den

    def divmod(self, x: QuadInt, y: QuadInt):
        """
        Division in a norm-Euclidean real quadratic ring.

        We use the absolute norm as the Euclidean function:
            f(z) = |N(z)|.

        For the known finite list of D where the ring of integers is norm-Euclidean,
        there exists q such that |N(x - qy)| < |N(y)|.

        Returns:
            q, r: The quotient and remainder

        Raises:
            ZeroDivisionError: If the magnitude of the divisor is 0.
            ArithmeticError: In the event of a parity mismatch.
        """
        y_norm = abs(y)  # signed norm (may be negative for D>0)
        abs_y_norm = abs(y_norm)
        if abs_y_norm == 0:
            raise ZeroDivisionError

        # Candidate center from x/y ≈ (x * conj(y)) / N(y)
        a1, b1 = x.a, x.b
        a2, b2 = y.a, y.b

        # num = x * conj(y), but computed in numerators directly
        # (a1+b1√D)(a2-b2√D) = (a1*a2 - b1*b2*D) + (a2*b1 - a1*b2)√D
        num_a = a1 * a2 - b1 * b2 * self.D
        num_b = a2 * b1 - a1 * b2

        if self.den != 1:
            num_a, rA = divmod(num_a, self.den)
            num_b, rB = divmod(num_b, self.den)

            if rA != 0 or rB != 0:
                raise ArithmeticError("Non-integral product; check ring parameters / parity")

        A0 = _round_div_ties_away_from_zero(num_a, y_norm)
        B0 = _round_div_ties_away_from_zero(num_b, y_norm)
        dd = self.den**2
        threshold = abs_y_norm * abs_y_norm * dd

        def B0_for_A(A: int) -> int:  # noqa: ARG001
            return B0

        # Prefer any norm-reducing remainder; among those, minimize |N(r)| then distance to (A0,B0).
        def score_for_AB(A: int, B: int) -> tuple[int, ...]:
            da = A * y_norm - num_a
            db = B * y_norm - num_b

            # numerator of N(w) where w=(da + db*sqrt(D))/den
            nw_num = da * da - self.D * (db * db)
            abs_nw_num = abs(nw_num)

            # norm-reducing condition: |N(w)| < |Ny|^2  <=>  |nw_num| < |Ny|^2 * den^2
            flag = 0 if abs_nw_num < threshold else 1

            dist2 = (A - A0) * (A - A0) + (B - B0) * (B - B0)
            return flag, abs_nw_num, dist2

        # Expand search radius until we find a norm-reducing remainder.
        search = _NeighborhoodSearch(
            A0=A0,
            B0_for_A=B0_for_A,
            score_for_AB=score_for_AB,
            den=self.den,
        )

        for rad in (1, 2, 3, 4, 6, 8):
            best_a, best_b = search.expand_to(rad)

            # score_for_AB returns (flag, abs_nw_num, dist2)
            best_score = search.best_score
            if best_score is not None and best_score[0] == 0:
                q = x._make(best_a, best_b)
                r = x - q * y
                return q, r

        raise NotImplementedError(
            f"No norm-reducing quotient found for D={self.D}, den={self.den} within search radii",
        )
