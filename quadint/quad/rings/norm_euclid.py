from __future__ import annotations

from math import isqrt
from typing import TYPE_CHECKING, ClassVar

from quadint.quad.rings.base import (
    QuadraticRing,
    _NeighborhoodSearch,
    _round_div_ties_away_from_zero,
)

if TYPE_CHECKING:
    from quadint.quad.int import QuadInt

NORM_EUCLID_D: set[int] = {-11, -7, -3, -2, -1, 2, 3, 5, 6, 7, 11, 13, 17, 19, 21, 29, 33, 37, 41, 57, 73}


def _hyperbola_branch_centers(num_b: int, da: int, y_norm: int, D: int) -> tuple[int, int]:
    """
    Return the integers nearest to (num_b - t) / y_norm and (num_b + t) / y_norm, where t = |da| / sqrt(D).

    For a fixed A in RealNormEuclidRing._divmod_on_branches, these are the two B values where the hyperbola branches
        db = +/- t are crossed (with db = B*y_norm - num_b). Floats lose precision past 2**53 and overflow past ~1e308,
        so this only uses integer arithmetic, with isqrt handling the square root exactly.

    Returns:
        tuple[int, int]: The two rounded B coordinates, in no particular order.
            Ties round up, but a tie needs t to be rational, which (for non-square D) only happens when da == 0.
    """
    if y_norm < 0:
        # (num_b -/+ t) / y_norm == (-num_b +/- t) / -y_norm, so this is the same pair of values
        num_b, y_norm = -num_b, -y_norm

    # The nearest integer to v = (num_b + u) / y_norm is floor(v + 1/2), which is
    #
    #     floor((2*num_b + y_norm + 2*u) / (2*y_norm))
    #
    # For integers m and k > 0, floor((m + w) / k) == (m + floor(w)) // k for ANY real w,
    #   so with u = +/- t all we need are floor(2t) and floor(-2t) == -ceil(2t), which isqrt gives exactly.
    m = 2 * num_b + y_norm
    k = 2 * y_norm

    q, rem = divmod(4 * da * da, D)
    floor_2t = isqrt(q)  # floor(sqrt(floor(z))) == floor(sqrt(z))

    # 2t is only an integer when 4*da**2 / D is a perfect square
    ceil_2t = floor_2t if rem == 0 and floor_2t * floor_2t == q else floor_2t + 1

    return (m - ceil_2t) // k, (m + floor_2t) // k


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
            It is usually near the rounded quotient, and otherwise out on the hyperbola branches.

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

        return self._divmod_on_branches(x, y, search, num_a, num_b, y_norm)

    def _divmod_on_branches(
        self,
        x: QuadInt,
        y: QuadInt,
        search: _NeighborhoodSearch,
        num_a: int,
        num_b: int,
        y_norm: int,
    ) -> tuple[QuadInt, QuadInt]:
        """
        Carry on a local quotient search that found nothing, out along the hyperbola branches.

        The real norm is indefinite, so reducing quotients are not always near the rounded quotient (A0, B0).
            With da = A*y_norm - num_a and db = B*y_norm - num_b, the candidate (A, B) is scored on
            da**2 - D*db**2 == (da - sqrt(D)*db) * (da + sqrt(D)*db), which stays small along db = +/- da/sqrt(D)
            however far A is from A0. Next to a crossing it is about 2*sqrt(D)*|da|*|y_norm| per unit of B,
            so once A is outside the local search, only the B values within 1 of a crossing can reduce |N|.

        This is shared by the subclasses, which score candidates with their own phi through `search`.
            (The imaginary rings never get here, their rounded quotient or one of its neighbors always reduces.)

        Returns:
            tuple[QuadInt, QuadInt]: The quotient and remainder.

        Raises:
            NotImplementedError: If no candidate reduces phi within the widest A range.
        """
        D, A0 = self.D, search.A0

        scanned = 0  # the local search already covered A0, whose crossings are right next to B0
        for rad in (64, 128, 256, 512, 1024, 2048, 4096, 65536):
            for d in range(scanned + 1, rad + 1):
                for A in (A0 - d, A0 + d):
                    da = A * y_norm - num_a
                    for Bc in _hyperbola_branch_centers(num_b, da, y_norm, D):
                        for B in (Bc - 1, Bc, Bc + 1):
                            search.consider(A, B)

            scanned = rad

            best_score = search.best_score
            if best_score is not None and best_score[0] == 0:
                best_a, best_b = search.best_ab
                q = x._make(best_a, best_b)
                return q, x - q * y

        raise NotImplementedError(
            f"No phi-reducing quotient found for D={D}, den={self.den} within search radii",
        )
