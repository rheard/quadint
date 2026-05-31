from __future__ import annotations

import warnings

from functools import cache
from math import gcd
from typing import TYPE_CHECKING, ClassVar, cast

from sympy import sieve
from sympy.ntheory import discrete_log, primitive_root

from quadint.quad.rings.base import (
    PrimeIdealData,
    QuadraticRing,
    _NeighborhoodSearch,
    _round_div_ties_away_from_zero,
)
from quadint.quad.rings.norm_euclid import NORM_EUCLID_D, RealNormEuclidRing
from quadint.utils import _is_squarefree

if TYPE_CHECKING:
    from quadint.quad.int import QuadInt


class Clark69Ring(RealNormEuclidRing):
    """
    Euclidean division for the maximal order of Q(sqrt(69)), i.e. Z[(1+sqrt(69))/2].

    This ring is Euclidean but not norm-Euclidean (Clark, 1994).

    A working Euclidean function can be taken as: |N(x)| with a single tweak that replaces each prime
        factor "23" in |N(x)| by "26". Equivalently: if v23(|N(x)|)=e then
        phi(x) = (|N(x)| / 23**e) * 26**e

    The obstruction to norm-Euclidean-ness lives entirely at the primes above 23;
        inflating 23->26 fixes Euclidean descent.
    """

    SUPPORTS_DIVISION: ClassVar[bool] = True

    _BAD_P: ClassVar[int] = 23
    _BAD_REPL: ClassVar[int] = 26

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:
        """Supported for the maximal order for D=69 only"""
        return D == 69 and den == default_den

    @classmethod
    def _phi_from_abs_norm(cls, abs_norm: int) -> int:
        """Compute Clark's adjusted Euclidean function value from the integer |N(x)|."""
        n = int(abs_norm)
        if n < 0:
            n = -n
        if n == 0:
            return 0

        p = cls._BAD_P
        e = 0
        qn, rn = divmod(n, p)
        while rn == 0:
            n = qn
            e += 1
            qn, rn = divmod(n, p)

        if e:
            n *= pow(cls._BAD_REPL, e)

        return n

    def phi(self, x: QuadInt) -> int:
        """Return Clark's adjusted Euclidean function for the `D=69` maximal order."""
        return self._phi_from_abs_norm(super().phi(x))

    def divmod(self, x: QuadInt, y: QuadInt) -> tuple[QuadInt, QuadInt]:
        """Divide `x` by `y` and return a Clark-admissible quotient and remainder."""
        # Same scaffolding as RealNormEuclidRing.divmod, but accept via phi() instead of |N|.
        y_norm = abs(y)  # signed norm (may be negative for D>0)
        abs_y_norm = abs(y_norm)
        if abs_y_norm == 0:
            raise ZeroDivisionError

        phi_y = self._phi_from_abs_norm(abs_y_norm)
        phi_y2 = phi_y * phi_y

        # Candidate center from x/y ≈ (x * conj(y)) / N(y)
        a1, b1 = x.a, x.b
        a2, b2 = y.a, y.b

        # num = x * conj(y), computed in numerators directly:
        # (a1+b1√D)(a2-b2√D) = (a1*a2 - b1*b2*D) + (a2*b1 - a1*b2)√D
        num_a = a1 * a2 - b1 * b2 * self.D
        num_b = a2 * b1 - a1 * b2

        if self.den != 1:
            if (num_a % self.den) != 0 or (num_b % self.den) != 0:
                raise ArithmeticError("Non-integral product; check ring parameters / parity")
            num_a //= self.den
            num_b //= self.den

        A0 = _round_div_ties_away_from_zero(num_a, y_norm)
        B0 = _round_div_ties_away_from_zero(num_b, y_norm)

        dd = self.den * self.den  # here dd=4

        def B0_for_A(A: int) -> int:  # noqa: ARG001
            return B0

        # Prefer any phi-reducing remainder; among those, minimize phi(w), then distance to (A0,B0).
        #
        # We work with w = q*N(y) - x*conj(y) = -(x-qy)*conj(y).
        # Since phi is multiplicative (by construction), phi(w) = phi(x-qy)*phi(y),
        # so phi(x-qy) < phi(y)  <=>  phi(w) < phi(y)^2.
        def score_for_AB(A: int, B: int) -> tuple[int, ...]:
            da = A * y_norm - num_a
            db = B * y_norm - num_b

            # numerator of N(w) where w=(da + db*sqrt(D))/den
            nw_num = da * da - self.D * (db * db)
            abs_nw_num = abs(nw_num)

            if abs_nw_num % dd:
                # Should not happen if parity/integrality is consistent, but be safe.
                return (1, abs_nw_num, (A - A0) * (A - A0) + (B - B0) * (B - B0))

            abs_nw = abs_nw_num // dd
            phi_w = self._phi_from_abs_norm(abs_nw)

            flag = 0 if phi_w < phi_y2 else 1
            dist2 = (A - A0) * (A - A0) + (B - B0) * (B - B0)
            return flag, phi_w, dist2

        # A slightly bigger radius schedule than the norm-euclid case, just in case.
        search = _NeighborhoodSearch(
            A0=A0,
            B0_for_A=B0_for_A,
            score_for_AB=score_for_AB,
            den=self.den,
        )

        for rad in (1, 2, 3, 4, 6, 8, 12, 16):
            best_a, best_b = search.expand_to(rad)

            best_score = search.best_score
            if best_score is not None and best_score[0] == 0:
                q = x._make(best_a, best_b)
                r = x - q * y
                return q, r

        raise NotImplementedError(
            f"No phi-reducing quotient found for D={self.D}, den={self.den} within search radii",
        )


class HarperRing(RealNormEuclidRing):
    """
    Harper-style Euclidean division for selected real quadratic maximal orders.

    This implementation uses admissible prime-pair witnesses to define a weighted
    Euclidean score `phi` and then performs a nearest-lattice quotient search.

    We can really only do the Harper-like method with cypari, and even then it requires care to
        test the witness speedup.

    Therefor without cypari, this will only work for the D values that have been hard-coded and validated.
    Even with cypari, the default behavior will be to tend towards accuracy, so division algorithms may be slow
        (or heck, untested or incorrect).

    CORRECTION: I have since written `Ideal`, `IdealClass`, and a whole host of classes and methods specifically
        to replace cypari. Yes, in calculating the `class_number`, but also explicitly here: when it comes to finding
        the admissible pairs.

        As such we can now find new admissible pairs without cypari at all. While it is a bit slower,
            it also means we can eliminate a dependency which is not supported on all platforms and versions.
    """

    SUPPORTS_DIVISION = True  # once divmod is implemented

    # According to the rules, any D value added here (with default den):
    #   * Must be square free (no prime factors with an exponent 2 or greater).
    #   * Must have class number 1 (_class_number_is_one is True).
    #   * Must have an admissible prime pair.
    #
    # This is a list of witness primes OR principal generators (which are defined in the _POST_HARDCODED list below)
    _HARDCODED: ClassVar[dict[tuple[int, int], tuple[int, int, int, int] | tuple[QuadInt, QuadInt]]] = {
        (14, 1): (5, 1, 43, 1),
        (22, 1): (3, 1, 29, 1),
        (23, 1): (11, 1, 13, 1),
        (31, 1): (3, 1, 5, 1),
        (43, 1): (7, 1, 53, 1),
        (46, 1): (3, 1, 5, 1),
        (47, 1): (11, 1, 53, 1),
        (53, 2): (11, 1, 29, 1),
        (59, 1): (5, 1, 47, 1),
        (61, 2): (3, 1, 5, 1),
        (62, 1): (13, 1, 23, 1),
        (67, 1): (7, 1, 149, 1),
        # While this algorithm can apply to D=69 (with den=2), it is less efficient.
        #   Its best to keep Clark69Ring for D=69 for efficiency
        #       It can also technically apply to norm-Euclidean rings, but we don't use it for them either...
        # (69, 2): (11, 1, 53, 1),
        #
        # D=71 was fun and required further investigation and expansion of this algorithm.
        #   Essentially for this ring, witnesses are simply not good enough for division.
        #   Instead they need to be converted to principal generators (which are quadratic integers),
        #       and use a slightly different (read slower) algorithm.
        #   See _POST_HARDCODED at end of file.
        # (71, 1): (5, 1, 23, 1),
        (77, 2): (13, 1, 23, 1),
        # D=83 was also quite fun, and required expansion of the heuristic search area to beyond 60,000.
        #   I worry this means the heuristic search area will need to increase for larger D values...
        #   But frankly I don't know what the theoretical max should be?
        (83, 1): (19, 1, 29, 1),
        (86, 1): (5, 1, 7, 1),
        (89, 2): (11, 1, 17, 1),
        (93, 2): (7, 1, 11, 1),
        (94, 1): (3, 1, 5, 1),
        (97, 2): (3, 1, 11, 1),
    }

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:
        """Read the above docstring and comments to understand the logic in this method."""
        # Harper-style only relevant for real quadratic maximal orders
        if D <= 0 or den != default_den:
            return False

        # Norm-Euclidean rings have a much cheaper direct division algorithm.
        # QuadraticRing checks subclasses before their parents, so without this guard
        # HarperRing would capture these D values before RealNormEuclidRing can.
        if D in NORM_EUCLID_D:
            return False

        if (D, den) in cls._HARDCODED:
            return True

        if not _is_squarefree(D):
            warnings.warn(
                "D is not squarefree, and may be an alias for a more primitive ring.",
                RuntimeWarning,
                stacklevel=3,
            )
            return False

        temp_ring: QuadraticRing = cls(D, den)
        disc = temp_ring.discriminant()

        # if isqrt(abs(disc)) ** 2 == abs(disc):
        #     return False

        # Must be a PID
        if temp_ring.class_number != 1:
            return False

        # Murty-Srinivas-Subramani state that Harper's thesis established
        #   all real quadratic fields with discriminant ≤ 500
        #   and class number one are Euclidean.
        if disc <= 500:
            return True

        return HarperRing._find_admissible_witness_primes(cast("HarperRing", temp_ring)) is not None

    # region Finding admissible primes
    #   The code in this region is the code needed to add new entries to the HARDCODED caches above and below.
    #   The process for that should be to find admissible witness primes, then add them to _HARDCODED above.
    #       Then run the test suite. If it does not pass then the witness speed-up trick will not work.
    #   You will need to turn the witness primess into the more reliable principal generators,
    #       by passing the witness primes set to _principal_generators_from_witness, and then add these
    #       to _POST_HARDCODED at the end of the file.
    @cache
    def _is_admissible_pair(
        self,
        P1: PrimeIdealData,
        P2: PrimeIdealData,
        epsilon: QuadInt,
    ) -> bool:
        """Return True iff P1, P2 satisfy Harper's admissible-pair unit condition."""
        p1 = P1.p
        p2 = P2.p

        if p1 == p2:
            return False
        if p1 == 2 or p2 == 2:
            return False
        if self.discriminant() % p1 == 0 or self.discriminant() % p2 == 0:
            return False
        if P1.root_p2() is None or P2.root_p2() is None:
            return False
        if P1.ideal.norm != p1 or P2.ideal.norm != p2:
            return False

        mod1 = p1 * p1
        mod2 = p2 * p2

        g1 = primitive_root(mod1)
        g2 = primitive_root(mod2)

        neg = -self.one

        neg_logs = (
            int(discrete_log(mod1, P1.residue_mod_p2(neg), g1)),
            int(discrete_log(mod2, P2.residue_mod_p2(neg), g2)),
        )
        eps_logs = (
            int(discrete_log(mod1, P1.residue_mod_p2(epsilon), g1)),
            int(discrete_log(mod2, P2.residue_mod_p2(epsilon), g2)),
        )

        n1 = p1 * (p1 - 1)
        n2 = p2 * (p2 - 1)

        a1, a2 = neg_logs
        b1, b2 = eps_logs

        index = abs(n1 * n2)
        index = gcd(index, abs(n1 * a2))
        index = gcd(index, abs(n1 * b2))
        index = gcd(index, abs(n2 * a1))
        index = gcd(index, abs(n2 * b1))
        index = gcd(index, abs(a1 * b2 - a2 * b1))

        return index == 1

    @cache
    def _find_admissible_witness_primes(
        self,
        *,
        prime_bound: int = 200,
    ) -> tuple[int, int, int, int] | None:
        """Search for a Harper admissible split-prime witness pair."""
        epsilon = self.fundamental_unit()

        candidates: list[PrimeIdealData] = []
        for p in sieve.primerange(3, prime_bound + 1):
            if self.discriminant() % p == 0:
                continue

            candidates.extend(
                data for data in self.prime_ideals_data_over(p) if data.ideal.norm == p and data.root_p2() is not None
            )

        for i, P1 in enumerate(candidates):
            for P2 in candidates[i + 1 :]:
                if self._is_admissible_pair(P1, P2, epsilon):
                    return P1.p, P1.index, P2.p, P2.index

        return None

    def _principal_generator_from_witness_prime(self, p: int, i: int) -> QuadInt:
        """Return a principal generator for the selected prime ideal over p."""
        data = self.prime_ideals_data_over(p)[i - 1]

        if data.ideal.norm != p:
            raise ValueError("witness does not select a norm-p split prime ideal")

        x = data.ideal.principal_generator()
        if x is None:
            raise ArithmeticError("selected witness prime ideal is not principal")

        candidates = (x, -x, x.conjugate(), (-x).conjugate())
        return min(candidates, key=lambda z: (abs(z.a), abs(z.b), z.a, z.b))

    def _principal_generators_from_witness(
        self,
        witness: tuple[int, int, int, int],
    ) -> tuple[QuadInt, QuadInt]:
        """Convert a full admissible witness (p1,i1,p2,i2) into two prime generators."""
        p1, i1, p2, i2 = witness
        pi1 = self._principal_generator_from_witness_prime(p1, i1)
        pi2 = self._principal_generator_from_witness_prime(p2, i2)

        # Order deterministically for easy comparison
        if (abs(pi2.a), abs(pi2.b), abs(abs(pi2))) < (abs(pi1.a), abs(pi1.b), abs(abs(pi1))):
            pi1, pi2 = pi2, pi1
        return pi1, pi2

    # endregion

    def _phi_from_abs_norm(self, abs_norm: int, witness: tuple) -> int:
        """
        Weighted-norm score used as a Harper-style search heuristic.

        Start from |N(x)| and replace selected rational-prime factors p by p+1
            (based on an admissible-pair witness), analogous in spirit to Clark69's 23->26 trick.

        Returns:
            int: Phi
        """
        n = int(abs_norm)
        if n < 0:
            n = -n
        if n == 0:
            return 0

        p1, _, p2, _ = witness
        replacements = {p1: p1 + 1, p2: p2 + 1}

        for p, p_new in replacements.items():
            e = 0
            qn, rn = divmod(n, p)
            while rn == 0:
                n = qn
                e += 1
                qn, rn = divmod(n, p)
            if e:
                n *= pow(p_new, e)

        return n

    def phi(self, x: QuadInt) -> int:
        """Return Harper's weighted Euclidean size for `x`."""
        n = super().phi(x)  # |N(x)| in your QuadraticRing base
        if n == 0:
            return 0

        cached_pair: tuple = self._HARDCODED.get((self.D, self.den), ())

        if len(cached_pair) == 4:
            return self._phi_from_abs_norm(n, cached_pair)  # faster fallback for witness cache entries

        if len(cached_pair) == 0:
            witness = self._find_admissible_witness_primes()
            if witness is None:
                raise RuntimeError("This should never happen but mypyc needs it.")

            cached_pair = self._principal_generators_from_witness(witness)

        out = n
        for pi in cached_pair:
            p = abs(abs(pi))  # |N(pi)| = rational prime p
            e = self._valuation_at_generator(x, pi)
            if e:
                out //= p**e
                out *= (p + 1) ** e
        return out

    # region Fallback helpers
    #   The methods in this region are only used if the witness primes speed-up trick fails,
    #       or obviously if that has not been manually validated.
    def _valuation_at_generator(self, x: QuadInt, pi: QuadInt) -> int:
        """v_pi(x) for a fixed chosen principal prime generator pi (ideal-specific)."""
        e = 0
        rem = x
        while rem:
            q = self.exact_div(rem, pi)
            if q is None:
                break
            rem = q
            e += 1
        return e

    # endregion

    def divmod(self, x: QuadInt, y: QuadInt) -> tuple[QuadInt, QuadInt]:
        """
        Practical Harper-style division search.

        This reuses the local lattice search used in RealNormEuclidRing / Clark69Ring,
        but scores candidates with a weighted norm heuristic based on an admissible-pair
        witness (when available). Empirical validation via tests is essential.

        Returns:
            tuple: The quotient and remainder.

        Raises:
            ZeroDivisionError: If y has an absolute norm of 0.
            ArithmeticError: TODO: Remove?
            NotImplementedError: If we were unable to find a quotient and remainder.
                Shouldn't happen. If it does, please contact a developer. Preferably one smarter than me.
        """
        y_norm = abs(y)  # signed norm (may be negative for D>0)
        abs_y_norm = abs(y_norm)
        if abs_y_norm == 0:
            raise ZeroDivisionError

        cached_pair: tuple = self._HARDCODED.get((self.D, self.den), ())
        witness: tuple[int, int, int, int] | None = cached_pair if len(cached_pair) == 4 else None

        if witness is None:
            # Use the actual Harper phi on the actual divisor.
            # (Do NOT use the old phi(w) < phi(y)^2 shortcut once phi is ideal-sensitive.)
            phi_y = self.phi(y)
            phi_y2 = 0
        else:
            phi_y = self._phi_from_abs_norm(abs_y_norm, witness)
            phi_y2 = phi_y * phi_y

        # Candidate center from x/y ≈ (x * conj(y)) / N(y)
        a1, b1 = x.a, x.b
        a2, b2 = y.a, y.b

        # num = x * conj(y), in numerator coordinates:
        # (a1+b1√D)(a2-b2√D) = (a1*a2 - b1*b2*D) + (a2*b1 - a1*b2)√D
        num_a = a1 * a2 - b1 * b2 * self.D
        num_b = a2 * b1 - a1 * b2

        if self.den != 1:
            if (num_a % self.den) != 0 or (num_b % self.den) != 0:
                raise ArithmeticError("Non-integral product; check ring parameters / parity")
            num_a //= self.den
            num_b //= self.den

        A0 = _round_div_ties_away_from_zero(num_a, y_norm)
        B0 = _round_div_ties_away_from_zero(num_b, y_norm)

        def B0_for_A(A: int) -> int:  # noqa: ARG001
            return B0

        dd = self.den * self.den

        def score_for_AB(A: int, B: int) -> tuple[int, ...]:
            dist2 = (A - A0) * (A - A0) + (B - B0) * (B - B0)

            if witness is not None:
                da = A * y_norm - num_a
                db = B * y_norm - num_b

                nw_num = da * da - self.D * db * db
                abs_nw_num = abs(nw_num)
                abs_nw, rem = divmod(abs_nw_num, dd)
                if rem:
                    return 1, abs_nw_num, dist2

                if abs_nw >= phi_y2:
                    return 1, abs_nw, dist2

                phi_w = self._phi_from_abs_norm(abs_nw, witness)
                flag = 0 if phi_w < phi_y2 else 1
                return flag, phi_w, dist2

            q = x._make(A, B)
            r = x - q * y
            pr = self.phi(r)
            flag = 0 if pr < phi_y else 1
            return flag, pr, dist2

        # Wider schedule than norm-euclid / Clark69; these cases are trickier.
        search = _NeighborhoodSearch(
            A0=A0,
            B0_for_A=B0_for_A,
            score_for_AB=score_for_AB,
            den=self.den,
        )

        for rad in (1, 2, 4, 8, 16, 32):
            best_a, best_b = search.expand_to(rad)

            best_score = search.best_score
            if best_score is not None and best_score[0] == 0:
                q = x._make(best_a, best_b)
                r = x - q * y
                return q, r

        # Branch-aware fallback for real quadratic indefinite norm.
        # For fixed A, small |da^2 - D*db^2| tends to occur near db ~= +/- |da|/sqrt(D),
        # which may correspond to B far away from the naive center B0.
        sqrtD = self.D**0.5
        best_a, best_b = search.best_ab
        best_q: QuadInt | None = None
        best_r: QuadInt | None = None

        phi = self.phi
        make = x._make
        den = self.den

        def consider(A: int, B: int) -> None:
            """Score one lattice candidate (A,B) exactly once."""
            nonlocal best_a, best_b, best_score, best_q, best_r

            if den == 2 and ((A ^ B) & 1):
                return

            if witness is not None:
                s = score_for_AB(A, B)
                if best_score is None or s < best_score:
                    best_score = s
                    best_a = A
                    best_b = B
                return

            q = make(A, B)
            r = x - q * y
            pr = phi(r)
            dist2 = (A - A0) * (A - A0) + (B - B0) * (B - B0)
            s = (0 if pr < phi_y else 1, pr, dist2)

            if best_score is None or s < best_score:
                best_score = s
                best_a = A
                best_b = B
                best_q = q
                best_r = r

        def has_reducing_best() -> bool:
            return best_score is not None and best_score[0] == 0

        @cache
        def _candidate_Bs_for_A(A: int) -> tuple[int, ...]:
            """Generate a small cached set of promising B values for this A."""
            da = A * y_norm - num_a
            cands: set[int] = set()

            if den == 1:
                # The local search already checked the center. In the branch pass
                # for indefinite norm forms, only the two hyperbola branches matter.
                t = abs(da) / sqrtD
                for sgn in (-1.0, 1.0):
                    db_target = sgn * t
                    Bf = (num_b + db_target) / y_norm
                    Bc = round(Bf)

                    cands.update(int(Bc) + dB for dB in (-1, 0, 1))

                return tuple(cands)

            def add_with_parity(Bcand: int):
                # den==2 parity constraint: A ≡ B (mod 2)
                if ((A ^ Bcand) & 1) == 0:
                    cands.add(Bcand)
                else:
                    cands.add(Bcand - 1)
                    cands.add(Bcand + 1)

            # Center-ish values
            for dB in (-2, -1, 0, 1, 2):
                add_with_parity(B0 + dB)

            # Hyperbola branch targets: db ~= +/- |da| / sqrt(D)
            # where db = B*y_norm - num_b
            t = abs(da) / sqrtD

            for sgn in (-1.0, 1.0):
                db_target = sgn * t
                Bf = (num_b + db_target) / y_norm
                Bc = round(Bf)

                for dB in range(-4, 5):
                    add_with_parity(int(Bc) + dB)

            # Midpoint spread
            mid = round(num_b / y_norm)
            for dB in (-3, -2, -1, 0, 1, 2, 3):
                add_with_parity(int(mid) + dB)

            return tuple(cands)

        prev_branch_rad = -1

        for rad in (64, 128, 256, 512, 1024, 2048, 4096, 65536):
            a_ranges: tuple[range, ...]
            if prev_branch_rad < 0:
                # First pass: scan full A range once
                a_ranges = (range(A0 - rad, A0 + rad + 1),)
            else:
                # Later passes: scan only the new A annulus
                a_ranges = (
                    range(A0 - rad, A0 - prev_branch_rad),
                    range(A0 + prev_branch_rad + 1, A0 + rad + 1),
                )

            for a_range in a_ranges:
                for A in a_range:
                    for B in _candidate_Bs_for_A(A):
                        consider(A, B)

            prev_branch_rad = rad

            if has_reducing_best():
                if best_q is not None and best_r is not None:
                    return best_q, best_r

                q = make(best_a, best_b)
                r = x - q * y
                return q, r

        raise NotImplementedError(
            f"No Harper-style phi-reducing quotient found for D={self.D}, den={self.den} "
            "within search radii; expand radius or refine weighted-phi construction",
        )


# While I've defined principal generators for all not-norm-Euclidean Euclidean fields with D<100 here,
#   the witness primes defined in _HARDCODED work for most D and the algorithm with them is faster.
_POST_HARDCODED = {
    # The following admissible primes are the ones Harper originally found:
    (14, 1): ((5, -1), (3, -2)),
    #   I've done my best to re-create the method Harper used to find these, and while I do find them as candidates,
    #       my best attempt at canonicalization of the candidates (using the code above) will find these:
    # (14, 1) = ((-1, -1), (-3, -2))
    #   While this works, I'm going to stick with Harper's originals here to pay homage.
    #       This is only used for validation of the above algorithm with the literature though anyway...
    (22, 1): ((-5, -1), (-13, -3)),
    (23, 1): ((-6, -1), (-9, -2)),
    (31, 1): ((-6, -1), (-11, -2)),
    (43, 1): ((-6, -1), (-15, -2)),
    (46, 1): ((-7, -1), (-61, -9)),
    (47, 1): ((-6, -1), (-10, -1)),
    (53, 2): ((-3, -1), (-13, -1)),
    (59, 1): ((-8, -1), (-22, -3)),
    (61, 2): ((-7, -1), (-9, -1)),
    (62, 1): ((-7, -1), (-15, -2)),
    (67, 1): ((-90, -11), (-156, -19)),
    # (69, 2): ((-5, -1), (-8, -2)),
    (71, 1): ((-17, -2), (-101, -12)),
    (77, 2): ((-5, -1), (-13, -1)),
    (83, 1): ((-8, -1), (-19, 23)),
    (86, 1): ((-9, -1), (-37, -4)),
    (89, 2): ((-20, -2), (-56, -6)),
    (93, 2): ((-7, -1), (-11, -1)),
    (94, 1): ((-29, -3), (-223, -23)),
    (97, 2): ((-20, -2), (-118, -12)),
}


for (D_, den_), ((p1_a_, p1_b_), (p2_a_, p2_b_)) in _POST_HARDCODED.items():
    if (D_, den_) in HarperRing._HARDCODED:
        continue

    _ring = HarperRing(D_, den_)
    HarperRing._HARDCODED[D_, den_] = (_ring(p1_a_, p1_b_), _ring(p2_a_, p2_b_))
