from __future__ import annotations

import warnings

from functools import cache
from typing import TYPE_CHECKING, ClassVar, cast

from quadint.quad.rings.base import (
    QuadraticRing,
    _NeighborhoodSearch,
    _round_div_ties_away_from_zero,
)
from quadint.quad.rings.norm_euclid import RealNormEuclidRing
from quadint.utils import requires_modules

if TYPE_CHECKING:
    from quadint.quad.int import QuadInt


def _is_squarefree(n: int) -> bool:
    from sympy import factorint  # noqa: PLC0415

    n = abs(n)
    if n <= 1:
        return False

    facts = factorint(n)
    return all(i < 2 for i in facts.values())


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
        while (n % p) == 0:
            n //= p
            e += 1

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
        # (71, 1): (5, 23),
        (77, 2): (13, 1, 23, 1),
        # D=83 was also quite fun, and required expansion of the heuristic search area to beyond 60,000.
        #   I worry this means the heuristic search area will need to increase for larger D values...
        #   But frankly I don't know what the theoretical max should be?
        (83, 1): (19, 1, 29, 1),
        (86, 1): (5, 1, 7, 1),
        (89, 2): (11, 1, 17, 1),
        (93, 2): (7, 1, 11, 1),
        (94, 1): (3, 1, 5, 1),
        (97, 2): (3, 1, 11, 2),
    }

    @classmethod
    def accept_override(cls, D: int, den: int, default_den: int) -> bool:
        """Read the above docstring and comments to understand the logic in this method."""
        # Harper-style only relevant for real quadratic maximal orders
        if D <= 0 or den != default_den:
            return False

        if (D, den) in cls._HARDCODED:
            return True

        try:
            import cypari  # noqa: PLC0415, F401
        except ImportError:
            warnings.warn(
                "This may be a Harper-like ring, however without cypari installed we cannot use it as such.",
                ImportWarning,
                stacklevel=3,
            )
            return False

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
        if temp_ring.class_number() != 1:
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
    @requires_modules(["cypari"])
    def _find_admissible_witness_primes(
        self,
        *,
        prime_bound: int = 200,
    ) -> tuple[int, int, int, int] | None:
        """
        Search for a Harper-style admissible prime-ideal pair for the real quadratic field
        with discriminant `disc`, and return a *Harper-like* witness (small generators).

        Returns:
            None if not found within prime_bound.
            Otherwise (p1, i1, p2, i2) where:
              - p1, p2 are rational primes
              - i1 is the 1-based index of the chosen prime ideal in idealprimedec(nf,p1)
              - i2 is the 1-based index of the chosen prime ideal in idealprimedec(nf,p2)
        """
        from cypari import pari  # noqa: PLC0415

        disc0 = int(self.discriminant())
        B = int(prime_bound)

        # We do the heavy lifting in GP to avoid depending on cypari object APIs.
        # Strategy:
        #  - Build bnf/nf and fundamental unit eps.
        #  - Loop over rational primes p1<p2 up to B.
        #  - For each split prime ideal P with Norm(P)=p, try pairs (P1,P2).
        #  - Let I=P1^2*P2^2, compute bid=idealstar(nf,I,2) with invariants cyc.
        #  - Compute v(-1), v(eps) = ideallog(...) vectors mod cyc.
        #  - Decide if these generate the full group via Smith normal form index test.
        #
        # Return [] if none found, or [p1,i1,p2,i2] if found.

        gp = f"""
    {{
      my(disc={disc0}, B={B});
      my(bnf = bnfinit(quadpoly(disc)));
      my(nf  = bnf.nf);
      my(eps = Vec(bnf.fu)[1]);   /* normalize */

      my(res = []);
      my(found = 0);

      forprime(p1=3, B,
        if(found, break);
        if(p1==2 || (disc % p1)==0, next);

        my(dec1 = Vec(idealprimedec(nf, p1)));
        for(i1=1, #dec1,
          if(found, break);
          my(P1 = dec1[i1]);
          if(idealnorm(nf, P1) != p1, next);  /* split prime ideal of norm p1 */

          forprime(p2=p1+1, B,
            if(found, break);
            if(p2==2 || (disc % p2)==0, next);

            my(dec2 = Vec(idealprimedec(nf, p2)));
            for(i2=1, #dec2,
              if(found, break);
              my(P2 = dec2[i2]);
              if(idealnorm(nf, P2) != p2, next);

              my(I = idealmul(nf, idealpow(nf, P1, 2), idealpow(nf, P2, 2)));
              my(bid = idealstar(nf, I, 2));

              my(cyc = Vec(bid.cyc));
              my(k = #cyc);

              /* trivial group => surjective */
              if(k==0, res = [p1,i1,p2,i2]; found=1; break);

              my(v1 = Vec(ideallog(nf, -1,  bid)));
              my(v2 = Vec(ideallog(nf, eps, bid)));

              /* M = [diag(cyc) | v1 | v2] is k x (k+2) over Z */
              my(M = matrix(k, k+2, i,j, 0));
              for(i=1,k,
                M[i,i]   = cyc[i];
                M[i,k+1] = v1[i];
                M[i,k+2] = v2[i];
              );

              /* Index of lattice generated by columns:
                 idx = abs(det(mathnf(M))) when rank = k */
              my(H = mathnf(M));
              my(idx = abs(matdet(H)));

              if(idx==1, res = [p1,i1,p2,i2]; found=1; break);
            );
          );
        );
      );

      res
    }}
    """
        out = pari(gp)

        # `out` is either [] (empty GP vector) or [p1,i1,p2,i2].
        try:
            if len(out) == 0:
                return None
            # PARI vectors are 1-indexed; cypari wrappers usually expose 0-indexed python access.
            # We therefore read by python indexing first, and fall back to 1-index style if needed.
            try:
                p1, i1, p2, i2 = (int(out[0]), int(out[1]), int(out[2]), int(out[3]))
            except (ValueError, TypeError):
                p1, i1, p2, i2 = (int(out[1]), int(out[2]), int(out[3]), int(out[4]))
        except Exception:
            # Extremely defensive: if wrapper doesn't support len()/indexing cleanly
            s = str(out)
            if s in ("[]", "Vecsmall([])", "vector([])"):
                return None
            raise

        return p1, i1, p2, i2

    @requires_modules(["cypari"])
    def _principal_generator_from_witness_prime(self, p: int, i: int) -> QuadInt:
        """
        Convert one PARI witness component (p, i) into a principal generator π of the
            chosen prime ideal P = idealprimedec(nf,p)[i], returned as a QuadInt.

        The result is only defined up to multiplication by a unit.

        Returns:
            QuadInt: The principal generator π.
        """
        from cypari import pari  # noqa: PLC0415

        disc = int(self.discriminant())
        p0 = int(p)
        i0 = int(i)
        den0 = int(self.den)

        # We use bnfinit(...,1) so PARI has exact algebraic data for generators/units.
        # Then:
        #   P = idealprimedec(nf,p)[i]
        #   [e,t] = bnfisprincipal(bnf,P)
        # In class number 1, e is empty (or zero vector), and t generates P up to units.
        #
        # Convert t to integral-basis coordinates c = [c0,c1] relative to nf.zk = [1,w]
        # where w = sqrt(D) if den=1, and w = (1+sqrt(D))/2 if den=2.
        # Then map to internal numerators (a + b*sqrt(D))/den:
        #   den=1: a=c0, b=c1
        #   den=2: c0 + c1*w = (2*c0 + c1 + c1*sqrt(D))/2
        #          so internal (a,b) = (2*c0 + c1, c1)
        gp = f"""
    {{
      my(disc={disc}, p={p0}, idx={i0}, den={den0});
      my(bnf = bnfinit(quadpoly(disc), 1));
      my(nf  = bnf.nf);

      my(dec = Vec(idealprimedec(nf, p)));
      if(idx < 1 || idx > #dec, error("witness index out of range"));

      my(P = dec[idx]);

      /* Must be a prime ideal of norm p (split case in your search). */
      if(idealnorm(nf, P) != p, error("idealprimedec witness is not norm-p prime ideal"));

      my(v = bnfisprincipal(bnf, P));  /* [e, t] */
      my(e = v[1], t = v[2]);

      /* In class number 1, PARI returns empty e-vector; in general, require trivial class. */
      if(#e > 0,
        for(k=1, #e, if(e[k] != 0, error("prime ideal is not principal in this field")))
      );

      my(c = Vec(nfalgtobasis(nf, t)));
      if(#c != 2, error("expected quadratic field basis coordinates"));

      /* c = [c0,c1] in integral basis [1,w]. Convert to your QuadInt numerators. */
      if(den == 1,
        [c[1], c[2]],
        [2*c[1] + c[2], c[2]]
      )
    }}
    """
        out = pari(gp)

        try:
            a = int(out[0])
            b = int(out[1])
        except (ValueError, TypeError):
            a = int(out[1])
            b = int(out[2])

        x = self(a, b)

        # Normalize only by ± and conjugation so literature comparison is obvious.
        # (Real quadratic torsion units are usually ±1, so this is the relevant ambiguity.)
        candidates = (x, -x, x.conjugate(), (-x).conjugate())

        def key(z: QuadInt) -> tuple[int, int, int, int]:
            # prefer smaller |a|, then |b|, then sign-tie-breakers
            return abs(z.a), abs(z.b), z.a, z.b

        return min(candidates, key=key)

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
            while n % p == 0:
                n //= p
                e += 1
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

        # Use the actual Harper phi on the actual divisor.
        # (Do NOT use the old phi(w) < phi(y)^2 shortcut once phi is ideal-sensitive.)
        phi_y = self.phi(y)

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

        def score_for_AB(A: int, B: int) -> tuple[int, ...]:
            q = x._make(A, B)
            r = x - q * y
            pr = self.phi(r)
            dist2 = (A - A0) * (A - A0) + (B - B0) * (B - B0)
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
        best_q: QuadInt | None = None
        best_r: QuadInt | None = None
        seen: set[tuple[int, int]] = set()

        phi = self.phi
        make = x._make
        den = self.den

        def consider(A: int, B: int) -> None:
            """Score one lattice candidate (A,B) exactly once."""
            nonlocal best_score, best_q, best_r

            if den == 2 and ((A ^ B) & 1):
                return

            key = (A, B)
            if key in seen:
                return
            seen.add(key)

            q = make(A, B)
            r = x - q * y
            pr = phi(r)
            dist2 = (A - A0) * (A - A0) + (B - B0) * (B - B0)
            s = (0 if pr < phi_y else 1, pr, dist2)

            if best_score is None or s < best_score:
                best_score = s
                best_q = q
                best_r = r

        def has_reducing_best() -> bool:
            return best_score is not None and best_score[0] == 0

        @cache
        def _candidate_Bs_for_A(A: int) -> tuple[int, ...]:
            """Generate a small cached set of promising B values for this A."""
            da = A * y_norm - num_a
            cands: set[int] = set()

            def add_with_parity(Bcand: int):
                if den == 1:
                    cands.add(Bcand)
                    return

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

            return tuple(sorted(cands))

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

            if has_reducing_best() and best_q is not None and best_r is not None:
                return best_q, best_r

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

    _ring = QuadraticRing(D_, den_)
    HarperRing._HARDCODED[D_, den_] = (_ring(p1_a_, p1_b_), _ring(p2_a_, p2_b_))
