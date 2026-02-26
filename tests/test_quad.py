from __future__ import annotations

import importlib.util
import os
import random

from math import gcd, isqrt, prod
from pathlib import Path

import pytest

import quadint

from quadint import QuadInt, complexint
from quadint.quad import Factorization, QuadraticRing
from quadint.quad.rings import HarperRing, _is_squarefree  # noqa: PLC2701

requires_cypari = pytest.mark.skipif(
    importlib.util.find_spec("cypari") is None,
    reason="requires cypari",
)


def norm_multiset(primes: dict[QuadInt, int]) -> list[int]:
    """Return sorted list of norms with multiplicity."""
    out: list[int] = []
    for p, k in primes.items():
        out.extend([abs(p)] * k)
    out.sort()
    return out


def brute_content(x: QuadInt) -> int:
    """Reference implementation: scan all divisors of gcd(a,b)."""
    g = gcd(abs(x.a), abs(x.b))
    if g <= 1:
        return 1

    den = x.ring.den
    best = 1
    r = isqrt(g)
    for d in range(1, r + 1):
        if g % d:
            continue
        for cand in (d, g // d):
            if cand <= best:
                continue
            a = x.a // cand
            b = x.b // cand
            if den == 2 and ((a ^ b) & 1):
                continue
            best = cand

    return best


@pytest.mark.skipif(
    os.getenv("CI", "").lower() not in {"1", "true", "yes"},
    reason="Compiled-only test",
)
def test_compiled_tests():
    """Verify that we are running these tests with a compiled version of quadint"""
    path = Path(quadint.__file__)
    assert path.suffix.lower() != ".py"


class RingTests:
    """Support methods for testing QuadraticRing singleton behavior"""

    @staticmethod
    def assert_same_ring_obj(a: QuadraticRing, b: QuadraticRing):
        """Validate rings are literally the same object and have same parameters"""
        assert a.D == b.D
        assert a.den == b.den
        assert a is b

    @staticmethod
    def assert_diff_ring_obj(a: QuadraticRing, b: QuadraticRing):
        """Validate rings are different objects (identity differs)"""
        assert a is not b


class TestQuadraticRing(RingTests):
    """Tests for the QuadraticRing behavior"""

    def test_same_instance_default_den(self):
        """QuadraticRing(D) should be a singleton per (D, default_den)"""
        q1 = QuadraticRing(-1)
        q2 = QuadraticRing(-1)
        self.assert_same_ring_obj(q1, q2)

    def test_same_instance_none_den(self):
        """QuadraticRing(D) and QuadraticRing(D, None) should be the same object"""
        q1 = QuadraticRing(5)
        q2 = QuadraticRing(5, None)
        self.assert_same_ring_obj(q1, q2)
        assert q1.den == 2  # 5 % 4 == 1 -> default den=2

    def test_same_instance_explicit_default_den(self):
        """QuadraticRing(D, den=<computed default>) should match QuadraticRing(D)"""
        # D=-3 -> default den=2
        q1 = QuadraticRing(-3)
        q2 = QuadraticRing(-3, 2)
        self.assert_same_ring_obj(q1, q2)

        # D=-1 -> default den=1
        q3 = QuadraticRing(-1)
        q4 = QuadraticRing(-1, 1)
        self.assert_same_ring_obj(q3, q4)

    def test_different_den(self):
        """QuadraticRing(D, den=...) must singleton by (D, den) key"""
        q_default = QuadraticRing(-3)  # default den=2
        q_other = QuadraticRing(-3, 1)  # non-maximal order (or at least non-default)
        self.assert_diff_ring_obj(q_default, q_other)

        assert q_default.D == q_other.D
        assert q_default.den != q_other.den

    def test_different_D(self):
        """Different D must produce different ring objects"""
        q1 = QuadraticRing(-1)
        q2 = QuadraticRing(-2)
        self.assert_diff_ring_obj(q1, q2)

    @pytest.mark.parametrize("den", [0, 3, -1])
    def test_invalid_den_rejected(self, den: int):
        """Only den=1 or den=2 are supported by QuadraticRing."""
        with pytest.raises(ValueError, match="den must be 1 or 2"):
            _ = QuadraticRing(5, den)

    def test_repeated_calls_do_not_mutate_cached_instance(self):
        """Repeated construction should not corrupt cached fields"""
        q = QuadraticRing(1)  # default den=2
        assert q.D == 1
        assert q.den == 2

        # Repeat a bunch of times; should remain stable
        for _ in range(100):
            q2 = QuadraticRing(1, None)
            self.assert_same_ring_obj(q, q2)


class TestRingCapabilities:
    """Tests for lightweight capability helpers on QuadraticRing."""

    @pytest.mark.parametrize(
        ("ring", "expected"),
        [
            (QuadraticRing(-1), True),
            (QuadraticRing(-7), True),
            (QuadraticRing(0), True),
            (QuadraticRing(1), True),
            (QuadraticRing(15), False),
            (QuadraticRing(16), False),
            (QuadraticRing(-17), False),
            (QuadraticRing(-3, 1), False),
        ],
        ids=str,
    )
    def test_supports_division(self, ring: QuadraticRing, *, expected: bool):
        """supports_division should mirror whether this ring has a divmod implementation."""
        assert ring.supports_division() is expected

    def test_supports_division_matches_runtime_behavior(self):
        """Unsupported rings should raise NotImplementedError from division operations."""
        supported = QuadraticRing(-1)
        unsupported = QuadraticRing(15)

        assert supported.supports_division() is True
        assert unsupported.supports_division() is False

        x = supported(5, 2)
        y = supported(3, -1)
        q, r = divmod(x, y)
        assert x == q * y + r

        with pytest.raises(NotImplementedError):
            _ = divmod(unsupported(5, 2), unsupported(3, -1))

    @pytest.mark.parametrize(
        ("ring", "expected"),
        [
            (QuadraticRing(-1), True),
            (QuadraticRing(-2), True),
            (QuadraticRing(-3), True),
            (QuadraticRing(-7), True),
            (QuadraticRing(-11), True),
            (QuadraticRing(2), False),
            (QuadraticRing(-3, 1), False),
            (QuadraticRing(-17), False),
        ],
        ids=str,
    )
    def test_supports_factorization(self, ring: QuadraticRing, *, expected: bool):
        """supports_factorization should mirror whether this ring has factorization support."""
        assert ring.supports_factorization() is expected

    def test_supports_factorization_matches_runtime_behavior(self):
        """Unsupported rings should raise NotImplementedError from factorization operations."""
        supported = QuadraticRing(-1)
        unsupported = QuadraticRing(2)

        assert supported.supports_factorization() is True
        assert unsupported.supports_factorization() is False

        factors = supported(5, 2).factor_detail()
        assert isinstance(factors, Factorization)

        with pytest.raises(NotImplementedError):
            _ = unsupported(5, 2).factor_detail()


class TestIdentityChecksWithQuadInt(RingTests):
    """Tests that rely on QuadInt.assert_same_ring using identity"""

    def test_elements_from_separate_ring_construction_can_mix(self):
        """If ring is cached, elements built via separate QuadraticRing(D) calls must interoperate."""
        Q1 = QuadraticRing(-1)
        Q2 = QuadraticRing(-1)

        # If caching is broken, this would raise TypeError in assert_same_ring (identity mismatch)
        a = QuadInt(1, 2, Q1)
        b = QuadInt(3, 4, Q2)

        c = a + b
        assert isinstance(c, QuadInt)
        assert c.a == 4
        assert c.b == 6
        assert c.ring is Q1  # result keeps self.ring

    def test_elements_from_different_den_do_not_mix(self):
        """Identity checks should still protect against mixing different rings (even with same D)."""
        Q_default = QuadraticRing(-3)  # den=2
        Q_other = QuadraticRing(-3, 1)  # den=1

        a = QuadInt(2, 0, Q_default)  # ok parity for den=2
        b = QuadInt(1, 0, Q_other)

        with pytest.raises(TypeError):
            _ = a + b


ZN17 = QuadraticRing(-17)
ZN11 = QuadraticRing(-11)
ZN7 = QuadraticRing(-7)
ZN2 = QuadraticRing(-2)
Z1 = QuadraticRing(1)
Z2 = QuadraticRing(2)
Z5 = QuadraticRing(5)
Z15 = QuadraticRing(15)
Z31 = QuadraticRing(31)
Z69 = QuadraticRing(69)

ZI = QuadraticRing(-1)
ZE = QuadraticRing(-3)


class QuadIntTests:
    """Tests for __div__"""

    a_int, b_int, a_cint, b_cint = None, None, None, None

    def setup_method(self, _):
        """Setup some test data"""
        self.a_int = Z2(5, 2)
        self.b_int = Z2(3, -2)

        self.a_cint = complexint(5, 2)
        self.b_cint = complexint(3, -2)

    @staticmethod
    def assert_quad_equal(res: tuple | QuadInt, res_int: QuadInt):
        """Validate the QuadInt is equal to the validation object, and that it is still backed by integers"""
        assert res[0] == res_int.a
        assert res[1] == res_int.b

        assert isinstance(res_int.a, int)
        assert isinstance(res_int.b, int)

        assert isinstance(res_int, QuadInt)

    def assert_factoring(self, n: QuadInt, factors: Factorization | dict):
        """Validate everything about the factoring is correct"""
        if isinstance(factors, Factorization):
            ans = factors.prod()
            primes = factors.primes
        else:
            ans = prod(p**k for p, k in factors.items())
            primes = factors

        self.assert_quad_equal(n, ans)

        for p in primes:
            # These _should_ all be primes and should be impossible to factor...
            if isinstance(factors, Factorization):
                prime_factors = p.factor_detail()

                assert abs(prime_factors.unit) == 1
                assert len(prime_factors.primes) == 1

                factored_p, factored_k = next(iter(prime_factors.primes.items()))
                assert factored_k == 1
                # Because the unit is not integrated, the prime returned may be an associate of the prime we care about
                assert any(p == factored_p * u for u in p.units)
            else:
                prime_factors = p.factor()

                assert len(prime_factors) == 1

                factored_p, factored_k = next(iter(prime_factors.items()))
                assert factored_k == 1
                assert factored_p == p


class TestDiv(QuadIntTests):
    """Tests for __div__"""

    def test_div(self):
        """Test QuadInt / QuadInt in QuadraticRing(2)"""
        mul_int = self.a_int * self.b_int

        res_int = mul_int / self.a_int
        self.assert_quad_equal((self.b_int.a, self.b_int.b), res_int)

        res_int = mul_int / self.b_int
        self.assert_quad_equal((self.a_int.a, self.a_int.b), res_int)

    def test_div_int(self):
        """Test QuadInt / int in QuadraticRing(2)"""
        mul_int = self.a_int * 3
        res_int = mul_int / 3
        self.assert_quad_equal((self.a_int.a, self.a_int.b), res_int)

    def test_div_float(self):
        """Test QuadInt / float in QuadraticRing(2)"""
        mul_int = self.a_int * 3
        res_int = mul_int / float(3)
        self.assert_quad_equal((self.a_int.a, self.a_int.b), res_int)

    def test_div_ring5(self):
        """Test QuadInt / QuadInt in QuadraticRing(5)"""
        a_int = Z5(7, 3)
        b_int = Z5(3, -5)

        mul_int = a_int * b_int

        res_int = mul_int / a_int
        self.assert_quad_equal((b_int.a, b_int.b), res_int)

        res_int = mul_int / b_int
        self.assert_quad_equal((a_int.a, a_int.b), res_int)

    def test_div_ring15(self):
        """Test QuadInt / QuadInt in QuadraticRing(15)"""
        a_int = Z15(5, 2)
        b_int = Z15(3, -2)

        mul_int = a_int * b_int

        with pytest.raises(NotImplementedError):
            _ = mul_int / a_int

    def test_div_ring_neg7(self):
        """Test QuadInt / QuadInt in QuadraticRing(-7)"""
        a_int = ZN7(7, 3)
        b_int = ZN7(3, -5)

        mul_int = a_int * b_int

        res_int = mul_int / a_int
        self.assert_quad_equal((b_int.a, b_int.b), res_int)

        res_int = mul_int / b_int
        self.assert_quad_equal((a_int.a, a_int.b), res_int)

    def test_div_ring_neg17(self):
        """Test QuadInt / QuadInt in QuadraticRing(-17) (-17 is not norm-Euclidean so this is not implemented)"""
        a_int = ZN17(5, 2)
        b_int = ZN17(3, -2)

        mul_int = a_int * b_int

        with pytest.raises(NotImplementedError):
            _ = mul_int / a_int

    def test_pow_mod_matches_pow_then_mod_gaussian(self):
        """pow(x, e, m) should match (x**e) % m in Gaussian integers."""
        x = ZI(5, 2)
        m = ZI(2, 1)

        got = pow(x, 13, m)
        expected = (x**13) % m

        self.assert_quad_equal((expected.a, expected.b), got)

    def test_pow_mod_exponent_zero(self):
        """pow(x, 0, m) should be 1 % m."""
        ZI = QuadraticRing(-1)
        x = ZI(5, 2)
        m = ZI(2, 1)

        got = pow(x, 0, m)
        expected = x.one % m

        self.assert_quad_equal((expected.a, expected.b), got)

    def test_pow_mod_accepts_int_modulus(self):
        """Allow int modulus (it should embed into the same ring)."""
        x = ZI(5, 2)

        got = pow(x, 20, 7)
        expected = (x**20) % 7

        self.assert_quad_equal((expected.a, expected.b), got)

    def test_pow_mod_requires_same_ring(self):
        """Modulus must be in the same QuadraticRing (identity check)."""
        x = ZI(5, 2)
        m_other_ring = Z2(2, 1)

        with pytest.raises(TypeError):
            pow(x, 5, m_other_ring)

    def test_pow_mod_zero_modulus_raises(self):
        """pow(x, e, 0) should raise like Python ints."""
        x = ZI(5, 2)

        with pytest.raises(ZeroDivisionError):
            pow(x, 5, 0)

    def test_pow_mod_negative_exponent_raises(self):
        """pow(x, -e, m) should raise."""
        x = ZI(5, 2)
        m = ZI(2, 1)

        with pytest.raises(ValueError, match="Negative powers not supported in quadratic integer rings"):
            pow(x, -1, m)

    def test_splitring_den2_does_not_force_qu_qv_parity(self):
        """The parity constraint for split integer division doesn't apply when den=2 (the default). Verify that"""
        # D=1 defaults to den=2 (maximal-order convention), so this uses SplitRing with den=2.
        Z = QuadraticRing(1)
        assert Z.den == 2

        # Counterexample where the best (qu, qv) in the 3x3 neighborhood has opposite parity.
        # With the current code, SplitRing.divmod forces same parity and picks a worse remainder.
        x = Z(-15, -15)
        y = Z(-15, -13)

        q, r = divmod(x, y)

        # sanity
        assert x == q * y + r

        # measure remainder size in split (u,v) coords: u=(a+b)/den, v=(a-b)/den
        den = Z.den
        ru = (r.a + r.b) // den
        rv = (r.a - r.b) // den

        # The truly-best local choice gives (ru, rv)=(-1, 0) => ru^2+rv^2 == 1.
        # The current parity-forced choice gives (ru, rv)=(-1, -1) => 2.
        assert ru * ru + rv * rv == 1

    @staticmethod
    def _rand_elem(rng: random.Random, Q: QuadraticRing, bound: int) -> QuadInt:
        """Create a random ring element while respecting den=2 parity constraints."""
        a = rng.randint(-bound, bound)
        b = rng.randint(-bound, bound)

        if Q.den == 2 and ((a ^ b) & 1):
            b += 1

        return Q(a, b)

    @pytest.mark.parametrize("D", [-11, -7, -3, -2, -1, 2, 3, 5, 6, 7, 11, 13], ids=str)
    def test_divmod_random_remainder_is_norm_reducing(self, D: int):
        """For supported norm-Euclidean orders, divmod should satisfy x=qy+r and |N(r)| < |N(y)|."""
        Q = QuadraticRing(D)
        rng = random.Random(10_000 + D)

        # Include both medium and larger coefficients to exercise neighborhood expansion.
        for bound in (50, 5000):
            for _ in range(20):
                x = self._rand_elem(rng, Q, bound)
                y = self._rand_elem(rng, Q, bound)

                while not y:
                    y = self._rand_elem(rng, Q, bound)

                q, r = divmod(x, y)

                assert x == q * y + r, f"division identity failed for D={D}, x={x}, y={y}"
                assert abs(abs(r)) < abs(abs(y)), f"non-reducing remainder for D={D}, x={x}, y={y}, r={r}"


def _rand_elem(rng: random.Random, Q: QuadraticRing, bound: int) -> QuadInt:
    """Create a random ring element while respecting den=2 parity constraints."""
    a = rng.randint(-bound, bound)
    b = rng.randint(-bound, bound)

    if Q.den == 2 and ((a ^ b) & 1):
        b += 1

    return Q(a, b)


class TestClark69EuclideanFunction:
    """Unit tests for the Euclidean function used by the D=69 division algorithm."""

    def test_phi69_basic_integer_values(self):
        """phi(0)=0, phi(1)=1, and the only tweak is 23 -> 26 (multiplicatively)."""
        assert Z69._phi_from_abs_norm(0) == 0
        assert Z69._phi_from_abs_norm(1) == 1
        assert Z69._phi_from_abs_norm(-1) == 1

        assert Z69._phi_from_abs_norm(23) == 26
        assert Z69._phi_from_abs_norm(-23) == 26

        assert Z69._phi_from_abs_norm(23 * 23) == 26 * 26
        assert Z69._phi_from_abs_norm(11) == 11
        assert Z69._phi_from_abs_norm(23 * 11) == 26 * 11

    def test_phi69_matches_known_prime_above_23(self):
        """
        In Z[(1+sqrt(69))/2], the elements (23 ± 3*sqrt(69))/2 have norm -23.
        So phi should map them to 26.
        """
        assert Z69.den == 2

        p = Z69(23, 3)
        assert abs(p) == -23
        assert Z69.phi(p) == 26

        p_conj = p.conjugate()
        assert abs(p_conj) == -23
        assert Z69.phi(p_conj) == 26

    def test_phi69_sign_and_conjugation_invariant(self):
        """Phi depends only on |N(x)|, so it's invariant under x -> -x and conjugation."""
        rng = random.Random(69_000)
        for _ in range(200):
            x = _rand_elem(rng, Z69, 10_000)
            assert Z69.phi(x) == Z69.phi(-x)
            assert Z69.phi(x) == Z69.phi(x.conjugate())

    def test_phi69_is_multiplicative(self):
        """Sanity: phi(xy) == phi(x)*phi(y) because |N| is multiplicative and v23 adds."""
        rng = random.Random(69_001)
        for _ in range(200):
            x = _rand_elem(rng, Z69, 2_000)
            y = _rand_elem(rng, Z69, 2_000)
            if not x or not y:
                continue
            assert Z69.phi(x * y) == Z69.phi(x) * Z69.phi(y)


class TestDivClark69:
    """Integration tests for the D=69 division algorithm."""

    def test_ring69_supports_division(self):
        """Once the D=69 ring override is installed, QuadraticRing(69) must support divmod."""
        assert Z69.den == 2
        assert Z69.supports_division() is True

    def test_divmod_zero_divisor_raises(self):
        """Division by 0 should raise."""
        x = Z69(10, 2)
        with pytest.raises(ZeroDivisionError):
            divmod(x, Z69.zero)

    def test_divmod_random_remainder_is_phi_reducing(self):
        """
        For a Euclidean function phi, divmod must satisfy x=qy+r and phi(r) < phi(y) (or r==0).
        Mirrors your existing norm-reduction randomized test, but uses phi69.
        """
        rng = random.Random(69_123)

        for bound in (50, 5000):
            for _ in range(40):
                x = _rand_elem(rng, Z69, bound)
                y = _rand_elem(rng, Z69, bound)

                while not y:
                    y = _rand_elem(rng, Z69, bound)

                # Skip torsion units; Euclidean condition would force remainder 0 anyway.
                if abs(abs(y)) == 1:
                    continue

                q, r = divmod(x, y)

                assert x == q * y + r, f"division identity failed: x={x}, y={y}, q={q}, r={r}"

                # Membership / parity sanity in den=2 ring.
                assert q.ring is Z69
                assert r.ring is Z69
                assert ((q.a ^ q.b) & 1) == 0
                assert ((r.a ^ r.b) & 1) == 0

                assert Z69.phi(r) < Z69.phi(y), f"phi did not reduce: x={x}, y={y}, q={q}, r={r}"

    @requires_cypari
    def test_harper_ring_similar(self):
        """While the Harper ring is not required to produce identical results, some are (and logically should be)"""
        # The only real problem with this is we NEED cypari to compute the admissible pairs in real time.
        #   They shouldn't be in the hardcoded list because then it may affect the subclass mechanism.
        #   Fwiw, the admissible pair that would be hardcoded is: ((-5-1*sqrt(69))/2, (-5-3*sqrt(69))/2)
        H69 = HarperRing(69)
        a1 = Z69(69 * 2, 420)
        a2 = H69(69 * 2, 420)

        b1 = Z69(13, 37)
        b2 = H69(13, 37)

        q1, r1 = divmod(a1, b1)
        q2, r2 = divmod(a2, b2)

        # Note that because we're forcing different rings, equality breaks down because we rely on instance checks.
        #   As long as users just stick to using QuadraticRing(69) then this won't be a problem for them...
        assert q1.a == q2.a
        assert q1.b == q2.b
        assert r1.a == r2.a
        assert r1.b == r2.b

    def test_clark_regression_pair_phi_fix(self):
        """
        Regression inspired by the standard D=69 obstruction near primes over 23.

        Let
            y = (23 + 3*sqrt(69))/2   with |N(y)| = 23  and phi(y)=26,
            x = 18 + 2*sqrt(69).

        The naive q=1 step gives remainder r0 = x - y = (13 + sqrt(69))/2 with |N(r0)|=25,
        which is not norm-reducing (25 > 23), but is phi-reducing (25 < 26).
        """
        y = Z69(23, 3)
        x = Z69(36, 4)  # == 18 + 2*sqrt(69)

        # Verify the "naive" step math.
        r0 = x - y
        assert abs(abs(r0)) == 25
        assert abs(abs(y)) == 23
        assert Z69.phi(r0) == 25
        assert Z69.phi(y) == 26
        assert Z69.phi(r0) < Z69.phi(y)

        # And verify divmod actually produces a phi-reducing remainder.
        q, r = divmod(x, y)
        assert x == q * y + r
        assert Z69.phi(r) < Z69.phi(y)

    def test_euclid_loop_terminates_and_phi_strictly_decreases(self):
        """A tiny Euclidean-algorithm loop should strictly decrease phi and terminate quickly."""
        a = Z69(36, 4)  # 18 + 2*sqrt(69)
        b = Z69(23, 3)  # (23 + 3*sqrt(69))/2

        steps = 0
        while b:
            q, r = divmod(a, b)
            assert a == q * b + r
            assert Z69.phi(r) < Z69.phi(b)

            a, b = b, r
            steps += 1
            assert steps < 200, "Euclidean descent did not terminate (phi not strictly decreasing?)"

        # a is the last nonzero remainder: it must divide the original inputs.
        x0 = Z69(36, 4)
        assert (x0 % a) == 0


class TestHarperHelpers:
    """Tests for the standalone helper functions used by HarperRing."""

    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (0, False),
            (1, False),
            (-1, False),
            (2, True),
            (3, True),
            (4, False),
            (5, True),
            (6, True),
            (8, False),
            (9, False),
            (10, True),
            (12, False),
            (14, True),
            (15, True),
            (16, False),
            (18, False),
            (21, True),
            (22, True),
            (23, True),
            (24, False),
            (29, True),
            (31, True),
            (45, False),
            (61, True),
            (69, True),  # 69 = 3 * 23, squarefree
            (-14, True),
            (-18, False),
        ],
        ids=str,
    )
    def test_squarefree(self, n: int, *, expected: bool):
        """Verify the squarefree helper handles signs and repeated prime factors correctly."""
        assert _is_squarefree(n) is expected

    @pytest.mark.parametrize(
        ("D", "den", "expected_disc"),
        [
            (14, None, 56),  # default den=1 -> disc = 4D
            (14, 1, 56),
            (14, 2, 14),  # explicit non-default order
            (23, None, 92),
            (23, 1, 92),
            (61, None, 61),  # default den=2 since 61 % 4 == 1
            (61, 2, 61),
            (61, 1, 244),  # explicit non-default order
            (69, None, 69),
            (69, 2, 69),
            (69, 1, 276),
        ],
        ids=str,
    )
    def test_disc_from_D_den(self, D: int, den: int | None, expected_disc: int):
        """Verify discriminant helper follows the den=1 vs den=2 convention."""
        assert QuadraticRing(D, den).discriminant() == expected_disc


@requires_cypari
class TestHarperPariHelpers:
    """Tests for PARI-backed helpers used by HarperRing."""

    @pytest.mark.parametrize(
        ("D", "den", "expected"),
        [
            (14, 1, True),  # Q(sqrt(14)) maximal order
            (61, 2, True),  # Q(sqrt(61)) maximal order
            (69, 2, True),  # Q(sqrt(69)) maximal order
            (15, 1, False),  # Q(sqrt(15)) has class number > 1
        ],
        ids=str,
    )
    def test_class_number_is_one(self, D: int, den: int, expected: bool):  # noqa: FBT001
        """Verify the class-number helper on a few known real quadratic discriminants."""
        assert (QuadraticRing(D, den).class_number() == 1) is expected

    @pytest.mark.parametrize(
        ("D", "den"),
        # Skip any principal generators
        [k for k, v in HarperRing._HARDCODED.items() if len(v) == 4],
        ids=str,
    )
    def test_find_admissible_witness_pair_known_cases(self, D: int, den: int):
        """Known Harper cases should yield an admissible prime pair via PARI search."""
        r = QuadraticRing(D, den)
        out = r._find_admissible_witness_primes()

        assert out is not None, f"expected admissible pair for D={D}, den={den}"
        assert isinstance(out, tuple)
        assert len(out) == 4
        assert all(isinstance(v, int) for v in out)

        p1, _, p2, _ = out
        assert HarperRing._HARDCODED[D, den] == out
        assert p1 > 1
        assert p2 > 1
        assert p1 != p2

    @pytest.mark.parametrize(
        ("D", "den"),
        # Only principal generators
        [k for k, v in HarperRing._HARDCODED.items() if len(v) == 2],
        ids=str,
    )
    def test_principal_generators_known_cases(self, D: int, den: int):
        """Known Harper cases should yield principal generators via PARI-backed search."""
        r = QuadraticRing(D, den)
        out = r._find_admissible_witness_primes()

        assert out is not None, f"expected admissible pair for D={D}, den={den}"
        assert isinstance(out, tuple)
        assert len(out) == 2
        assert all(isinstance(v, QuadInt) for v in out)

        p1, p2 = out
        assert HarperRing._HARDCODED[D, den] == out
        assert p1 != p2


class TestHarperAcceptOverride(RingTests):
    """Tests for HarperRing selection logic and hardcoded coverage."""

    @pytest.mark.parametrize(
        ("D", "den"),
        HarperRing._HARDCODED,
        ids=str,
    )
    def test_hardcoded_cases_are_selected(self, D: int, den: int):
        """QuadraticRing should select HarperRing for every hardcoded Harper case."""
        Q = QuadraticRing(D, den)
        assert isinstance(Q, HarperRing)
        assert Q.supports_division() is True

    @pytest.mark.parametrize(
        ("D", "den"),
        [(14, 2), (22, 2), (23, 2), (61, 1)],
        ids=str,
    )
    def test_nonmax_orders_not_selected(self, D: int, den: int):
        """HarperRing should only apply to the maximal order (default denominator)."""
        default_den = 2 if (D % 4) == 1 else 1
        assert den != default_den

        Q = QuadraticRing(D, den)
        assert not isinstance(Q, HarperRing)

    @pytest.mark.parametrize(
        ("D", "den", "default_den"),
        [
            (0, 1, 1),
            (-14, 1, 1),
            (-61, 2, 2),
            (14, 2, 1),  # wrong denominator for maximal order
        ],
        ids=str,
    )
    def test_accept_override_rejects_wrong_domain(self, D: int, den: int, default_den: int):
        """HarperRing.accept_override should reject non-real or non-maximal inputs."""
        assert HarperRing.accept_override(D, den, default_den) is False

    def test_hardcoded_subset_contains_expected_literature_values(self):
        """The hardcoded list should include the standard Harper/Conrad examples."""
        expected = {(14, 1), (22, 1), (23, 1), (61, 2)}
        assert expected.issubset(HarperRing._HARDCODED)


class TestHarperDiv:
    """Tests for HarperRing.divmod and Euclidean reduction on hardcoded Harper rings."""

    @staticmethod
    def _rand_elem(rng: random.Random, Q: QuadraticRing, bound: int) -> QuadInt:
        """Create a random ring element while respecting den=2 parity constraints."""
        a = rng.randint(-bound, bound)
        b = rng.randint(-bound, bound)

        if Q.den == 2 and ((a ^ b) & 1):
            b += 1

        return Q(a, b)

    @pytest.mark.parametrize(
        ("D", "den"),
        HarperRing._HARDCODED,
        ids=str,
    )
    def test_zero_divisor(self, D: int, den: int):
        """Division by zero should raise ZeroDivisionError."""
        Q = QuadraticRing(D, den)
        x = Q(7, 3)
        with pytest.raises(ZeroDivisionError):
            _ = divmod(x, Q.zero)

    @pytest.mark.parametrize(
        ("D", "den", "xa", "xb", "ya", "yb"),
        [
            (14, 1, 69, 420, 13, 37),
            (22, 1, 91, 315, 11, 17),
            (23, 1, 77, 221, 9, 10),
            (61, 2, 69, 421, 13, 37),  # odd/odd parity for den=2
        ],
        ids=str,
    )
    def test_example_division_identity_and_phi_reduction(
        self,
        D: int,
        den: int,
        xa: int,
        xb: int,
        ya: int,
        yb: int,
    ):
        """A few concrete examples should satisfy x=qy+r and reduce the Euclidean size."""
        Q = QuadraticRing(D, den)
        x = Q(xa, xb)
        y = Q(ya, yb)

        q, r = divmod(x, y)

        assert x == q * y + r
        assert Q.phi(r) < Q.phi(y), f"phi did not decrease for D={D}, x={x}, y={y}, r={r}"

    @pytest.mark.parametrize(
        ("D", "den"),
        HarperRing._HARDCODED,
        ids=str,
    )
    def test_exact_products_divide_with_zero_remainder(self, D: int, den: int):
        """If x = a*b exactly then divmod(x, a) should return (b, 0)."""
        Q = QuadraticRing(D, den)

        # Hand-picked exact factors with valid parity in den=2 case.
        examples = [
            (Q(5, 1), Q(3, 1)),
            (Q(7, -3), Q(9, 1)),
            (Q(11, 5), Q(-3, 7)),
        ]

        for a, b in examples:
            x = a * b
            q, r = divmod(x, a)

            assert r == Q.zero, f"expected exact division remainder 0 for D={D}, x={x}, a={a}"
            assert q == b, f"expected exact quotient b for D={D}, x={x}, a={a}, got q={q}"

            # Also verify __truediv__ integration on exact divisions.
            assert x / a == b

    @pytest.mark.parametrize(
        ("D", "den"),
        HarperRing._HARDCODED,
        ids=str,
    )
    def test_random_phi_reducing(self, D: int, den: int):
        """Randomized regression: Harper divmod should always preserve identity and reduce phi."""
        Q = QuadraticRing(D, den)
        rng = random.Random(70_000 + 100 * D + den)

        # Use both small and medium sizes to exercise quotient-neighborhood searches.
        for bound in (40, 300):
            for _ in range(50):
                x = self._rand_elem(rng, Q, bound)
                y = self._rand_elem(rng, Q, bound)

                while not y:
                    y = self._rand_elem(rng, Q, bound)

                q, r = divmod(x, y)

                assert x == q * y + r, f"division identity failed for D={D}, x={x}, y={y}, q={q}, r={r}"
                assert Q.phi(r) < Q.phi(y), f"non-reducing remainder for D={D}, x={x}, y={y}, r={r}"

    @pytest.mark.parametrize(
        ("D", "den"),
        # I have cached all values below 100.
        #   These are the first couple rings passed 100 that should also work (if cypari is installed).
        [(101, 2), (103, 1)],
        ids=str,
    )
    @requires_cypari
    def test_random_phi_reducing_not_cached(self, D: int, den: int):
        """Randomized regression: Harper divmod should always preserve identity and reduce phi."""
        assert (D, den) not in HarperRing._HARDCODED

        Q = QuadraticRing(D, den)
        assert isinstance(Q, HarperRing)
        rng = random.Random(70_000 + 100 * D + den)

        # Use both small and medium sizes to exercise quotient-neighborhood searches.
        for bound in (40, 300):
            for _ in range(50):
                x = self._rand_elem(rng, Q, bound)
                y = self._rand_elem(rng, Q, bound)

                while not y:
                    y = self._rand_elem(rng, Q, bound)

                q, r = divmod(x, y)

                assert x == q * y + r, f"division identity failed for D={D}, x={x}, y={y}, q={q}, r={r}"
                assert Q.phi(r) < Q.phi(y), f"non-reducing remainder for D={D}, x={x}, y={y}, r={r}"

    def test_try_exact_quotient_success_and_failure(self):
        """_try_exact_quotient should return q for exact division and None otherwise."""
        Q = QuadraticRing(14, 1)
        assert isinstance(Q, HarperRing)

        y = Q(5, 1)
        q_expected = Q(3, 2)
        x = q_expected * y

        assert Q._try_exact_quotient(x, y) == q_expected
        assert Q._try_exact_quotient(x + Q.one, y) is None

    def test_valuation_at_generator_counts_exact_powers(self):
        """_valuation_at_generator should count repeated exact divisibility by a generator."""
        Q = QuadraticRing(14, 1)
        assert isinstance(Q, HarperRing)

        pi = Q(5, 1)
        x = pi * pi * pi

        assert Q._valuation_at_generator(x, pi) == 3
        assert Q._valuation_at_generator(Q.one, pi) == 0


class TestUnits:
    """Tests for the units"""

    def test_units_sizes(self):
        """Verify Gaussian and Eisenstein rings have the correct unit lengths"""
        assert len(Z1(2, 0).units) == 4
        assert len(ZI(1, 0).units) == 4
        assert len(ZE(2, 0).units) == 6

        assert len(Z2(1, 0).units) == 2

    @pytest.mark.parametrize("x", [ZI(3, 2), ZE(5, 1), Z2(7, 3)], ids=str)
    def test_canonical_associate_is_idempotent_and_unit_invariant(self, x: QuadInt):
        """Canonical associate selection should be stable across repeated calls and unit multiples."""
        base = x._canonical_associate()
        assert base == base._canonical_associate()

        for u in x.units:
            y = x * u
            assert y._canonical_associate() == base


class TestContent:
    """Tests for the content method"""

    def test_zero(self):
        """Use the standard content(0)=0 convention."""
        assert complexint(0, 0).content() == 0

    def test_gaussian(self):
        """Verify content is produced correctly for Gaussian rings"""
        x = complexint(4, 53)
        assert x.content() == 1

        y = complexint(6, 0)
        assert y.content() == 6

    def test_parity_matters(self):
        """Verify content is parity-aware and takes the den value into account"""
        # D=-3 defaults to den=2
        assert ZE.den == 2

        one = ZE(2, 0)  # (2+0*sqrt(-3))/2 == 1
        two = ZE(4, 0)  # == 2
        three = ZE(6, 0)  # == 3

        # You can't pull out a factor of 2 from "1" in den=2 ring because it breaks parity.
        assert one.content() == 1
        # But you can pull out 2 from "2"
        assert two.content() == 2
        # And you can pull out 3 from "3"
        assert three.content() == 3

    @pytest.mark.parametrize("k", [-7, -2, -1, 1, 2, 9])
    def test_content_scales(self, k: int):
        """Verify content(k*x)=|k|*content(x) for Gaussian integers."""
        x = complexint(4, 6)
        assert x.content() == 2

        y = x * k
        assert y.content() == abs(k) * x.content()

    @pytest.mark.parametrize("k", [-7, -2, -1, 1, 2, 9])
    def test_content_scales_den2(self, k: int):
        """Verify content(k*x)=|k|*content(x) in den=2 rings too."""
        x = ZE(4, 0)  # == 2, content 2
        assert x.content() == 2

        y = x * k
        assert y.content() == abs(k) * x.content()

    @pytest.mark.parametrize(
        "x",
        [
            complexint(4, 6),
            complexint(6, 0),
            ZE(4, 2),
            ZE(6, 6),
        ],
    )
    def test_content_divides_numerators(self, x: QuadInt):
        """Verify the returned content divides stored ring numerators."""
        c = x.content()
        assert c > 0
        assert x.a % c == 0
        assert x.b % c == 0

    @pytest.mark.skipif(
        os.getenv("CI", "").lower() in {"1", "true", "yes"},
        reason="Local-only test",
    )
    @pytest.mark.parametrize(
        "x",
        [
            complexint(1_999_966_000_289, 999_983_000_138),
            complexint(2_000_000_000_000, 1_500_000_000_000),
            ZE(1_800_000_000_000, 600_000_000_000),
            ZE(2_400_000_000_000, 1_200_000_000_000),
        ],
        ids=str,
    )
    def test_content_matches_reference_large_values(self, x: QuadInt):
        """Large-value regression: optimized content matches the old divisor-scan logic."""
        assert x.content() == brute_content(x)


# region Factoring tests
class TestFactorDetail(QuadIntTests):
    """Tests for the factor_detail method"""

    @pytest.mark.parametrize(
        "x",
        [
            ZN2(1, 0),
            ZN2(-1, 0),
            ZN2(0, 1),
            ZN2(5, 2),
            ZN2(4, 53),
            ZN2(6, 0),
            ZN7(2, 0),
            ZN7(-2, 0),
            ZN7(0, 2),
            ZN7(10, 2),
            ZN7(4, 56),
            ZN7(6, 0),
            ZN11(2, 0),
            ZN11(-2, 0),
            ZN11(0, 2),
            ZN11(10, 2),
            ZN11(4, 56),
            ZN11(6, 0),
        ],
        ids=str,
    )
    def test_examples(self, x: QuadInt):
        """Validate some given examples"""
        f = x.factor_detail()
        self.assert_factoring(x, f)

    @pytest.mark.parametrize(
        "x",
        [
            ZN2(0, 0),
            ZN7(0, 0),
            ZN11(0, 0),
        ],
        ids=str,
    )
    def test_zero_raises(self, x: QuadInt):
        """Validate zero cannot be factored"""
        with pytest.raises(ValueError, match="0 does not have a finite factorization"):
            _ = x.factor_detail()

    @pytest.mark.parametrize(
        "x",
        [
            # This test only works with D=-2 because the other two require same parity, and I can't
            #   create the same parity as 0 with 2 odd primes...
            ZN2(17 * 31, 0),
        ],
        ids=str,
    )
    def test_527_primitive_vs_full(self, x: QuadInt):
        """Test factoring with a composite wholly real number"""
        # 17 splits (norm 17 twice), 31 stays Gaussian prime (norm 31^2)
        f_full = x.factor_detail()
        self.assert_factoring(x, f_full)
        assert len(f_full.primes) == 3
        assert norm_multiset(f_full.primes) == [17, 17, 31 * 31]

    @pytest.mark.parametrize(
        ("z", "expected_prime_norm"),
        [
            (ZN2(1, 1), 3),
            (ZN7(-1, 1), 2),
            (ZN11(3, 1), 5),
        ],
        ids=str,
    )
    def test_square(self, z: QuadInt, expected_prime_norm: int):
        """This is designed to catch candidate pruning that accidentally drops a needed divisor."""
        x = z * z

        f = x.factor_detail()
        self.assert_factoring(x, f)

        norms = norm_multiset(f.primes)
        assert norms.count(expected_prime_norm) == 2, (
            f"expected two norm-{expected_prime_norm} primes, got norms={norms} primes={f.primes}"
        )

    @pytest.mark.parametrize("a", [-4, -2, -1, 1, 2, 5])
    @pytest.mark.parametrize("b", [-5, -3, -1, 1, 3, 4])
    @pytest.mark.parametrize("test_klass", [ZN2, ZN7, ZN11])
    def test_small_grid(self, a: int, b: int, test_klass: QuadraticRing):
        """Check factor_detail().prod() round-trips for a small Gaussian grid."""
        try:
            x = test_klass(a, b)
        except ValueError:
            return

        if not x:
            return

        f = x.factor_detail()
        self.assert_factoring(x, f)

    def test_notimplemented_for_positive_D(self):
        """Factorization implemented only for imaginary D"""
        x = Z2(5, 2)
        with pytest.raises(NotImplementedError):
            _ = x.factor_detail()

    def test_notimplemented_for_non_norm_euclid_ring(self):
        """Factorization only reliably make sense for norm-Euclidean rings."""
        # D=-17 is proven to be not norm-Euclidean
        x = ZN17(5, 2)
        with pytest.raises(NotImplementedError):
            _ = x.factor_detail()

    @pytest.mark.parametrize("test_klass", [ZN2, ZN7, ZN11])
    def test_associate_factorization_norm_invariant(self, test_klass: QuadraticRing):
        """Associates should keep the same factor-norm multiset."""
        x = test_klass(4, 56)
        fx = x.factor_detail()
        base_norms = norm_multiset(fx.primes)

        for u in x.units:
            y = x * u
            fy = y.factor_detail()
            assert fy.prod() == y
            assert norm_multiset(fy.primes) == base_norms


class TestFactor(QuadIntTests):
    """Tests for factor"""

    @pytest.mark.parametrize(
        "x",
        [
            ZN2(4, 53),
            ZN2(6, 0),
            ZN2(17 * 31, 0),
            ZN2(5, 2),
            ZN7(4, 56),
            ZN7(6, 0),
            ZN7(17, 7),
            ZN11(4, 56),
            ZN11(6, 0),
            ZN11(17, 7),
        ],
        ids=str,
    )
    def test_examples(self, x: QuadInt):
        """Validate some given examples"""
        factors = x.factor()
        assert isinstance(factors, dict)
        self.assert_factoring(x, factors)


# endregion
