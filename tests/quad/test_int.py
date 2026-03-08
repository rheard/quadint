from __future__ import annotations

import os
import random

from math import gcd, isqrt, prod

import pytest

from quadint import QuadInt, complexint
from quadint.quad import Factorization, QuadraticRing


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


ZN17 = QuadraticRing(-17)
ZN7 = QuadraticRing(-7)
Z1 = QuadraticRing(1)
Z2 = QuadraticRing(2)
Z5 = QuadraticRing(5)
Z15 = QuadraticRing(15)

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

    def test_units_are_units(self):
        """Every element in .units should be a unit."""
        for ring in [ZI, ZE, Z1, Z2, ZN7]:
            one = ring.one
            for u in one.units:
                assert u.is_unit(), f"{u} in {ring} should be a unit"

    def test_zero_is_not_unit(self):
        """Zero is never a unit."""
        assert not complexint(0, 0).is_unit()
        assert not ZE(0, 0).is_unit()

    def test_primes_are_not_units(self):
        """Non-unit elements should return False."""
        assert not complexint(1, 1).is_unit()  # norm 2
        assert not complexint(2, 0).is_unit()  # norm 4
        assert not Z2(3, 1).is_unit()  # norm 7

    def test_gaussian_units(self):
        """Verify the four Gaussian units."""
        assert complexint(1, 0).is_unit()
        assert complexint(-1, 0).is_unit()
        assert complexint(0, 1).is_unit()
        assert complexint(0, -1).is_unit()

    def test_eisenstein_units(self):
        """Verify the six Eisenstein units."""
        # In internal numerator coords (den=2): units are ±1, ±ω, ±ω²
        assert ZE(2, 0).is_unit()  # 1
        assert ZE(-2, 0).is_unit()  # -1
        assert ZE(-1, 1).is_unit()  # ω
        assert ZE(1, -1).is_unit()  # -ω


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
