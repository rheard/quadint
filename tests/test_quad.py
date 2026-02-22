import os

from math import gcd, isqrt, prod
from pathlib import Path
from typing import Union

import pytest

import quadint

from quadint import QuadInt, complexint
from quadint.quad import Factorization, QuadraticRing


def id_generator(value: str):
    """I want to see the examples in PyCharm, and this enables that..."""
    return str(value)


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


class TestQuadraticRingSingleton(RingTests):
    """Tests for the QuadraticRing __new__ cache singleton behavior"""

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

    def test_repeated_calls_do_not_mutate_cached_instance(self):
        """Repeated construction should not corrupt cached fields"""
        q = QuadraticRing(1)  # default den=2
        assert q.D == 1
        assert q.den == 2

        # Repeat a bunch of times; should remain stable
        for _ in range(100):
            q2 = QuadraticRing(1, None)
            self.assert_same_ring_obj(q, q2)


class TestIdentityChecksWithQuadInt(RingTests):
    """Tests that rely on QuadInt.assert_same_ring using identity"""

    def test_elements_from_separate_ring_construction_can_mix(self):
        """If ring is cached, elements built via separate QuadraticRing(D) calls must interoperate."""
        Q1 = QuadraticRing(-1)
        Q2 = QuadraticRing(-1)

        # If caching is broken, this would raise TypeError in assert_same_ring (identity mismatch)
        a = QuadInt(Q1, 1, 2)
        b = QuadInt(Q2, 3, 4)

        c = a + b
        assert isinstance(c, QuadInt)
        assert c.a == 4
        assert c.b == 6
        assert c.ring is Q1  # result keeps self.ring

    def test_elements_from_different_den_do_not_mix(self):
        """Identity checks should still protect against mixing different rings (even with same D)."""
        Q_default = QuadraticRing(-3)  # den=2
        Q_other = QuadraticRing(-3, 1)  # den=1

        a = QuadInt(Q_default, 2, 0)  # ok parity for den=2
        b = QuadInt(Q_other, 1, 0)

        with pytest.raises(TypeError):
            _ = a + b


sqrtNeg17 = QuadraticRing(-17)
sqrtNeg11 = QuadraticRing(-11)
sqrtNeg7 = QuadraticRing(-7)
sqrtNeg2 = QuadraticRing(-2)
sqrt1 = QuadraticRing(1)
sqrt2 = QuadraticRing(2)
sqrt5 = QuadraticRing(5)
sqrt31 = QuadraticRing(31)

ZI = QuadraticRing(-1)
ZE = QuadraticRing(-3)


class QuadIntTests:
    """Tests for __div__"""

    a_int, b_int, a_cint, b_cint = None, None, None, None

    def setup_method(self, _):
        """Setup some test data"""
        self.a_int = sqrt2(5, 2)
        self.b_int = sqrt2(3, -2)

        self.a_cint = complexint(5, 2)
        self.b_cint = complexint(3, -2)

    @staticmethod
    def assert_quad_equal(res: Union[tuple, QuadInt], res_int: QuadInt):
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
        a_int = sqrt5(7, 3)
        b_int = sqrt5(3, -5)

        mul_int = a_int * b_int

        res_int = mul_int / a_int
        self.assert_quad_equal((b_int.a, b_int.b), res_int)

        res_int = mul_int / b_int
        self.assert_quad_equal((a_int.a, a_int.b), res_int)

    def test_div_ring31(self):
        """Test QuadInt / QuadInt in QuadraticRing(31) (31 is not norm-Euclidean so this is not implemented)"""
        a_int = sqrt31(5, 2)
        b_int = sqrt31(3, -2)

        mul_int = a_int * b_int

        with pytest.raises(NotImplementedError):
            _ = mul_int / a_int

    def test_div_ring_neg7(self):
        """Test QuadInt / QuadInt in QuadraticRing(-7)"""
        a_int = sqrtNeg7(7, 3)
        b_int = sqrtNeg7(3, -5)

        mul_int = a_int * b_int

        res_int = mul_int / a_int
        self.assert_quad_equal((b_int.a, b_int.b), res_int)

        res_int = mul_int / b_int
        self.assert_quad_equal((a_int.a, a_int.b), res_int)

    def test_div_ring_neg17(self):
        """Test QuadInt / QuadInt in QuadraticRing(-17) (-17 is not norm-Euclidean so this is not implemented)"""
        a_int = sqrtNeg17(5, 2)
        b_int = sqrtNeg17(3, -2)

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
        m_other_ring = sqrt2(2, 1)

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


class TestUnits:
    """Tests for the units"""

    def test_units_sizes(self):
        """Verify Gaussian and Eisenstein rings have the correct unit lengths"""
        assert len(sqrt1(2, 0).units) == 4
        assert len(ZI(1, 0).units) == 4
        assert len(ZE(2, 0).units) == 6

        assert len(sqrt2(1, 0).units) == 2

    @pytest.mark.parametrize("x", [ZI(3, 2), ZE(5, 1), sqrt2(7, 3)], ids=id_generator)
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
        ids=id_generator,
    )
    def test_content_matches_reference_large_values(self, x: QuadInt):
        """Large-value regression: optimized content matches the old divisor-scan logic."""
        assert x.content() == brute_content(x)


class TestFactorDetail(QuadIntTests):
    """Tests for the factor_detail method"""

    @pytest.mark.parametrize(
        "x",
        [
            sqrtNeg2(1, 0),
            sqrtNeg2(-1, 0),
            sqrtNeg2(0, 1),
            sqrtNeg2(5, 2),
            sqrtNeg2(4, 53),
            sqrtNeg2(6, 0),
            sqrtNeg7(2, 0),
            sqrtNeg7(-2, 0),
            sqrtNeg7(0, 2),
            sqrtNeg7(10, 2),
            sqrtNeg7(4, 56),
            sqrtNeg7(6, 0),
            sqrtNeg11(2, 0),
            sqrtNeg11(-2, 0),
            sqrtNeg11(0, 2),
            sqrtNeg11(10, 2),
            sqrtNeg11(4, 56),
            sqrtNeg11(6, 0),
        ],
        ids=id_generator,
    )
    def test_examples(self, x: QuadInt):
        """Validate some given examples"""
        f = x.factor_detail()
        self.assert_factoring(x, f)

    @pytest.mark.parametrize(
        "x",
        [
            sqrtNeg2(0, 0),
            sqrtNeg7(0, 0),
            sqrtNeg11(0, 0),
        ],
        ids=id_generator,
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
            sqrtNeg2(17 * 31, 0),
        ],
        ids=id_generator,
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
            (sqrtNeg2(1, 1), 3),
            (sqrtNeg7(-1, 1), 2),
            (sqrtNeg11(3, 1), 5),
        ],
        ids=id_generator,
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
    @pytest.mark.parametrize("test_klass", [sqrtNeg2, sqrtNeg7, sqrtNeg11])
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
        x = sqrt2(5, 2)
        with pytest.raises(NotImplementedError):
            _ = x.factor_detail()

    def test_notimplemented_for_non_norm_euclid_ring(self):
        """Factorization only reliably make sense for norm-Euclidean rings."""
        # D=-17 is proven to be not norm-Euclidean
        x = sqrtNeg17(5, 2)
        with pytest.raises(NotImplementedError):
            _ = x.factor_detail()

    @pytest.mark.parametrize("test_klass", [sqrtNeg2, sqrtNeg7, sqrtNeg11])
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
            sqrtNeg2(4, 53),
            sqrtNeg2(6, 0),
            sqrtNeg2(17 * 31, 0),
            sqrtNeg2(5, 2),
            sqrtNeg7(4, 56),
            sqrtNeg7(6, 0),
            sqrtNeg7(17, 7),
            sqrtNeg11(4, 56),
            sqrtNeg11(6, 0),
            sqrtNeg11(17, 7),
        ],
        ids=id_generator,
    )
    def test_examples(self, x: QuadInt):
        """Validate some given examples"""
        factors = x.factor()
        assert isinstance(factors, dict)
        self.assert_factoring(x, factors)
