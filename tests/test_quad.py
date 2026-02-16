"""
Tests for QuadraticRing singleton caching / identity semantics.

These tests are specifically meant to catch regressions that might only show up
after mypyc compilation: repeated construction must return the same object, and
ring identity checks must remain valid.
"""

import os

from pathlib import Path
from typing import Union

import pytest

import quadint

from quadint import QuadInt
from quadint.quad import QuadraticRing

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
        q_default = QuadraticRing(-3)      # default den=2
        q_other = QuadraticRing(-3, 1)     # non-maximal order (or at least non-default)
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
        q = QuadraticRing(1)          # default den=2
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
        Q_default = QuadraticRing(-3)      # den=2
        Q_other = QuadraticRing(-3, 1)     # den=1

        a = QuadInt(Q_default, 2, 0)       # ok parity for den=2
        b = QuadInt(Q_other, 1, 0)

        with pytest.raises(TypeError):
            _ = a + b


sqrtNeg17 = QuadraticRing(-17)
sqrtNeg7 = QuadraticRing(-7)
sqrt2 = QuadraticRing(2)
sqrt5 = QuadraticRing(5)
sqrt31 = QuadraticRing(31)


class TestDiv:
    """Tests for __div__"""
    a, b, a_int, b_int = None, None, None, None

    def setup_method(self, _):
        """Setup some test data"""
        self.a_int = sqrt2(5, 2)
        self.b_int = sqrt2(3, -2)

    @staticmethod
    def assert_quad_equal(res: Union[tuple, QuadInt], res_int: QuadInt):
        """Validate the QuadInt is equal to the validation object, and that it is still backed by integers"""
        assert res[0] == res_int.a
        assert res[1] == res_int.b

        assert isinstance(res_int.a, int)
        assert isinstance(res_int.b, int)

        assert isinstance(res_int, QuadInt)

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
        ZI = QuadraticRing(-1)
        x = ZI(5, 2)
        m = ZI(2, 1)

        got = pow(x, 13, m)
        expected = (x ** 13) % m

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
        ZI = QuadraticRing(-1)
        x = ZI(5, 2)

        got = pow(x, 20, 7)
        expected = (x ** 20) % 7

        self.assert_quad_equal((expected.a, expected.b), got)

    def test_pow_mod_requires_same_ring(self):
        """Modulus must be in the same QuadraticRing (identity check)."""
        ZI = QuadraticRing(-1)
        ZS2 = QuadraticRing(-2)

        x = ZI(5, 2)
        m_other_ring = ZS2(2, 1)

        with pytest.raises(TypeError):
            pow(x, 5, m_other_ring)

    def test_pow_mod_zero_modulus_raises(self):
        """pow(x, e, 0) should raise like Python ints."""
        ZI = QuadraticRing(-1)
        x = ZI(5, 2)

        with pytest.raises(ZeroDivisionError):
            pow(x, 5, 0)

    def test_pow_mod_negative_exponent_raises(self):
        """pow(x, -e, m) should raise."""
        ZI = QuadraticRing(-1)
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
