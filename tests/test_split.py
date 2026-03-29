"""These are simple tests to verify splitint acts like split-complex numbers, but with int output"""

from __future__ import annotations

import os
import random

from math import gcd as igcd
from pathlib import Path

import pytest

import quadint.split

from quadint import QuadInt, QuadraticRing
from quadint.split import splitint


@pytest.mark.skipif(os.getenv("CI", "").lower() not in {"1", "true", "yes"}, reason="Compiled-only test")
def test_compiled_tests():
    """Verify that we are running these tests with a compiled version of splitint"""
    path = Path(quadint.split.__file__)
    assert path.suffix.lower() != ".py"


def test_is_instance():
    """Verify that basic isinstance checks work"""
    assert isinstance(splitint(1, 2), splitint)
    assert not isinstance(complex(1, 2), splitint)


def test_ring_is_singleton():
    """Splitint should use the cached D=1 ring and match QuadraticRing(1, den=1)."""
    w = splitint(1, 2)
    assert w.ring is QuadraticRing(1, den=1)


def test_alias():
    """Verify the hyperbolic/j aliases"""
    a = splitint(1, 2)
    assert a.hyper == a.b
    assert a.j == a.b


class SplitIntTests:
    """Support methods for testing splitint"""

    a, b, a_int, b_int = None, None, None, None

    def setup_method(self, _):
        """Setup some test data"""
        self.a_int = splitint(5, 2)
        self.b_int = splitint(3, -2)

    @staticmethod
    def assert_split_equal(res: tuple | splitint, res_int: splitint | QuadInt):
        """Validate the splitint is equal to the validation object, and that it is still backed by integers"""
        assert res[0] == res_int.real
        assert res[1] == res_int.hyper

        assert isinstance(res_int.real, int)
        assert isinstance(res_int.hyper, int)

        assert isinstance(res_int, splitint)


class TestEq(SplitIntTests):
    """Tests for __eq__"""

    def test_main(self):
        """Basic equals tests"""
        c = splitint(5, 2)
        assert self.a_int == c
        assert self.b_int != c

    def test_quadint(self):
        """Validate that a D=1 den=1 quadint equals a split int"""
        # In the strictest sense this is True: split-complex integers are D=1 quadratic integers with den=1.
        Z1 = QuadraticRing(1, den=1)
        c = Z1(self.a_int.a, self.a_int.b)
        assert self.a_int == c
        assert self.b_int != c


class TestAdd(SplitIntTests):
    """Tests for __add__"""

    def test_add(self):
        """Test splitint + splitint"""
        res_int = self.a_int + self.b_int
        self.assert_split_equal((self.a_int.real + self.b_int.real, self.a_int.hyper + self.b_int.hyper), res_int)

    def test_add_int(self):
        """Test splitint + int"""
        for i in range(100):
            res_int = self.a_int + i
            self.assert_split_equal((self.a_int.real + i, self.a_int.hyper), res_int)

    def test_add_int_reversed(self):
        """Test int + splitint"""
        for i in range(100):
            res_int = i + self.a_int
            self.assert_split_equal((self.a_int.real + i, self.a_int.hyper), res_int)

    def test_add_float(self):
        """Test splitint + float"""
        for i in range(100):
            res_int = self.a_int + float(i)
            self.assert_split_equal((self.a_int.real + i, self.a_int.hyper), res_int)

    def test_add_float_reversed(self):
        """Test float + splitint"""
        for i in range(100):
            res_int = float(i) + self.a_int
            self.assert_split_equal((self.a_int.real + i, self.a_int.hyper), res_int)


class TestSub(SplitIntTests):
    """Tests for __sub__"""

    def test_sub(self):
        """Test splitint - splitint"""
        res_int = self.a_int - self.b_int
        self.assert_split_equal((self.a_int.real - self.b_int.real, self.a_int.hyper - self.b_int.hyper), res_int)

    def test_sub_int(self):
        """Test splitint - int"""
        for i in range(100):
            res_int = self.a_int - i
            self.assert_split_equal((self.a_int.real - i, self.a_int.hyper), res_int)

    def test_sub_int_reversed(self):
        """Test int - splitint"""
        for i in range(100):
            res_int = i - self.a_int
            self.assert_split_equal((i - self.a_int.real, -self.a_int.hyper), res_int)

    def test_sub_float(self):
        """Test splitint - float"""
        for i in range(100):
            res_int = self.a_int - float(i)
            self.assert_split_equal((self.a_int.real - i, self.a_int.hyper), res_int)

    def test_sub_float_reversed(self):
        """Test float - splitint"""
        for i in range(100):
            res_int = float(i) - self.a_int
            self.assert_split_equal((i - self.a_int.real, -self.a_int.hyper), res_int)


class TestNegPos(SplitIntTests):
    """Tests for __neg__ and __pos__"""

    def test_neg(self):
        """Test -splitint"""
        res_int = -self.a_int
        self.assert_split_equal((-self.a_int.real, -self.a_int.hyper), res_int)

    def test_pos(self):
        """Test +splitint"""
        res_int = +self.a_int
        self.assert_split_equal((self.a_int.real, self.a_int.hyper), res_int)


class TestMul(SplitIntTests):
    """Tests for __mul__"""

    def test_mul(self):
        """Test splitint * splitint"""
        res_int = self.a_int * self.b_int

        self.assert_split_equal((11, -4), res_int)

    def test_zero_mul(self):
        """Split integers have the interesting property where odd numbers have norm 0, and multiply to 0"""
        self.assert_split_equal((0, 0), splitint(1, 1) * splitint(1, -1))

    def test_mul_int(self):
        """Test splitint * int"""
        for i in range(100):
            res_int = self.a_int * i
            self.assert_split_equal((self.a_int.real * i, self.a_int.hyper * i), res_int)

    def test_mul_int_reversed(self):
        """Test int * splitint"""
        for i in range(100):
            res_int = i * self.a_int
            self.assert_split_equal((self.a_int.real * i, self.a_int.hyper * i), res_int)

    def test_mul_float(self):
        """Test splitint * float"""
        for i in range(100):
            res_int = self.a_int * float(i)
            self.assert_split_equal((self.a_int.real * i, self.a_int.hyper * i), res_int)

    def test_mul_float_reversed(self):
        """Test float * splitint"""
        for i in range(100):
            res_int = float(i) * self.a_int
            self.assert_split_equal((self.a_int.real * i, self.a_int.hyper * i), res_int)


class TestAbs(SplitIntTests):
    """Tests for __abs__ (norm)"""

    def test_norm_sign(self):
        """Split norm is indefinite: a^2 - b^2 can be negative"""
        assert abs(splitint(5, 2)) == 21
        assert abs(splitint(2, 5)) == -21
        assert isinstance(abs(splitint(2, 5)), int)


class TestDiv(SplitIntTests):
    """Tests for __div__"""

    def test_div(self):
        """Test splitint / splitint (exact cancellation cases)"""
        mul_int = self.a_int * self.b_int

        res_int = mul_int / self.a_int
        self.assert_split_equal((self.b_int.real, self.b_int.hyper), res_int)

        res_int = mul_int / self.b_int
        self.assert_split_equal((self.a_int.real, self.a_int.hyper), res_int)

    def test_div_int(self):
        """Test splitint / int"""
        mul_int = self.a_int * 3
        res_int = mul_int / 3
        self.assert_split_equal((self.a_int.real, self.a_int.hyper), res_int)

    def test_div_float(self):
        """Test splitint / float"""
        mul_int = self.a_int * 3
        res_int = mul_int / float(3)
        self.assert_split_equal((self.a_int.real, self.a_int.hyper), res_int)

    def test_div_by_zero_divisor_raises(self):
        """Division by a zero divisor should raise"""
        z = splitint(1, 1)  # norm 0
        with pytest.raises(ZeroDivisionError):
            _ = self.a_int / z


class TestConjugate(SplitIntTests):
    """Tests for conjugate"""

    def test_examples(self):
        """Test known examples that conceptually make sense to me"""
        a = splitint(1, 1)
        a_conj = a.conjugate()

        assert a_conj.real == 1
        assert a_conj.hyper == -1

        a = splitint(0, 1)
        a_conj = a.conjugate()

        assert a_conj.real == 0
        assert a_conj.hyper == -1


class TestRepr(SplitIntTests):
    """Validate the repr matches existing solutions"""

    @pytest.mark.parametrize(
        ("x", "expected_repr"),
        [
            (splitint(-9, 12), "(-9+12j)"),
            (splitint(0, -10), "-10j"),
            (splitint(0, 10), "10j"),
            (splitint(10, 0), "(10+0j)"),
            (splitint(5, -5), "(5-5j)"),
        ],
        ids=str,
    )
    def test_examples(self, x: QuadInt, expected_repr: str):
        """Verify some given examples"""
        assert repr(x) == expected_repr


class TestExactDiv(SplitIntTests):
    """Tests for exact_div in split integers."""

    def test_exact_div(self):
        """Test splitint.exact_div on exact cancellation cases."""
        mul_int = self.a_int * self.b_int

        res_int = mul_int.exact_div(self.a_int)
        assert res_int is not None
        self.assert_split_equal((self.b_int.real, self.b_int.hyper), res_int)

        res_int = mul_int.exact_div(self.b_int)
        assert res_int is not None
        self.assert_split_equal((self.a_int.real, self.a_int.hyper), res_int)

    def test_exact_div_returns_none_when_not_divisible(self):
        """Non-divisible split integers should return None."""
        assert self.a_int.exact_div(self.b_int) is None
        assert self.b_int.exact_div(self.a_int) is None

    def test_exact_div_int(self):
        """Test splitint.exact_div(int)."""
        mul_int = self.a_int * 3
        res_int = mul_int.exact_div(3)

        assert res_int is not None
        self.assert_split_equal((self.a_int.real, self.a_int.hyper), res_int)

    def test_exact_div_float(self):
        """Test splitint.exact_div(float)."""
        mul_int = self.a_int * 3
        res_int = mul_int.exact_div(float(3))

        assert res_int is not None
        self.assert_split_equal((self.a_int.real, self.a_int.hyper), res_int)

    def test_exact_div_zero_divisor_returns_none(self):
        """Division by a nonzero zero divisor should return None (not a unique quotient)."""
        z = splitint(1, 1)  # split coords (2, 0), norm 0
        x = splitint(3, 3)  # split coords (6, 0), so z divides x non-uniquely

        assert abs(z) == 0
        assert x.exact_div(z) is None
        assert z.divides(x) is False

    def test_exact_div_zero_raises_not_implemented(self):
        """Division by literal zero is currently unsupported by exact_div."""
        with pytest.raises(NotImplementedError):
            _ = self.a_int.exact_div(splitint(0, 0))

    def test_exact_div_rejects_parity_mismatch_quotient(self):
        """Exact split-coordinate quotients must still lie in the den=1 sublattice."""
        # In split coords:
        #   y = splitint(2, 0)  -> (u, v) = (2, 2)
        #   x = splitint(3, -1) -> (u, v) = (2, 4)
        #
        # Componentwise quotient would be (1, 2), which has opposite parity,
        # so it is not a valid splitint quotient.
        x = splitint(3, -1)
        y = splitint(2, 0)

        assert x.exact_div(y) is None
        assert y.divides(x) is False

    def test_exact_div_matches_constructed_products(self):
        """If x = q*y with y a non-zero-divisor, exact_div should recover q."""
        rng = random.Random(123)

        for _ in range(50):
            q = splitint(rng.randint(-20, 20), rng.randint(-20, 20))
            y = splitint(rng.randint(-20, 20), rng.randint(-20, 20))

            if not y or abs(y) == 0:
                continue

            x = q * y
            out = x.exact_div(y)

            assert out is not None, f"expected exact quotient for x={x}, y={y}"
            assert out == q


class TestGcd(SplitIntTests):
    """Tests for gcd in split integers."""

    def test_gcd_coprime(self):
        """GCD of coprime elements should be a unit."""
        a = splitint(3, 2)  # split coords: (5, 1)
        b = splitint(2, 1)  # split coords: (3, 1)
        g = a.gcd(b)
        assert g.is_unit()

    def test_gcd_common_factor(self):
        """GCD should find common factors."""
        # a = 3+2j -> (5, 1), b = 3-2j -> (1, 5)
        # a*b = (5, 5) = 5*(1,1) = splitint(5, 0)
        # gcd(a*2, b*2) should give 2 (up to unit)
        c = splitint(5, 0)  # the integer 5, split coords (5, 5)
        d = splitint(10, 0)  # the integer 10, split coords (10, 10)
        g = c.gcd(d)
        assert g.divides(c)
        assert g.divides(d)
        # gcd(5, 10) = 5 in both components
        assert abs(abs(g)) == abs(abs(c))  # g is associate of 5

    def test_gcd_with_zero(self):
        """GCD with zero should give the other element (up to associate)."""
        a = splitint(3, 2)
        z = splitint(0, 0)
        g = a.gcd(z)
        assert g.divides(a)

    def test_gcd_both_zero(self):
        """GCD(0, 0) = 0."""
        z = splitint(0, 0)
        g = z.gcd(z)
        assert g == z

    def test_gcd_symmetry(self):
        """gcd(a, b) should be associate of gcd(b, a)."""
        a = splitint(6, 2)
        b = splitint(4, 2)
        g1 = a.gcd(b)
        g2 = b.gcd(a)
        assert g1.divides(g2)
        assert g2.divides(g1)

    def test_gcd_rational_integers(self):
        """GCD of rational integers should match integer GCD."""
        a = splitint(12, 0)
        b = splitint(8, 0)
        g = a.gcd(b)
        # Should be associate of 4
        assert abs(abs(g)) == igcd(12, 8) ** 2  # norm of gcd(12,8)=4 is 16

    def test_gcd_random(self):
        """Property test: g divides both a and b."""
        rng = random.Random(42)
        for _ in range(50):
            a = splitint(rng.randint(-50, 50), rng.randint(-50, 50))
            b = splitint(rng.randint(-50, 50), rng.randint(-50, 50))

            # Skip zero divisors
            if not a or not b:
                continue
            if abs(a) == 0 or abs(b) == 0:
                continue

            g = a.gcd(b)
            assert g.divides(a), f"gcd({a}, {b}) = {g} does not divide {a}"
            assert g.divides(b), f"gcd({a}, {b}) = {g} does not divide {b}"


class TestXgcd(SplitIntTests):
    """Tests for extended GCD in split integers."""

    def test_xgcd_not_supported_den1(self):
        """Xgcd is not supported for den=1 splitint (not a PID)."""
        a = splitint(6, 2)
        b = splitint(4, 2)
        with pytest.raises(NotImplementedError):
            _ = a.xgcd(b)

    def test_xgcd_works_den2(self):
        """Xgcd works for den=2 maximal order (Z*Z)."""
        Z1_max = QuadraticRing(1)  # den=2 by default
        a = Z1_max(5, 3)  # den=2: both odd, same parity
        b = Z1_max(3, 1)
        g, s, t = a.xgcd(b)
        assert s * a + t * b == g
