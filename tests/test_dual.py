"""These are simple tests to verify complexint acts very similar to complex, but just with int output"""

from __future__ import annotations

import os

from pathlib import Path

import pytest

import quadint.dual

from quadint import QuadInt, QuadraticRing
from quadint.dual import dualint


@pytest.mark.skipif(os.getenv("CI", "").lower() not in {"1", "true", "yes"}, reason="Compiled-only test")
def test_compiled_tests():
    """Verify that we are running these tests with a compiled version of dualint"""
    path = Path(quadint.dual.__file__)
    assert path.suffix.lower() != ".py"


def test_is_instance():
    """Verify that basic isinstance checks work"""
    assert isinstance(dualint(1, 2), dualint)
    assert not isinstance(complex(1, 2), dualint)


def test_ring_is_singleton():
    """Dualint should use the cached D=0 ring and match QuadraticRing(0)."""
    w = dualint(1, 2)
    assert w.ring is QuadraticRing(0)


def test_alias():
    """Verify the epsilon alias"""
    a = dualint(1, 2)
    assert a.dual == a.b
    assert a.epsilon == a.b


class DualIntTests:
    """Support methods for testing dualint"""

    a, b, a_int, b_int = None, None, None, None

    def setup_method(self, _):
        """Setup some test data"""
        self.a_int = dualint(5, 2)
        self.b_int = dualint(3, -2)

    @staticmethod
    def assert_dual_equal(res: tuple | dualint, res_int: dualint | QuadInt):
        """Validate the complexint is equal to the validation object, and that it is still backed by integers"""
        assert res[0] == res_int.real
        assert res[1] == res_int.dual

        assert isinstance(res_int.real, int)
        assert isinstance(res_int.dual, int)

        assert isinstance(res_int, dualint)


class TestEq(DualIntTests):
    """Tests for __eq__"""

    def test_main(self):
        """Basic equals tests"""
        c = dualint(5, 2)
        assert self.a_int == c
        assert self.b_int != c

    def test_quadint(self):
        """Validate that a D=0 quadint equals a dual int"""
        # TODO: In the strictest sense this is True, as dual integers are D=0 quadratic integers
        #  However I'm uncomfortable with this.
        #   I've gone back and forth on if this should work or not.
        #   For now I'm leaving it... For it to not work will require a new __eq__
        Z0 = QuadraticRing(0)
        c = Z0(self.a_int.a, self.a_int.b)
        assert self.a_int == c
        assert self.b_int != c


class TestAdd(DualIntTests):
    """Tests for __add__"""

    def test_add(self):
        """Test dualint + dualint"""
        res_int = self.a_int + self.b_int

        self.assert_dual_equal((self.a_int.real + self.b_int.real, self.a_int.dual + self.b_int.dual), res_int)

    def test_add_int(self):
        """Test dualint + int"""
        for i in range(100):
            res_int = self.a_int + i

            self.assert_dual_equal((self.a_int.real + i, self.a_int.dual), res_int)

    def test_add_int_reversed(self):
        """Test int + dualint"""
        for i in range(100):
            res_int = i + self.a_int

            self.assert_dual_equal((self.a_int.real + i, self.a_int.dual), res_int)

    def test_add_float(self):
        """Test dualint + float"""
        for i in range(100):
            res_int = self.a_int + float(i)

            self.assert_dual_equal((self.a_int.real + i, self.a_int.dual), res_int)

    def test_add_float_reversed(self):
        """Test float + dualint"""
        for i in range(100):
            res_int = float(i) + self.a_int

            self.assert_dual_equal((self.a_int.real + i, self.a_int.dual), res_int)


class TestSub(DualIntTests):
    """Tests for __sub__"""

    def test_sub(self):
        """Test dualint - dualint"""
        res_int = self.a_int - self.b_int

        self.assert_dual_equal((self.a_int.real - self.b_int.real, self.a_int.dual - self.b_int.dual), res_int)

    def test_sub_int(self):
        """Test dualint - int"""
        for i in range(100):
            res_int = self.a_int - i

            self.assert_dual_equal((self.a_int.real - i, self.a_int.dual), res_int)

    def test_sub_int_reversed(self):
        """Test int - dualint"""
        for i in range(100):
            res_int = i - self.a_int

            self.assert_dual_equal((i - self.a_int.real, -self.a_int.dual), res_int)

    def test_sub_float(self):
        """Test dualint - float"""
        for i in range(100):
            res_int = self.a_int - float(i)

            self.assert_dual_equal((self.a_int.real - i, self.a_int.dual), res_int)

    def test_sub_float_reversed(self):
        """Test float - dualint"""
        for i in range(100):
            res_int = float(i) - self.a_int

            self.assert_dual_equal((i - self.a_int.real, -self.a_int.dual), res_int)


class TestNegPos(DualIntTests):
    """Tests for __neg__ and __pos__"""

    def test_neg(self):
        """Test -dualint"""
        res_int = -self.a_int

        self.assert_dual_equal((-self.a_int.real, -self.a_int.dual), res_int)

    def test_pos(self):
        """Test +dualint"""
        res_int = +self.a_int

        self.assert_dual_equal((self.a_int.real, self.a_int.dual), res_int)


class TestMul(DualIntTests):
    """Tests for __mul__"""

    def test_mul(self):
        """Test dualint * dualint"""
        res_int = self.a_int * self.b_int

        self.assert_dual_equal((15, -4), res_int)

    def test_mul_int(self):
        """Test dualint * int"""
        for i in range(100):
            res_int = self.a_int * i

            self.assert_dual_equal((self.a_int.real * i, self.a_int.dual * i), res_int)

    def test_mul_int_reversed(self):
        """Test int * dualint"""
        for i in range(100):
            res_int = i * self.a_int

            self.assert_dual_equal((self.a_int.real * i, self.a_int.dual * i), res_int)

    def test_mul_float(self):
        """Test dualint * float"""
        for i in range(100):
            res_int = self.a_int * float(i)

            self.assert_dual_equal((self.a_int.real * i, self.a_int.dual * i), res_int)

    def test_mul_float_reversed(self):
        """Test float * dualint"""
        for i in range(100):
            res_int = float(i) * self.a_int

            self.assert_dual_equal((self.a_int.real * i, self.a_int.dual * i), res_int)


class TestDiv(DualIntTests):
    """Tests for __div__"""

    def test_div(self):
        """Test dualint / dualint"""
        mul_int = self.a_int * self.b_int
        res_int = mul_int / self.a_int

        self.assert_dual_equal((self.b_int.real, self.b_int.dual), res_int)

        res_int = mul_int / self.b_int

        self.assert_dual_equal((self.a_int.real, self.a_int.dual), res_int)

    def test_div_int(self):
        """Test dualint / int"""
        mul_int = self.a_int * 3
        res_int = mul_int / 3

        self.assert_dual_equal((self.a_int.real, self.a_int.dual), res_int)

    def test_div_float(self):
        """Test dualint / float"""
        mul_int = self.a_int * 3
        res_int = mul_int / float(3)

        self.assert_dual_equal((self.a_int.real, self.a_int.dual), res_int)

    def test_divmod_by_epsilon_raises(self):
        """A multiple of ε only has multiples with no real part, so there is no remainder to shrink."""
        with pytest.raises(ZeroDivisionError, match="zero divisor"):
            _ = divmod(dualint(3, 2), dualint(0, 1))


class TestExactDiv(DualIntTests):
    """Tests for exact_div and divides"""

    def test_exact_div(self):
        """Exact quotients should come back, and inexact ones should give None."""
        res_int = (self.a_int * self.b_int).exact_div(self.a_int)

        assert res_int is not None
        self.assert_dual_equal((self.b_int.real, self.b_int.dual), res_int)
        assert self.a_int.exact_div(self.b_int) is None

    def test_exact_div_by_epsilon(self):
        """
        A multiple of ε divides the multiples of ε whose ε part it divides, even though ε is a zero divisor.

        The ε part of the quotient could be anything then (6ε == (3 + kε) * 2ε for every k), so it is 0.
        """
        eps = dualint(0, 1)
        assert abs(eps) == 0

        res_int = dualint(0, 6).exact_div(dualint(0, 2))
        assert res_int is not None
        self.assert_dual_equal((3, 0), res_int)
        assert eps.divides(dualint(0, 7)) is True

        assert dualint(5, 7).exact_div(eps) is None  # every multiple of ε has no real part
        assert dualint(0, 7).exact_div(dualint(0, 2)) is None
        assert dualint(0, 2).divides(dualint(0, 7)) is False

    def test_divides_matches_brute_force(self):
        """y.divides(x) should hold exactly when some q has q*y == x, for zero divisors like ε too."""
        small = [dualint(a, b) for a in range(-4, 5) for b in range(-4, 5)]
        # Every quotient between two small elements, when there is one, has a representative in this range
        wide = range(-20, 21)
        quotients = [dualint(a, b) for a in wide for b in wide]

        for y in small:
            if not y:
                continue

            multiples = {q * y for q in quotients}
            for x in small:
                q = x.exact_div(y)
                assert y.divides(x) is (x in multiples)
                assert (q is not None) is (x in multiples)
                assert q is None or q * y == x


class TestConjugate(DualIntTests):
    """Tests for conjugate"""

    def test_examples(self):
        """Test known examples that conceptually make sense to me"""
        a = dualint(1, 1)
        a_conj = a.conjugate()

        assert a_conj.real == 1
        assert a_conj.dual == -1

        a = dualint(0, 1)
        a_conj = a.conjugate()

        assert a_conj.real == 0
        assert a_conj.dual == -1


class TestRepr(DualIntTests):
    """Validate the repr matches existing solutions"""

    @pytest.mark.parametrize(
        ("x", "expected_repr"),
        [
            (dualint(-9, 12), "(-9+12ε)"),
            (dualint(0, -10), "-10ε"),
            (dualint(0, 10), "10ε"),
            (dualint(10, 0), "(10+0ε)"),
            (dualint(5, -5), "(5-5ε)"),
        ],
        ids=str,
    )
    def test_examples(self, x: QuadInt, expected_repr: str):
        """Verify some given examples"""
        assert repr(x) == expected_repr
