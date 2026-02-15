"""These are simple tests to verify splitint acts like split-complex numbers, but with int output"""

import os

from pathlib import Path
from typing import Union

import pytest

import quadint.split

from quadint import QuadInt
from quadint.quad import QuadraticRing
from quadint.split import splitint

@pytest.mark.skipif(os.getenv("CI", "").lower() not in {"1", "true", "yes"},
                    reason="Compiled-only test")
def test_compiled_tests():
    """Verify that we are running these tests with a compiled version of splitint"""
    path = Path(quadint.split.__file__)
    assert path.suffix.lower() != '.py'


def test_is_instance():
    """Verify that basic isinstance checks work"""
    assert isinstance(splitint(1, 2), splitint)
    assert not isinstance(complex(1, 2), splitint)


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
    def assert_split_equal(res: Union[tuple, splitint], res_int: Union[splitint, QuadInt]):
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
        self.assert_split_equal((self.a_int.real + self.b_int.real, self.a_int.hyper + self.b_int.hyper),
                                res_int)

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
        self.assert_split_equal((self.a_int.real - self.b_int.real, self.a_int.hyper - self.b_int.hyper),
                                res_int)

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
        z = splitint(1, 1)   # norm 0
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
