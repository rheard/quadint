"""These are simple tests to verify complexint acts very similar to complex, but just with int output"""

import os

from pathlib import Path
from typing import Union

import pytest

import quadint.eisenstein

from quadint import QuadInt, QuadraticRing
from quadint.eisenstein import eisensteinint as eisenstein

@pytest.mark.skipif(os.getenv("CI", "").lower() not in {"1", "true", "yes"},
                    reason="Compiled-only test")
def test_compiled_tests():
    """Verify that we are running these tests with a compiled version of eisensteinint"""
    path = Path(quadint.eisenstein.__file__)
    assert path.suffix.lower() != '.py'


def test_is_instance():
    """Verify that basic isinstance checks work"""
    assert isinstance(eisenstein(1, 2), eisenstein)
    assert not isinstance(complex(1, 2), eisenstein)


def test_ring_is_singleton():
    """Eisensteinint should use the cached D=-3 ring and match QuadraticRing(-3)."""
    w = eisenstein(1, 2)
    assert w.ring is QuadraticRing(-3)


class EisensteinIntTests:
    """Support methods for testing eisensteinint"""
    a, b, a_int, b_int = None, None, None, None

    def setup_method(self, _):
        """Setup some test data"""
        self.a_int = eisenstein(5, 2)
        self.b_int = eisenstein(3, -2)

    @staticmethod
    def assert_eisenstein_equal(res: Union[tuple, eisenstein], res_int: Union[eisenstein, QuadInt]):
        """Validate the complexint is equal to the validation object, and that it is still backed by integers"""
        assert res[0] == res_int.real
        assert res[1] == res_int.omega

        assert isinstance(res_int.real, int)
        assert isinstance(res_int.omega, int)

        assert isinstance(res_int, eisenstein)


class TestEq(EisensteinIntTests):
    """Tests for __eq__"""

    def test_main(self):
        """Basic equals tests"""
        c = eisenstein(5, 2)
        assert self.a_int == c
        assert self.b_int != c

    def test_quadint(self):
        """Validate that a D=-3 quadint equals an eisenstein int"""
        # TODO: In the strictest sense this is True, as Eisenstein integers are D=-3 quadratic integers
        #   with a different basis vector.... However I'm uncomfortable with this.
        #   I've gone back and forth on if this should work or not.
        #   For now I'm leaving it... For it to not work will require a new __eq__
        Z3 = QuadraticRing(-3)
        c = Z3(self.a_int.a, self.a_int.b)
        assert self.a_int == c
        assert self.b_int != c


class TestAdd(EisensteinIntTests):
    """Tests for __add__"""

    def test_add(self):
        """Test eisensteinint + eisensteinint"""
        res_int = self.a_int + self.b_int

        self.assert_eisenstein_equal((self.a_int.real + self.b_int.real, self.a_int.omega + self.b_int.omega),
                                     res_int)

    def test_add_int(self):
        """Test eisensteinint + int"""
        for i in range(100):
            res_int = self.a_int + i

            self.assert_eisenstein_equal((self.a_int.real + i, self.a_int.omega), res_int)

    def test_add_int_reversed(self):
        """Test int + eisensteinint"""
        for i in range(100):
            res_int = i + self.a_int

            self.assert_eisenstein_equal((self.a_int.real + i, self.a_int.omega), res_int)

    def test_add_float(self):
        """Test eisensteinint + float"""
        for i in range(100):
            res_int = self.a_int + float(i)

            self.assert_eisenstein_equal((self.a_int.real + i, self.a_int.omega), res_int)

    def test_add_float_reversed(self):
        """Test float + eisensteinint"""
        for i in range(100):
            res_int = float(i) + self.a_int

            self.assert_eisenstein_equal((self.a_int.real + i, self.a_int.omega), res_int)


class TestSub(EisensteinIntTests):
    """Tests for __sub__"""

    def test_sub(self):
        """Test eisensteinint - eisensteinint"""
        res_int = self.a_int - self.b_int

        self.assert_eisenstein_equal((self.a_int.real - self.b_int.real, self.a_int.omega - self.b_int.omega),
                                     res_int)

    def test_sub_int(self):
        """Test eisensteinint - int"""
        for i in range(100):
            res_int = self.a_int - i

            self.assert_eisenstein_equal((self.a_int.real - i, self.a_int.omega), res_int)

    def test_sub_int_reversed(self):
        """Test int - eisensteinint"""
        for i in range(100):
            res_int = i - self.a_int

            self.assert_eisenstein_equal((i - self.a_int.real, -self.a_int.omega), res_int)

    def test_sub_float(self):
        """Test eisensteinint - float"""
        for i in range(100):
            res_int = self.a_int - float(i)

            self.assert_eisenstein_equal((self.a_int.real - i, self.a_int.omega), res_int)

    def test_sub_float_reversed(self):
        """Test float - eisensteinint"""
        for i in range(100):
            res_int = float(i) - self.a_int

            self.assert_eisenstein_equal((i - self.a_int.real, -self.a_int.omega), res_int)


class TestNegPos(EisensteinIntTests):
    """Tests for __neg__ and __pos__"""

    def test_neg(self):
        """Test -eisensteinint"""
        res_int = -self.a_int

        self.assert_eisenstein_equal((-self.a_int.real, -self.a_int.omega), res_int)

    def test_pos(self):
        """Test +eisensteinint"""
        res_int = +self.a_int

        self.assert_eisenstein_equal((self.a_int.real, self.a_int.omega), res_int)


class TestMul(EisensteinIntTests):
    """Tests for __mul__"""

    def test_mul(self):
        """Test eisensteinint * eisensteinint"""
        res_int = self.a_int * self.b_int

        self.assert_eisenstein_equal((19, 0), res_int)

    def test_mul_int(self):
        """Test eisensteinint * int"""
        for i in range(100):
            res_int = self.a_int * i

            self.assert_eisenstein_equal((self.a_int.real * i, self.a_int.omega * i), res_int)

    def test_mul_int_reversed(self):
        """Test int * eisensteinint"""
        for i in range(100):
            res_int = i * self.a_int

            self.assert_eisenstein_equal((self.a_int.real * i, self.a_int.omega * i), res_int)

    def test_mul_float(self):
        """Test eisensteinint * float"""
        for i in range(100):
            res_int = self.a_int * float(i)

            self.assert_eisenstein_equal((self.a_int.real * i, self.a_int.omega * i), res_int)

    def test_mul_float_reversed(self):
        """Test float * eisensteinint"""
        for i in range(100):
            res_int = float(i) * self.a_int

            self.assert_eisenstein_equal((self.a_int.real * i, self.a_int.omega * i), res_int)

    def test_rotations(self):
        """Validate multiplying by omega to rotate around the origin"""
        # This is just a logical test that is satisfying to me geometrically
        n = eisenstein(1, 0)
        omega = eisenstein(0, 1)

        n *= omega
        self.assert_eisenstein_equal((0, 1), n)

        n *= omega
        self.assert_eisenstein_equal((-1, -1), n)

        n *= omega
        self.assert_eisenstein_equal((1, 0), n)

        n = eisenstein(1, 1)
        n *= omega
        self.assert_eisenstein_equal((-1, 0), n)

        n *= omega
        self.assert_eisenstein_equal((0, -1), n)

        n *= omega
        self.assert_eisenstein_equal((1, 1), n)


class TestDiv(EisensteinIntTests):
    """Tests for __div__"""

    def test_div(self):
        """Test eisensteinint / eisensteinint"""
        mul_int = self.a_int * self.b_int
        res_int = mul_int / self.a_int

        self.assert_eisenstein_equal((self.b_int.real, self.b_int.omega), res_int)

        res_int = mul_int / self.b_int

        self.assert_eisenstein_equal((self.a_int.real, self.a_int.omega), res_int)

    def test_div_int(self):
        """Test eisensteinint / int"""
        mul_int = self.a_int * 3
        res_int = mul_int / 3

        self.assert_eisenstein_equal((self.a_int.real, self.a_int.omega), res_int)

    def test_div_float(self):
        """Test eisensteinint / float"""
        mul_int = self.a_int * 3
        res_int = mul_int / float(3)

        self.assert_eisenstein_equal((self.a_int.real, self.a_int.omega), res_int)


class TestConjugate(EisensteinIntTests):
    """Tests for conjugate"""

    def test_examples(self):
        """Test known examples that conceptually make sense to me"""
        a = eisenstein(1, 1)
        a_conj = a.conjugate()

        assert a_conj.real == 0
        assert a_conj.omega == -1

        a = eisenstein(0, 1)
        a_conj = a.conjugate()

        assert a_conj.real == -1
        assert a_conj.omega == -1


class TestRepr(EisensteinIntTests):
    """Validate the repr matches existing solutions"""

    def test_examples(self):
        """Verify some given examples"""
        examples = [
            (eisenstein(-9, 12), "(-9+12ω)"),
            (eisenstein(0, -10), "-10ω"),
            (eisenstein(0, 10), "10ω"),
            (eisenstein(10, 0), "(10+0ω)"),
            (eisenstein(5, -5), "(5-5ω)"),
        ]

        for example, expected in examples:
            assert repr(example) == expected
