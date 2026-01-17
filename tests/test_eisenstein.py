"""These are simple tests to verify complexint acts very similar to complex, but just with int output"""

from pathlib import Path

import quadint.eisenstein

from quadint.eisenstein import eisensteinint as eisenstein

def test_compiled_tests():
    """Verify that we are running these tests with a compiled version of eisensteinint"""
    path = Path(quadint.eisenstein.__file__)
    assert path.suffix.lower() != '.py'


class EisensteinIntTests:
    """Support methods for testing eisensteinint"""
    a, b, a_int, b_int = None, None, None, None

    def setup_method(self, _):
        """Setup some test data"""
        self.a_int = eisenstein(5, 2)
        self.b_int = eisenstein(3, -2)


class TestAdd(EisensteinIntTests):
    """Tests for __add__"""

    def test_add(self):
        """Test eisensteinint + eisensteinint"""
        res_int = self.a_int + self.b_int

        assert res_int.real == self.a_int.real + self.b_int.real
        assert res_int.omega == self.a_int.omega + self.b_int.omega

    def test_add_int(self):
        """Test eisensteinint + int"""
        for i in range(100):
            res_int = self.a_int + i

            assert res_int.real == self.a_int.real + i
            assert res_int.omega == self.a_int.omega

    def test_add_int_reversed(self):
        """Test int + eisensteinint"""
        for i in range(100):
            res_int = i + self.a_int

            assert res_int.real == self.a_int.real + i
            assert res_int.omega == self.a_int.omega

    def test_add_float(self):
        """Test eisensteinint + float"""
        for i in range(100):
            res_int = self.a_int + float(i)

            assert res_int.real == self.a_int.real + i
            assert res_int.omega == self.a_int.omega

    def test_add_float_reversed(self):
        """Test float + eisensteinint"""
        for i in range(100):
            res_int = float(i) + self.a_int

            assert res_int.real == self.a_int.real + i
            assert res_int.omega == self.a_int.omega


class TestSub(EisensteinIntTests):
    """Tests for __sub__"""

    def test_sub(self):
        """Test eisensteinint - eisensteinint"""
        res_int = self.a_int - self.b_int

        assert res_int.real == self.a_int.real - self.b_int.real
        assert res_int.omega == self.a_int.omega - self.b_int.omega

    def test_sub_int(self):
        """Test eisensteinint - int"""
        for i in range(100):
            res_int = self.a_int - i

            assert res_int.real == self.a_int.real - i
            assert res_int.omega == self.a_int.omega

    def test_sub_int_reversed(self):
        """Test int - eisensteinint"""
        for i in range(100):
            res_int = i - self.a_int

            assert res_int.real == i - self.a_int.real
            assert res_int.omega == -self.a_int.omega

    def test_sub_float(self):
        """Test eisensteinint - float"""
        for i in range(100):
            res_int = self.a_int - float(i)

            assert res_int.real == self.a_int.real - i
            assert res_int.omega == self.a_int.omega

    def test_sub_float_reversed(self):
        """Test float - eisensteinint"""
        for i in range(100):
            res_int = float(i) - self.a_int

            assert res_int.real == i - self.a_int.real
            assert res_int.omega == -self.a_int.omega


class TestNegPos(EisensteinIntTests):
    """Tests for __neg__ and __pos__"""

    def test_neg(self):
        """Test -eisensteinint"""
        res_int = -self.a_int

        assert res_int.real == -self.a_int.real
        assert res_int.omega == -self.a_int.omega

    def test_pos(self):
        """Test +eisensteinint"""
        res_int = +self.a_int

        assert res_int.real == self.a_int.real
        assert res_int.omega == self.a_int.omega


class TestMul(EisensteinIntTests):
    """Tests for __mul__"""

    def test_mul(self):
        """Test eisensteinint * eisensteinint"""
        res_int = self.a_int * self.b_int

        assert res_int.real == 19
        assert res_int.omega == 0

    def test_mul_int(self):
        """Test eisensteinint * int"""
        for i in range(100):
            res_int = self.a_int * i

            assert res_int.real == self.a_int.real * i
            assert res_int.omega == self.a_int.omega * i

    def test_mul_int_reversed(self):
        """Test int * eisensteinint"""
        for i in range(100):
            res_int = i * self.a_int

            assert res_int.real == self.a_int.real * i
            assert res_int.omega == self.a_int.omega * i

    def test_mul_float(self):
        """Test eisensteinint * float"""
        for i in range(100):
            res_int = self.a_int * float(i)

            assert res_int.real == self.a_int.real * i
            assert res_int.omega == self.a_int.omega * i

    def test_mul_float_reversed(self):
        """Test float * eisensteinint"""
        for i in range(100):
            res_int = float(i) * self.a_int

            assert res_int.real == self.a_int.real * i
            assert res_int.omega == self.a_int.omega * i

    def test_rotations(self):
        """Validate multiplying by omega to rotate around the origin"""
        # This is just a logical test that is satisfying to me geometrically
        n = eisenstein(1, 0)
        omega = eisenstein(0, 1)

        n *= omega
        assert n.real == 0
        assert n.omega == 1

        n *= omega
        assert n.real == -1
        assert n.omega == -1

        n *= omega
        assert n.real == 1
        assert n.omega == 0

        n = eisenstein(1, 1)
        n *= omega
        assert n.real == -1
        assert n.omega == 0

        n *= omega
        assert n.real == 0
        assert n.omega == -1

        n *= omega
        assert n.real == 1
        assert n.omega == 1


class TestDiv(EisensteinIntTests):
    """Tests for __div__"""

    def test_div(self):
        """Test eisensteinint / eisensteinint"""
        mul_int = self.a_int * self.b_int
        res_int = mul_int / self.a_int

        assert res_int.real == self.b_int.real
        assert res_int.omega == self.b_int.omega

        res_int = mul_int / self.b_int

        assert res_int.real == self.a_int.real
        assert res_int.omega == self.a_int.omega

    def test_div_int(self):
        """Test eisensteinint / int"""
        mul_int = self.a_int * 3
        res_int = mul_int / 3

        assert res_int.real == self.a_int.real
        assert res_int.omega == self.a_int.omega

    def test_div_float(self):
        """Test eisensteinint / float"""
        mul_int = self.a_int * 3
        res_int = mul_int / float(3)

        assert res_int.real == self.a_int.real
        assert res_int.omega == self.a_int.omega
