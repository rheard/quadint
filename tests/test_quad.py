import pytest

from quadint import make_quadint

Q2 = make_quadint(-2)


def test_aliases():
    """Validate that the complexint aliases do not work when D != -1"""
    a = Q2(1, 2)

    with pytest.raises(AttributeError):
        assert a.real == 1

    with pytest.raises(AttributeError):
        assert a.imag == 2
