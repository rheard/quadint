"""Pickling and copying, which is what multiprocessing (and copy.deepcopy) rely on"""

from __future__ import annotations

import copy
import pickle

import pytest

from quadint import Ideal, QuadInt, QuadraticRing, complexint, dualint, eisensteinint, splitint

ZI = QuadraticRing(-1)
ZE = QuadraticRing(-3)
ZN5 = QuadraticRing(-5)
ZN19 = QuadraticRing(-19)
ZN23 = QuadraticRing(-23)
Z10 = QuadraticRing(10)
Z69 = QuadraticRing(69)
Z71 = QuadraticRing(71)


def _round_trips(obj: object) -> list:
    """Return obj after a pickle round trip at every protocol, a deepcopy, and a shallow copy."""
    out = [pickle.loads(pickle.dumps(obj, protocol)) for protocol in range(pickle.HIGHEST_PROTOCOL + 1)]
    out.extend((copy.deepcopy(obj), copy.copy(obj)))
    return out


class TestPickle:
    """Tests for pickling and copying every kind of quadint object."""

    @pytest.mark.parametrize(
        "x",
        [
            complexint(1, 2),
            eisensteinint(2, 3),
            dualint(3, 4),
            splitint(-3, 4),
            ZE(1, 3),
            ZN5(2, -7),
            ZN19(5, 1),
            Z69(5, 1),
            Z71(17, 2),
            QuadraticRing(1)(3, 1),
            ZN5(0),
        ],
        ids=repr,
    )
    def test_elements(self, x: QuadInt):
        """Elements come back equal, with the same type, and in the very same ring object, so they still mix."""
        for y in _round_trips(x):
            assert type(y) is type(x)
            assert y == x
            assert y.ring is x.ring
            assert y - x == x.zero

    @pytest.mark.parametrize(
        "ring",
        [ZI, ZE, ZN5, ZN19, Z10, Z69, Z71, QuadraticRing(0), QuadraticRing(1), QuadraticRing(1, den=1)],
        ids=repr,
    )
    def test_rings(self, ring: QuadraticRing):
        """Rings are singletons, so they come back as the very same object (with its specialized type)."""
        for r in _round_trips(ring):
            assert r is ring

    @pytest.mark.parametrize(
        "ideal",
        [ZN5.ideal(3, ZN5(1, 1)), ZN5.zero_ideal(), ZN5.unit_ideal(), ZE.ideal(7), Z10.ideal(2, Z10(0, 1))],
        ids=repr,
    )
    def test_ideals(self, ideal: Ideal):
        """Ideals come back equal, and in the very same ring object."""
        for copied in _round_trips(ideal):
            assert copied == ideal
            assert copied.ring is ideal.ring

    @pytest.mark.parametrize("ring", [ZN5, ZN23, Z10], ids=repr)
    def test_class_groups(self, ring: QuadraticRing):
        """Class groups come back as the very same object, and each ideal class as an equal class."""
        group = ring.class_group
        for copied in _round_trips(group):
            assert copied is group

        for ideal_class in group:
            for copied_class in _round_trips(ideal_class):
                assert copied_class == ideal_class
                assert copied_class.ring is ring

    @pytest.mark.parametrize("x", [complexint(10, 0), complexint(-27, 36), ZE(14, 4), ZN19(10, 2)], ids=repr)
    def test_factorization(self, x: QuadInt):
        """Factorizations come back with the same unit and primes."""
        factorization = x.factor_detail()
        for copied in _round_trips(factorization):
            assert copied.unit == factorization.unit
            assert copied.primes == factorization.primes
            assert copied.prod() == x
