from __future__ import annotations

import pytest

from quadint import QuadraticRing
from quadint.quad.ideal import ClassGroup, IdealClass

ZI = QuadraticRing(-1)
ZE = QuadraticRing(-3)
ZN19 = QuadraticRing(-19)
ZN7 = QuadraticRing(-7)
ZN5 = QuadraticRing(-5)

Z5 = QuadraticRing(5)
Z14 = QuadraticRing(14)
Z15 = QuadraticRing(15)


class TestConstruct:
    """Tests for constructing class groups."""

    def test_imaginary(self):
        """A class group should remember the imaginary quadratic ring it belongs to."""
        group = ClassGroup(ZN5)

        assert group.ring is ZN5
        assert repr(group) == f"ClassGroup({ZN5!r})"
        assert hash(group) == hash(ZN5)

    def test_real(self):
        """A class group should also be constructible for real quadratic fields."""
        group = ClassGroup(Z15)

        assert group.ring is Z15
        assert repr(group) == f"ClassGroup({Z15!r})"
        assert hash(group) == hash(Z15)

    @pytest.mark.parametrize(
        "ring",
        [
            QuadraticRing(0),
            QuadraticRing(1),
            QuadraticRing(4),
            QuadraticRing(12),
            QuadraticRing(-8),
            QuadraticRing(-9),
        ],
        ids=str,
    )
    def test_not_field(self, ring: QuadraticRing):
        """Class groups should reject dual, split, square, and nonsquarefree D cases."""
        with pytest.raises(NotImplementedError, match="quadratic field"):
            ClassGroup(ring)


class TestBounds:
    """Tests for Minkowski bounds."""

    @pytest.mark.parametrize(
        ("ring", "expected"),
        [
            (ZI, 2),
            (ZE, 2),
            (ZN5, 3),
            (ZN7, 2),
            (Z5, 2),
            (Z14, 4),
            (Z15, 4),
        ],
        ids=str,
    )
    def test_minkowski_bound(self, ring: QuadraticRing, expected: int):
        """The Minkowski bound should use the imaginary or real quadratic formula as appropriate."""
        assert ClassGroup(ring).minkowski_bound == expected


class TestEquality:
    """Tests for class group equality."""

    def test_equal(self):
        """Class groups for the same ring should compare equal even when not identical."""
        first = ClassGroup(ZN5)
        key = (type(ZN5), ZN5.D, ZN5.den)

        ClassGroup._CACHE.pop(key)
        try:
            second = ClassGroup(ZN5)

            assert first is not second
            assert first == second
            assert hash(first) == hash(second)
        finally:
            ClassGroup._CACHE[key] = first

    def test_different(self):
        """Class groups for different rings should not compare equal."""
        assert ClassGroup(ZN5) != ClassGroup(Z15)
        assert ClassGroup(ZN5) != object()


class TestTrivialGroups:
    """Tests for rings whose ideal class group is trivial."""

    @pytest.mark.parametrize(
        "ring",
        [
            ZI,
            ZE,
            ZN7,
            ZN19,
            Z5,
            Z14,
        ],
        ids=str,
    )
    def test_class_number_one(self, ring: QuadraticRing):
        """Class-number-one rings should have only the principal ideal class."""
        group = ClassGroup(ring)

        assert group.order == 1
        assert group.class_number() == 1
        assert len(group) == 1
        assert group.classes == (IdealClass(ring.unit_ideal()),)
        assert group.generators == ()

    def test_gaussian_prime_ideals_are_principal(self):
        """In Z[i], the prime ideals over 5 should not create nontrivial ideal classes."""
        group = ClassGroup(ZI)
        primes = ZI.prime_ideals_over(5)

        assert len(primes) == 2
        assert all(prime.is_principal() for prime in primes)
        assert all(IdealClass(prime) in group for prime in primes)
        assert all(IdealClass(prime) == IdealClass(ZI.unit_ideal()) for prime in primes)

    def test_real_class_number_one(self):
        """In Z[sqrt(14)], prime ideals up to the bound should all be principal."""
        group = ClassGroup(Z14)

        assert group.ring.discriminant() == 56
        assert group.minkowski_bound == 4
        assert group.order == 1
        assert group.generators == ()

        for p in (2, 3):
            for ideal in Z14.prime_ideals_over(p):
                if ideal.norm <= group.minkowski_bound:
                    assert ideal.is_principal()
                    assert IdealClass(ideal) == IdealClass(Z14.unit_ideal())


class TestNontrivialGroups:
    """Tests for rings with nontrivial ideal class groups."""

    def test_zsqrt_minus_five(self):
        """Z[sqrt(-5)] should have class group of order two."""
        group = ClassGroup(ZN5)
        prime_two = ZN5.prime_ideals_over(2)[0]
        prime_three = next(ideal for ideal in ZN5.prime_ideals_over(3) if ideal.hnf == (3, 1, 1))

        assert group.ring.discriminant() == -20
        assert group.minkowski_bound == 3
        assert group.order == 2
        assert group.class_number() == 2
        assert len(group.classes) == 2
        assert len(group.generators) == 1

        assert not prime_two.is_principal()
        assert not prime_three.is_principal()

        assert IdealClass(prime_two) in group
        assert IdealClass(prime_three) in group
        assert IdealClass(prime_two) == IdealClass(prime_three)

        assert IdealClass(prime_two).order == 2
        assert (IdealClass(prime_two) ** 2).is_trivial()

    def test_zsqrt_fifteen(self):
        """Z[sqrt(15)] should have class group of order two."""
        group = ClassGroup(Z15)
        prime_two = Z15.prime_ideals_over(2)[0]
        prime_three = Z15.prime_ideals_over(3)[0]

        assert group.ring.discriminant() == 60
        assert group.minkowski_bound == 4
        assert group.order == 2
        assert group.class_number() == 2
        assert len(group.classes) == 2
        assert len(group.generators) == 1

        assert prime_two.norm == 2
        assert prime_three.norm == 3
        assert not prime_two.is_principal()
        assert not prime_three.is_principal()

        assert IdealClass(prime_two) in group
        assert IdealClass(prime_three) in group
        assert IdealClass(prime_two) == IdealClass(prime_three)

        assert (IdealClass(prime_two) ** 2).is_trivial()
        assert (IdealClass(prime_three) ** 2).is_trivial()


class TestGroupBehavior:
    """Tests for basic class group behavior."""

    def test_iter(self):
        """Iterating over a class group should iterate over its computed classes."""
        group = ClassGroup(ZN5)

        assert tuple(group) == group.classes
        assert all(isinstance(cls, IdealClass) for cls in group)

    def test_contains(self):
        """Containment should recognize ideal classes from the same ring only."""
        group = ClassGroup(ZN5)
        prime = ZN5.prime_ideals_over(2)[0]

        assert IdealClass(ZN5.unit_ideal()) in group
        assert IdealClass(prime) in group
        assert IdealClass(ZI.unit_ideal()) not in group
        assert object() not in group

    def test_closed(self):
        """The computed classes should be closed under ideal-class multiplication."""
        group = ClassGroup(ZN5)

        for left in group:
            for right in group:
                assert left * right in group

    def test_cached(self):
        """Generator and class computation should be cached on the ClassGroup instance."""
        group = ClassGroup(ZN5)

        assert group.generators is group.generators
        assert group.classes is group.classes
