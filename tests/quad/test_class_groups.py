from __future__ import annotations

import math
import random

import pytest

from quadint import Ideal, QuadraticRing
from quadint.quad.ideal import ClassGroup, IdealClass, _class_form, _cycle_form, _reduce_form, _reduced_forms  # ruff: ignore[import-private-name]
from quadint.utils import _is_squarefree  # ruff: ignore[import-private-name]
from tests.quad.test_rings import _rand_elem

ZI = QuadraticRing(-1)
ZE = QuadraticRing(-3)
ZN19 = QuadraticRing(-19)
ZN7 = QuadraticRing(-7)
ZN5 = QuadraticRing(-5)

Z5 = QuadraticRing(5)
Z14 = QuadraticRing(14)
Z15 = QuadraticRing(15)


def _kronecker(a: int, n: int) -> int:
    """Return the Kronecker symbol (a / n) for n > 0, from the reciprocity of the Jacobi symbol."""
    result = 1
    while n % 2 == 0:
        if a % 2 == 0:
            return 0

        n //= 2
        if a % 8 in {3, 5}:
            result = -result

    a %= n
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in {3, 5}:
                result = -result

        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result

        a %= n

    return result if n == 1 else 0


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

    @pytest.mark.parametrize(
        "ring",
        [
            QuadraticRing(-3, den=1),
            QuadraticRing(-7, den=1),
            QuadraticRing(-15, den=1),
            QuadraticRing(5, den=1),
        ],
        ids=str,
    )
    def test_not_maximal_order(self, ring: QuadraticRing):
        """Non-maximal orders should be rejected (their non-invertible primes used to make these loop forever)."""
        with pytest.raises(NotImplementedError, match="maximal orders"):
            ClassGroup(ring)

        with pytest.raises(NotImplementedError, match="maximal orders"):
            _ = ring.class_number


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
        assert group.class_number == 1
        assert len(group) == 1
        assert group.classes == (IdealClass(ring.unit_ideal()),)
        assert group.generators == ()

    def test_is_trivial_matches_order(self):
        """is_trivial has to agree with the class number, which it gets to without finding every class."""
        for D in [*range(-300, 0), *range(2, 300)]:
            if D != -1 and not _is_squarefree(D):
                continue

            ring = QuadraticRing(D)
            key = (type(ring), ring.D, ring.den)
            cached = ClassGroup._CACHE.pop(key, None)  # a fresh group, so is_trivial can't just count its classes
            try:
                group = ClassGroup(ring)
                trivial = group.is_trivial
                assert trivial is (group.order == 1), f"Wrong for D={D}"
                assert group.is_trivial is trivial  # and again, from the classes it has now
            finally:
                if cached is not None:
                    ClassGroup._CACHE[key] = cached

    def test_gaussian_prime_ideals_are_principal(self):
        """In Z[i], the prime ideals over 5 should not create nontrivial ideal classes."""
        group = ClassGroup(ZI)
        primes = ZI.prime_ideals_over(5)

        assert len(primes) == 2
        assert all(prime.is_principal for prime in primes)
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
                    assert ideal.is_principal
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
        assert group.class_number == 2
        assert len(group.classes) == 2
        assert len(group.generators) == 1

        assert not prime_two.is_principal
        assert not prime_three.is_principal

        assert IdealClass(prime_two) in group
        assert IdealClass(prime_three) in group
        assert IdealClass(prime_two) == IdealClass(prime_three)

        assert IdealClass(prime_two).order == 2
        assert (IdealClass(prime_two) ** 2).is_trivial

    def test_zsqrt_fifteen(self):
        """Z[sqrt(15)] should have class group of order two."""
        group = ClassGroup(Z15)
        prime_two = Z15.prime_ideals_over(2)[0]
        prime_three = Z15.prime_ideals_over(3)[0]

        assert group.ring.discriminant() == 60
        assert group.minkowski_bound == 4
        assert group.order == 2
        assert group.class_number == 2
        assert len(group.classes) == 2
        assert len(group.generators) == 1

        assert prime_two.norm == 2
        assert prime_three.norm == 3
        assert not prime_two.is_principal
        assert not prime_three.is_principal

        assert IdealClass(prime_two) in group
        assert IdealClass(prime_three) in group
        assert IdealClass(prime_two) == IdealClass(prime_three)

        assert (IdealClass(prime_two) ** 2).is_trivial
        assert (IdealClass(prime_three) ** 2).is_trivial

    @pytest.mark.parametrize(("D", "expected"), [(-23, 3), (-47, 5), (-71, 7), (-199, 9), (-89, 12), (-254, 16)])
    def test_imaginary_class_numbers(self, D: int, expected: int):
        """Imaginary class numbers should match the count of reduced binary quadratic forms."""
        assert ClassGroup(QuadraticRing(D)).order == expected

    def test_small_imaginary_class_numbers(self):
        """The imaginary fields with class number 1, 2 or 3 should be exactly the known lists (all with d < 1000)."""
        # Heegner and Stark (h = 1), Baker and Stark (h = 2), Oesterle (h = 3), as d for Q(sqrt(-d))
        known = {
            1: {1, 2, 3, 7, 11, 19, 43, 67, 163},
            2: {5, 6, 10, 13, 15, 22, 35, 37, 51, 58, 91, 115, 123, 187, 235, 267, 403, 427},
            3: {23, 31, 59, 83, 107, 139, 211, 283, 307, 331, 379, 499, 547, 643, 883, 907},
        }

        found: dict[int, set[int]] = {1: set(), 2: set(), 3: set()}
        for d in range(1, 1000):
            if d == 1 or _is_squarefree(d):
                h = ClassGroup(QuadraticRing(-d)).order
                if h in found:
                    found[h].add(d)

        assert found == known

    @pytest.mark.parametrize(
        ("D", "expected"),
        [(10, 2), (79, 3), (82, 4), (226, 8), (401, 5), (3315, 8), (10001, 16)],
    )
    def test_real_class_numbers(self, D: int, expected: int):
        """Real class numbers should match the known values, which take their classes' reduced cycles to tell apart."""
        group = ClassGroup(QuadraticRing(D))

        assert group.order == expected
        assert len(set(group.classes)) == expected  # hashable, and distinct

    def test_real_class_numbers_match_dirichlet(self):
        """
        Every real class number up to D=300 should match Dirichlet's class number formula, which needs no ideals:
            h = -(sum of chi(k) * log(sin(pi*k / disc)) for 0 < k < disc) / (2 * log(eps)), with chi(k) = (disc / k)
        """
        for D in range(2, 300):
            if not _is_squarefree(D):
                continue

            ring = QuadraticRing(D)
            disc = ring.discriminant()
            eps = ring.fundamental_unit()
            log_eps = math.log((eps.a + eps.b * math.sqrt(D)) / ring.den)
            total = sum(_kronecker(disc, k) * math.log(math.sin(math.pi * k / disc)) for k in range(1, disc))

            assert ClassGroup(ring).order == round(-total / (2 * log_eps)), f"Wrong for D={D}"


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


class TestReducedForms:
    """Tests for the reduced binary quadratic forms behind ideal classes, imaginary and real."""

    @pytest.mark.parametrize(
        ("form", "expected"),
        [
            ((1, 0, 5), (1, 0, 5)),  # already reduced
            ((5, 0, 1), (1, 0, 5)),  # swap a and c
            ((2, -2, 3), (2, 2, 3)),  # b == -a moves up to a
            ((2, -3, 4), (2, 1, 3)),  # shift b into (-a, a]
            ((3, 1, 2), (2, -1, 3)),  # swap, which negates b
            ((2, -1, 2), (2, 1, 2)),  # a == c needs b >= 0
            ((12, 19, 8), (1, 1, 6)),  # several rounds, back to the principal form of disc -23
        ],
        ids=str,
    )
    def test_reduce_form_examples(self, form: tuple[int, int, int], expected: tuple[int, int, int]):
        """_reduce_form should find the reduced form, keeping the discriminant."""
        a, b, c = form
        assert _reduce_form(a, b, c) == expected
        assert expected[1] ** 2 - 4 * expected[0] * expected[2] == b * b - 4 * a * c

    @pytest.mark.parametrize("disc", [-3, -4, -20, -23, -47, -56, -71, -84, -199, -420, -4004], ids=str)
    def test_reduce_form_ignores_changes_of_variables(self, disc: int):
        """Every form properly equivalent to a reduced form should reduce back to it, and the forms are distinct."""
        rng = random.Random(-disc)
        forms = list(_reduced_forms(disc))
        assert len(set(forms)) == len(forms)

        for a, b, c in forms:
            assert b * b - 4 * a * c == disc
            assert abs(b) <= a <= c

            for _ in range(30):
                # A random change of variables (x, y) -> (p*x + q*y, r*x + s*y) with p*s - q*r == 1
                p, q, r, s = 1, 0, 0, 1
                for _ in range(rng.randint(1, 8)):
                    k = rng.randint(-5, 5)
                    p, q, r, s = (q, -p + k * q, s, -r + k * s) if rng.random() < 0.5 else (p, q + k * p, r, s + k * r)

                assert p * s - q * r == 1
                moved = (
                    a * p * p + b * p * r + c * r * r,
                    2 * a * p * q + b * (p * s + q * r) + 2 * c * r * s,
                    a * q * q + b * q * s + c * s * s,
                )
                assert _reduce_form(*moved) == (a, b, c)

    @pytest.mark.parametrize("D", [10, 79, 82, 145, 226, 229, 3315], ids=str)
    def test_cycle_form_ignores_changes_of_variables(self, D: int):
        """Every form properly equivalent to a real class's form should go back to it, and the classes' forms differ."""
        rng = random.Random(D)
        forms = [_class_form(cls.representative) for cls in ClassGroup(QuadraticRing(D))]
        assert len(set(forms)) == len(forms)

        for a, b, c in forms:
            assert _cycle_form(a, b, c) == (a, b, c)

            for _ in range(30):
                # A random change of variables (x, y) -> (p*x + q*y, r*x + s*y) with p*s - q*r == 1, which can take a
                #   negative a, since an indefinite form takes both signs
                p, q, r, s = 1, 0, 0, 1
                for _ in range(rng.randint(1, 8)):
                    k = rng.randint(-5, 5)
                    p, q, r, s = (q, -p + k * q, s, -r + k * s) if rng.random() < 0.5 else (p, q + k * p, r, s + k * r)

                moved = (
                    a * p * p + b * p * r + c * r * r,
                    2 * a * p * q + b * (p * s + q * r) + 2 * c * r * s,
                    a * q * q + b * q * s + c * s * s,
                )
                assert _cycle_form(*moved) == (a, b, c)

    @pytest.mark.parametrize(
        "ring",
        [
            *(ZN5, ZN19, QuadraticRing(-23), QuadraticRing(-105), QuadraticRing(-15, den=1), QuadraticRing(-12)),
            *(Z15, QuadraticRing(10), QuadraticRing(79), QuadraticRing(82), QuadraticRing(229)),
            *(QuadraticRing(5, den=1), QuadraticRing(13, den=1), QuadraticRing(85, den=1)),
        ],
        ids=str,
    )
    def test_class_equality_matches_principal_test(self, ring: QuadraticRing):
        """Comparing reduced forms should agree with testing whether I * conj(J) is principal, and so should hashes."""
        rng = random.Random(ring.D)

        ideals = []
        while len(ideals) < 25:
            ideal = ring.ideal(_rand_elem(rng, ring, 30), _rand_elem(rng, ring, 30))
            if not ideal.norm:
                continue

            try:
                IdealClass(ideal)
            except ValueError:
                continue  # not invertible, in the non-maximal orders

            ideals.append(ideal)

        for left in ideals:
            assert IdealClass(left).is_trivial == left.is_principal
            for right in ideals:
                same = (left * right.conjugate()).is_principal
                assert (IdealClass(left) == IdealClass(right)) == same
                if same:
                    assert hash(IdealClass(left)) == hash(IdealClass(right))

    def test_classes_are_represented_by_smallest_ideals(self):
        """Each listed class should be represented by the ideal of its reduced form, which has the smallest norm."""
        ring = QuadraticRing(-4999)
        group = ClassGroup(ring)

        assert group.order == 33
        assert group.classes[0] == IdealClass(ring.unit_ideal())
        assert len(set(group.classes)) == 33  # hashable, and distinct

        # Find the smallest norm in each class from every ideal of norm below 82, which is past every reduced form's a
        #   (a <= sqrt(19996 / 3)). An ideal's hnf (A, B, k) has norm A*k, with k dividing A and B.
        smallest: dict[IdealClass, int] = {}
        for n in range(1, 82):
            for k in range(1, n + 1):
                if n % (k * k) == 0:
                    for B in range(0, n // k, k):
                        lattice = Ideal(ring, _hnf=(n // k, B, k))
                        if ring.ideal(*lattice.basis) == lattice:  # closed under multiplication, so an ideal
                            smallest.setdefault(IdealClass(lattice), n)

        assert len(smallest) == 33
        for cls in group.classes:
            assert cls.representative.norm == smallest[cls]

    def test_products_and_powers_stay_reduced(self):
        """Products and powers of imaginary classes should land in the right class, with a reduced representative."""
        ring = QuadraticRing(-10007)  # class number 77
        group = ClassGroup(ring)
        rng = random.Random(10007)

        for _ in range(100):
            left, right = rng.choice(group.classes), rng.choice(group.classes)
            product = left * right

            assert product == IdealClass(left.representative * right.representative)
            assert 3 * product.representative.norm**2 <= 10007  # a reduced form has 3*a**2 <= |disc|

        cls = group.classes[5]
        k = cls.order
        assert k > 1
        assert (cls**k).is_trivial
        assert cls**3 == cls * cls * cls
        assert cls ** (k + 3) == cls**3
        assert cls**-1 == ~cls

        # Without reducing along the way, this representative would be an ideal whose norm has about 10**18 digits
        huge = cls ** (10**18)
        assert huge == cls ** (10**18 % k)
        assert 3 * huge.representative.norm**2 <= 10007

    @pytest.mark.parametrize("D", [-10007, 10001], ids=str)
    def test_order_adds_nothing_to_the_generator_cache(self, D: int):
        """Finding a class's order should compare forms, not cache every power as an ideal (real ones used to)."""
        cache_info = Ideal.principal_generator.cache_info
        before = cache_info().currsize
        group = ClassGroup(QuadraticRing(D))

        orders = [IdealClass(cls.representative).order for cls in group.classes]

        assert cache_info().currsize == before
        assert orders[0] == 1
        assert all(group.order % order == 0 for order in orders)  # Lagrange: every order divides the class number

    def test_real_products_and_powers_stay_reduced(self):
        """Products and powers of real classes should land in the right class, with a reduced representative."""
        ring = QuadraticRing(10001)  # class number 16, a cyclic group
        disc = ring.discriminant()
        group = ClassGroup(ring)
        rng = random.Random(10001)

        for _ in range(100):
            left, right = rng.choice(group.classes), rng.choice(group.classes)
            product = left * right

            assert product == IdealClass(left.representative * right.representative)
            assert product.representative.norm**2 < disc  # a reduced ideal's norm is below sqrt(disc)

        cls = next(cls for cls in group.classes if cls.order == 16)
        assert cls**3 == cls * cls * cls == IdealClass(cls.representative**3)
        assert cls**19 == cls**3
        assert cls**-1 == ~cls
        assert (cls**16).is_trivial

        # Without reducing along the way, this representative would be an ideal whose norm has about 10**18 digits
        huge = cls ** (10**18)
        assert huge == cls ** (10**18 % 16)
        assert huge.representative.norm**2 < disc


class TestNonMaximalOrders:
    """Tests for ideal classes in orders that are not maximal."""

    def test_non_invertible_ideal_rejected(self):
        """A non-invertible ideal has no class, and no power of it is principal (so its order would never return)."""
        ring = QuadraticRing(-3, den=1)  # Z[sqrt(-3)], the order of conductor 2 in the Eisenstein integers
        prime = ring.prime_ideals_over(2)[0]

        assert prime * prime == ring.ideal(2) * prime  # so prime cannot be invertible
        with pytest.raises(ValueError, match="not invertible"):
            IdealClass(prime)

    def test_invertible_ideal_accepted(self):
        """Invertible ideals of a non-maximal order still have classes, with finite orders."""
        ring = QuadraticRing(-15, den=1)  # Z[sqrt(-15)], whose Picard group has order 2
        prime = ring.prime_ideals_over(3)[0]

        assert not prime.is_principal
        assert IdealClass(prime).order == 2
