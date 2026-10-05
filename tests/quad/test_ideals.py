from __future__ import annotations

import operator
import random

from functools import reduce
from itertools import combinations, islice
from math import gcd

import pytest

from quadint import Ideal, QuadraticRing
from quadint.quad.ideal import IdealClass, _bezout_coefficients, _canonical_hnf, _lattice_hnf  # ruff: ignore[import-private-name]
from tests.quad.test_rings import _rand_elem, ideal_prod

ZN7 = QuadraticRing(-7)
ZN5 = QuadraticRing(-5)
ZI = QuadraticRing(-1)
Z2 = QuadraticRing(2)


class TestConstruct:
    """Tests for constructing and normalizing ideals."""

    def test_normalize(self):
        """Equivalent generating sets should normalize to the same HNF."""
        expected = ZN5.ideal(3, ZN5(1, 1))

        assert expected.hnf == (3, 1, 1)
        assert expected.norm == 3
        assert expected.basis == (ZN5(3), ZN5(1, 1))

        assert ZN5.ideal(3, ZN5(4, 1)) == expected
        assert ZN5.ideal(ZN5(3), ZN5(1, 1), ZN5(10, 4)) == expected

    def test_zero(self):
        """The zero ideal should normalize to the zero HNF."""
        zero = ZN5.zero_ideal()

        assert zero == ZN5.ideal(0)
        assert zero.hnf == (0, 0, 0)
        assert zero.norm == 0
        assert zero.basis == (ZN5.zero, ZN5.zero)

    def test_unit(self):
        """The unit ideal should contain the standard integral basis."""
        unit = ZN5.unit_ideal()

        assert unit == ZN5.ideal(1)
        assert unit.hnf == (1, 0, 1)
        assert unit.norm == 1
        assert ZN5.one in unit
        assert ZN5(0, 1) in unit

    def test_hnf(self):
        """The internal HNF path should normalize signs and residues, without changing the lattice."""
        # -3 and -4 - sqrt(-5) span the same lattice as 3 and 1 + sqrt(-5): negate both, then add 3 to the second
        ideal = Ideal(ZN5, _hnf=(-3, -4, -1))

        assert ideal.hnf == (3, 1, 1)
        assert ideal.norm == 3
        assert ideal == ZN5.ideal(3, ZN5(1, 1))
        assert ZN5(-3) in ideal
        assert ZN5(-4, -1) in ideal

    def test_canonical_hnf_keeps_the_lattice(self):
        """Normalizing a basis (a, 0), (b, c) should give a basis of exactly the same lattice, whatever the signs."""
        rng = random.Random(49)
        for _ in range(2_000):
            a = rng.choice((-1, 1)) * rng.randint(1, 40)
            b = rng.randint(-300, 300)
            c = rng.choice((-1, 1)) * rng.randint(1, 40)

            a2, b2, c2 = _canonical_hnf(a, b, c)
            assert a2 > 0
            assert c2 > 0
            assert 0 <= b2 < a2

            # The new lattice contains both old basis vectors, so it contains the old lattice. Its basis also has the
            #   same |determinant| (the area of a cell), so it can't be any bigger: the two lattices are the same.
            assert a2 * c2 == abs(a * c)
            assert a % a2 == 0
            n, r = divmod(c, c2)
            assert r == 0
            assert (b - n * b2) % a2 == 0

    def test_lattice_hnf_spans_the_same_lattice(self):
        """_lattice_hnf should give the normalized basis of exactly the lattice its vectors span, or reject rank 1."""
        rng = random.Random(50)
        for _ in range(3_000):
            vectors = [(rng.randint(-60, 60), rng.randint(-60, 60)) for _ in range(rng.randint(1, 6))]
            if rng.random() < 0.2:
                # Every vector on one line through the origin (or all of them zero)
                dx, dy = rng.randint(-5, 5), rng.randint(-5, 5)
                vectors = [(k * dx, k * dy) for k in (rng.randint(-9, 9) for _ in vectors)]

            # The lattice the vectors span has covolume (the area of a cell) gcd(det(v, v')) over all pairs of them,
            #   which is 0 when they don't span the plane
            covolume = reduce(gcd, (x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in combinations(vectors, 2)), 0)
            if covolume == 0:
                if any(v != (0, 0) for v in vectors):
                    with pytest.raises(ValueError, match="rank-2"):
                        _lattice_hnf(vectors)
                else:
                    assert _lattice_hnf(vectors) == (0, 0, 0)

                continue

            a, b, c = _lattice_hnf(vectors)
            assert a > 0
            assert c > 0
            assert 0 <= b < a

            # The new lattice contains every old vector, so it contains the old lattice, and with the same covolume
            #   it can't be any bigger: they are the same lattice
            assert a * c == covolume
            for x, y in vectors:
                n, r = divmod(y, c)
                assert r == 0
                assert (x - b * n) % a == 0

    def test_invalid(self):
        """Invalid constructor combinations should raise clear errors."""
        with pytest.raises(TypeError, match="expected at least one generator"):
            Ideal(ZN5)

        with pytest.raises(TypeError, match="cannot pass both"):
            Ideal(ZN5, 1, _hnf=(1, 0, 1))

        with pytest.raises(ValueError, match="rank 2"):
            Ideal(ZN5, _hnf=(0, 1, 0))


class TestMembership:
    """Tests for ideal membership."""

    def test_contains(self):
        """Membership should follow the normalized lattice, not the original generators."""
        ideal = ZN5.ideal(3, ZN5(1, 1))

        assert ZN5.zero in ideal
        assert ZN5(3) in ideal
        assert ZN5(1, 1) in ideal
        assert ZN5(4, 1) in ideal
        assert ZN5(1) not in ideal
        assert ZN5(1, 0) not in ideal
        assert object() not in ideal

    def test_wrong_ring(self):
        """Elements from another ring should not be treated as members."""
        ideal = ZN5.ideal(3, ZN5(1, 1))

        assert ZI(1, 1) not in ideal

    def test_den_two(self):
        """Membership should work in the integral basis when den is 2."""
        w = ZN7.DEFAULT_KLASS(1, 1, ZN7, skip_basis=True)
        ideal = ZN7.ideal(2, w)

        assert ideal.hnf == (2, 0, 1)
        assert ideal.norm == 2
        assert ZN7.zero in ideal
        assert 2 in ideal
        assert w in ideal
        assert ZN7.DEFAULT_KLASS(3, 1, ZN7, skip_basis=True) not in ideal

    def test_python_numbers(self):
        """Python numbers are members exactly when they equal an element of the ideal, not after truncation."""
        two = ZI.ideal(2)

        assert 2 in two
        assert 2.0 in two
        assert (4 - 6j) in two
        assert 3.0 not in two
        assert (2 + 1j) not in two

        # These used to be truncated into members, the way arithmetic truncates them (2.5 -> 2)
        assert 2.5 not in two
        assert (2 + 0.5j) not in two
        assert 1.5 not in ZI.unit_ideal()
        assert 1.5j not in ZI.unit_ideal()
        assert float("inf") not in ZI.unit_ideal()

        # Complex numbers only reach past the real axis in the Gaussian integers, but 3+0j is plain 3 in any ring
        assert (3 + 0j) in Z2.ideal(3)  # this one used to be rejected
        assert 3j not in Z2.ideal(3)

    @pytest.mark.parametrize(
        "ideal",
        [
            ZI.ideal(2),
            ZI.ideal(ZI(1, 1)),
            ZN5.ideal(3, ZN5(1, 1)),
            ZN7.ideal(2, ZN7.DEFAULT_KLASS(1, 1, ZN7, skip_basis=True)),
            Z2.ideal(3),
            ZI.zero_ideal(),
        ],
        ids=str,
    )
    def test_python_numbers_match_equality(self, ideal: Ideal):
        """A Python number is in an ideal exactly when it == one of the ideal's elements."""
        elements = list(islice(ideal, 41 * 41))  # every element with basis coefficients up to 20

        for real in (-4, -3, -2.5, -2, -1, -0.5, 0, 1, 1.5, 2, 3, 4, 6):
            for value in (real, float(real), complex(real, 0), complex(real, 1), complex(real, 2), complex(real, 0.5)):
                assert (value in ideal) == any(z == value for z in elements), value


class TestBezoutCoefficients:
    """Tests for writing an element of the ideal (a, b) as s*a + t*b."""

    @pytest.mark.parametrize("ring", [ZI, ZN5, ZN7, QuadraticRing(14), QuadraticRing(57)], ids=str)
    def test_recovers_combinations(self, ring: QuadraticRing):
        """Any g == s0*a + t0*b should come back as some s*a + t*b == g."""
        rng = random.Random(1_234 + ring.D)

        for bound in (10, 10**30):
            for _ in range(40):
                a = _rand_elem(rng, ring, bound)
                b = _rand_elem(rng, ring, bound)
                if not a or not b:
                    continue

                g = _rand_elem(rng, ring, bound) * a + _rand_elem(rng, ring, bound) * b
                bezout = _bezout_coefficients(ring, a, b, g)

                assert bezout is not None
                s, t = bezout
                assert s * a + t * b == g

    def test_outside_the_ideal(self):
        """Elements that are not in (a, b) have no Bezout coefficients."""
        # (3, 1 + sqrt(-5)) is a non-principal prime ideal of norm 3, so 1 and 2 are not in it
        assert _bezout_coefficients(ZN5, ZN5(3), ZN5(1, 1), ZN5.one) is None
        assert _bezout_coefficients(ZN5, ZN5(3), ZN5(1, 1), ZN5(2)) is None

        # ...but 4 + sqrt(-5) == 3 + (1 + sqrt(-5)) is
        bezout = _bezout_coefficients(ZN5, ZN5(3), ZN5(1, 1), ZN5(4, 1))
        assert bezout is not None
        s, t = bezout
        assert s * ZN5(3) + t * ZN5(1, 1) == ZN5(4, 1)

    def test_den_two(self):
        """The integral basis for den=2 is 1, (1 + sqrt(D))/2, which the coefficients have to respect."""
        w = ZN7.DEFAULT_KLASS(1, 1, ZN7, skip_basis=True)  # (1 + sqrt(-7)) / 2, a prime of norm 2
        two = ZN7.from_obj(2)

        # 2 == w * conj(w), and w and conj(w) are coprime, so (2, w**2) == (w)
        bezout = _bezout_coefficients(ZN7, two, w * w, w)
        assert bezout is not None
        s, t = bezout
        assert s * two + t * (w * w) == w

        # but w**2 alone is not enough
        assert _bezout_coefficients(ZN7, w * w, ZN7.from_obj(4), w) is None


class TestIter:
    """Tests for ideal iteration."""

    def test_zero(self):
        """The zero ideal should yield zero and then stop."""
        assert list(ZN5.zero_ideal()) == [ZN5.zero]

    def test_nonzero(self):
        """A nonzero ideal iterator should yield distinct ideal elements."""
        ideal = ZN5.ideal(3, ZN5(1, 1))
        values = list(islice(ideal, 25))

        assert values[0] == ZN5.zero
        assert len(values) == len(set(values))
        assert all(x in ideal for x in values)


class TestPrimeIdeals:
    """Tests for rational-prime decomposition into prime ideals."""

    def test_split(self):
        """A split rational prime should produce two prime ideals of norm p."""
        ideals = ZN5.prime_ideals_over(3)

        assert len(ideals) == 2
        assert {ideal.hnf for ideal in ideals} == {(3, 1, 1), (3, 2, 1)}
        assert all(ideal.norm == 3 for ideal in ideals)
        assert all(ideal.is_prime for ideal in ideals)
        assert ideal_prod(ZN5, ideals) == ZN5.ideal(3)

    def test_inert(self):
        """An inert rational prime should stay prime with norm p squared."""
        ideals = ZI.prime_ideals_over(3)

        assert ideals == (ZI.ideal(3),)
        assert ideals[0].norm == 9
        assert ideals[0].is_prime
        assert ZI.ideal(3).factor() == ideals

    def test_ramified(self):
        """A ramified rational prime should factor as a repeated prime ideal."""
        ideals = ZI.prime_ideals_over(2)

        assert len(ideals) == 1
        assert ideals[0].norm == 2
        assert ideals[0].is_prime
        assert ZI.ideal(2).factor() == (ideals[0], ideals[0])
        assert ideals[0] ** 2 == ZI.ideal(2)

    def test_invalid(self):
        """Only rational primes should be accepted."""
        with pytest.raises(ValueError, match="prime"):
            ZN5.prime_ideals_over(9)


class TestPrincipal:
    """Tests for principal ideal detection."""

    def test_gaussian(self):
        """A Gaussian ideal generated by one element should find a generator."""
        ideal = ZI.ideal(ZI(1, 1))
        generator = ideal.principal_generator()

        assert generator is not None
        assert ZI.ideal(generator) == ideal
        assert ideal.is_principal

    def test_nonprincipal(self):
        """The standard non-principal ideal in Z[sqrt(-5)] should not look principal."""
        ideal = ZN5.ideal(3, ZN5(1, 1))

        assert ideal.principal_generator() is None
        assert not ideal.is_principal

    def test_imaginary_den_two(self):
        """A den=2 generator's numerator a can reach isqrt(4*norm), which is past 2*isqrt(norm)."""
        ring = QuadraticRing(-19)
        alpha = ring.DEFAULT_KLASS(19, -1, ring, skip_basis=True)  # norm 95, and 19 > 2*isqrt(95) == 18

        assert ring.ideal(alpha).principal_generator() == alpha

    def test_imaginary_huge(self):
        """Imaginary generators should be found (or ruled out) without a search that grows with the norm."""
        alpha = ZN5(3**90 + 1, 2**140 - 1)
        prime_two = ZN5.prime_ideals_over(2)[0]

        assert ZN5.ideal(alpha).principal_generator() == alpha._canonical_associate()
        assert (prime_two * ZN5.ideal(alpha)).principal_generator() is None  # still in the nontrivial class

    def test_real(self):
        """A principal ideal in a real quadratic ring should solve the norm equation."""
        ring = QuadraticRing(10)
        alpha = ring(1, 1)
        ideal = ring.ideal(alpha)

        generator = ideal.principal_generator()

        assert abs(abs(alpha)) == 9
        assert ideal.norm == 9
        assert generator is not None
        assert generator in ideal
        assert abs(abs(generator)) == ideal.norm
        assert generator.a * generator.a - 10 * generator.b * generator.b in {9, -9}
        assert ring.ideal(generator) == ideal

    def test_real_den_two(self):
        """A real quadratic ring with den 2 should solve the norm equation in the integral basis."""
        ring = QuadraticRing(77)
        w = ring.DEFAULT_KLASS(1, 1, ring, skip_basis=True)
        ideal = ring.ideal(w)

        generator = ideal.principal_generator()

        assert abs(abs(w)) == 19
        assert ideal.norm == 19
        assert generator is not None
        assert generator in ideal
        assert abs(abs(generator)) == ideal.norm
        assert generator.a * generator.a - 77 * generator.b * generator.b in {76, -76}
        assert ring.ideal(generator) == ideal

    def test_real_nonprincipal(self):
        """The ramified prime over 2 in Z[sqrt(10)] should not be principal."""
        ring = QuadraticRing(10)
        ideal = ring.prime_ideals_over(2)[0]

        # If this ideal were principal, some a + b*sqrt(10) would have norm ±2.
        # Modulo 5, that would require a square to be congruent to 2 or -2.
        residues = {x * x % 5 for x in range(5)}

        assert ideal == ring.ideal(2, ring(0, 1))
        assert ideal.norm == 2
        assert residues == {0, 1, 4}
        assert 2 not in residues
        assert -2 % 5 not in residues
        assert ideal.principal_generator() is None
        assert not ideal.is_principal

    def test_dual_refused(self):
        """
        Dual-number ideals get NotImplementedError like the other square D, since their norm a**2 is not definite.

        That includes principal ones, like (2 + ε). They used to go through the imaginary lattice reduction, which
            divided by zero.
        """
        ring = QuadraticRing(0)
        for ideal in (ring.ideal(2, ring(0, 1)), ring.ideal(ring(2, 1)), ring.ideal(3, ring(0, 2))):
            with pytest.raises(NotImplementedError, match="nonsquare D"):
                ideal.principal_generator()

            with pytest.raises(NotImplementedError, match="nonsquare D"):
                _ = ideal.is_principal


class TestOperations:
    """Tests for ideal arithmetic."""

    def test_multiply(self):
        """Multiplication should agree with rational-prime factorization."""
        left, right = ZN5.prime_ideals_over(3)

        assert left * right == ZN5.ideal(3)
        assert right * left == ZN5.ideal(3)
        assert left * 2 == ZN5.ideal(*(2 * x for x in left.basis))
        assert 2 * left == left * 2

    def test_unsupported_multiplications(self):
        """
        * with anything but an ideal or a ring element raises Python's usual TypeError in both builds, on either side,
            and a type with its own __rmul__ gets to handle it. Compiled, every one of these used to raise mypyc's
            "Ideal object expected; got NotImplementedType" instead, before the other operand had a turn.
        """

        class WithRmul:
            def __rmul__(self, _: object) -> str:
                return "rmul"

        ideal = ZN5.ideal(3, ZN5(1, 1))
        for other in (None, IdealClass(ideal), ZN5.class_group):
            with pytest.raises(TypeError, match="unsupported operand"):
                operator.mul(ideal, other)

            with pytest.raises(TypeError, match="unsupported operand"):
                operator.mul(other, ideal)

        # A sequence gets to try repeating itself, which the two builds word differently (compiled, the ideal looks
        #   like it might be an index to CPython, as an ideal class did already)
        for other in ("a", [2]):
            with pytest.raises(TypeError):
                operator.mul(ideal, other)

            with pytest.raises(TypeError):
                operator.mul(other, ideal)

        assert ideal * WithRmul() == "rmul"

    def test_power(self):
        """Powers should use repeated ideal multiplication."""
        ideal = ZN5.ideal(3, ZN5(1, 1))

        assert ideal**0 == ZN5.unit_ideal()
        assert ideal**1 == ideal
        assert ideal**2 == ideal * ideal
        assert ideal**3 == ideal * ideal * ideal

        with pytest.raises(ValueError, match="Negative"):
            ideal**-1

    def test_non_int_exponents(self):
        """
        An exponent that isn't an int raises TypeError in both builds, a float included (which pure Python used to
            truncate), and a type with its own __rpow__ gets to handle it
        """

        class WithRpow:
            def __rpow__(self, _: object) -> str:
                return "rpow"

        ideal = ZN5.ideal(3, ZN5(1, 1))
        for exp in (2.0, 2.5, "2", None, [2]):
            with pytest.raises(TypeError):
                operator.pow(ideal, exp)

        assert ideal**True == ideal
        assert ideal ** WithRpow() == "rpow"

    def test_ideal_exponents(self):
        """
        Nothing takes an ideal as an exponent, so whatever the base, ** raises TypeError in both builds (the compiled
            one used to recurse until RecursionError, see the TODO on QuadInt.__rpow__)
        """
        ideal = ZN5.ideal(3, ZN5(1, 1))
        for base in (ideal, IdealClass(ideal), ZN5(1, 1), 2, 2.5, None, "a"):
            with pytest.raises(TypeError):
                operator.pow(base, ideal)

        with pytest.raises(TypeError):
            operator.ipow(ideal, ideal)

    def test_conjugate(self):
        """Conjugating a split prime ideal should produce its opposite factor."""
        left, right = ZN5.prime_ideals_over(3)

        assert left.conjugate() == right
        assert right.conjugate() == left
        assert left * left.conjugate() == ZN5.ideal(3)

    def test_divides(self):
        """Ideal divisibility should match containment of generated lattices."""
        left, right = ZN5.prime_ideals_over(3)
        product = left * right

        assert left.divides(product)
        assert right.divides(product)
        assert not product.divides(left)

        with pytest.raises(TypeError, match="different rings"):
            left.divides(ZI.ideal(2))

    def test_factor(self):
        """Factoring an ideal should reconstruct the original ideal."""
        ideal = ZN5.ideal(3) * ZN5.prime_ideals_over(7)[0]
        factors = ideal.factor()

        assert factors
        assert all(factor.is_prime for factor in factors)
        assert ideal_prod(ZN5, factors) == ideal
        assert ideal.factor() == factors

    def test_factor_trivial(self):
        """The unit and zero ideals should have special factorization behavior."""
        assert ZN5.unit_ideal().factor() == ()

        with pytest.raises(ValueError, match="zero ideal"):
            ZN5.zero_ideal().factor()


class TestQuotient:
    """Tests for colon ideals and exact ideal quotients."""

    def test_colon(self):
        """The colon ideal should recover the missing factor from a product."""
        left, right = ZN5.prime_ideals_over(3)
        product = left * right

        assert product.colon(left) == right
        assert product.colon(right) == left
        assert product.colon(ZN5.unit_ideal()) == product

    def test_exact(self):
        """Exact division should return q when self equals other times q."""
        left, right = ZN5.prime_ideals_over(3)
        ideal = left * left * right

        quotient = ideal.exact_div(left)

        assert quotient == left * right
        assert left * quotient == ideal
        assert ideal // left == quotient
        assert ideal.norm == left.norm * quotient.norm

    def test_inexact(self):
        """Exact division should reject non-integral ideal quotients."""
        left, right = ZN5.prime_ideals_over(3)

        with pytest.raises(ValueError, match="not exact"):
            left.exact_div(right)

        with pytest.raises(ValueError, match="not exact"):
            left // right

    def test_zero(self):
        """Zero ideal cases should follow the integral quotient conventions."""
        left = ZN5.prime_ideals_over(3)[0]
        zero = ZN5.zero_ideal()

        assert left.colon(zero) == ZN5.unit_ideal()
        assert zero.colon(left) == zero
        assert zero.exact_div(left) == zero

        with pytest.raises(ZeroDivisionError):
            left.exact_div(zero)

    def test_wrong_ring(self):
        """Quotients of ideals from different rings should be rejected."""
        left = ZN5.prime_ideals_over(3)[0]

        with pytest.raises(TypeError, match="different rings"):
            left.colon(ZI.ideal(2))

        with pytest.raises(TypeError, match="different rings"):
            left.exact_div(ZI.ideal(2))

    def test_unsupported_operands(self):
        """
        // with anything but an ideal raises TypeError in both builds, on either side, and a type with its own
            __rfloordiv__ gets to handle it (compiled, 5 // ideal used to recurse until RecursionError)
        """

        class WithRfloordiv:
            def __rfloordiv__(self, _: object) -> str:
                return "rfloordiv"

        ideal = ZN5.ideal(3, ZN5(1, 1))
        for other in (5, 2.5, None, "a", ZN5(1, 1), IdealClass(ideal)):
            with pytest.raises(TypeError):
                operator.floordiv(other, ideal)

            with pytest.raises(TypeError):
                operator.floordiv(ideal, other)

        assert ideal // WithRfloordiv() == "rfloordiv"


class TestIdealMath:
    """Tests for concrete mathematical facts about ideals."""

    def test_principal_norm(self):
        """A principal ideal should have norm equal to the absolute field norm of its generator."""
        alpha = ZN5(1, 1)
        ideal = ZN5.ideal(alpha)

        assert abs(abs(alpha)) == 6
        assert ideal.norm == 6
        assert ideal.hnf == (6, 1, 1)

    def test_gaussian_split(self):
        """In Z[i], the prime 5 should split as (2 + i)(2 - i)."""
        left = ZI.ideal(ZI(2, 1))
        right = ZI.ideal(ZI(2, -1))

        assert left.norm == 5
        assert right.norm == 5
        assert left * right == ZI.ideal(5)
        assert {left, right} == set(ZI.prime_ideals_over(5))

    def test_ramified_five(self):
        """In Z[sqrt(-5)], the prime 5 should ramify as (sqrt(-5)) squared."""
        w = ZN5(0, 1)
        prime = ZN5.prime_ideals_over(5)[0]

        assert abs(abs(w)) == 5
        assert prime == ZN5.ideal(w)
        assert prime.norm == 5
        assert prime**2 == ZN5.ideal(5)

    def test_nonunique_integer_factorization(self):
        """The equality 2 * 3 = (1 + sqrt(-5)) * (1 - sqrt(-5)) should agree as ideals."""
        w_plus = ZN5(1, 1)
        w_minus = ZN5(1, -1)

        assert ZN5(2) * ZN5(3) == w_plus * w_minus
        assert ZN5.ideal(2) * ZN5.ideal(3) == ZN5.ideal(w_plus) * ZN5.ideal(w_minus)
        assert ZN5.ideal(6) == ZN5.ideal(w_plus) * ZN5.ideal(w_minus)

    def test_six_factorization(self):
        """The ideal (6) in Z[sqrt(-5)] should factor as P2^2 * P3 * conjugate(P3)."""
        prime_two = ZN5.prime_ideals_over(2)[0]
        prime_three, prime_three_conj = ZN5.prime_ideals_over(3)

        ideal = ZN5.ideal(6)
        factors = ideal.factor()

        assert prime_two.norm == 2
        assert prime_three.norm == 3
        assert prime_three_conj.norm == 3

        assert prime_two**2 == ZN5.ideal(2)
        assert prime_three * prime_three_conj == ZN5.ideal(3)
        assert prime_two**2 * prime_three * prime_three_conj == ideal

        assert len(factors) == 4
        assert factors.count(prime_two) == 2
        assert factors.count(prime_three) == 1
        assert factors.count(prime_three_conj) == 1

    def test_den_two_split(self):
        """In the full ring of integers of Q(sqrt(-7)), 2 should split as w * conjugate(w)."""
        w = ZN7.DEFAULT_KLASS(1, 1, ZN7, skip_basis=True)
        left = ZN7.ideal(w)
        right = ZN7.ideal(w.conjugate())

        assert abs(abs(w)) == 2
        assert left.norm == 2
        assert right.norm == 2
        assert left * right == ZN7.ideal(2)
        assert {left, right} == set(ZN7.prime_ideals_over(2))

    def test_shell_order(self):
        """A nonzero ideal iterator should expand through deterministic lattice shells."""
        ideal = ZN5.ideal(3, ZN5(1, 1))

        values = list(islice(ideal, 9))

        assert ideal.basis == (ZN5(3), ZN5(1, 1))
        assert values == [
            ZN5.zero,
            ZN5(-4, -1),
            ZN5(-2, 1),
            ZN5(-1, -1),
            ZN5(1, 1),
            ZN5(2, -1),
            ZN5(4, 1),
            ZN5(-3),
            ZN5(3),
        ]
        assert all(x in ideal for x in values)

    def test_even_numbers(self):
        """
        This test (and the next one) I devised after reading the Wikipedia article on ideals, which says:
            "Ideals generalize certain subsets of the integers, such as the even numbers or the multiples of 3."

        So in theory I should be able to create an ideal that represents these with my class?
            However the Ideal class in this package is a class for rank-2 lattice ideals,
                for quadratic integers (the point of the library after all).
            So instead of getting all even numbers, we would get all even numbers a+bi (where a and b are both even).
        """
        I = ZI.ideal(2)  # ruff: ignore[ambiguous-variable-name]

        found_numbers = list(islice(I, 100))

        assert [x for x in found_numbers if x.b == 0] == [0, -2, 2, -4, 4, -6, 6, -8, 8]
        assert all(x.a % 2 == 0 and x.b % 2 == 0 for x in found_numbers)

    def test_multiples_3(self):
        """See above docstring"""
        I = ZI.ideal(3)  # ruff: ignore[ambiguous-variable-name]

        found_numbers = list(islice(I, 100))

        assert [x for x in found_numbers if x.b == 0] == [0, -3, 3, -6, 6, -9, 9, -12, 12]
        assert all(x.a % 3 == 0 and x.b % 3 == 0 for x in found_numbers)


class TestIdealClassConstruct:
    """Tests for constructing ideal classes."""

    def test_zero(self):
        """The zero ideal should not define an ideal class."""
        with pytest.raises(ValueError, match="zero ideal"):
            IdealClass(ZN5.zero_ideal())

    def test_principal(self):
        """Principal ideals should represent the trivial ideal class."""
        unit_class = IdealClass(ZN5.unit_ideal())
        rational_class = IdealClass(ZN5.ideal(3))
        element_class = IdealClass(ZN5.ideal(ZN5(1, 1)))

        assert unit_class.is_trivial
        assert rational_class.is_trivial
        assert element_class.is_trivial

        assert unit_class.order == 1
        assert rational_class.order == 1
        assert element_class.order == 1

        assert rational_class == unit_class
        assert element_class == unit_class

    def test_nonprincipal(self):
        """The ramified prime over 2 in Z[sqrt(-5)] should be the nontrivial class."""
        prime = ZN5.prime_ideals_over(2)[0]
        ideal_class = IdealClass(prime)

        assert prime.hnf == (2, 1, 1)
        assert prime.norm == 2
        assert not prime.is_principal

        assert prime**2 == ZN5.ideal(2)
        assert not ideal_class.is_trivial
        assert ideal_class.order == 2


class TestIdealClassMath:
    """Tests for concrete ideal-class arithmetic."""

    def test_equal_nonprincipal(self):
        """The prime ideals over 2 and 3 should represent the same nontrivial class."""
        prime_two = ZN5.prime_ideals_over(2)[0]
        prime_three = next(ideal for ideal in ZN5.prime_ideals_over(3) if ideal.hnf == (3, 1, 1))

        assert not prime_two.is_principal
        assert not prime_three.is_principal

        assert prime_two * prime_three.conjugate() == ZN5.ideal(ZN5(1, -1))
        assert IdealClass(prime_two) == IdealClass(prime_three)

    def test_hash(self):
        """Equal ideal classes should have equal hashes even with different representatives."""
        prime_two = ZN5.prime_ideals_over(2)[0]
        prime_three = next(ideal for ideal in ZN5.prime_ideals_over(3) if ideal.hnf == (3, 1, 1))

        left = IdealClass(prime_two)
        right = IdealClass(prime_three)

        assert left == right
        assert hash(left) == hash(right)

    def test_inverse(self):
        """The nontrivial class in Z[sqrt(-5)] should be its own inverse."""
        prime = ZN5.prime_ideals_over(2)[0]
        ideal_class = IdealClass(prime)

        assert ~ideal_class == ideal_class
        assert ideal_class * ~ideal_class == IdealClass(ZN5.unit_ideal())

    def test_power(self):
        """Powers of the nontrivial class should follow the class group of order two."""
        prime = ZN5.prime_ideals_over(2)[0]
        ideal_class = IdealClass(prime)

        assert (ideal_class**0).is_trivial
        assert ideal_class**1 == ideal_class
        assert (ideal_class**2).is_trivial
        assert ideal_class**3 == ideal_class
        assert ideal_class**-1 == ideal_class
        assert (ideal_class**-2).is_trivial

    def test_non_int_exponents(self):
        """An exponent that isn't an int raises TypeError in both builds, a float included (see the Ideal test)."""
        ideal_class = IdealClass(ZN5.prime_ideals_over(2)[0])
        for exp in (2.0, 2.5, "2", None, [2]):
            with pytest.raises(TypeError):
                operator.pow(ideal_class, exp)

        assert ideal_class**True == ideal_class

    def test_ideal_class_exponents(self):
        """Nothing takes an ideal class as an exponent, so whatever the base, ** raises TypeError (like an ideal)."""
        ideal = ZN5.prime_ideals_over(2)[0]
        ideal_class = IdealClass(ideal)
        for base in (ideal_class, ideal, ZN5(1, 1), 2, 2.5, None, "a"):
            with pytest.raises(TypeError):
                operator.pow(base, ideal_class)

        with pytest.raises(TypeError):
            operator.ipow(ideal_class, ideal_class)

    def test_unsupported_operands(self):
        """
        * with anything but an ideal class raises TypeError in both builds, on either side, and a type with its own
            __rmul__ gets to handle it (compiled, 5 * cls used to recurse until RecursionError)
        """

        class WithRmul:
            def __rmul__(self, _: object) -> str:
                return "rmul"

        ideal = ZN5.prime_ideals_over(2)[0]
        ideal_class = IdealClass(ideal)
        for other in (5, 2.5, None, "a", ZN5(1, 1), ideal):
            with pytest.raises(TypeError):
                operator.mul(other, ideal_class)

            with pytest.raises(TypeError):
                operator.mul(ideal_class, other)

        assert ideal_class * WithRmul() == "rmul"

    def test_gaussian(self):
        """Prime ideals in the Gaussian integers should represent the trivial class."""
        primes = ZI.prime_ideals_over(5)

        assert len(primes) == 2

        for prime in primes:
            assert prime.norm == 5
            assert prime.is_principal

            ideal_class = IdealClass(prime)

            assert ideal_class.is_trivial
            assert ideal_class.order == 1
            assert ideal_class == IdealClass(ZI.unit_ideal())
