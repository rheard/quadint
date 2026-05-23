from __future__ import annotations

from itertools import islice

import pytest

from quadint import Ideal, QuadraticRing
from quadint.quad.ideal import IdealClass
from tests.quad.test_rings import ideal_prod

ZN7 = QuadraticRing(-7)
ZN5 = QuadraticRing(-5)
ZI = QuadraticRing(-1)


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
        """The internal HNF path should normalize signs and residues."""
        ideal = Ideal(ZN5, _hnf=(-3, -4, -1))

        assert ideal.hnf == (3, 2, 1)
        assert ideal.norm == 3
        assert ideal == ZN5.ideal(3, ZN5(2, 1))

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
        assert all(ideal.is_prime() for ideal in ideals)
        assert ideal_prod(ZN5, ideals) == ZN5.ideal(3)

    def test_inert(self):
        """An inert rational prime should stay prime with norm p squared."""
        ideals = ZI.prime_ideals_over(3)

        assert ideals == (ZI.ideal(3),)
        assert ideals[0].norm == 9
        assert ideals[0].is_prime()
        assert ZI.ideal(3).factor() == ideals

    def test_ramified(self):
        """A ramified rational prime should factor as a repeated prime ideal."""
        ideals = ZI.prime_ideals_over(2)

        assert len(ideals) == 1
        assert ideals[0].norm == 2
        assert ideals[0].is_prime()
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
        assert ideal.is_principal()

    def test_nonprincipal(self):
        """The standard non-principal ideal in Z[sqrt(-5)] should not look principal."""
        ideal = ZN5.ideal(3, ZN5(1, 1))

        assert ideal.principal_generator() is None
        assert not ideal.is_principal()

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
        assert not ideal.is_principal()


class TestOperations:
    """Tests for ideal arithmetic."""

    def test_multiply(self):
        """Multiplication should agree with rational-prime factorization."""
        left, right = ZN5.prime_ideals_over(3)

        assert left * right == ZN5.ideal(3)
        assert right * left == ZN5.ideal(3)
        assert left * 2 == ZN5.ideal(*(2 * x for x in left.basis))
        assert 2 * left == left * 2

    def test_power(self):
        """Powers should use repeated ideal multiplication."""
        ideal = ZN5.ideal(3, ZN5(1, 1))

        assert ideal**0 == ZN5.unit_ideal()
        assert ideal**1 == ideal
        assert ideal**2 == ideal * ideal
        assert ideal**3 == ideal * ideal * ideal

        with pytest.raises(ValueError, match="Negative"):
            ideal**-1

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
        assert all(factor.is_prime() for factor in factors)
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
        I = ZI.ideal(2)  # noqa: E741

        found_numbers = list(islice(I, 100))

        assert [x for x in found_numbers if x.b == 0] == [0, -2, 2, -4, 4, -6, 6, -8, 8]
        assert all(x.a % 2 == 0 and x.b % 2 == 0 for x in found_numbers)

    def test_multiples_3(self):
        """See above docstring"""
        I = ZI.ideal(3)  # noqa: E741

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

        assert unit_class.is_trivial()
        assert rational_class.is_trivial()
        assert element_class.is_trivial()

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
        assert not prime.is_principal()

        assert prime**2 == ZN5.ideal(2)
        assert not ideal_class.is_trivial()
        assert ideal_class.order == 2


class TestIdealClassMath:
    """Tests for concrete ideal-class arithmetic."""

    def test_equal_nonprincipal(self):
        """The prime ideals over 2 and 3 should represent the same nontrivial class."""
        prime_two = ZN5.prime_ideals_over(2)[0]
        prime_three = next(ideal for ideal in ZN5.prime_ideals_over(3) if ideal.hnf == (3, 1, 1))

        assert not prime_two.is_principal()
        assert not prime_three.is_principal()

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

        assert ideal_class.inverse() == ideal_class
        assert ideal_class * ideal_class.inverse() == IdealClass(ZN5.unit_ideal())

    def test_power(self):
        """Powers of the nontrivial class should follow the class group of order two."""
        prime = ZN5.prime_ideals_over(2)[0]
        ideal_class = IdealClass(prime)

        assert (ideal_class**0).is_trivial()
        assert ideal_class**1 == ideal_class
        assert (ideal_class**2).is_trivial()
        assert ideal_class**3 == ideal_class
        assert ideal_class**-1 == ideal_class
        assert (ideal_class**-2).is_trivial()

    def test_gaussian(self):
        """Prime ideals in the Gaussian integers should represent the trivial class."""
        primes = ZI.prime_ideals_over(5)

        assert len(primes) == 2

        for prime in primes:
            assert prime.norm == 5
            assert prime.is_principal()

            ideal_class = IdealClass(prime)

            assert ideal_class.is_trivial()
            assert ideal_class.order == 1
            assert ideal_class == IdealClass(ZI.unit_ideal())
