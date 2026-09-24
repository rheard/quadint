from __future__ import annotations

import os
import random

from itertools import product
from math import gcd, isclose, isqrt, prod
from typing import TYPE_CHECKING

import pytest

from quadint import QuadInt, complexint, eisensteinint
from quadint.quad import Factorization, QuadraticRing
from quadint.quad.rings import NORM_EUCLID_D

if TYPE_CHECKING:
    from collections.abc import Callable


def brute_content(x: QuadInt) -> int:
    """Reference implementation: scan all divisors of gcd(a,b)."""
    g = gcd(abs(x.a), abs(x.b))
    if g <= 1:
        return 1

    den = x.ring.den
    best = 1
    r = isqrt(g)
    for d in range(1, r + 1):
        if g % d:
            continue
        for cand in (d, g // d):
            if cand <= best:
                continue
            a = x.a // cand
            b = x.b // cand
            if den == 2 and ((a ^ b) & 1):
                continue
            best = cand

    return best


ZN17 = QuadraticRing(-17)
ZN7 = QuadraticRing(-7)
ZN6 = QuadraticRing(-6)
ZN5 = QuadraticRing(-5)
ZN2 = QuadraticRing(-2)
Z1 = QuadraticRing(1)
Z2 = QuadraticRing(2)
Z5 = QuadraticRing(5)
Z15 = QuadraticRing(15)

ZI = QuadraticRing(-1)
ZE = QuadraticRing(-3)


class QuadIntTests:
    """Tests for __div__"""

    a_int, b_int, a_cint, b_cint = None, None, None, None

    def setup_method(self, _):
        """Setup some test data"""
        self.a_int = Z2(5, 2)
        self.b_int = Z2(3, -2)

        self.a_cint = complexint(5, 2)
        self.b_cint = complexint(3, -2)

    @staticmethod
    def assert_quad_equal(res: tuple | QuadInt, res_int: QuadInt):
        """Validate the QuadInt is equal to the validation object, and that it is still backed by integers"""
        assert res[0] == res_int.basis_a
        assert res[1] == res_int.basis_b

        assert isinstance(res_int.a, int)
        assert isinstance(res_int.b, int)
        assert isinstance(res_int.basis_a, int)
        assert isinstance(res_int.basis_b, int)

        assert isinstance(res_int, QuadInt)

    def assert_factoring(self, n: QuadInt, factors: Factorization | dict):
        """Validate everything about the factoring is correct"""
        if isinstance(factors, Factorization):
            ans = factors.prod()
            primes = factors.primes
        else:
            ans = prod(p**k for p, k in factors.items())
            primes = factors

        self.assert_quad_equal(n, ans)

        for p in primes:
            # These _should_ all be primes and should be impossible to factor...
            if isinstance(factors, Factorization):
                prime_factors = p.factor_detail()

                assert abs(prime_factors.unit) == 1
                assert len(prime_factors.primes) == 1

                factored_p, factored_k = next(iter(prime_factors.primes.items()))
                assert factored_k == 1
                # Because the unit is not integrated, the prime returned may be an associate of the prime we care about
                assert any(p == factored_p * u for u in p.units)
            else:
                prime_factors = p.factor()

                assert len(prime_factors) == 1

                factored_p, factored_k = next(iter(prime_factors.items()))
                assert factored_k == 1
                assert factored_p == p


class TestArithmetic:
    """Tests for direct quadratic-integer arithmetic."""

    @pytest.mark.parametrize(
        ("x", "y", "expected"),
        [
            (ZN2(1, 3), ZN2(5, -2), ZN2(17, 13)),
            (Z2(5, 2), Z2(3, -2), Z2(7, -4)),
            (Z15(5, 2), Z15(3, -2), Z15(-45, -4)),
            (ZN7(1, 1), ZN7(3, -1), ZN7(5, 1)),
            (Z5(1, 1), Z5(3, -1), Z5(-1, 1)),
        ],
        ids=str,
    )
    def test_mul(self, x: QuadInt, y: QuadInt, expected: QuadInt):
        """Multiplication should match hand-calculated products in representative rings."""
        assert x * y == expected
        assert y * x == expected

    @pytest.mark.parametrize(
        ("x", "expected"),
        [
            (ZI(2, 3), 13),
            (ZN2(1, 3), 19),
            (Z2(5, 2), 17),
            (ZE(1, 1), 1),
            (ZN7(1, 1), 2),
            (Z5(1, 1), -1),
            (Z1(1, 1), 0),
        ],
        ids=str,
    )
    def test_norm_hand_calculated(self, x: QuadInt, expected: int):
        """Norms should use the ring's D and denominator exactly."""
        assert abs(x) == expected

    @pytest.mark.parametrize(
        ("x", "y"),
        [
            (ZI(2, 3), ZI(4, -5)),
            (ZN2(1, 3), ZN2(5, -2)),
            (Z2(5, 2), Z2(3, -2)),
            (Z15(5, 2), Z15(3, -2)),
            (ZE(1, 1), ZE(1, -1)),
            (ZN7(1, 1), ZN7(3, -1)),
            (Z5(1, 1), Z5(3, -1)),
        ],
        ids=str,
    )
    def test_norm_is_multiplicative(self, x: QuadInt, y: QuadInt):
        """Norms should multiply exactly, including real and den=2 rings."""
        assert abs(x * y) == abs(x) * abs(y)

    @pytest.mark.parametrize(
        "x",
        [
            ZI(2, 3),
            ZN2(1, 3),
            Z2(5, 2),
            Z15(5, 2),
            ZE(1, 1),
            ZN7(1, 1),
            Z5(1, 1),
        ],
        ids=str,
    )
    def test_conjugate_product_is_norm(self, x: QuadInt):
        """x*conjugate(x) should be the embedded rational norm."""
        assert x * x.conjugate() == x.ring.from_ab(abs(x), 0)


class TestDiv(QuadIntTests):
    """Tests for __div__"""

    def test_div(self):
        """Test QuadInt / QuadInt in QuadraticRing(2)"""
        mul_int = self.a_int * self.b_int

        res_int = mul_int / self.a_int
        self.assert_quad_equal((self.b_int.a, self.b_int.b), res_int)

        res_int = mul_int / self.b_int
        self.assert_quad_equal((self.a_int.a, self.a_int.b), res_int)

    def test_div_int(self):
        """Test QuadInt / int in QuadraticRing(2)"""
        mul_int = self.a_int * 3
        res_int = mul_int / 3
        self.assert_quad_equal((self.a_int.a, self.a_int.b), res_int)

    def test_rdivmod_int(self):
        """Test divmod(int, QuadInt) embeds the int into the divisor's ring."""
        y = ZI(2, 1)
        expected = divmod(ZI(7), y)

        got = divmod(7, y)

        assert got == expected

    def test_rdivmod_complex(self):
        """Test divmod(complex, complexint) embeds the complex number into the Gaussian integers."""
        y = complexint(2, 1)
        expected = divmod(complexint(7, 4), y)

        got = divmod(7 + 4j, y)

        assert got == expected

    def test_div_float(self):
        """Test QuadInt / float in QuadraticRing(2)"""
        mul_int = self.a_int * 3
        res_int = mul_int / float(3)
        self.assert_quad_equal((self.a_int.a, self.a_int.b), res_int)

    @pytest.mark.parametrize(
        ("x", "y", "expected"),
        [
            (ZN2(17, 13), ZN2(1, 3), ZN2(5, -2)),
            (Z2(7, -4), Z2(5, 2), Z2(3, -2)),
            (ZN7(5, 1), ZN7(1, 1), ZN7(3, -1)),
            (Z5(-1, 1), Z5(1, 1), Z5(3, -1)),
        ],
        ids=str,
    )
    def test_exact_quotients(self, x: QuadInt, y: QuadInt, expected: QuadInt):
        """Exact quotients should match hand-calculated results, not just reverse a product."""
        q, r = divmod(x, y)

        assert q == expected
        assert r == x.ring.zero
        assert x / y == expected
        assert x.exact_div(y) == expected
        assert y.divides(x) is True

    @pytest.mark.parametrize(
        ("x", "y", "expected_q", "expected_r"),
        [
            (Z2(7, 5), Z2(3, 1), Z2(2, 1), Z2(-1, 0)),
            (ZN2(7, 5), ZN2(3, 1), ZN2(3, 1), ZN2(0, -1)),
        ],
        ids=str,
    )
    def test_divmod_remainders(
        self,
        x: QuadInt,
        y: QuadInt,
        expected_q: QuadInt,
        expected_r: QuadInt,
    ):
        """Non-exact Euclidean division should return known quotients and remainders."""
        q, r = divmod(x, y)

        assert q == expected_q
        assert r == expected_r
        assert x == q * y + r
        assert abs(abs(r)) < abs(abs(y))

    def test_div_ring5(self):
        """Test QuadInt / QuadInt in QuadraticRing(5)"""
        a_int = Z5(7, 3)
        b_int = Z5(3, -5)

        mul_int = a_int * b_int

        res_int = mul_int / a_int
        self.assert_quad_equal((b_int.a, b_int.b), res_int)

        res_int = mul_int / b_int
        self.assert_quad_equal((a_int.a, a_int.b), res_int)

    def test_div_ring15(self):
        """Test QuadInt / QuadInt in QuadraticRing(15)"""
        a_int = Z15(5, 2)
        b_int = Z15(3, -2)

        mul_int = a_int * b_int

        with pytest.raises(NotImplementedError):
            _ = mul_int / a_int

    def test_div_ring_neg7(self):
        """Test QuadInt / QuadInt in QuadraticRing(-7)"""
        a_int = ZN7(7, 3)
        b_int = ZN7(3, -5)

        mul_int = a_int * b_int

        res_int = mul_int / a_int
        self.assert_quad_equal((b_int.a, b_int.b), res_int)

        res_int = mul_int / b_int
        self.assert_quad_equal((a_int.a, a_int.b), res_int)

    def test_div_ring_neg17(self):
        """Test QuadInt / QuadInt in QuadraticRing(-17) (-17 is not norm-Euclidean so this is not implemented)"""
        a_int = ZN17(5, 2)
        b_int = ZN17(3, -2)

        mul_int = a_int * b_int

        with pytest.raises(NotImplementedError):
            _ = mul_int / a_int

    def test_pow_mod_matches_pow_then_mod_gaussian(self):
        """pow(x, e, m) should match (x**e) % m in Gaussian integers."""
        x = ZI(5, 2)
        m = ZI(2, 1)

        got = pow(x, 13, m)
        expected = (x**13) % m

        self.assert_quad_equal((expected.a, expected.b), got)

    def test_pow_mod_exponent_zero(self):
        """pow(x, 0, m) should be 1 % m."""
        ZI = QuadraticRing(-1)
        x = ZI(5, 2)
        m = ZI(2, 1)

        got = pow(x, 0, m)
        expected = x.one % m

        self.assert_quad_equal((expected.a, expected.b), got)

    def test_pow_mod_accepts_int_modulus(self):
        """Allow int modulus (it should embed into the same ring)."""
        x = ZI(5, 2)

        got = pow(x, 20, 7)
        expected = (x**20) % 7

        self.assert_quad_equal((expected.a, expected.b), got)

    @pytest.mark.parametrize("e", [2**64, 2**1100], ids=["2**64", "2**1100"])
    def test_pow_mod_huge_exponent(self, e: int):
        """pow(x, e, m) should use every bit of e, even past what a float holds (doubles round past 2**53)."""
        x = ZI(5, 2)
        m = ZI(7, 0)

        assert pow(x, e + 1, m) == (pow(x, e, m) * x) % m
        assert pow(x, e + 1, m) != pow(x, e, m)

    def test_pow_mod_requires_same_ring(self):
        """Modulus must be in the same QuadraticRing (identity check)."""
        x = ZI(5, 2)
        m_other_ring = Z2(2, 1)

        with pytest.raises(TypeError):
            pow(x, 5, m_other_ring)

    def test_pow_mod_zero_modulus_raises(self):
        """pow(x, e, 0) should raise like Python ints."""
        x = ZI(5, 2)

        with pytest.raises(ZeroDivisionError):
            pow(x, 5, 0)

    def test_splitring_den2_does_not_force_qu_qv_parity(self):
        """The parity constraint for split integer division doesn't apply when den=2 (the default). Verify that"""
        # D=1 defaults to den=2 (maximal-order convention), so this uses SplitRing with den=2.
        Z = QuadraticRing(1)
        assert Z.den == 2

        # Counterexample where the best (qu, qv) in the 3x3 neighborhood has opposite parity.
        # With the current code, SplitRing.divmod forces same parity and picks a worse remainder.
        x = Z(-15, -15)
        y = Z(-15, -13)

        q, r = divmod(x, y)

        # sanity
        assert x == q * y + r

        # measure remainder size in split (u,v) coords: u=(a+b)/den, v=(a-b)/den
        den = Z.den
        ru = (r.a + r.b) // den
        rv = (r.a - r.b) // den

        # The truly-best local choice gives (ru, rv)=(-1, 0) => ru^2+rv^2 == 1.
        # The current parity-forced choice gives (ru, rv)=(-1, -1) => 2.
        assert ru * ru + rv * rv == 1

    @staticmethod
    def _rand_elem(rng: random.Random, Q: QuadraticRing, bound: int) -> QuadInt:
        """Create a random ring element while respecting den=2 parity constraints."""
        a = rng.randint(-bound, bound)
        b = rng.randint(-bound, bound)

        if Q.den == 2 and ((a ^ b) & 1):
            b += 1

        return Q(a, b)

    @pytest.mark.parametrize("D", sorted(NORM_EUCLID_D), ids=str)
    def test_divmod_random_remainder_is_norm_reducing(self, D: int):
        """For supported norm-Euclidean orders, divmod should satisfy x=qy+r and |N(r)| < |N(y)|."""
        Q = QuadraticRing(D)
        rng = random.Random(10_000 + D)

        # Include both medium and larger coefficients to exercise neighborhood expansion.
        for bound in (50, 5000):
            for _ in range(20):
                x = self._rand_elem(rng, Q, bound)
                y = self._rand_elem(rng, Q, bound)

                while not y:
                    y = self._rand_elem(rng, Q, bound)

                q, r = divmod(x, y)

                assert x == q * y + r, f"division identity failed for D={D}, x={x}, y={y}"
                assert abs(abs(r)) < abs(abs(y)), f"non-reducing remainder for D={D}, x={x}, y={y}, r={r}"

    @pytest.mark.parametrize(
        ("D", "xa", "xb", "ya", "yb"),
        [
            # The only norm-reducing quotients for these are out on the hyperbola branches, past the local search
            (19, -14, -9, -14, 0),
            (57, -9, -7, -9, 3),
            (73, -7, -5, -6, 2),
        ],
        ids=str,
    )
    def test_divmod_quotient_on_hyperbola_branch(self, D: int, xa: int, xb: int, ya: int, yb: int):
        """Real norm-Euclidean divmod should still find a norm-reducing remainder when no nearby quotient has one."""
        Q = QuadraticRing(D)
        x = Q(xa, xb)
        y = Q(ya, yb)

        q, r = divmod(x, y)

        assert x == q * y + r
        assert abs(abs(r)) < abs(abs(y))


class TestUnits:
    """Tests for the units"""

    @pytest.mark.parametrize(
        ("x", "expected"),
        [
            (ZI(0, 1), ZI(0, -1)),
            (ZE(-1, 1), ZE(-1, -1)),
            (Z2(3, 2), Z2(3, -2)),
        ],
        ids=str,
    )
    def test_invert_norm_one_units(self, x: QuadInt, expected: QuadInt):
        """The ~ dunder should return the multiplicative inverse for norm-one units."""
        assert abs(x) == 1
        assert ~x == expected
        assert x * ~x == x.one
        assert ~x * x == x.one

    def test_invert_norm_negative_one_unit(self):
        """The ~ dunder should handle units with norm -1."""
        x = Z2(1, 1)

        assert abs(x) == -1
        assert ~x == Z2(-1, 1)
        assert x * ~x == x.one
        assert ~x * x == x.one

    def test_invert_preserves_subclass(self):
        """The ~ dunder should preserve the concrete QuadInt subclass."""
        x = complexint(0, 1)

        assert ~x == complexint(0, -1)
        assert isinstance(~x, complexint)

    def test_invert_non_unit_raises(self):
        """The ~ dunder should reject non-units."""
        x = ZI(1, 1)

        with pytest.raises(ValueError, match="is not a unit"):
            _ = ~x

    def test_units_sizes(self):
        """Verify Gaussian and Eisenstein rings have the correct unit lengths"""
        assert len(Z1(2, 0).units) == 4
        assert len(ZI(1, 0).units) == 4
        assert len(ZE(2, 0).units) == 6

        assert len(Z2(1, 0).units) == 2

    @pytest.mark.parametrize("x", [ZI(3, 2), ZE(5, 1), Z2(7, 3)], ids=str)
    def test_canonical_associate_is_idempotent_and_unit_invariant(self, x: QuadInt):
        """Canonical associate selection should be stable across repeated calls and unit multiples."""
        base = x._canonical_associate()
        assert base == base._canonical_associate()

        for u in x.units:
            y = x * u
            assert y._canonical_associate() == base

    @pytest.mark.parametrize("x", [Z2(7, 3), Z5(9, 1), QuadraticRing(57)(9, 3), QuadraticRing(14)(5, 1)], ids=str)
    def test_canonical_associate_is_invariant_under_the_fundamental_unit(self, x: QuadInt):
        """In real rings every power of the fundamental unit gives another associate, with the same canonical one."""
        eps = x.ring.fundamental_unit()
        base = x._canonical_associate()

        for k in (1, 2, 5):
            assert (x * eps**k)._canonical_associate() == base
            assert (x * (~eps) ** k)._canonical_associate() == base

    @pytest.mark.parametrize(
        "ring",
        [ZI, ZE, ZN2, ZN5, ZN7, QuadraticRing(-3, 1), QuadraticRing(0), Z1, QuadraticRing(1, 1), Z2, Z5, Z15],
        ids=str,
    )
    def test_canonical_associate_has_positive_leading_coefficient(self, ring: QuadraticRing):
        """The canonical associate has a > 0 (or a == 0 and b > 0), so integers come out positive and units as 1."""
        rng = random.Random(9_000 + 10 * ring.D + ring.den)
        assert ring.one._canonical_associate() == ring.one
        assert (-ring.one)._canonical_associate() == ring.one
        assert ring.from_obj(-6)._canonical_associate() == 6

        for _ in range(300):
            a, b = rng.randint(-60, 60), rng.randint(-60, 60)
            if ring.den == 2 and (a ^ b) & 1:
                b += 1
            x = ring(a, b)
            if not x:
                continue

            c = x._canonical_associate()
            assert c.a > 0 or (c.a == 0 and c.b > 0)
            assert c._canonical_associate() == c
            for u in x.units:
                assert (x * u)._canonical_associate() == c

    @pytest.mark.parametrize(
        ("ring", "in_region"),
        [
            (ZI, lambda a, b: a > 0 and b >= 0),  # the first quadrant
            (ZE, lambda a, b: 0 <= b < a),  # 0 <= arg < 60 degrees, since z = a/2 + (b*sqrt(3)/2)*i
            (Z1, lambda a, b: a > 0 and abs(b) <= a),  # split coordinates u = (a + b)/2 and v = (a - b)/2 both >= 0
        ],
        ids=["gaussian", "eisenstein", "split"],
    )
    def test_canonical_associate_region_with_extra_units(self, ring: QuadraticRing, in_region: Callable):
        """With more units than +/-1, exactly one associate of each nonzero element is in a region, and that's it."""
        for a, b in product(range(-12, 13), repeat=2):
            if ring.den == 2 and (a ^ b) & 1:
                continue
            x = ring(a, b)
            if not x:
                continue

            associates = {(w.a, w.b) for w in (x * u for u in x.units)}  # a set, since zero divisors can repeat
            assert sum(1 for a2, b2 in associates if in_region(a2, b2)) == 1

            c = x._canonical_associate()
            assert in_region(c.a, c.b)

    def test_eisenstein_canonical_associate_in_omega_basis(self):
        """In the w-basis x + y*w, the first sextant 0 <= arg < 60 degrees is 0 <= y < x."""
        for x, y in product(range(-9, 10), repeat=2):
            z = eisensteinint(x, y)
            if not z:
                continue

            c = z._canonical_associate()
            assert 0 <= c.omega < c.real

    def test_units_are_units(self):
        """Every element in .units should be a unit."""
        for ring in [ZI, ZE, Z1, Z2, ZN7]:
            one = ring.one
            for u in one.units:
                assert u.is_unit(), f"{u} in {ring} should be a unit"

    def test_zero_is_not_unit(self):
        """Zero is never a unit."""
        assert not complexint(0, 0).is_unit()
        assert not ZE(0, 0).is_unit()

    def test_primes_are_not_units(self):
        """Non-unit elements should return False."""
        assert not complexint(1, 1).is_unit()  # norm 2
        assert not complexint(2, 0).is_unit()  # norm 4
        assert not Z2(3, 1).is_unit()  # norm 7

    def test_gaussian_units(self):
        """Verify the four Gaussian units."""
        assert complexint(1, 0).is_unit()
        assert complexint(-1, 0).is_unit()
        assert complexint(0, 1).is_unit()
        assert complexint(0, -1).is_unit()

    def test_eisenstein_units(self):
        """Verify the six Eisenstein units."""
        # In internal numerator coords (den=2): units are ±1, ±ω, ±ω²
        assert ZE(2, 0).is_unit()  # 1
        assert ZE(-2, 0).is_unit()  # -1
        assert ZE(-1, 1).is_unit()  # ω
        assert ZE(1, -1).is_unit()  # -ω


class TestIrreducible:
    """Tests for QuadInt.is_irreducible"""

    def test_zero_is_not_irreducible(self):
        """The zero element is not irreducible."""
        assert not ZI(0, 0).is_irreducible()

    def test_gaussian_units_are_not_irreducible(self):
        """Gaussian units are not irreducible because irreducible elements must be non-units."""
        assert not ZI(1, 0).is_irreducible()
        assert not ZI(-1, 0).is_irreducible()
        assert not ZI(0, 1).is_irreducible()
        assert not ZI(0, -1).is_irreducible()

    def test_eisenstein_units_are_not_irreducible(self):
        """Eisenstein units are not irreducible."""
        for unit in ZE.elements_with_norm(1):
            assert unit.is_unit()
            assert not unit.is_irreducible()

    def test_prime_norm_element_is_irreducible(self):
        """An element with rational-prime norm is irreducible."""
        z = ZN2(1, 1)

        assert abs(z) == 3
        assert z.is_irreducible()

    def test_rational_two_is_reducible(self):
        """The rational integer 2 factors as (1 + i)(1 - i) in Z[i]."""
        two = ZI(2, 0)

        assert ZI(1, 1) * ZI(1, -1) == two
        assert not two.is_irreducible()

    def test_rational_three_is_irreducible(self):
        """The rational integer 3 is irreducible in Z[i] because a**2 + b**2 = 3 has no solution."""
        three = ZI(3, 0)

        assert abs(three) == 9
        assert not ZI.has_element_with_norm(3)
        assert three.is_irreducible()

    def test_rational_five_is_reducible(self):
        """The rational integer 5 factors as (2 + i)(2 - i) in Z[i]."""
        five = ZI(5, 0)

        assert ZI(2, 1) * ZI(2, -1) == five
        assert not five.is_irreducible()

    def test_composite_product_is_reducible(self):
        """The element 6 is reducible because it is the product of the non-units 2 and 3."""
        six = ZI(6, 0)

        assert ZI(2, 0) * ZI(3, 0) == six
        assert not six.is_irreducible()

    def test_three_is_reducible(self):
        """The rational integer 3 factors as (1 + sqrt(-2))(1 - sqrt(-2))."""
        three = ZN2(3, 0)

        assert ZN2(1, 1) * ZN2(1, -1) == three
        assert not three.is_irreducible()

    def test_five_is_irreducible(self):
        """The rational integer 5 is irreducible in Z[sqrt(-2)] because there is no element of norm 5."""
        five = ZN2(5, 0)

        assert abs(five) == 25
        assert not ZN2.has_element_with_norm(5)
        assert five.is_irreducible()

    def test_seven_is_irreducible(self):
        """The rational integer 7 is irreducible in Z[sqrt(-2)] because a**2 + 2*b**2 = 7 has no solution."""
        seven = ZN2(7, 0)

        assert abs(seven) == 49
        assert not ZN2.has_element_with_norm(7)
        assert seven.is_irreducible()

    def test_two_is_irreducible_in_z_sqrt_minus_five(self):
        """The element 2 is irreducible in Z[sqrt(-5)] because there are no elements of norm 2 or -2."""
        two = ZN5(2, 0)

        assert abs(two) == 4
        assert not ZN5.has_element_with_norm(2)
        assert not ZN5.has_element_with_norm(-2)
        assert two.is_irreducible()

    def test_three_is_irreducible_in_z_sqrt_minus_five(self):
        """The element 3 is irreducible in Z[sqrt(-5)] because there are no elements of norm 3 or -3."""
        three = ZN5(3, 0)

        assert abs(three) == 9
        assert not ZN5.has_element_with_norm(3)
        assert not ZN5.has_element_with_norm(-3)
        assert three.is_irreducible()

    def test_six_has_two_distinct_factorizations(self):
        """The classic equality 2*3 = (1 + sqrt(-5))*(1 - sqrt(-5)) holds in Z[sqrt(-5)]."""
        assert ZN5(2, 0) * ZN5(3, 0) == ZN5(6, 0)
        assert ZN5(1, 1) * ZN5(1, -1) == ZN5(6, 0)

    def test_composite_unknown_case_raises(self):
        """Unsupported composite-norm cases should raise instead of guessing incorrectly."""
        six = ZN5(6, 0)

        assert abs(six) == 36

        with pytest.raises(NotImplementedError):
            six.is_irreducible()

    def test_unsupported_composite_norm_raises(self):
        """An unsupported composite-norm case should raise instead of returning a false mathematical answer."""
        z = ZN6(4, 1)

        assert abs(z) == 22

        with pytest.raises(NotImplementedError):
            z.is_irreducible()

    def test_unsupported_prime_norm_still_returns_true(self):
        """A prime norm is a ring-independent certificate of irreducibility."""
        z = ZN6(1, 1)

        assert abs(z) == 7
        assert z.is_irreducible()


class TestIndex:
    """Tests for __index__ (int conversion)"""

    def test_pure_integer(self):
        """A QuadInt with b=0 should convert to int."""
        assert int(complexint(5, 0)) == 5
        assert int(ZE(6, 0)) == 3  # den=2, so 6/2 = 3
        assert int(Z2(7, 0)) == 7

    def test_negative(self):
        """Negative pure integers should convert correctly."""
        assert int(complexint(-3, 0)) == -3

    def test_zero(self):
        """Zero should convert to 0."""
        assert int(complexint(0, 0)) == 0

    def test_non_integer_raises(self):
        """A QuadInt with b!=0 should raise TypeError."""
        with pytest.raises(TypeError):
            int(complexint(1, 2))

        with pytest.raises(TypeError):
            int(Z2(3, 1))


class TestFloat:
    """Tests for __float__ conversion"""

    def test_pure_integer(self):
        """A QuadInt with b=0 should convert to float."""
        assert isclose(float(complexint(5, 0)), 5.0)
        assert isclose(float(ZE(6, 0)), 3.0)  # den=2

    def test_non_integer_raises(self):
        """A QuadInt with b!=0 should raise TypeError."""
        with pytest.raises(TypeError):
            float(complexint(1, 2))


class TestComplex:
    """Tests for __complex__ conversion"""

    def test_gaussian(self):
        """Gaussian integers should convert to complex."""
        assert complex(complexint(3, 4)) == (3 + 4j)  # ruff: ignore[float-equality-comparison]
        assert complex(complexint(-1, 0)) == (-1 + 0j)  # ruff: ignore[float-equality-comparison]
        assert complex(complexint(0, -2)) == -2j  # ruff: ignore[float-equality-comparison]

    def test_non_gaussian_raises(self):
        """Non-Gaussian quadratic integers should raise TypeError."""
        with pytest.raises(TypeError):
            complex(ZE(2, 0))

        with pytest.raises(TypeError):
            complex(Z2(1, 0))

    def test_roundtrip(self):
        """Complex -> complexint -> complex should roundtrip."""
        c = 3 + 7j
        x = complexint.DEFAULT_RING.from_obj(c)
        assert complex(x) == c  # ruff: ignore[float-equality-comparison]


class TestEqualityWithNumbers:
    """Tests for __eq__ and __hash__ against plain Python numbers"""

    @pytest.mark.parametrize(
        ("x", "n"),
        [
            (complexint(5, 0), 5),
            (ZI(-4, 0), -4),
            (ZE(6, 0), 3),  # den=2
            (Z5(-2, 0), -1),  # den=2, and -1 is the one int whose hash is not itself
            (Z2(0, 0), 0),
            (Z1(8, 0), 4),  # split integers, den=2
            (ZN5(7, 0), 7),
            (Z2(-6, 0), -6),
        ],
        ids=repr,
    )
    def test_integers_equal_and_hash_like_int(self, x: QuadInt, n: int):
        """A plain integer equals the same int, float, and complex, and must hash like them to be the same dict key."""
        for number in (n, float(n), complex(n, 0)):
            assert x == number
            assert number == x
            assert hash(x) == hash(number)

        assert {n: "int"}[x] == "int"
        assert len({x, n, float(n), complex(n, 0)}) == 1

    @pytest.mark.parametrize(("a", "b"), [(1, 2), (-3, 5), (0, -7), (2**52 + 1, -(2**50)), (-(2**53), 2**53)], ids=str)
    def test_gaussian_equals_and_hashes_like_complex(self, a: int, b: int):
        """Gaussian integers equal the matching complex, so they must hash like it too."""
        x = complexint(a, b)
        c = complex(a, b)

        assert x == c
        assert hash(x) == hash(c)
        assert {c: "complex"}[x] == "complex"

    def test_gaussian_hash_keeps_complex_minus_one_rule(self):
        """CPython never returns -1 as a hash (it signals an error), so a complex that would hash to -1 hashes to -2."""
        c = complex(-1000004, 1)  # hash(-1000004) + 1000003 * hash(1) == -1
        assert hash(c) == -2
        assert hash(complexint(-1000004, 1)) == -2

    @pytest.mark.parametrize(
        ("x", "number"),
        [
            (complexint(1), 1.9),  # arithmetic truncates floats with int(), but equality is exact
            (complexint(1), 1.5 + 2j),
            (complexint(1, 2), 1 + 2.5j),
            (ZE(2, 0), 1.25),
            (complexint(1), float("inf")),
            (complexint(0), float("nan")),
            (Z2(0, 1), 1j),  # only the Gaussian integers equal a complex off the real axis
            (ZN5(1, 1), 1 + 1j),
            (ZE(1, 1), 0.5),
            (Z2(3, 1), 3),
        ],
        ids=repr,
    )
    def test_unequal_numbers(self, x: QuadInt, number: complex):
        """Equality with Python numbers is exact, and never raises (which would break `in` checks on lists)."""
        assert x != number
        assert number != x


class TestContent:
    """Tests for the content method"""

    def test_zero(self):
        """Use the standard content(0)=0 convention."""
        assert complexint(0, 0).content() == 0

    def test_gaussian(self):
        """Verify content is produced correctly for Gaussian rings"""
        x = complexint(4, 53)
        assert x.content() == 1

        y = complexint(6, 0)
        assert y.content() == 6

    def test_parity_matters(self):
        """Verify content is parity-aware and takes the den value into account"""
        # D=-3 defaults to den=2
        assert ZE.den == 2

        one = ZE(2, 0)  # (2+0*sqrt(-3))/2 == 1
        two = ZE(4, 0)  # == 2
        three = ZE(6, 0)  # == 3

        # You can't pull out a factor of 2 from "1" in den=2 ring because it breaks parity.
        assert one.content() == 1
        # But you can pull out 2 from "2"
        assert two.content() == 2
        # And you can pull out 3 from "3"
        assert three.content() == 3

    @pytest.mark.parametrize("k", [-7, -2, -1, 1, 2, 9])
    def test_content_scales(self, k: int):
        """Verify content(k*x)=|k|*content(x) for Gaussian integers."""
        x = complexint(4, 6)
        assert x.content() == 2

        y = x * k
        assert y.content() == abs(k) * x.content()

    @pytest.mark.parametrize("k", [-7, -2, -1, 1, 2, 9])
    def test_content_scales_den2(self, k: int):
        """Verify content(k*x)=|k|*content(x) in den=2 rings too."""
        x = ZE(4, 0)  # == 2, content 2
        assert x.content() == 2

        y = x * k
        assert y.content() == abs(k) * x.content()

    @pytest.mark.parametrize(
        "x",
        [
            complexint(4, 6),
            complexint(6, 0),
            ZE(4, 2),
            ZE(6, 6),
        ],
    )
    def test_content_divides_numerators(self, x: QuadInt):
        """Verify the returned content divides stored ring numerators."""
        c = x.content()
        assert c > 0
        assert x.a % c == 0
        assert x.b % c == 0

    @pytest.mark.skipif(
        os.getenv("CI", "").lower() in {"1", "true", "yes"},
        reason="Local-only test",
    )
    @pytest.mark.parametrize(
        "x",
        [
            complexint(1_999_966_000_289, 999_983_000_138),
            complexint(2_000_000_000_000, 1_500_000_000_000),
            ZE(1_800_000_000_000, 600_000_000_000),
            ZE(2_400_000_000_000, 1_200_000_000_000),
        ],
        ids=str,
    )
    def test_content_matches_reference_large_values(self, x: QuadInt):
        """Large-value regression: optimized content matches the old divisor-scan logic."""
        assert x.content() == brute_content(x)
