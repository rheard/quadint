from __future__ import annotations

import importlib.util
import os
import random

from math import prod
from pathlib import Path

import pytest

import quadint

from quadint import Ideal, QuadInt, complexint
from quadint.quad import Factorization, QuadraticRing
from quadint.quad.rings import HarperRing
from tests.quad.test_int import QuadIntTests

ZN19 = QuadraticRing(-19)
ZN17 = QuadraticRing(-17)
ZN11 = QuadraticRing(-11)
ZN7 = QuadraticRing(-7)
ZN5 = QuadraticRing(-5)
ZN2 = QuadraticRing(-2)
Z1 = QuadraticRing(1)
Z2 = QuadraticRing(2)
Z5 = QuadraticRing(5)
Z15 = QuadraticRing(15)
Z69 = QuadraticRing(69)

ZI = QuadraticRing(-1)
ZE = QuadraticRing(-3)


requires_cypari = pytest.mark.skipif(
    importlib.util.find_spec("cypari") is None,
    reason="requires cypari",
)


def ideal_prod(ring: QuadraticRing, factors: tuple[Ideal, ...]) -> Ideal:
    """Return the product of a tuple of ideals."""
    return prod(factors, start=ring.unit_ideal())


def norm_multiset(primes: dict[QuadInt, int]) -> list[int]:
    """Return sorted list of norms with multiplicity."""
    out: list[int] = []
    for p, k in primes.items():
        out.extend([abs(p)] * k)
    out.sort()
    return out


@pytest.mark.skipif(
    os.getenv("CI", "").lower() not in {"1", "true", "yes"},
    reason="Compiled-only test",
)
def test_compiled_tests():
    """Verify that we are running these tests with a compiled version of quadint"""
    path = Path(quadint.__file__)
    assert path.suffix.lower() != ".py"


class RingTests:
    """Support methods for testing QuadraticRing singleton behavior"""

    @staticmethod
    def assert_same_ring_obj(a: QuadraticRing, b: QuadraticRing):
        """Validate rings are literally the same object and have same parameters"""
        assert a.D == b.D
        assert a.den == b.den
        assert a is b

    @staticmethod
    def assert_diff_ring_obj(a: QuadraticRing, b: QuadraticRing):
        """Validate rings are different objects (identity differs)"""
        assert a is not b


class TestQuadraticRing(RingTests):
    """Tests for the QuadraticRing behavior"""

    def test_same_instance_default_den(self):
        """QuadraticRing(D) should be a singleton per (D, default_den)"""
        q1 = QuadraticRing(-1)
        q2 = QuadraticRing(-1)
        self.assert_same_ring_obj(q1, q2)

    def test_same_instance_none_den(self):
        """QuadraticRing(D) and QuadraticRing(D, None) should be the same object"""
        q1 = QuadraticRing(5)
        q2 = QuadraticRing(5, None)
        self.assert_same_ring_obj(q1, q2)
        assert q1.den == 2  # 5 % 4 == 1 -> default den=2

    def test_same_instance_explicit_default_den(self):
        """QuadraticRing(D, den=<computed default>) should match QuadraticRing(D)"""
        # D=-3 -> default den=2
        q1 = QuadraticRing(-3)
        q2 = QuadraticRing(-3, 2)
        self.assert_same_ring_obj(q1, q2)

        # D=-1 -> default den=1
        q3 = QuadraticRing(-1)
        q4 = QuadraticRing(-1, 1)
        self.assert_same_ring_obj(q3, q4)

    def test_different_den(self):
        """QuadraticRing(D, den=...) must singleton by (D, den) key"""
        q_default = QuadraticRing(-3)  # default den=2
        q_other = QuadraticRing(-3, 1)  # non-maximal order (or at least non-default)
        self.assert_diff_ring_obj(q_default, q_other)

        assert q_default.D == q_other.D
        assert q_default.den != q_other.den

    def test_different_D(self):
        """Different D must produce different ring objects"""
        q1 = QuadraticRing(-1)
        q2 = QuadraticRing(-2)
        self.assert_diff_ring_obj(q1, q2)

    @pytest.mark.parametrize("den", [0, 3, -1])
    def test_invalid_den_rejected(self, den: int):
        """Only den=1 or den=2 are supported by QuadraticRing."""
        with pytest.raises(ValueError, match="den must be 1 or 2"):
            _ = QuadraticRing(5, den)

    def test_repeated_calls_do_not_mutate_cached_instance(self):
        """Repeated construction should not corrupt cached fields"""
        q = QuadraticRing(1)  # default den=2
        assert q.D == 1
        assert q.den == 2

        # Repeat a bunch of times; should remain stable
        for _ in range(100):
            q2 = QuadraticRing(1, None)
            self.assert_same_ring_obj(q, q2)

    def test_registered_subclasses(self):
        """
        Validate that subclasses of QuadInt are registered with their default rings,
            and they use that type successfully.
        """
        assert isinstance(ZI(1, 2), complexint)
        assert ZI.DEFAULT_KLASS is complexint

        assert isinstance(ZI.one, complexint)

        # Now verify complexint isn't returned for the non-default ring
        non_default_ZI = QuadraticRing(-1, 2)
        assert non_default_ZI.DEFAULT_KLASS is not complexint

        a = non_default_ZI(2, 4)
        assert not isinstance(a, complexint)
        assert not isinstance(non_default_ZI.one, complexint)


class TestRingCapabilities:
    """Tests for lightweight capability helpers on QuadraticRing."""

    @pytest.mark.parametrize(
        ("ring", "expected"),
        [
            (QuadraticRing(-1), True),
            (QuadraticRing(-7), True),
            (QuadraticRing(0), True),
            (QuadraticRing(1), True),
            (QuadraticRing(15), False),
            (QuadraticRing(16), False),
            (QuadraticRing(-17), False),
            (QuadraticRing(-19), False),
            (QuadraticRing(-3, 1), False),
        ],
        ids=str,
    )
    def test_supports_division(self, ring: QuadraticRing, *, expected: bool):
        """supports_division should mirror whether this ring has a divmod implementation."""
        assert ring.supports_division() is expected

    def test_supports_division_matches_runtime_behavior(self):
        """Unsupported rings should raise NotImplementedError from division operations."""
        supported = QuadraticRing(-1)
        unsupported = QuadraticRing(15)

        assert supported.supports_division() is True
        assert unsupported.supports_division() is False

        x = supported(5, 2)
        y = supported(3, -1)
        q, r = divmod(x, y)
        assert x == q * y + r

        with pytest.raises(NotImplementedError):
            _ = divmod(unsupported(5, 2), unsupported(3, -1))

    @pytest.mark.parametrize(
        ("ring", "expected"),
        [
            (QuadraticRing(-1), True),
            (QuadraticRing(-2), True),
            (QuadraticRing(-3), True),
            (QuadraticRing(-7), True),
            (QuadraticRing(-11), True),
            (QuadraticRing(-19), True),
            (QuadraticRing(2), False),
            (QuadraticRing(-3, 1), False),
            (QuadraticRing(-17), False),
        ],
        ids=str,
    )
    def test_supports_factorization(self, ring: QuadraticRing, *, expected: bool):
        """supports_factorization should mirror whether this ring has factorization support."""
        assert ring.supports_factorization() is expected

    def test_supports_factorization_matches_runtime_behavior(self):
        """Unsupported rings should raise NotImplementedError from factorization operations."""
        supported = QuadraticRing(-1)
        unsupported = QuadraticRing(2)

        assert supported.supports_factorization() is True
        assert unsupported.supports_factorization() is False

        factors = supported(5, 2).factor_detail()
        assert isinstance(factors, Factorization)

        with pytest.raises(NotImplementedError):
            _ = unsupported(5, 2).factor_detail()


class TestIdentityChecksWithQuadInt(RingTests):
    """Tests that rely on QuadInt.assert_same_ring using identity"""

    def test_elements_from_separate_ring_construction_can_mix(self):
        """If ring is cached, elements built via separate QuadraticRing(D) calls must interoperate."""
        Q1 = QuadraticRing(-1)
        Q2 = QuadraticRing(-1)

        # If caching is broken, this would raise TypeError in assert_same_ring (identity mismatch)
        a = QuadInt(1, 2, Q1)
        b = QuadInt(3, 4, Q2)

        c = a + b
        assert isinstance(c, QuadInt)
        assert c.a == 4
        assert c.b == 6
        assert c.ring is Q1  # result keeps self.ring

    def test_elements_from_different_den_do_not_mix(self):
        """Identity checks should still protect against mixing different rings (even with same D)."""
        Q_default = QuadraticRing(-3)  # den=2
        Q_other = QuadraticRing(-3, 1)  # den=1

        a = QuadInt(2, 0, Q_default)  # ok parity for den=2
        b = QuadInt(1, 0, Q_other)

        with pytest.raises(TypeError):
            _ = a + b


def _rand_elem(rng: random.Random, Q: QuadraticRing, bound: int) -> QuadInt:
    """Create a random ring element while respecting den=2 parity constraints."""
    a = rng.randint(-bound, bound)
    b = rng.randint(-bound, bound)

    if Q.den == 2 and ((a ^ b) & 1):
        b += 1

    return Q(a, b)


class TestClark69EuclideanFunction:
    """Unit tests for the Euclidean function used by the D=69 division algorithm."""

    def test_phi69_basic_integer_values(self):
        """phi(0)=0, phi(1)=1, and the only tweak is 23 -> 26 (multiplicatively)."""
        assert Z69._phi_from_abs_norm(0) == 0
        assert Z69._phi_from_abs_norm(1) == 1
        assert Z69._phi_from_abs_norm(-1) == 1

        assert Z69._phi_from_abs_norm(23) == 26
        assert Z69._phi_from_abs_norm(-23) == 26

        assert Z69._phi_from_abs_norm(23 * 23) == 26 * 26
        assert Z69._phi_from_abs_norm(11) == 11
        assert Z69._phi_from_abs_norm(23 * 11) == 26 * 11

    def test_phi69_matches_known_prime_above_23(self):
        """
        In Z[(1+sqrt(69))/2], the elements (23 ± 3*sqrt(69))/2 have norm -23.
        So phi should map them to 26.
        """
        assert Z69.den == 2

        p = Z69(23, 3)
        assert abs(p) == -23
        assert Z69.phi(p) == 26

        p_conj = p.conjugate()
        assert abs(p_conj) == -23
        assert Z69.phi(p_conj) == 26

    def test_phi69_sign_and_conjugation_invariant(self):
        """Phi depends only on |N(x)|, so it's invariant under x -> -x and conjugation."""
        rng = random.Random(69_000)
        for _ in range(200):
            x = _rand_elem(rng, Z69, 10_000)
            assert Z69.phi(x) == Z69.phi(-x)
            assert Z69.phi(x) == Z69.phi(x.conjugate())

    def test_phi69_is_multiplicative(self):
        """Sanity: phi(xy) == phi(x)*phi(y) because |N| is multiplicative and v23 adds."""
        rng = random.Random(69_001)
        for _ in range(200):
            x = _rand_elem(rng, Z69, 2_000)
            y = _rand_elem(rng, Z69, 2_000)
            if not x or not y:
                continue
            assert Z69.phi(x * y) == Z69.phi(x) * Z69.phi(y)


class TestDivClark69:
    """Integration tests for the D=69 division algorithm."""

    def test_ring69_supports_division(self):
        """Once the D=69 ring override is installed, QuadraticRing(69) must support divmod."""
        assert Z69.den == 2
        assert Z69.supports_division() is True

    def test_divmod_zero_divisor_raises(self):
        """Division by 0 should raise."""
        x = Z69(10, 2)
        with pytest.raises(ZeroDivisionError):
            divmod(x, Z69.zero)

    def test_divmod_random_remainder_is_phi_reducing(self):
        """
        For a Euclidean function phi, divmod must satisfy x=qy+r and phi(r) < phi(y) (or r==0).
        Mirrors your existing norm-reduction randomized test, but uses phi69.
        """
        rng = random.Random(69_123)

        for bound in (50, 5000):
            for _ in range(40):
                x = _rand_elem(rng, Z69, bound)
                y = _rand_elem(rng, Z69, bound)

                while not y:
                    y = _rand_elem(rng, Z69, bound)

                # Skip torsion units; Euclidean condition would force remainder 0 anyway.
                if abs(abs(y)) == 1:
                    continue

                q, r = divmod(x, y)

                assert x == q * y + r, f"division identity failed: x={x}, y={y}, q={q}, r={r}"

                # Membership / parity sanity in den=2 ring.
                assert q.ring is Z69
                assert r.ring is Z69
                assert ((q.a ^ q.b) & 1) == 0
                assert ((r.a ^ r.b) & 1) == 0

                assert Z69.phi(r) < Z69.phi(y), f"phi did not reduce: x={x}, y={y}, q={q}, r={r}"

    @requires_cypari
    def test_harper_ring_similar(self):
        """While the Harper ring is not required to produce identical results, some are (and logically should be)"""
        # The only real problem with this is we NEED cypari to compute the admissible pairs in real time.
        #   They shouldn't be in the hardcoded list because then it may affect the subclass mechanism.
        #   Fwiw, the admissible pair that would be hardcoded is: ((-5-1*sqrt(69))/2, (-5-3*sqrt(69))/2)
        H69 = HarperRing(69)
        a1 = Z69(69 * 2, 420)
        a2 = H69(69 * 2, 420)

        b1 = Z69(13, 37)
        b2 = H69(13, 37)

        q1, r1 = divmod(a1, b1)
        q2, r2 = divmod(a2, b2)

        # Note that because we're forcing different rings, equality breaks down because we rely on instance checks.
        #   As long as users just stick to using QuadraticRing(69) then this won't be a problem for them...
        assert q1.a == q2.a
        assert q1.b == q2.b
        assert r1.a == r2.a
        assert r1.b == r2.b

    def test_clark_regression_pair_phi_fix(self):
        """
        Regression inspired by the standard D=69 obstruction near primes over 23.

        Let
            y = (23 + 3*sqrt(69))/2   with |N(y)| = 23  and phi(y)=26,
            x = 18 + 2*sqrt(69).

        The naive q=1 step gives remainder r0 = x - y = (13 + sqrt(69))/2 with |N(r0)|=25,
        which is not norm-reducing (25 > 23), but is phi-reducing (25 < 26).
        """
        y = Z69(23, 3)
        x = Z69(36, 4)  # == 18 + 2*sqrt(69)

        # Verify the "naive" step math.
        r0 = x - y
        assert abs(abs(r0)) == 25
        assert abs(abs(y)) == 23
        assert Z69.phi(r0) == 25
        assert Z69.phi(y) == 26
        assert Z69.phi(r0) < Z69.phi(y)

        # And verify divmod actually produces a phi-reducing remainder.
        q, r = divmod(x, y)
        assert x == q * y + r
        assert Z69.phi(r) < Z69.phi(y)

    def test_euclid_loop_terminates_and_phi_strictly_decreases(self):
        """A tiny Euclidean-algorithm loop should strictly decrease phi and terminate quickly."""
        a = Z69(36, 4)  # 18 + 2*sqrt(69)
        b = Z69(23, 3)  # (23 + 3*sqrt(69))/2

        steps = 0
        while b:
            q, r = divmod(a, b)
            assert a == q * b + r
            assert Z69.phi(r) < Z69.phi(b)

            a, b = b, r
            steps += 1
            assert steps < 200, "Euclidean descent did not terminate (phi not strictly decreasing?)"

        # a is the last nonzero remainder: it must divide the original inputs.
        x0 = Z69(36, 4)
        assert (x0 % a) == 0


class TestHarperHelpers:
    """Tests for the standalone helper functions used by HarperRing."""

    @pytest.mark.parametrize(
        ("D", "den", "expected_disc"),
        [
            (14, None, 56),  # default den=1 -> disc = 4D
            (14, 1, 56),
            (14, 2, 14),  # explicit non-default order
            (23, None, 92),
            (23, 1, 92),
            (61, None, 61),  # default den=2 since 61 % 4 == 1
            (61, 2, 61),
            (61, 1, 244),  # explicit non-default order
            (69, None, 69),
            (69, 2, 69),
            (69, 1, 276),
        ],
        ids=str,
    )
    def test_disc_from_D_den(self, D: int, den: int | None, expected_disc: int):
        """Verify discriminant helper follows the den=1 vs den=2 convention."""
        assert QuadraticRing(D, den).discriminant() == expected_disc


@requires_cypari
class TestHarperPariHelpers:
    """Tests for PARI-backed helpers used by HarperRing."""

    @pytest.mark.parametrize(
        ("D", "den"),
        # Skip any principal generators
        [k for k, v in HarperRing._HARDCODED.items() if len(v) == 4],
        ids=str,
    )
    def test_find_admissible_witness_pair_known_cases(self, D: int, den: int):
        """Known Harper cases should yield an admissible prime pair via PARI search."""
        r = QuadraticRing(D, den)
        out = r._find_admissible_witness_primes()

        assert out is not None, f"expected admissible pair for D={D}, den={den}"
        assert isinstance(out, tuple)
        assert len(out) == 4
        assert all(isinstance(v, int) for v in out)

        p1, _, p2, _ = out
        assert HarperRing._HARDCODED[D, den] == out
        assert p1 > 1
        assert p2 > 1
        assert p1 != p2

    @pytest.mark.parametrize(
        ("D", "den"),
        # Only principal generators
        [k for k, v in HarperRing._HARDCODED.items() if len(v) == 2],
        ids=str,
    )
    def test_principal_generators_known_cases(self, D: int, den: int):
        """Known Harper cases should yield principal generators via PARI-backed search."""
        r = QuadraticRing(D, den)
        out = r._principal_generators_from_witness(r._find_admissible_witness_primes())

        assert out is not None, f"expected admissible pair for D={D}, den={den}"
        assert isinstance(out, tuple)
        assert len(out) == 2
        assert all(isinstance(v, QuadInt) for v in out)
        assert HarperRing._HARDCODED[D, den] == out

        p1, p2 = out
        assert p1 != p2


class TestPrimeIdealsOver:
    """Tests for QuadraticRing.prime_ideals_over."""

    @pytest.mark.parametrize(
        ("ring", "p", "expected_norms"),
        [
            (ZI, 5, (5, 5)),  # split in Z[i]
            (ZI, 3, (9,)),  # inert in Z[i]
            (ZI, 2, (2,)),  # ramified in Z[i]
            (ZN7, 2, (2, 2)),  # split in den=2 imaginary order
            (QuadraticRing(10), 3, (3, 3)),  # split in a real quadratic order
            (QuadraticRing(10), 7, (49,)),  # inert in a real quadratic order
            (Z5, 5, (5,)),  # ramified in a den=2 real quadratic order
        ],
        ids=str,
    )
    def test_decomposition(self, ring: QuadraticRing, p: int, expected_norms: tuple[int, ...]):
        """Prime ideals over p should have the expected splitting type and reconstruct (p)."""
        ideals = ring.prime_ideals_over(p)

        assert tuple(ideal.norm for ideal in ideals) == expected_norms
        assert all(ideal.is_prime() for ideal in ideals)
        assert all(p in ideal for ideal in ideals)

        if len(ideals) == 1 and ideals[0].norm == p:
            assert ideals[0] ** 2 == ring.ideal(p)
        else:
            assert ideal_prod(ring, ideals) == ring.ideal(p)

    @pytest.mark.parametrize(
        ("ring", "p"),
        [
            (ZI, 5),
            (ZN7, 2),
            (QuadraticRing(10), 3),
            (QuadraticRing(10), 7),
            (Z5, 5),
        ],
        ids=str,
    )
    def test_roots(self, ring: QuadraticRing, p: int):
        """Prime ideals over p should correspond to roots of the integral-basis minimal polynomial mod p."""
        roots = set()
        for ideal in ring.prime_ideals_over(p):
            _, b, c = ideal.hnf
            if c == 1:
                roots.add((-b) % p)

        if ring.den == 1:
            expected_roots = {r for r in range(p) if (r * r - ring.D) % p == 0}
        else:
            constant = (1 - ring.D) // 4
            expected_roots = {r for r in range(p) if (r * r - r + constant) % p == 0}

        assert roots == expected_roots

    @pytest.mark.parametrize("p", [-3, 0, 1, 4, 9], ids=str)
    def test_invalid(self, p: int):
        """Only rational primes should be accepted."""
        with pytest.raises(ValueError, match="prime"):
            ZI.prime_ideals_over(p)


class TestClassNumber:
    """Tests for QuadraticRing.class_number."""

    @pytest.mark.parametrize(
        "ring",
        [
            ZI,
            ZE,
            ZN2,
            ZN7,
            ZN11,
            ZN19,
        ],
        ids=str,
    )
    def test_imaginary_class_number_one(self, ring: QuadraticRing):
        """The standard Heegner/PID examples should have class number one."""
        assert ring.class_number == 1
        assert len(ring.class_group) == 1

    def test_imaginary_class_number_two(self):
        """Z[sqrt(-5)] should have class number two."""
        assert ZN5.class_number == 2
        assert len(ZN5.class_group) == 2

    @pytest.mark.parametrize(
        ("D", "den"),
        HarperRing._HARDCODED,
        ids=str,
    )
    def test_harper_hardcoded_class_number_one(self, D: int, den: int):
        """Every hardcoded Harper ring should have class number one."""
        assert QuadraticRing(D, den).class_number == 1

    @pytest.mark.parametrize(
        "ring",
        [
            QuadraticRing(0),
            QuadraticRing(1),
            QuadraticRing(4),
        ],
        ids=str,
    )
    def test_non_field_rejected(self, ring: QuadraticRing):
        """Class numbers should not be defined for dual, split, or positive-square rings."""
        with pytest.raises(NotImplementedError, match="quadratic field"):
            _ = ring.class_number

    @pytest.mark.parametrize(
        ("D", "den", "expected"),
        [
            (14, 1, True),  # Q(sqrt(14)) maximal order
            (61, 2, True),  # Q(sqrt(61)) maximal order
            (69, 2, True),  # Q(sqrt(69)) maximal order
            (15, 1, False),  # Q(sqrt(15)) has class number > 1
        ],
        ids=str,
    )
    def test_class_number_is_one(self, D: int, den: int, expected: bool):  # noqa: FBT001
        """Verify the class-number helper on a few known real quadratic discriminants."""
        assert (QuadraticRing(D, den).class_number == 1) is expected


class TestFundamentalUnit:
    """Tests QuadraticRing.fundamental_unit."""

    def test_den_one(self):
        """The fundamental unit in Z[sqrt(2)] is the smallest unit greater than 1."""
        ring = QuadraticRing(2)

        unit = ring.fundamental_unit()

        assert unit == ring(1, 1)
        assert abs(unit) == -1

    def test_den_two(self):
        """The fundamental unit respects the half-integral basis when den is 2."""
        ring = QuadraticRing(5)

        unit = ring.fundamental_unit()

        assert unit == ring(1, 1)
        assert abs(unit) == -1

    def test_can_have_norm_one(self):
        """Some real quadratic rings have fundamental unit of norm 1."""
        ring = QuadraticRing(3)

        unit = ring.fundamental_unit()

        assert unit == ring(2, 1)
        assert abs(unit) == 1

    def test_uses_order_not_just_field(self):
        """The non-maximal order Z[sqrt(5)] has a different fundamental unit than O_Q(sqrt(5))."""
        ring = QuadraticRing(5, den=1)

        unit = ring.fundamental_unit()

        assert unit == ring(2, 1)
        assert abs(unit) == -1

    def test_is_cached(self):
        """The cached method returns the same object on repeated access."""
        ring = QuadraticRing(69)

        assert ring.fundamental_unit() is ring.fundamental_unit()


class TestHarperAcceptOverride(RingTests):
    """Tests for HarperRing selection logic and hardcoded coverage."""

    @pytest.mark.parametrize(
        ("D", "den"),
        HarperRing._HARDCODED,
        ids=str,
    )
    def test_hardcoded_cases_are_selected(self, D: int, den: int):
        """QuadraticRing should select HarperRing for every hardcoded Harper case."""
        Q = QuadraticRing(D, den)
        assert isinstance(Q, HarperRing)
        assert Q.supports_division() is True

    @pytest.mark.parametrize(
        ("D", "den"),
        [(14, 2), (22, 2), (23, 2), (61, 1)],
        ids=str,
    )
    def test_nonmax_orders_not_selected(self, D: int, den: int):
        """HarperRing should only apply to the maximal order (default denominator)."""
        default_den = 2 if (D % 4) == 1 else 1
        assert den != default_den

        Q = QuadraticRing(D, den)
        assert not isinstance(Q, HarperRing)

    @pytest.mark.parametrize(
        ("D", "den", "default_den"),
        [
            (0, 1, 1),
            (-14, 1, 1),
            (-61, 2, 2),
            (14, 2, 1),  # wrong denominator for maximal order
        ],
        ids=str,
    )
    def test_accept_override_rejects_wrong_domain(self, D: int, den: int, default_den: int):
        """HarperRing.accept_override should reject non-real or non-maximal inputs."""
        assert HarperRing.accept_override(D, den, default_den) is False

    def test_hardcoded_subset_contains_expected_literature_values(self):
        """The hardcoded list should include the standard Harper/Conrad examples."""
        expected = {(14, 1), (22, 1), (23, 1), (61, 2)}
        assert expected.issubset(HarperRing._HARDCODED)


class TestHarperDiv:
    """Tests for HarperRing.divmod and Euclidean reduction on hardcoded Harper rings."""

    @staticmethod
    def _rand_elem(rng: random.Random, Q: QuadraticRing, bound: int) -> QuadInt:
        """Create a random ring element while respecting den=2 parity constraints."""
        a = rng.randint(-bound, bound)
        b = rng.randint(-bound, bound)

        if Q.den == 2 and ((a ^ b) & 1):
            b += 1

        return Q(a, b)

    @pytest.mark.parametrize(
        ("D", "den"),
        HarperRing._HARDCODED,
        ids=str,
    )
    def test_zero_divisor(self, D: int, den: int):
        """Division by zero should raise ZeroDivisionError."""
        Q = QuadraticRing(D, den)
        x = Q(7, 3)
        with pytest.raises(ZeroDivisionError):
            _ = divmod(x, Q.zero)

    @pytest.mark.parametrize(
        ("D", "den", "xa", "xb", "ya", "yb"),
        [
            (14, 1, 69, 420, 13, 37),
            (22, 1, 91, 315, 11, 17),
            (23, 1, 77, 221, 9, 10),
            (61, 2, 69, 421, 13, 37),  # odd/odd parity for den=2
        ],
        ids=str,
    )
    def test_example_division_identity_and_phi_reduction(
        self,
        D: int,
        den: int,
        xa: int,
        xb: int,
        ya: int,
        yb: int,
    ):
        """A few concrete examples should satisfy x=qy+r and reduce the Euclidean size."""
        Q = QuadraticRing(D, den)
        x = Q(xa, xb)
        y = Q(ya, yb)

        q, r = divmod(x, y)

        assert x == q * y + r
        assert Q.phi(r) < Q.phi(y), f"phi did not decrease for D={D}, x={x}, y={y}, r={r}"

    @pytest.mark.parametrize(
        ("D", "den"),
        HarperRing._HARDCODED,
        ids=str,
    )
    def test_exact_products_divide_with_zero_remainder(self, D: int, den: int):
        """If x = a*b exactly then divmod(x, a) should return (b, 0)."""
        Q = QuadraticRing(D, den)

        # Hand-picked exact factors with valid parity in den=2 case.
        examples = [
            (Q(5, 1), Q(3, 1)),
            (Q(7, -3), Q(9, 1)),
            (Q(11, 5), Q(-3, 7)),
        ]

        for a, b in examples:
            x = a * b
            q, r = divmod(x, a)

            assert r == Q.zero, f"expected exact division remainder 0 for D={D}, x={x}, a={a}"
            assert q == b, f"expected exact quotient b for D={D}, x={x}, a={a}, got q={q}"

            # Also verify __truediv__ integration on exact divisions.
            assert x / a == b

    @pytest.mark.parametrize(
        ("D", "den"),
        HarperRing._HARDCODED,
        ids=str,
    )
    def test_random_phi_reducing(self, D: int, den: int):
        """Randomized regression: Harper divmod should always preserve identity and reduce phi."""
        Q = QuadraticRing(D, den)
        rng = random.Random(70_000 + 100 * D + den)

        # Use both small and medium sizes to exercise quotient-neighborhood searches.
        for bound in (40, 300):
            for _ in range(50):
                x = self._rand_elem(rng, Q, bound)
                y = self._rand_elem(rng, Q, bound)

                while not y:
                    y = self._rand_elem(rng, Q, bound)

                q, r = divmod(x, y)

                assert x == q * y + r, f"division identity failed for D={D}, x={x}, y={y}, q={q}, r={r}"
                assert Q.phi(r) < Q.phi(y), f"non-reducing remainder for D={D}, x={x}, y={y}, r={r}"

    @pytest.mark.parametrize(
        ("D", "den"),
        # I have cached all values below 100.
        #   These are the first couple rings passed 100 that should also work (if cypari is installed).
        [(101, 2), (103, 1)],
        ids=str,
    )
    @requires_cypari
    def test_random_phi_reducing_not_cached(self, D: int, den: int):
        """Randomized regression: Harper divmod should always preserve identity and reduce phi."""
        assert (D, den) not in HarperRing._HARDCODED

        Q = QuadraticRing(D, den)
        assert isinstance(Q, HarperRing)
        rng = random.Random(70_000 + 100 * D + den)

        # Use both small and medium sizes to exercise quotient-neighborhood searches.
        for bound in (40, 300):
            for _ in range(50):
                x = self._rand_elem(rng, Q, bound)
                y = self._rand_elem(rng, Q, bound)

                while not y:
                    y = self._rand_elem(rng, Q, bound)

                q, r = divmod(x, y)

                assert x == q * y + r, f"division identity failed for D={D}, x={x}, y={y}, q={q}, r={r}"
                assert Q.phi(r) < Q.phi(y), f"non-reducing remainder for D={D}, x={x}, y={y}, r={r}"

    def test_valuation_at_generator_counts_exact_powers(self):
        """_valuation_at_generator should count repeated exact divisibility by a generator."""
        Q = QuadraticRing(14, 1)
        assert isinstance(Q, HarperRing)

        pi = Q(5, 1)
        x = pi * pi * pi

        assert Q._valuation_at_generator(x, pi) == 3
        assert Q._valuation_at_generator(Q.one, pi) == 0


class TestExactDivAndDivides(QuadIntTests):
    """Tests for QuadInt.exact_div and QuadInt.divides."""

    @pytest.mark.parametrize(
        ("Q", "y", "q"),
        [
            # Gaussian integers (den=1)
            (ZI, ZI(1, 1), ZI(5, 2)),  # (1+i) divides (5+2i)(1+i)
            (ZI, ZI(2, -1), ZI(-3, 4)),  # random-ish
            # Eisenstein integers (den=2) - good for parity/den regressions
            (ZE, ZE(3, 1), ZE(5, 1)),  # ramified-ish generator and a nontrivial quotient
            (ZE, ZE(2, 2), ZE(7, -1)),  # another den=2 case
            # Real quadratic den=2 with negative norm unit (exercises N<0 branch)
            (Z5, Z5(3, 7), Z5(9, 3)),
            # Non-division ring: exact_div should still work (divmod not implemented)
            (Z15, Z15(5, 1), Z15(7, 2)),
            (ZN17, ZN17(5, 2), ZN17(3, -1)),
        ],
        ids=str,
    )
    def test_exact_div_roundtrip_on_products(self, Q: QuadraticRing, y: QuadInt, q: QuadInt):
        """If we construct x=q*y, then x.exact_div(y) must return q and y.divides(x) must be True."""
        assert y.ring is Q
        assert q.ring is Q
        assert y  # nonzero

        x = q * y

        got = x.exact_div(y)
        assert got == q
        assert got is not None
        assert got.ring is Q
        assert x == got * y

        Ny = abs(y)  # signed norm
        assert abs(Ny) != 1  # avoid units in real quadratic rings

        assert y.divides(x) is True
        assert y.divides(x + Q.one) is False
        assert (x + Q.one).exact_div(y) is None

    def test_exact_div_unit_divides_everything(self):
        """Every torsion unit u should divide every x (exactly)."""
        for Q in (ZI, ZE, Z5, Z15, ZN17):
            x = Q(37, -11)
            for u in x.units:
                q = x.exact_div(u)
                assert q is not None
                assert x == q * u
                assert u.divides(x) is True

    def test_exact_div_works_even_if_divmod_not_supported(self):
        """
        exact_div is independent of Euclidean division support.
        QuadraticRing(15) has supports_division()==False but exact_div should still work.
        """
        assert Z15.supports_division() is False

        y = Z15(3, 0)  # the integer 3
        x = Z15(21, 0)  # the integer 21

        q = x.exact_div(y)
        assert q == Z15(7, 0)
        assert y.divides(x) is True

        assert (x + Z15.one).exact_div(y) is None
        assert y.divides(x + Z15.one) is False

    def test_divides_is_equivalent_to_exact_div_not_none(self):
        """Coherence: y.divides(x) <=> x.exact_div(y) is not None."""
        rng = random.Random(12345)
        for Q in (ZI, ZE, Z5, Z15, ZN17):
            for _ in range(200):
                x = _rand_elem(rng, Q, 200)
                y = _rand_elem(rng, Q, 200)
                if not y:
                    continue

                ok = y.divides(x)
                q = x.exact_div(y)

                assert ok == (q is not None)
                if q is not None:
                    assert x == q * y

    def test_exact_div_rejects_mixed_rings(self):
        """exact_div/divides should still enforce the identity ring check."""
        x = ZI(5, 2)
        y_other = Z2(5, 2)

        with pytest.raises(TypeError):
            _ = x.exact_div(y_other)

        with pytest.raises(TypeError):
            _ = x.divides(y_other)

    def test_exact_div_accepts_int_and_float(self):
        """exact_div/divides should accept int/float inputs via embedding."""
        x = ZI(12, 6)
        assert x.exact_div(2) == ZI(6, 3)
        assert x.exact_div(2.0) == ZI(6, 3)
        assert (2).__class__ is int  # sanity: we're not relying on weird coercions

        y = ZI(3, 0)
        assert y.divides(x) is True
        assert ZI(5, 0).divides(x) is False

    def test_exact_div_den2_parity_rejection_finds_a_witness(self):
        """
        Find an example in a den=2 ring where the field quotient would have wrong parity,
        so exact_div must return None (it enforces the den=2 integrality constraint).
        """
        Q = ZE  # D=-3, den=2
        assert Q.den == 2

        # Search small values for a (x,y) where:
        #  - computed (qa,qb) are integers (pass divisibility-by-norm check)
        #  - but qa,qb have opposite parity (not in the order)
        for ya in range(-9, 10):
            for yb in range(-9, 10):
                if (ya ^ yb) & 1:
                    continue
                y = Q(ya, yb)
                if not y:
                    continue
                N = abs(y)
                if N == 0:
                    continue

                m = abs(N)

                for xa in range(-15, 16):
                    for xb in range(-15, 16):
                        if (xa ^ xb) & 1:
                            continue
                        x = Q(xa, xb)

                        num_a = x.a * y.a - x.b * y.b * Q.D
                        num_b = y.a * x.b - x.a * y.b

                        if (num_a % m) or (num_b % m):
                            continue

                        qa = num_a // m
                        qb = num_b // m
                        if N < 0:
                            qa = -qa
                            qb = -qb

                        # This is the specific failure mode we want to witness:
                        if (qa ^ qb) & 1:
                            assert x.exact_div(y) is None
                            assert y.divides(x) is False
                            return

        pytest.skip("No den=2 parity-mismatch witness found in the search window (unexpected).")

    def test_exact_div_zero_norm_divisor_not_supported(self):
        """
        In rings with zero divisors (DualRing D=0, SplitRing D=1), exact_div should reject
        divisors with norm 0 (currently NotImplementedError in QuadraticRing.exact_div).
        """
        # Dual numbers: epsilon has a=0 => norm 0
        Q0 = QuadraticRing(0)
        x0 = Q0(5, 7)
        eps = Q0(0, 1)
        assert abs(eps) == 0
        with pytest.raises(NotImplementedError):
            _ = x0.exact_div(eps)

        # Split-complex: a=±b => norm 0
        Q1 = QuadraticRing(1)  # default den=2
        x1 = Q1(6, 2)
        z = Q1(0, 0)
        assert abs(z) == 0
        with pytest.raises(NotImplementedError):
            _ = x1.exact_div(z)


class TestGcdXgcd(QuadIntTests):
    """Tests for gcd/xgcd."""

    @staticmethod
    def _is_associate(x: QuadInt, y: QuadInt) -> bool:
        """Return True iff x and y differ by multiplication by a torsion unit."""
        return any(x == y * u for u in y.units)

    def test_xgcd_requires_division(self):
        """xgcd/gcd should raise in rings without divmod support."""
        a = Z15(5, 2)
        b = Z15(3, -2)
        with pytest.raises(NotImplementedError):
            _ = a.xgcd(b)
        with pytest.raises(NotImplementedError):
            _ = a.gcd(b)

    def test_xgcd_rejects_zero_divisor_rings(self):
        """Verify xgcd is intentionally not implemented for D=0 (dual numbers)."""
        Q = QuadraticRing(0)
        a = Q(5, 9)
        b = Q(3, 1)
        with pytest.raises(NotImplementedError):
            _ = a.xgcd(b)
        with pytest.raises(NotImplementedError):
            _ = a.gcd(b)

    def test_xgcd_trivial_cases_obey_bezout(self):
        """Verify xgcd should satisfy Bézout identity even when one argument is 0."""
        Q = ZI  # Gaussian integers have nontrivial torsion units, good stress-test.
        a = Q(5, 2)
        z = Q.zero

        g1, s1, t1 = a.xgcd(z)
        assert s1 * a + t1 * z == g1
        assert g1 == a._canonical_associate()
        assert a.gcd(z) == g1

        g2, s2, t2 = z.xgcd(a)
        assert s2 * z + t2 * a == g2
        assert g2 == a._canonical_associate()
        assert z.gcd(a) == g2

    @pytest.mark.parametrize("D", [-11, -7, -3, -2, -1, 2, 5, 69], ids=str)
    def test_xgcd_bezout_and_divisibility_random(self, D: int):
        """
        Property test:
          - s*a + t*b == g
          - g divides a and b
          - gcd agrees with xgcd()[0]
          - gcd is symmetric (stable canonical associate)
        """
        Q = QuadraticRing(D)
        rng = random.Random(99_000 + D)

        for bound in (50, 5000):
            for _ in range(30):
                a = _rand_elem(rng, Q, bound)
                b = _rand_elem(rng, Q, bound)
                if not b:
                    continue

                g, s, t = a.xgcd(b)

                # Bézout identity
                assert s * a + t * b == g

                # Stable canonical representative (torsion-unit canonicalization)
                assert g == g._canonical_associate()

                # Divisibility
                assert g.divides(a) is True
                assert g.divides(b) is True

                # gcd wrappers agree
                assert a.gcd(b) == g
                assert Q.gcd(a, b) == g

                # symmetry; gcds should be associates: each divides the other
                g2 = b.gcd(a)
                assert g.divides(g2)
                assert g2.divides(g)

    @pytest.mark.parametrize("Q", [ZI, ZE, Z5], ids=str)
    def test_gcd_contains_common_factor(self, Q: QuadraticRing):
        """If d divides both a and b, then d should divide gcd(a,b)."""
        rng = random.Random(12_345 + Q.D)

        # Pick a small non-unit d
        d = _rand_elem(rng, Q, 20)
        while not d or abs(abs(d)) == 1:
            d = _rand_elem(rng, Q, 20)

        u = _rand_elem(rng, Q, 200)
        v = _rand_elem(rng, Q, 200)
        a = d * u
        b = d * v

        g = a.gcd(b)
        assert d.divides(g) is True


class TestInvModAndNegativePow(QuadIntTests):
    """Tests for inv_mod and negative exponents in pow(x, e, mod)."""

    @staticmethod
    def _embed_int(Q: QuadraticRing, n: int) -> QuadInt:
        """Embed Python int n as an element of Q (works for den=1 and den=2)."""
        return Q(n * Q.den, 0)

    @staticmethod
    def _find_invertible(Q: QuadraticRing, mod: QuadInt) -> tuple[QuadInt, QuadInt]:
        """
        Find a small element a with inverse modulo mod.

        Returns:
            (a, inv): with a*inv ≡ 1 (mod mod).
        """
        for a0 in range(-6, 7):
            for b0 in range(-6, 7):
                if a0 == 0 and b0 == 0:
                    continue
                if Q.den == 2 and ((a0 ^ b0) & 1):
                    continue
                a = Q(a0, b0)
                try:
                    inv = a.inv_mod(mod)
                except ValueError:
                    continue
                return a, inv

        pytest.skip(f"Could not find invertible element modulo {mod} in {Q}")

    @pytest.mark.parametrize("Q", [ZI, ZE, Z2, Z5, Z69], ids=str)
    def test_inv_mod_matches_pow_negative_one(self, Q: QuadraticRing):
        """inv_mod agrees with pow(a, -1, m), and a*inv ≡ 1 (mod m)."""
        assert Q.supports_division() is True

        m = self._embed_int(Q, 29)  # integer modulus
        a, inv = self._find_invertible(Q, m)

        one_mod = a.one % m
        assert (a * inv) % m == one_mod
        assert (inv * a) % m == one_mod

        assert pow(a, -1, m) == inv
        assert pow(a, -1, m) * a % m == one_mod

    @pytest.mark.parametrize("Q", [ZI, ZE, Z2, Z5, Z69], ids=str)
    def test_pow_negative_exponents_use_inverse(self, Q: QuadraticRing):
        """pow(a, -e, m) equals pow(a.inv_mod(m), e, m) for e>0."""
        assert Q.supports_division() is True

        m = self._embed_int(Q, 31)
        a, inv = self._find_invertible(Q, m)

        for e in [1, 2, 5, 17]:
            left = pow(a, -e, m)
            right = pow(inv, e, m)
            assert left == right

    def test_pow_negative_without_mod_raises(self):
        """Negative exponent without a modulus should still be rejected."""
        a = ZI(3, 2)
        error_msg = r"Negative powers not supported.* without a modulus"
        with pytest.raises(ValueError, match=error_msg):
            _ = a**-1
        with pytest.raises(ValueError, match=error_msg):
            _ = pow(a, -7)

    @pytest.mark.parametrize("Q", [ZI, ZE, Z2, Z5, Z69], ids=str)
    def test_inv_mod_noninvertible_raises(self, Q: QuadraticRing):
        """If gcd(a,m) is not a unit, inv_mod / pow(-1, mod) must raise ValueError."""
        assert Q.supports_division() is True

        m = self._embed_int(Q, 6)
        a = self._embed_int(Q, 2)  # shares a non-unit gcd with 6
        error_msg = r".* is not invertible mod .* \(gcd\=.*\)"

        with pytest.raises(ValueError, match=error_msg):
            _ = a.inv_mod(m)
        with pytest.raises(ValueError, match=error_msg):
            _ = pow(a, -1, m)

    def test_inv_mod_requires_division(self):
        """Rings without divmod/xgcd should reject inv_mod and negative modular pow."""
        assert Z15.supports_division() is False

        a = Z15(5, 2)
        m = self._embed_int(Z15, 7)

        with pytest.raises(NotImplementedError):
            _ = a.inv_mod(m)
        with pytest.raises(NotImplementedError):
            _ = pow(a, -1, m)

    @pytest.mark.parametrize("Q", [ZI, Z2, Z5], ids=str)
    def test_inv_mod_and_pow_reject_modulus_zero(self, Q: QuadraticRing):
        """Modulus == 0 should raise ZeroDivisionError (both inv_mod and pow)."""
        assert Q.supports_division() is True

        a = Q(3 * Q.den, Q.den)  # nonzero element
        z = Q.zero

        with pytest.raises(ZeroDivisionError):
            _ = a.inv_mod(z)
        with pytest.raises(ZeroDivisionError):
            _ = pow(a, -1, z)


# region Factoring tests
class TestFactorDetail(QuadIntTests):
    """Tests for the factor_detail method"""

    @pytest.mark.parametrize(
        "x",
        [
            ZN2(1, 0),
            ZN2(-1, 0),
            ZN2(0, 1),
            ZN2(5, 2),
            ZN2(4, 53),
            ZN2(6, 0),
            ZN7(2, 0),
            ZN7(-2, 0),
            ZN7(0, 2),
            ZN7(10, 2),
            ZN7(4, 56),
            ZN7(6, 0),
            ZN11(2, 0),
            ZN11(-2, 0),
            ZN11(0, 2),
            ZN11(10, 2),
            ZN11(4, 56),
            ZN11(6, 0),
        ],
        ids=str,
    )
    def test_examples(self, x: QuadInt):
        """Validate some given examples"""
        f = x.factor_detail()
        self.assert_factoring(x, f)

    @pytest.mark.parametrize(
        "x",
        [
            ZN2(0, 0),
            ZN7(0, 0),
            ZN11(0, 0),
        ],
        ids=str,
    )
    def test_zero_raises(self, x: QuadInt):
        """Validate zero cannot be factored"""
        with pytest.raises(ValueError, match="0 does not have a finite factorization"):
            _ = x.factor_detail()

    @pytest.mark.parametrize(
        "x",
        [
            # This test only works with D=-2 because the other two require same parity, and I can't
            #   create the same parity as 0 with 2 odd primes...
            ZN2(17 * 31, 0),
        ],
        ids=str,
    )
    def test_527_primitive_vs_full(self, x: QuadInt):
        """Test factoring with a composite wholly real number"""
        # 17 splits (norm 17 twice), 31 stays Gaussian prime (norm 31^2)
        f_full = x.factor_detail()
        self.assert_factoring(x, f_full)
        assert len(f_full.primes) == 3
        assert norm_multiset(f_full.primes) == [17, 17, 31 * 31]

    @pytest.mark.parametrize(
        ("z", "expected_prime_norm"),
        [
            (ZN2(1, 1), 3),
            (ZN7(-1, 1), 2),
            (ZN11(3, 1), 5),
        ],
        ids=str,
    )
    def test_square(self, z: QuadInt, expected_prime_norm: int):
        """This is designed to catch candidate pruning that accidentally drops a needed divisor."""
        x = z * z

        f = x.factor_detail()
        self.assert_factoring(x, f)

        norms = norm_multiset(f.primes)
        assert norms.count(expected_prime_norm) == 2, (
            f"expected two norm-{expected_prime_norm} primes, got norms={norms} primes={f.primes}"
        )

    @pytest.mark.parametrize("b", [-5, -3, -1, 1, 3, 4])
    @pytest.mark.parametrize("a", [-4, -2, -1, 1, 2, 5])
    @pytest.mark.parametrize("test_klass", [ZN2, ZN7, ZN11])
    def test_small_grid(self, a: int, b: int, test_klass: QuadraticRing):
        """Check factor_detail().prod() round-trips for a small Gaussian grid."""
        try:
            x = test_klass(a, b)
        except ValueError:
            return

        if not x:
            return

        f = x.factor_detail()
        self.assert_factoring(x, f)

    def test_notimplemented_for_positive_D(self):
        """Factorization not implemented for positive D (except D=1)."""
        x = Z2(5, 2)
        with pytest.raises(NotImplementedError):
            _ = x.factor_detail()

    def test_notimplemented_for_non_norm_euclid_ring(self):
        """Non-UFD / non-supported imaginary rings should still reject factorization."""
        # D=-17 is proven to be not norm-Euclidean
        x = ZN17(5, 2)
        with pytest.raises(NotImplementedError):
            _ = x.factor_detail()

    def test_heegner_non_euclid_ramified_prime_factorization(self):
        """For D=-19, sqrt(D) should be a ramified prime element (norm 19)."""
        x = ZN19(0, 2)  # == sqrt(-19)
        f = x.factor_detail()
        self.assert_factoring(x, f)
        assert norm_multiset(f.primes) == [19]

    def test_heegner_non_euclid_inert_prime_factorization(self):
        """Inert rational primes should stay prime as elements (norm p^2)."""
        # For D=-19, p=3 is inert (since -19 is not a square mod 3).
        x = ZN19(6, 0)  # == 3
        f = x.factor_detail()
        self.assert_factoring(x, f)
        assert norm_multiset(f.primes) == [9]

    def test_heegner_non_euclid_69_1337(self):
        """Regression: ensure (69 + 1337*sqrt(-19))/2 factors and round-trips."""
        x = ZN19(69, 1337)
        f = x.factor_detail()
        self.assert_factoring(x, f)

        # From observed output: N(x)=11^2 * 70183, so expect norms [11, 11, 70183].
        assert norm_multiset(f.primes) == [11, 11, 70183]

    @pytest.mark.parametrize("test_klass", [ZN2, ZN7, ZN11])
    def test_associate_factorization_norm_invariant(self, test_klass: QuadraticRing):
        """Associates should keep the same factor-norm multiset."""
        x = test_klass(4, 56)
        fx = x.factor_detail()
        base_norms = norm_multiset(fx.primes)

        for u in x.units:
            y = x * u
            fy = y.factor_detail()
            assert fy.prod() == y
            assert norm_multiset(fy.primes) == base_norms


class TestFactor(QuadIntTests):
    """Tests for factor"""

    @pytest.mark.parametrize(
        "x",
        [
            ZN2(4, 53),
            ZN2(6, 0),
            ZN2(17 * 31, 0),
            ZN2(5, 2),
            ZN7(4, 56),
            ZN7(6, 0),
            ZN7(17, 7),
            ZN11(4, 56),
            ZN11(6, 0),
            ZN11(17, 7),
        ],
        ids=str,
    )
    def test_examples(self, x: QuadInt):
        """Validate some given examples"""
        factors = x.factor()
        assert isinstance(factors, dict)
        self.assert_factoring(x, factors)


# endregion
