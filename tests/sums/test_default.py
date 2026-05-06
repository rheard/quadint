import math
import os
import random

from functools import cache
from pathlib import Path

import pytest

from pytest import mark, raises
from sympy import primerange

import quadint.sums

from quadint.sums import decompose_number, decompose_prime


@pytest.mark.skipif(os.getenv("CI", "").lower() not in {"1", "true", "yes"}, reason="Compiled-only test")
def test_compiled_tests():
    """Verify that we are running these tests with a compiled version of twosquares"""
    path = Path(quadint.sums.__file__)
    assert path.suffix.lower() != ".py"


@cache
def _brute_force(
    n: int,
    d: int,
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()

    for x in range(math.isqrt(n) + 1):
        rem = n - x * x
        if rem < 0 or rem % d:
            continue

        y2 = rem // d
        y = math.isqrt(y2)
        if y * y != y2:
            continue

        if d == 1 and x > y:
            continue

        out.add((x, y))

    return out


def brute_force_quadratic_form(
    n: int,
    d: int = 1,
    *,
    no_trivial_solutions: bool = True,
):
    """Brute-force all canonical nonnegative (x,y) with x^2 + d*y^2 = n."""
    sols = _brute_force(n, d)
    if no_trivial_solutions:
        sols = {(x, y) for x, y in sols if not (d == 1 and x == y) and x != 0 and y != 0}
    return sols


class TestPrimeDecomposition:
    """Tests for decompose_prime"""

    def test_primes_below_1000(self):
        """Verify small primes"""

        for i in primerange(1000):
            if i % 4 != 1:  # Primes that are not 1 mod 4 will produce an error (see source for decompose_prime)
                continue

            x, y = decompose_prime(i)
            assert i == x**2 + y**2

    def test_invalid_prime(self):
        """Verify invalid primes"""
        with raises(ValueError, match="Could not decompose"):
            decompose_prime(11)

    def test_high_range(self):
        """Verify large primes"""

        found_one = False
        for i in primerange(2**31 + 1, 2**31 + 1001):
            if i % 4 != 1:
                continue

            x, y = decompose_prime(i)
            assert i == x**2 + y**2
            found_one = True

        assert found_one

    def test_examples(self):
        """Test some verified examples"""
        examples = {
            19889: (17, 140),
        }

        for example_p, decomposition in examples.items():
            assert decompose_prime(example_p) == decomposition

    def test_d_examples(self):
        """Test some verified examples of higher d values"""
        examples = {
            (41, 2): (3, 4),
            (43, 2): (5, 3),
            (19, 3): (4, 1),
            (37, 3): (5, 2),
            (157, 12): (7, 3),
            (181, 12): (13, 1),
            (2147483929, 10000): (34173, 313),
        }

        for (example_p, example_d), decomposition in examples.items():
            assert decompose_prime(example_p, example_d) == decomposition, f"Not matching for {example_p}, {example_d}"

    def test_two(self):
        """Verify two"""
        assert decompose_prime(2) == (1, 1)
        assert decompose_prime(2, 2) == (0, 1)
        with raises(ValueError, match="Could not decompose"):
            decompose_prime(2, 3)

    def test_three(self):
        """Verify three"""
        with raises(ValueError, match="Could not decompose"):
            decompose_prime(3)
        assert decompose_prime(3, 2) == (1, 1)
        assert decompose_prime(3, 3) == (0, 1)
        with raises(ValueError, match="Could not decompose"):
            decompose_prime(3, 4)

    def test_five(self):
        """Verify five"""
        assert decompose_prime(5) == (1, 2)
        with raises(ValueError, match="Could not decompose"):
            decompose_prime(5, 2)
        with raises(ValueError, match="Could not decompose"):
            decompose_prime(5, 3)
        assert decompose_prime(5, 4) == (1, 1)
        assert decompose_prime(5, 5) == (0, 1)

    @mark.parametrize(
        ("p", "d", "den", "expected"),
        [
            # Genuine denominator-2 cases: p is not x^2 + d*y^2,
            # but den^2*p is A^2 + d*B^2.
            (2, 7, 2, (1, 1)),
            (5, 11, 2, (3, 1)),
            (11, 19, 2, (5, 1)),
            # Lifted denominator-2 case:
            # 70183 = 262^2 + 19*9^2, so 4*70183 = 524^2 + 19*18^2.
            (70183, 19, 2, (524, 18)),
        ],
        ids=str,
    )
    def test_den2_examples(self, p: int, d: int, den: int, expected: tuple[int, int]):
        """Verify denominator-2 prime decompositions return numerator coordinates."""
        A, B = decompose_prime(p, d, den)

        assert expected == (A, B)
        assert A * A + d * B * B == den * den * p
        assert ((A ^ B) & 1) == 0

    @mark.parametrize(
        ("p", "d", "den"),
        [
            (3, 3, 1),
            (5, 5, 1),
            (2, 7, 2),
            (5, 11, 2),
            (11, 19, 2),
            (70183, 19, 2),
        ],
        ids=str,
    )
    def test_decompose_prime_invariant(self, p: int, d: int, den: int):
        """Every returned decomposition should satisfy A^2 + d*B^2 = den^2*p."""
        A, B = decompose_prime(p, d, den)

        assert A >= 0
        assert B >= 0
        assert A * A + d * B * B == den * den * p

        if den == 2:
            assert ((A ^ B) & 1) == 0

    @mark.parametrize(
        ("p", "d", "den"),
        [
            (3, 7, 2),  # 3 is inert in D=-7
            (2, 11, 2),  # 2 is inert in D=-11
            (3, 19, 2),  # 3 is inert in D=-19
            (5, 2, 1),  # existing den=1 non-representation
        ],
        ids=str,
    )
    def test_decompose_prime_invalid_generalized_examples(self, p: int, d: int, den: int):
        """Verify known non-representable primes still raise after generalizing den."""
        with raises(ValueError, match="Could not decompose"):
            decompose_prime(p, d, den)

    def test_decompose_prime_den2_uses_complementary_root(self):
        """D=-11, p=5 needs A=3; using only the smaller root representative misses it."""
        assert decompose_prime(5, 11, 2) == (3, 1)

    def test_invalid_parameters(self):
        """Validate parameter guards for generalized decomposition."""
        with raises(ValueError, match="d must be >= 1"):
            decompose_prime(5, 0)

        with raises(ValueError, match="den must be 1 or 2"):
            decompose_prime(5, 1, 3)


class TestNumberDecomposition:
    """Tests for decompose_number"""

    def test_small_numbers(self):
        """Verify small numbers"""
        max_n = 10_000 if os.getenv("CI") else 50_000

        for n in range(1, max_n + 1):
            got = decompose_number(n)
            expect = brute_force_quadratic_form(n)

            assert got == expect, f"Mismatch for n={n}: missing={expect - got}, extra={got - expect}"

    def test_all_small_numbers(self):
        """Verify all solutions for small numbers"""
        max_n = 10_000 if os.getenv("CI") else 50_000

        for n in range(1, max_n + 1):
            got = decompose_number(n, no_trivial_solutions=False)
            expect = brute_force_quadratic_form(n, no_trivial_solutions=False)

            assert got == expect, f"Mismatch for n={n}: missing={expect - got}, extra={got - expect}"

    def test_outside_range(self):
        """Verify large numbers"""

        for i in range(2**31 + 1, 2**31 + 1001):
            for x, y in decompose_number(i):
                assert i == x**2 + y**2

    def test_example(self):
        """Test the example from my documentation"""
        answers = decompose_number(19890)

        assert len(answers) == 4
        for x, y in answers:
            assert x**2 + y**2 == 19890

    def test_four(self):
        """Verify four"""
        assert decompose_number(4) == set()
        assert decompose_number(4, no_trivial_solutions=False) == {(0, 2)}
        # assert decompose_number(4, 2) == {(2, 0)}
        # assert decompose_number(4, 3) == {(2, 0), (1, 1)}
        # assert decompose_number(4, 4) == {(2, 0), (0, 1)}

    @mark.parametrize(
        "n",
        [
            # Larger hand-picked composites / squares / near-32bit boundary
            10**6,
            10**6 + 1,
            999_999,
            2**31 - 1,
            2**31 + 1,
            2**31 + 12345,
        ],
    )
    def test_all_examples(self, n: int):
        """Validate completeness for the given examples"""
        got = decompose_number(n, no_trivial_solutions=False)
        expect = brute_force_quadratic_form(n, no_trivial_solutions=False)
        assert got == expect, f"Mismatch for n={n}: missing={expect - got}, extra={got - expect}"

    def test_fuzzed_large_numbers_match_bruteforce(self):
        """Validate completeness for some random examples"""
        rng = random.Random(0)
        count = 20 if os.getenv("CI") else 200

        for _ in range(count):
            n = rng.randrange(1, 2**31 + 100_000)
            got = decompose_number(n, no_trivial_solutions=False)
            expect = brute_force_quadratic_form(n, no_trivial_solutions=False)
            assert got == expect, f"Mismatch for n={n}: missing={expect - got}, extra={got - expect}"

    @mark.parametrize(
        ("n", "d", "expected"),
        [
            (19, 3, {(4, 1)}),
            (20, 11, {(3, 1)}),
            # These are denominator-2 algebraic decompositions, but not integer
            # solutions to x^2 + d*y^2 = n.
            (2, 7, set()),
            (5, 11, set()),
            (11, 19, set()),
        ],
        ids=str,
    )
    def test_general_d_prime_shortcut_semantics(self, n: int, d: int, expected: set[tuple[int, int]]):
        """Prime shortcuts should return integer-form decompositions, not raw den=2 numerator coords."""
        assert decompose_number(n, d, no_trivial_solutions=False, warn=False) == expected

    @mark.parametrize("no_trivial_solutions", [True, False], ids=str)
    @mark.parametrize("d", [2, 3, 4, 7, 11, 19], ids=str)
    def test_small_numbers_general_d_match_bruteforce(self, d: int, *, no_trivial_solutions: bool):
        """Verify small generalized x^2 + d*y^2 decompositions against brute force."""
        for n in range(1, 151):
            got = decompose_number(
                n,
                d,
                no_trivial_solutions=no_trivial_solutions,
                warn=False,
            )
            expect = brute_force_quadratic_form(
                n,
                d,
                no_trivial_solutions=no_trivial_solutions,
            )

            assert got == expect, f"Mismatch for n={n}, d={d}: missing={expect - got}, extra={got - expect}"

    def test_eisenstein_unit_orbit_for_pure_inert_square(self):
        """Pure inert-even factors still need unit orbits in D=-3."""
        assert decompose_number(4, 3, no_trivial_solutions=False, warn=False) == {
            (2, 0),
            (1, 1),
        }
        assert decompose_number(4, 3, no_trivial_solutions=True, warn=False) == {
            (1, 1),
        }

    def test_eisenstein_unit_orbit_with_ramified_axis_factor(self):
        """Unit orbits should also be applied after product enumeration, not only scalar cases."""
        assert decompose_number(12, 3, no_trivial_solutions=False, warn=False) == {
            (0, 2),
            (3, 1),
        }
        assert decompose_number(12, 3, no_trivial_solutions=True, warn=False) == {
            (3, 1),
        }

    def test_square_factor_reduction_preserves_orientation(self):
        """Reducing d=4 to d=1 must try both square-sum orientations."""
        assert decompose_number(1, 4, no_trivial_solutions=False, warn=False) == {
            (1, 0),
        }
        assert decompose_number(8, 4, no_trivial_solutions=True, warn=False) == {
            (2, 1),
        }
        assert decompose_number(13, 4, no_trivial_solutions=True, warn=False) == {
            (3, 1),
        }

    def test_inert_even_scalar_branch_without_extra_units(self):
        """When no split primes exist, the simplified scalar-orbit branch is enough."""
        assert decompose_number(9, 2, no_trivial_solutions=False, warn=False) == {
            (1, 2),
            (3, 0),
        }
        assert decompose_number(9, 2, no_trivial_solutions=True, warn=False) == {
            (1, 2),
        }
