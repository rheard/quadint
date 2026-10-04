import math
import os
import random
import warnings

from functools import cache
from pathlib import Path

import pytest

from pytest import mark, raises
from sympy import factorint, isprime, legendre_symbol, nextprime, primerange

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

    @mark.parametrize(("p", "expected"), [(179, (21, 5)), (317, (27, 7)), (643, (41, 9)), (983, (51, 11))], ids=str)
    def test_decompose_prime_den2_needs_euclid_on_2p(self, p: int, expected: tuple[int, int]):
        """Euclid on (p, root) misses these (about 6% of split primes for d=11); running it on (2p, root) does not."""
        assert decompose_prime(p, 11, 2) == expected

    @mark.parametrize("d", [3, 7, 11, 19, 43, 67, 163], ids=str)
    def test_decompose_prime_den2_every_split_prime(self, d: int):
        """These rings have class number one, so every prime that splits or ramifies there is a norm."""
        disc = -d  # d == 3 mod 4, so the ring of integers has discriminant -d
        for p in primerange(2, 20_000):
            splits = disc % 8 == 1 if p == 2 else disc % p == 0 or legendre_symbol(disc % p, p) == 1
            if not splits:
                continue

            A, B = decompose_prime(p, d, 2)
            assert A * A + d * B * B == 4 * p, f"Wrong decomposition of {p} with d={d}"

    @mark.parametrize("d", [1, 2, 3, 4, 5, 6, 7, 11, 12, 19, 163], ids=str)
    def test_every_prime_matches_bruteforce(self, d: int):
        """
        A prime has at most one decomposition (up to order, when d == 1), and every one there is has to be found.

        Every other prime has to raise, including the ones that divide d, and 2.
        """
        max_p = 3_000 if os.getenv("CI") else 20_000
        for p in primerange(2, max_p):
            expect = brute_force_quadratic_form(p, d, no_trivial_solutions=False)
            assert len(expect) <= 1

            if expect:
                assert decompose_prime(p, d) == next(iter(expect)), f"Wrong decomposition of {p} with d={d}"
            else:
                with raises(ValueError, match="Could not decompose"):
                    decompose_prime(p, d)

    @mark.parametrize("d", [3, 7, 11, 15, 19, 23, 31, 35, 43, 67, 163], ids=str)
    def test_every_prime_den2_matches_bruteforce(self, d: int):
        """With den=2, 4*p has to be decomposed whenever it can be, for every d, not just the class-number-one ones."""
        max_p = 3_000 if os.getenv("CI") else 10_000
        for p in primerange(2, max_p):
            expect = brute_force_quadratic_form(4 * p, d, no_trivial_solutions=False)
            if expect:
                assert decompose_prime(p, d, 2) in expect, f"Wrong decomposition of 4*{p} with d={d}"
            else:
                with raises(ValueError, match="Could not decompose"):
                    decompose_prime(p, d, 2)

    def test_large_primes(self):
        """Big primes, including ones with many factors of 2 in p - 1, which take the most Tonelli-Shanks steps."""
        rng = random.Random(2)
        primes = [65537, 7 * 2**26 + 1, 119 * 2**23 + 1, 3 * 2**30 + 1, 2**64 - 2**32 + 1]
        primes += [p for p in (nextprime(rng.randrange(10**15, 10**40)) for _ in range(100)) if p % 4 == 1]

        for p in primes:
            assert isprime(p)
            x, y = decompose_prime(p)
            assert x * x + y * y == p, f"Wrong decomposition of {p}"
            assert 0 < x < y

    def test_invalid_parameters(self):
        """Validate parameter guards for generalized decomposition."""
        with raises(ValueError, match="d must be >= 1"):
            decompose_prime(5, 0)

        with raises(ValueError, match="den must be 1 or 2"):
            decompose_prime(5, 1, 3)

    @pytest.mark.usefixtures("hang_guard")
    @mark.parametrize("d", [1, 2, 3, 5, 7, 11, 19], ids=str)
    def test_composite_numbers_finish(self, d: int):
        """
        A composite p is not supported, but it still has to get an answer, which 9, 25 and 65 (and many more) never did.

        That answer is ValueError, or a decomposition of the number it was given.
        """
        for n in range(4, 5_000):
            if isprime(n):
                continue

            for den in (1, 2):
                try:
                    x, y = decompose_prime(n, d, den)
                except ValueError:
                    continue

                assert x * x + d * y * y == den * den * n, f"Wrong decomposition of {n} with d={d}, den={den}"


class TestNumberDecomposition:
    """Tests for decompose_number"""

    def test_small_numbers(self):
        """Verify small numbers"""
        max_n = 10_000 if os.getenv("CI") else 50_000

        for n in range(max_n + 1):
            got = decompose_number(n)
            expect = brute_force_quadratic_form(n)

            assert got == expect, f"Mismatch for n={n}: missing={expect - got}, extra={got - expect}"

    def test_all_small_numbers(self):
        """Verify all solutions for small numbers"""
        max_n = 10_000 if os.getenv("CI") else 50_000

        for n in range(max_n + 1):
            got = decompose_number(n, no_trivial_solutions=False)
            expect = brute_force_quadratic_form(n, no_trivial_solutions=False)

            assert got == expect, f"Mismatch for n={n}: missing={expect - got}, extra={got - expect}"

    @mark.parametrize("d", [1, 3, 12], ids=str)
    def test_zero_and_negative(self, d: int):
        """x**2 + d*y**2 is never negative, and it is only 0 at (0, 0), which is a trivial solution."""
        assert decompose_number(0, d) == set()
        assert decompose_number(0, d, no_trivial_solutions=False) == {(0, 0)}
        assert decompose_number(0, d, check_count=2, no_trivial_solutions=False) == set()
        assert decompose_number(-5, d, no_trivial_solutions=False) == set()

    @mark.parametrize("d", [0, -1, -4, -5], ids=str)
    def test_invalid_d(self, d: int):
        """Only d >= 1 is supported, which used to quietly return no solutions instead of raising."""
        with raises(ValueError, match="d must be >= 1"):
            decompose_number(25, d)

        with raises(ValueError, match="d must be >= 1"):
            decompose_number({5: 2}, d, no_trivial_solutions=False)

        with raises(ValueError, match="d must be >= 1"):
            decompose_number(0, d)

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

    def test_twenty_five(self):
        """25 and its factorization {5: 2} give the same answer ({25: 1} is something else: it claims 25 is prime)."""
        assert decompose_number(25) == decompose_number({5: 2}) == {(3, 4)}
        assert decompose_number(25, no_trivial_solutions=False) == decompose_number({5: 2}, no_trivial_solutions=False)
        assert decompose_number(25, no_trivial_solutions=False) == {(0, 5), (3, 4)}

    @pytest.mark.usefixtures("hang_guard")
    @mark.parametrize(
        "factors",
        [{9: 1}, {21: 1}, {25: 1}, {65: 1}, {1729: 1}, {3277: 1}, {25: 1, 13: 1}, {65: 3, 2: 1}, {21: 2, 5: 1}],
        ids=str,
    )
    def test_composite_keys_finish(self, factors: dict[int, int]):
        """
        A dict with a composite key is not a factorization, but it still has to get an answer, which these never did.

        The answer can miss solutions, or be ValueError, but every pair in it still has to be a solution.
        """
        n = math.prod(p**k for p, k in factors.items())
        for d in (1, 2, 3, 4, 5, 7, 12):
            for no_trivial_solutions in (True, False):
                try:
                    got = decompose_number(factors, d, no_trivial_solutions=no_trivial_solutions)
                except ValueError:
                    continue

                expect = brute_force_quadratic_form(n, d, no_trivial_solutions=no_trivial_solutions)
                assert got <= expect, f"Not solutions for {factors}, d={d}: {got - expect}"

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
        "n",
        [
            5**61,
            2**60 * 5,  # 2 ramifies, so it only has one option
            2**40 * 3**10 * 5**12 * 13**9 * 17**6,
        ],
        ids=["5**61", "2**60 * 5", "mixed"],
    )
    def test_high_exponents_are_complete(self, n: int):
        """
        High prime powers should be quick (these used to take 2**60 or more combinations), and still complete.

        They are far too big for brute force, so completeness is checked by count instead: x**2 + y**2 == n has
            r2(n) == 4 * prod(e + 1 for each p**e exactly dividing n with p % 4 == 1) integer solutions
            (when every p % 4 == 3 has an even exponent), and each canonical pair with 0 < x < y accounts for 8 of them.
        """
        got = decompose_number(n)

        r2 = 4 * math.prod(e + 1 for p, e in factorint(n).items() if p % 4 == 1)
        square = math.isqrt(n) ** 2 == n
        twice_square = n % 2 == 0 and math.isqrt(n // 2) ** 2 == n // 2
        assert len(got) == (r2 - 4 * square - 4 * twice_square) // 8
        assert all(x * x + y * y == n and 0 < x < y for x, y in got)

    def test_many_split_primes_are_complete(self):
        """
        Numbers with many primes that are 1 mod 4, which have the most solutions, have to get every one of them.

        These are too big for brute force, so like test_high_exponents_are_complete this counts the solutions instead,
            and checks that their prepared factorization and check_count agree.
        """
        rng = random.Random(1)
        split = [p for p in primerange(5, 2_000) if p % 4 == 1]
        inert = [q for q in primerange(3, 200) if q % 4 == 3]

        for _ in range(40 if os.getenv("CI") else 200):
            factors = {p: rng.randint(1, 3) for p in rng.sample(split, rng.randint(1, 6))}
            factors |= {q: 2 * rng.randint(1, 2) for q in rng.sample(inert, rng.randint(0, 2))}
            if rng.random() < 0.5:
                factors[2] = rng.randint(1, 5)

            n = math.prod(p**k for p, k in factors.items())
            got = decompose_number(n)

            r2 = 4 * math.prod(k + 1 for p, k in factors.items() if p % 4 == 1)
            square = math.isqrt(n) ** 2 == n
            twice_square = n % 2 == 0 and math.isqrt(n // 2) ** 2 == n // 2
            assert len(got) == (r2 - 4 * square - 4 * twice_square) // 8, f"Missing solutions for {factors}"
            assert all(x * x + y * y == n and 0 < x < y for x, y in got)

            assert decompose_number(factors) == got
            assert decompose_number(factors, check_count=len(got)) == got
            assert len(decompose_number(factors, no_trivial_solutions=False)) == len(got) + square + twice_square

    def test_factored_input_variants(self):
        """
        A prepared factorization has to give the same answer with its primes in any order, or with exponents of 0.

        The order picks the prime that decompose_number fixes to halve its work, and an exponent of 0 can move a prime
            number off its shortcut.
        """
        for n in range(1, 3_000 if os.getenv("CI") else 10_000):
            factors = factorint(n)
            reordered = dict(reversed(factors.items()))
            padded = {**factors, 1_000_003: 0, 13: factors.get(13, 0)}

            for no_trivial_solutions in (True, False):
                expect = decompose_number(n, no_trivial_solutions=no_trivial_solutions)
                for variant in (reordered, padded):
                    got = decompose_number(variant, no_trivial_solutions=no_trivial_solutions)
                    assert got == expect, f"Mismatch for {variant}: missing={expect - got}, extra={got - expect}"

    @mark.parametrize("d", [1, 2, 3, 7], ids=str)
    def test_limited_checks_with_prepared_input(self, d: int):
        """limited_checks only skips checks that prepared input passes, so it can't change the answer when n has one."""
        max_n = 3_000 if os.getenv("CI") else 20_000
        for n in range(1, max_n):
            if not brute_force_quadratic_form(n, d, no_trivial_solutions=False):
                continue  # not prepared input, where limited_checks can give false positives (as documented)

            factors = factorint(n)
            for no_trivial_solutions in (True, False):
                expect = decompose_number(n, d, no_trivial_solutions=no_trivial_solutions)
                got = decompose_number(factors, d, limited_checks=True, no_trivial_solutions=no_trivial_solutions)
                assert got == expect, f"Mismatch for n={n}: missing={expect - got}, extra={got - expect}"

    @mark.parametrize(
        ("n", "d"),
        [
            (7**8 * 3**3 * 13**2, 3),  # 7 and 13 split, 3 ramifies
            (2**24 * 11**2 * 23, 7),  # 2 splits for d=7 (in den=2 coordinates)
        ],
        ids=str,
    )
    def test_high_exponents_general_d_match_bruteforce(self, n: int, d: int):
        """High prime powers for other d should still find every solution."""
        for no_trivial_solutions in (True, False):
            got = decompose_number(n, d, no_trivial_solutions=no_trivial_solutions)
            expect = brute_force_quadratic_form(n, d, no_trivial_solutions=no_trivial_solutions)

            assert got == expect, f"Mismatch for n={n}, d={d}: missing={expect - got}, extra={got - expect}"

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
        assert decompose_number(n, d, no_trivial_solutions=False) == expected

    @mark.parametrize("no_trivial_solutions", [True, False], ids=str)
    @mark.parametrize("d", [2, 3, 4, 7, 11, 19], ids=str)
    def test_small_numbers_general_d_match_bruteforce(self, d: int, *, no_trivial_solutions: bool):
        """Verify small generalized x^2 + d*y^2 decompositions against brute force."""
        for n in range(1, 151):
            got = decompose_number(
                n,
                d,
                no_trivial_solutions=no_trivial_solutions,
            )
            expect = brute_force_quadratic_form(
                n,
                d,
                no_trivial_solutions=no_trivial_solutions,
            )

            assert got == expect, f"Mismatch for n={n}, d={d}: missing={expect - got}, extra={got - expect}"

    def test_d11_matches_bruteforce(self):
        """d=11 used to miss every solution involving a split prime like 179 (537 == 19**2 + 11*4**2 == 3 * 179)."""
        assert decompose_number(537, 11) == {(19, 4)}

        for n in range(1, 3_001):
            for no_trivial_solutions in (True, False):
                got = decompose_number(n, 11, no_trivial_solutions=no_trivial_solutions)
                expect = brute_force_quadratic_form(n, 11, no_trivial_solutions=no_trivial_solutions)

                assert got == expect, f"Mismatch for n={n}: missing={expect - got}, extra={got - expect}"

    def test_eisenstein_unit_orbit_for_pure_inert_square(self):
        """Pure inert-even factors still need unit orbits in D=-3."""
        assert decompose_number(4, 3, no_trivial_solutions=False) == {
            (2, 0),
            (1, 1),
        }
        assert decompose_number(4, 3, no_trivial_solutions=True) == {
            (1, 1),
        }

    def test_eisenstein_unit_orbit_with_ramified_axis_factor(self):
        """Unit orbits should also be applied after product enumeration, not only scalar cases."""
        assert decompose_number(12, 3, no_trivial_solutions=False) == {
            (0, 2),
            (3, 1),
        }
        assert decompose_number(12, 3, no_trivial_solutions=True) == {
            (3, 1),
        }

    def test_square_factor_reduction_preserves_orientation(self):
        """Reducing d=4 to d=1 must try both square-sum orientations."""
        assert decompose_number(1, 4, no_trivial_solutions=False) == {
            (1, 0),
        }
        assert decompose_number(8, 4, no_trivial_solutions=True) == {
            (2, 1),
        }
        assert decompose_number(13, 4, no_trivial_solutions=True) == {
            (3, 1),
        }

    def test_inert_even_scalar_branch_without_extra_units(self):
        """When no split primes exist, the simplified scalar-orbit branch is enough."""
        assert decompose_number(9, 2, no_trivial_solutions=False) == {
            (1, 2),
            (3, 0),
        }
        assert decompose_number(9, 2, no_trivial_solutions=True) == {
            (1, 2),
        }

    def test_products_of_primes_without_elements(self):
        """
        With class number above one, primes can have no element of their norm, but products of them can.

        Neither 3 nor 7 is x**2 + 5*y**2, since the prime ideals over them have no generators, but 21 == 4**2 + 5*1**2
            == 1**2 + 5*2**2, from the two ways to multiply those ideals into principal ones. These used to give no
            solutions at all, since 3 and 7 looked inert.
        """
        assert decompose_number(6, 5) == {(1, 1)}
        assert decompose_number(21, 5) == {(1, 2), (4, 1)}

        # Eight solutions, from four primes that are none of them x**2 + 5*y**2
        assert len(decompose_number(3 * 7 * 23 * 103, 5)) == 8
        assert decompose_number(3 * 7 * 23 * 103, 5) == brute_force_quadratic_form(3 * 7 * 23 * 103, 5)

    @mark.parametrize("no_trivial_solutions", [True, False], ids=str)
    @mark.parametrize("d", [5, 6, 10, 14, 15, 17, 21, 23, 26, 30, 47, 71, 20, 24, 45], ids=str)
    def test_class_number_above_one_matches_bruteforce(self, d: int, *, no_trivial_solutions: bool):
        """Every solution should be found for d with class number above one, and for their multiples by squares."""
        for n in range(1, 801):
            got = decompose_number(n, d, no_trivial_solutions=no_trivial_solutions)
            expect = brute_force_quadratic_form(n, d, no_trivial_solutions=no_trivial_solutions)

            assert got == expect, f"Mismatch for n={n}, d={d}: missing={expect - got}, extra={got - expect}"

    @mark.parametrize(
        ("n", "d"),
        [
            (2 * 3**3 * 7 * 23 * 47 * 89 * 103, 5),
            (2 * 5 * 7**3 * 11 * 59 * 83 * 101, 6),
            (2 * 3**2 * 5**3 * 13**2 * 19 * 23 * 113, 14),
            (2 * 5**2 * 11 * 17 * 19 * 31**2 * 107, 21),
        ],
        ids=str,
    )
    def test_class_number_above_one_large_numbers(self, n: int, d: int):
        """Numbers with 64 or 72 solutions each, from many primes that mostly have no element of their norm."""
        for no_trivial_solutions in (True, False):
            got = decompose_number(n, d, no_trivial_solutions=no_trivial_solutions)
            expect = brute_force_quadratic_form(n, d, no_trivial_solutions=no_trivial_solutions)

            assert got == expect, f"Mismatch for n={n}, d={d}: missing={expect - got}, extra={got - expect}"

    @mark.parametrize("d", [1, 3, 5, 12, 19, 20, 23], ids=str)
    def test_check_count_only_skips_numbers_with_fewer_solutions(self, d: int):
        """check_count may give an empty set, but only when there really are fewer solutions than it asks for."""
        for n in range(1, 801):
            expect = brute_force_quadratic_form(n, d)
            for check_count in (1, 2, 3):
                got = decompose_number(n, d, check_count=check_count)
                if len(expect) >= check_count:
                    assert got == expect, f"n={n}, d={d}, check_count={check_count} skipped {expect}"
                else:
                    assert got in (expect, set())

    def test_check_count_eisenstein_orbits(self):
        """
        For d=3, one product can give three solutions, which check_count used to count as one, and so skip.

        28 == 5**2 + 3*1**2 == 4**2 + 3*2**2 == 1**2 + 3*3**2 all come from 2 * (2 + sqrt(-3)), times units of Z[ω].
        """
        expected = {(1, 3), (4, 2), (5, 1)}
        assert decompose_number(28, 3) == expected
        assert decompose_number(28, 3, check_count=3) == expected
        assert decompose_number(4, 3, check_count=2, no_trivial_solutions=False) == {(1, 1), (2, 0)}

    def test_nothing_warns(self):
        """No d can miss solutions anymore, so there is nothing left to warn about."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            assert decompose_number(21, 5) == {(1, 2), (4, 1)}
            assert decompose_number(11 * 17, 19) == {(4, 3)}
