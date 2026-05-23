import math
import os

from functools import cache
from pathlib import Path

import pytest

from pytest import mark, raises
from sympy import factorint, primerange

import quadint.sums.eisenstein

from quadint.sums.eisenstein import (
    decompose_number as decompose_eisenstein_number,
    decompose_prime as decompose_eisenstein_prime,
)


@pytest.mark.skipif(os.getenv("CI", "").lower() not in {"1", "true", "yes"}, reason="Compiled-only test")
def test_compiled_tests():
    """Verify that we are running these tests with a compiled version of quadint.sums.eisenstein."""
    path = Path(quadint.sums.eisenstein.__file__)
    assert path.suffix.lower() != ".py"


def eisenstein_norm(a: int, b: int) -> int:
    """Return the Eisenstein norm a^2 - ab + b^2."""
    return a * a - a * b + b * b


def _unit_associates(a: int, b: int) -> tuple[tuple[int, int], ...]:
    """Return unit associates of a + bω in coefficient coordinates."""
    return (
        (a, b),  # 1
        (-a, -b),  # -1
        (-b, a - b),  # ω
        (b, -a + b),  # -ω
        (b - a, -a),  # ω^2
        (a - b, a),  # -ω^2
    )


def _conjugate(a: int, b: int) -> tuple[int, int]:
    """Return the conjugate of a + bω in coefficient coordinates."""
    return a - b, -b


def _canonical_eisenstein_pair(
    a: int,
    b: int,
    *,
    no_trivial_solutions: bool,
) -> tuple[int, int] | None:
    """Return the canonical nonnegative representative of an associate/conjugate orbit."""
    candidates: list[tuple[int, int]] = []

    for base_a, base_b in ((a, b), _conjugate(a, b)):
        for x, y in _unit_associates(base_a, base_b):
            if x < 0 or y < 0:
                continue

            if no_trivial_solutions and (x == 0 or y in (0, x)):
                continue

            candidates.append((x, y))

    return min(candidates) if candidates else None


@cache
def _brute_force_eisenstein(
    n: int,
    *,
    no_trivial_solutions: bool,
) -> frozenset[tuple[int, int]]:
    """Brute-force canonical nonnegative pairs with a^2 - ab + b^2 = n."""
    if n < 1:
        return frozenset()

    found: set[tuple[int, int]] = set()
    # The Eisenstein norm is positive definite, and this intentionally favors clarity
    # over a tight bound. These brute-force tests are kept small enough for this to be cheap.
    lim = 2 * math.isqrt(n) + 2

    for a in range(-lim, lim + 1):
        for b in range(-lim, lim + 1):
            if eisenstein_norm(a, b) != n:
                continue

            sol = _canonical_eisenstein_pair(
                a,
                b,
                no_trivial_solutions=no_trivial_solutions,
            )
            if sol is not None:
                found.add(sol)

    return frozenset(found)


def brute_force_eisenstein(
    n: int,
    *,
    no_trivial_solutions: bool = True,
) -> set[tuple[int, int]]:
    """Return brute-force canonical Eisenstein norm-form decompositions."""
    return set(_brute_force_eisenstein(n, no_trivial_solutions=no_trivial_solutions))


class TestEisensteinPrimeDecomposition:
    """Tests for quadint.sums.eisenstein.decompose_prime."""

    def test_primes_below_1000(self):
        """Verify small Eisenstein-split and ramified primes."""
        for p in primerange(2, 1000):
            if p != 3 and p % 3 != 1:
                continue

            a, b = decompose_eisenstein_prime(p)
            assert eisenstein_norm(a, b) == p
            assert (a, b) == next(iter(brute_force_eisenstein(p, no_trivial_solutions=False)))

    def test_invalid_primes_below_1000(self):
        """Verify Eisenstein-inert primes do not decompose."""
        for p in primerange(2, 1000):
            if p == 3 or p % 3 == 1:
                continue

            with raises(ValueError, match="Could not decompose"):
                decompose_eisenstein_prime(p)

    def test_invalid_non_prime_like_values(self):
        """Verify values less than 2 are rejected by prime decomposition."""
        for p in (-7, 0, 1):
            with raises(ValueError, match="Could not decompose"):
                decompose_eisenstein_prime(p)

    def test_examples(self):
        """Test some verified prime examples."""
        for p in (3, 7, 13, 19, 31, 37, 43, 97, 109):
            a, b = decompose_eisenstein_prime(p)
            assert eisenstein_norm(a, b) == p
            assert (a, b) == next(iter(brute_force_eisenstein(p, no_trivial_solutions=False)))


class TestEisensteinNumberDecomposition:
    """Tests for quadint.sums.eisenstein.decompose_number."""

    @mark.parametrize("no_trivial_solutions", [True, False], ids=str)
    def test_small_numbers_match_bruteforce(self, *, no_trivial_solutions: bool):
        """Verify small Eisenstein norm-form decompositions against brute force."""
        max_n = 500 if os.getenv("CI") else 2_000

        for n in range(1, max_n + 1):
            got = decompose_eisenstein_number(
                n,
                no_trivial_solutions=no_trivial_solutions,
            )
            expect = brute_force_eisenstein(
                n,
                no_trivial_solutions=no_trivial_solutions,
            )

            assert got == expect, f"Mismatch for n={n}: missing={expect - got}, extra={got - expect}"

    def test_examples(self):
        """Test a few hand-picked examples with multiple factor types."""
        examples = [
            3,
            4,
            7,
            12,
            21,
            28,
            49,
            91,
            19890,
        ]

        for n in examples:
            got = decompose_eisenstein_number(n, no_trivial_solutions=False)
            expect = brute_force_eisenstein(n, no_trivial_solutions=False)
            assert got == expect, f"Mismatch for n={n}: missing={expect - got}, extra={got - expect}"
            for a, b in got:
                assert eisenstein_norm(a, b) == n

    def test_factored_input(self):
        """Verify a pre-factored input produces the same results as the integer input."""
        n = 19890
        assert decompose_eisenstein_number(factorint(n), no_trivial_solutions=False) == decompose_eisenstein_number(
            n,
            no_trivial_solutions=False,
        )

    def test_check_count(self):
        """Verify check_count can skip numbers whose predicted solution count is too small."""
        answers = decompose_eisenstein_number(7)

        assert answers
        assert decompose_eisenstein_number(7, check_count=1) == answers

        # 7 has only one split-prime factor, so asking for a much larger count
        # should be rejected by the cheap count check.
        assert decompose_eisenstein_number(7, check_count=3) == set()

    def test_inert_prime_odd_exponent_blocks_solutions(self):
        """Verify primes congruent to 2 mod 3 must occur to even exponent."""
        assert decompose_eisenstein_number(2, no_trivial_solutions=False) == set()
        assert decompose_eisenstein_number(2 * 7, no_trivial_solutions=False) == set()

    def test_inert_prime_even_exponent_scales_solutions(self):
        """Verify inert primes with even exponent scale existing solutions."""
        n = 2 * 2 * 7
        got = decompose_eisenstein_number(n, no_trivial_solutions=False)
        expect = brute_force_eisenstein(n, no_trivial_solutions=False)

        assert got == expect
        assert got

    def test_no_trivial_solutions(self):
        """Verify axis and diagonal square-like representations are filtered by default."""
        assert decompose_eisenstein_number(1, no_trivial_solutions=False) == brute_force_eisenstein(
            1,
            no_trivial_solutions=False,
        )
        assert decompose_eisenstein_number(1) == set()

        assert decompose_eisenstein_number(4, no_trivial_solutions=False) == brute_force_eisenstein(
            4,
            no_trivial_solutions=False,
        )
        assert decompose_eisenstein_number(4) == set()
