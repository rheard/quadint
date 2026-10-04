import random

import pytest

from sympy import nextprime, primerange

from quadint.utils import _is_squarefree, _sqrt_mod_prime  # ruff: ignore[import-private-name]


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, False),
        (1, False),
        (-1, False),
        (2, True),
        (3, True),
        (4, False),
        (5, True),
        (6, True),
        (8, False),
        (9, False),
        (10, True),
        (12, False),
        (14, True),
        (15, True),
        (16, False),
        (18, False),
        (21, True),
        (22, True),
        (23, True),
        (24, False),
        (29, True),
        (31, True),
        (45, False),
        (61, True),
        (69, True),  # 69 = 3 * 23, squarefree
        (-14, True),
        (-18, False),
        ({2: 1}, True),
        ({2: 2}, False),
        ({2: 1, 3: 1}, True),
        ({2: 2, 3: 1}, False),
        ({3: 1, 23: 1}, True),  # 69 = 3 * 23
        ({2: 1, 3: 2}, False),  # 18 = 2 * 3**2
        ({5: 1, 7: 1, 11: 1}, True),
        ({5: 1, 7: 1, 11: 2}, False),
    ],
    ids=str,
)
def test_squarefree(n: int, *, expected: bool):
    """Verify the squarefree helper handles signs and repeated prime factors correctly."""
    assert _is_squarefree(n) is expected


def test_sqrt_mod_prime_small_primes():
    """Every residue modulo every prime below 400 should get a square root exactly when it is a square."""
    for p in primerange(2, 400):
        squares = {x * x % p for x in range(p)}
        for a in range(-p, 2 * p):
            r = _sqrt_mod_prime(a, p)
            if a % p in squares:
                assert r is not None
                assert r * r % p == a % p, f"{r}**2 is not {a} mod {p}"
            else:
                assert r is None, f"{a} is not a square mod {p}, but got {r}"


def test_sqrt_mod_prime_large_primes():
    """Big primes too, including ones with many factors of 2 in p - 1, which take the most Tonelli-Shanks steps."""
    rng = random.Random(0)
    primes = [nextprime(rng.randrange(10**12, 10**40)) for _ in range(200)]
    primes += [2**127 - 1, 2**64 - 59, 3 * 2**30 + 1, 7 * 2**26 + 1, 119 * 2**23 + 1]

    for p in primes:
        for _ in range(10):
            x = rng.randrange(1, p)
            r = _sqrt_mod_prime(x * x, p)
            assert r in (x, p - x)

            # Exactly half of the nonzero residues are squares, and Euler's criterion tells them apart
            a = rng.randrange(1, p)
            assert (_sqrt_mod_prime(a, p) is None) is (pow(a, (p - 1) // 2, p) == p - 1)
