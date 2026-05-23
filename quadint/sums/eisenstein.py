from __future__ import annotations

import math

from itertools import product

from quadint.eisenstein import eisensteinint
from quadint.sums import (
    _factor_input,
    decompose_prime as _decompose_quadratic_prime,
)


def _norm(a: int, b: int) -> int:
    """Return the Eisenstein norm `a**2 - a*b + b**2`."""
    return a * a - a * b + b * b


def _canonical_pair(
    z: eisensteinint,
    *,
    no_trivial_solutions: bool = True,
) -> tuple[int, int] | None:
    """
    Return one canonical nonnegative pair from the associate/conjugate orbit of `z`.

    The Eisenstein norm has a six-fold unit symmetry, plus conjugation. This helper
        treats all of those as the same representation and returns the lexicographically
        smallest nonnegative pair `(a, b)` in that orbit.

    Args:
        z: The integer to get the canonical pair for.
        no_trivial_solutions (bool): When true, the axis solutions `a == 0` and `b == 0` are discarded,
            as is the diagonal solution `a == b`. These are all unit-associate forms of a plain integer square.

    Returns:
        tuple: The found canonical pair.
    """
    candidates: list[tuple[int, int]] = []

    for base in (z, z.conjugate()):
        for unit in base.units:
            w = base * unit

            a = w.real
            b = w.omega

            if a < 0 or b < 0:
                continue

            if no_trivial_solutions and (a == 0 or b in (0, a)):
                continue

            candidates.append((a, b))

    return min(candidates) if candidates else None


def _prime_element(p: int) -> eisensteinint:
    """Return an Eisenstein integer with norm `p` for split or ramified primes."""
    A, B = _decompose_quadratic_prime(p, 3, 2)
    return eisensteinint(A, B, skip_basis=True)


def decompose_prime(p: int) -> tuple[int, int]:
    """Decompose a rational prime as `p = a**2 - a*b + b**2`."""
    if p < 2:
        raise ValueError(f"Could not decompose {p!r}")

    try:
        z = _prime_element(p)
    except ValueError as exc:
        raise ValueError(f"Could not decompose {p!r}") from exc

    sol = _canonical_pair(z, no_trivial_solutions=False)
    if sol is None:
        raise ValueError(f"Could not decompose {p!r}")

    return sol


def decompose_number(
    n: dict[int, int] | int,
    check_count: int | None = None,
    *,
    limited_checks: bool = False,
    no_trivial_solutions: bool = True,
) -> set[tuple[int, int]]:
    """
    Decompose a number into canonical Eisenstein norm-form solutions.

    Returns pairs `(a, b)` such that:

        a**2 - a*b + b**2 == n

    The returned pairs are canonical representatives under the Eisenstein unit and
    conjugation symmetries. By default, trivial square-like solutions with
    `a == 0`, `b == 0`, or `a == b` are omitted.

    Args:
        n: An integer to factor, or an already factored dictionary `{prime: exp}`.
        check_count: If provided, return `set()` when the quick upper-bound on
            the number of split-prime choices is less than this value.
        limited_checks: Accepted for API symmetry with `quadint.sums.decompose_number`.
            Fundamental inert-prime parity checks are still performed.
        no_trivial_solutions: Whether to discard axis and diagonal square-like
            solutions.

    Returns:
        A set of canonical nonnegative pairs `(a, b)`.
    """
    n_int, factors = _factor_input(n)
    if n_int < 1:
        return set()

    base = eisensteinint(1, 0)
    split_parts: list[list[eisensteinint]] = []

    for p, k in sorted(factors.items()):
        if k <= 0:
            continue

        if p == 3:
            base *= _prime_element(3) ** k
            continue

        if p % 3 == 2:
            if k & 1:
                return set()
            base *= p ** (k // 2)
            continue

        if p % 3 == 1:
            z = _prime_element(p)
            zc = z.conjugate()
            split_parts.append([z**i * zc ** (k - i) for i in range(k + 1)])
            continue

        # This should only be reachable for invalid factorizations.
        if not limited_checks:
            return set()

    if check_count is not None:
        predicted = math.prod(len(parts) for parts in split_parts)
        if predicted < check_count:
            return set()

    found: set[tuple[int, int]] = set()
    for choices in product(*split_parts):
        total = base
        for part in choices:
            total *= part

        sol = _canonical_pair(total, no_trivial_solutions=no_trivial_solutions)
        if sol is not None:
            found.add(sol)

    return found
