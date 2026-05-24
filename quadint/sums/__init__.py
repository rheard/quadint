from __future__ import annotations

import math
import warnings

from itertools import product
from typing import TYPE_CHECKING

from sympy import factorint, sqrt_mod

from quadint.quad.rings.base import QuadraticRing

if TYPE_CHECKING:
    from quadint import QuadInt

_HEEGNER_D = {1, 2, 3, 7, 11, 19, 43, 67, 163}
_EUCLIDEAN_HEEGNER_D = {x for x in _HEEGNER_D if x < 15}  # Only the first 5 are Euclidean


def _factor_input(n: dict[int, int] | int) -> tuple[int, dict[int, int]]:
    """Return `(n, factorization)` whether the caller provided n or its factors."""
    if not isinstance(n, dict):
        return n, factorint(n)

    n_int = math.prod(p**k for p, k in n.items())
    return n_int, n


def _canonical_pair(
    A: int,
    B: int,
    d: int,
    den: int,
    *,
    no_trivial_solutions: bool,
) -> tuple[int, int] | None:
    """Convert numerator coordinates into a returned integer solution."""
    x, rA = divmod(abs(A), den)
    y, rB = divmod(abs(B), den)
    if rA or rB:
        return None

    if d == 1 and y < x:
        x, y = y, x

    if no_trivial_solutions and ((d == 1 and x == y) or x == 0 or y == 0):
        return None

    return x, y


def _orbit(z: QuadInt, *, no_trivial_solutions: bool = True) -> set[tuple[int, int]]:
    """
    Convert the torsion-unit orbit of a quadratic integer into integer-form solutions.

    The input `z` is an element of `QuadraticRing(-d)` whose norm represents a
    candidate value of `x^2 + d*y^2`. Because different unit multiples can produce
    distinct nonnegative integer-coordinate solutions, especially in the Eisenstein
    case `d == 3`, this helper tries every torsion-unit multiple of `z`.

    Args:
        z: A quadratic integer whose unit orbit should be converted to solutions.
        no_trivial_solutions: Whether to discard zero-coordinate and symmetric
            trivial solutions.

    Returns:
        A set of canonical nonnegative integer pairs `(x, y)` satisfying
            `x^2 + d*y^2 == abs(z)` whenever the conversion is integral.
    """
    d = abs(z.ring.D)
    den = z.ring.den
    out: set[tuple[int, int]] = set()

    for u in z.units:
        w = z * u
        sol = _canonical_pair(
            w.a,
            w.b,
            d,
            den,
            no_trivial_solutions=no_trivial_solutions,
        )
        if sol is not None:
            out.add(sol)

    return out


def _squarefree_part_and_scale(d: int) -> tuple[int, int]:
    """Return (sf, scale) such that d == sf * scale**2, with sf squarefree."""
    sf = 1
    scale = 1

    for p, k in factorint(d).items():
        if k & 1:
            sf *= p
        scale *= p ** (k // 2)

    return sf, scale


def _euclids_algorithm(a: int, b: int, c: int) -> int | None:
    """Runs Euclid's algorithm and returns remainder"""
    while b > c:
        r = a % b
        a, b = b, r
        if not b:
            return None

    return b


def _decompose_prime_den1(p: int, d: int = 1) -> tuple[int, int]:
    """decompose_prime when den=1, this is the original algorithm"""
    if p == 2:
        if d == 1:
            return 1, 1
        if d == 2:
            return 0, 1

        raise ValueError(f"Could not decompose {p!r} with d={d!r}")

    # If sqrt(-d) mod p doesn't exist, no solution for this prime
    t = sqrt_mod(-d, p, all_roots=False)
    if t is None:
        raise ValueError(f"Could not decompose {p!r} with d={d!r}")

    def _try_cornacchia_root(p: int, d: int, t: int) -> tuple[int, int] | None:
        p_sqrt = math.isqrt(p)

        x = _euclids_algorithm(p, t, p_sqrt)
        if x is None:
            return None

        rhs = p - x * x
        if rhs < 0:
            return None

        y2, r_rhs = divmod(rhs, d)
        if r_rhs != 0:
            return None

        y = math.isqrt(y2)
        if y * y != y2:
            return None

        return abs(x), abs(y)

    res = _try_cornacchia_root(p, d, t)
    if res is None:
        res = _try_cornacchia_root(p, d, (p - t) % p)

    if res is None:
        raise ValueError(f"Could not decompose {p!r} with d={d!r}")

    if res[0] > res[1] and d == 1:
        return res[1], res[0]

    return res


def _decompose_prime_den2(p: int, d: int = 1) -> tuple[int, int]:
    """decompose_prime when den=2"""
    den = 2

    # Pass 1: if p = x^2 + d*y^2, lift to den=2 numerator coordinates.
    try:
        x, y = decompose_prime(p, d)
    except ValueError:
        pass
    else:
        A = den * x
        B = den * y
        return abs(A), abs(B)

    # Pass 2: genuinely den=2 case, e.g. 11 in D=-19:
    #     5^2 + 19*1^2 = 4*11
    target = 4 * p
    roots = sqrt_mod(-d, p, all_roots=True)
    if roots is None:
        raise ValueError(f"Could not decompose {p!r} with d={d!r}, den={den!r}")

    def _cornacchia_prime_remainder_candidates(p: int, root: int):
        """Yield Euclidean remainders from (p, root), preserving the chosen root."""
        a = p
        b = int(root) % p

        while True:
            yield b
            if not b:
                return
            a, b = b, a % b

    for root in roots:
        for A in _cornacchia_prime_remainder_candidates(p, int(root)):
            if target < A * A:
                continue

            rhs = target - A * A
            B2, B2_r = divmod(rhs, d)
            if B2_r:
                continue

            B = math.isqrt(B2)
            if B * B != B2:
                continue

            if den == 2 and ((A ^ B) & 1):
                continue

            return abs(A), abs(B)

    raise ValueError(f"Could not decompose {p!r} with d={d!r}, den={den!r}")


def decompose_prime(p: int, d: int = 1, den: int = 1) -> tuple[int, int]:
    """
    Decompose a prime number into (a**2 + d * b**2) / den**2

    There will be at most 1 solution for primes.
        If d == 1, this will only be if the prime is equal 1 mod 4 according to Fermat's theorem on sums of two squares.

    This is based on the algorithm described by Stan Wagon (1990),
        based on work by Serret and Hermite (1848), and Cornacchia (1908)

    Returns:
        tuple<int, int>: a and b

    Raises:
        ValueError: If p cannot be decomposed because it is 3 mod 4
    """
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d!r}")

    if den not in (1, 2):
        raise ValueError(f"den must be 1 or 2, got {den!r}")

    if den == 1:
        return _decompose_prime_den1(p, d)

    return _decompose_prime_den2(p, d)


def decompose_number(
    n: dict[int, int] | int,
    d: int = 1,
    check_count: int | None = None,
    *,
    limited_checks: bool = False,
    no_trivial_solutions: bool = True,
    warn: bool = True,
) -> set[tuple[int, int]]:
    """
    Decompose any number into all possible integer (x, y) solutions to:

        x^2 + d*y^2 = n

    Notes on correctness/completeness:
      - This function only produces TRUE solutions (no false positives) as long as it only
        multiplies ring elements whose norms match the intended prime powers.
      - Completeness (“enumerate all solutions”) is guaranteed only in the nicest cases
        (roughly: when the relevant quadratic integer ring is a UFD and prime decomposition
        exists for the necessary primes). Otherwise, it is still a good heuristic and often works,
        but can miss solutions.

    Args:
        n (int, dict): The number to decompose. Can be an integer which will be factored,
            or the already factored number.
        d: coefficient in x^2 + d*y^2 (d >= 1).
        check_count (int): If provided, and it is predicted that a number will have fewer than this many solutions,
            that number is skipped and an empty list is returned instead.
        limited_checks (bool): Only run limited checks. Should only be used with prepared input
            or false positive will appear.
        no_trivial_solutions (bool): Exclude trivial solutions? Defined as any symmetrical solution, or any
            solution with 0. Essentially excludes perfect squares and doubles of perfect squares.
            Note that a value of False will make the algorithm quite a bit slower.
        warn: if True, emits a warning about when the enumeration is exact vs heuristic.

    Returns:
        set<tuple<int, int>>: All unique solutions (x, y)
    """

    # Step 1: Factor n. This is the most time consuming step, especially on larger numbers. Avoid if possible
    n_int, factors = _factor_input(n)

    # Step 1.1: Sanitize d
    sf_d, y_scale = _squarefree_part_and_scale(d)
    if y_scale != 1:
        raw = decompose_number(
            n_int,
            sf_d,
            check_count=None,
            limited_checks=limited_checks,
            no_trivial_solutions=False,
            warn=warn,
        )

        out: set[tuple[int, int]] = set()

        for x, z in raw:
            candidates = [(x, z)]

            # When the reduced form is x^2 + z^2, the variables are symmetric.
            # decompose_number(..., d=1) canonicalizes to x <= z, but after
            # substituting z = y_scale*y, the orientation matters again.
            if sf_d == 1 and x != z:
                candidates.append((z, x))

            for x0, z0 in candidates:
                y, y_r = divmod(z0, y_scale)
                if y_r:
                    continue

                # d here is the original d, not sf_d. For d=4, x/y are not symmetric.
                if d == 1 and y < x0:
                    x0, y = y, x0

                if no_trivial_solutions and ((d == 1 and x0 == y) or x0 == 0 or y == 0):
                    continue

                out.add((x0, y))

        if check_count is not None and len(out) < check_count:
            return set()

        return out

    if warn and d not in _EUCLIDEAN_HEEGNER_D:
        if d not in _HEEGNER_D:
            warnings.warn(
                "decompose_number: d is NOT a Heegner (class number 1) value. Completeness is not guarenteed.",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                "decompose_number: d is not a Euclidean quadratic field. Completeness is not guarenteed.",
                UserWarning,
                stacklevel=2,
            )

    Q = QuadraticRing(-d)
    den = Q.den

    # Look for shortcuts
    if len(factors) == 0:  # p=1
        if not no_trivial_solutions:
            if d == 1:
                return {(0, 1)}  # d=1, solutions are symmetrical. Return sorted solution
            return {(1, 0)}  # For all other d, the only possible solution is 1**2 + d*0**2

        return set()

    if len(factors) == 1 and sum(factors.values()) == 1:
        # Only 1 factor with a power of 1, this is a prime number
        p = next(iter(factors))
        if check_count and check_count > 1:
            return set()  # There will only be 1 solution. If check_count is greater than that, do nothing

        try:
            A, B = decompose_prime(p, d, den)
        except ValueError:
            return set()

        sol = _canonical_pair(
            A,
            B,
            d,
            den,
            no_trivial_solutions=no_trivial_solutions,
        )
        return {sol} if sol else set()

    # Split factors into:
    #   - representable primes (we can get (a,b) with a^2 + d b^2 = p)
    #   - inert-ish primes (cannot represent p itself; require even exponent so we can scale by p^(k/2))
    representable: dict[int, int] = {}
    inert_even_scale: dict[int, int] = {}
    p_decompositions: dict[int, tuple[int, int]] = {}

    for p, k in factors.items():
        if k <= 0:
            continue

        # Prime shortcut for p itself: if decompose_prime succeeds, we can treat it as representable.
        # If it fails, then we only know how to deal with it safely when exponent is even (scale).
        try:
            decomposition = decompose_prime(p, d, den)
        except ValueError:
            inert_even_scale[p] = k
        else:
            p_decompositions[p] = decomposition
            representable[p] = k

    # If we have any “non-representable” primes with odd exponent, we cannot build the right norm.
    if (not limited_checks or no_trivial_solutions) and any(k % 2 == 1 for k in inert_even_scale.values()):
        return set()

    # Predicted upper bound on #solutions from conjugate-choice enumeration:
    # product over representable primes of (k+1)
    if check_count:
        predicted = math.prod(k + 1 for k in representable.values())
        if predicted < check_count:
            return set()

    # Scalar coefficient from “inert-even” primes: scale by p^(k/2)
    base = math.prod(p ** (k // 2) for p, k in inert_even_scale.items())

    if not representable:
        return _orbit(Q(base * den, 0), no_trivial_solutions=no_trivial_solutions)

    p_ring_pairs = {}
    for p, (a, b) in p_decompositions.items():
        # Represent a + b*sqrt(-d) in quadint's numerator-scaled storage:
        z_ = Q(a, b)
        p_ring_pairs[p] = (z_, z_.conjugate())

    # This is purely to help mypyc with type-checking,
    #   to guarantee that base will be a QuadInt
    base_quad: QuadInt = Q.one
    base_quad *= base

    if no_trivial_solutions:
        # Base-item trick: fix one factor to reduce symmetry.
        first_p = next(iter(representable))
        representable[first_p] -= 1  # consume one occurrence as the fixed base
        base_quad *= p_ring_pairs[first_p][0]

    found: set[tuple[int, int]] = set()

    total_slots = sum(representable.values())
    for choices in product([0, 1], repeat=total_slots):  # runs once if repeat=0
        total = base_quad
        choice_i = 0
        for p, k in representable.items():
            for _ in range(k):
                total *= p_ring_pairs[p][choices[choice_i]]
                choice_i += 1

        found |= _orbit(total, no_trivial_solutions=no_trivial_solutions)

    return found
