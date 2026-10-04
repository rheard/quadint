from __future__ import annotations

import math

from itertools import product
from typing import TYPE_CHECKING

from sympy import factorint, sqrt_mod

from quadint.quad.rings.base import QuadraticRing

if TYPE_CHECKING:
    from quadint import QuadInt
    from quadint.quad.ideal import Ideal

# The d where QuadraticRing(-d) has class number one (Heegner, Stark): every prime ideal there has a generator, so the
#   elements of norm p for the primes p of n multiply out to every solution. Any other d goes through ideals instead.
_HEEGNER_D = {1, 2, 3, 7, 11, 19, 43, 67, 163}


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
    # This is Cornacchia's algorithm modified for A^2 + d*B^2 = 4p (Cohen, "A Course in Computational Algebraic
    #   Number Theory", Algorithm 1.5.3): take the square root of -d mod p with the same parity as d, and run Euclid
    #   on (2p, root) until the remainder is at most 2*sqrt(p). If there is a solution, that remainder is its A.
    #   (Euclid on (p, root) finds most of them, but not all: 4*179 == 21^2 + 11*5^2 is missed that way.)
    root = sqrt_mod(-d, p, all_roots=False)
    if root is None:
        raise ValueError(f"Could not decompose {p!r} with d={d!r}, den={den!r}")

    a, A = 2 * p, int(root)
    if (A ^ d) & 1:
        A = p - A

    limit = math.isqrt(4 * p)  # floor(2*sqrt(p))
    while limit < A:
        a, A = A, a % A

    B2, B2_r = divmod(4 * p - A * A, d)
    B = math.isqrt(B2)
    if B2_r or B * B != B2 or ((A ^ B) & 1):
        raise ValueError(f"Could not decompose {p!r} with d={d!r}, den={den!r}")

    return A, B


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


def _decompose_by_ideals(
    ring: QuadraticRing,
    factors: dict[int, int],
    check_count: int | None,
    *,
    no_trivial_solutions: bool,
) -> set[tuple[int, int]]:
    """
    Find every solution of x**2 + d*y**2 == n through the ideals of norm n, for d with class number above one.

    There, some primes p have no element of norm p (the prime ideals over them have no generator), but products of those
        ideals can still have one, like 6 == 1**2 + 5*1**2 with neither 2 nor 3 of the form x**2 + 5*y**2. Every element
        of norm n generates an ideal of norm n, and those are the products of one choice per prime power p**k of n:
        P**i * conj(P)**(k - i) for any i when p splits into P and conj(P), P**k when p ramifies as P**2, and
        (p)**(k/2) when p is inert, which needs an even k. So the solutions come from the generators of the principal
        ones, each of which is only unique up to a unit (which _orbit tries).

    Returns:
        set[tuple[int, int]]: The solutions, in the form decompose_number returns them.
    """
    # Every ideal over an inert p has an even power of p as its norm, so an inert p with an odd exponent rules out all
    #   of them. Most n fail like this, and Euler's criterion finds out without building any ideals: a p that does not
    #   divide the discriminant is inert when the discriminant is not a square mod p (for p == 2: when it is 5 mod 8).
    disc = ring.discriminant()
    for p, k in factors.items():
        if k % 2 and disc % p and (disc % 8 == 5 if p == 2 else pow(disc, (p - 1) // 2, p) == p - 1):
            return set()

    choices_by_prime: list[list[Ideal]] = []
    for p, k in factors.items():
        if k <= 0:
            continue

        # The uncached _prime_ideals_data_over, since a long run over many n would otherwise keep every prime it meets
        prime_ideals = [data.ideal for data in ring._prime_ideals_data_over(p)]
        if len(prime_ideals) == 2:
            P, P_bar = prime_ideals
            choices_by_prime.append([P**i * P_bar ** (k - i) for i in range(k + 1)])
        elif prime_ideals[0].norm == p:
            choices_by_prime.append([prime_ideals[0] ** k])
        else:
            choices_by_prime.append([ring.ideal(p ** (k // 2))])  # inert, so k is even (see above)

    # Each of these ideals gives at most one solution, since its generator is only unique up to sign (d > 3 here)
    if check_count and math.prod(len(choices) for choices in choices_by_prime) < check_count:
        return set()

    ideals = [ring.unit_ideal()]
    for choices in choices_by_prime:
        ideals = [ideal * choice for ideal in ideals for choice in choices]

    found: set[tuple[int, int]] = set()
    for ideal in ideals:
        generator = ideal._generator()  # uncached, for the same reason as above
        if generator is not None:
            found |= _orbit(generator, no_trivial_solutions=no_trivial_solutions)

    return found


def decompose_number(
    n: dict[int, int] | int,
    d: int = 1,
    check_count: int | None = None,
    *,
    limited_checks: bool = False,
    no_trivial_solutions: bool = True,
    warn: bool = True,  # ruff: ignore[unused-function-argument]
) -> set[tuple[int, int]]:
    """
    Decompose any number into all possible integer (x, y) solutions to:

        x^2 + d*y^2 = n

    Every solution is found, for every d. When QuadraticRing(-d) has class number one (d is 1, 2, 3, 7, 11, 19, 43, 67
        or 163, once square factors are taken out of d), each solution is a product of elements with prime norms,
        which this multiplies out directly from the factorization of n. Any other d goes through the ideals of norm n
        instead, since some of its primes have no element of that norm (see _decompose_by_ideals), which is slower.

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
        warn: Ignored, and only kept so existing calls still work. It used to warn that a d could miss solutions.

    Returns:
        set<tuple<int, int>>: All unique solutions (x, y)

    Raises:
        ValueError: If d < 1, where there can be infinitely many solutions (like 5**2 + 0*y**2 == 25 for every y).
    """
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d!r}")

    # Step 1: Factor n. This is the most time consuming step, especially on larger numbers. Avoid if possible
    n_int, factors = _factor_input(n)

    # x^2 + d*y^2 is never negative, and it is only 0 at (0, 0), which is a trivial solution
    if n_int <= 0:
        if n_int < 0 or no_trivial_solutions or (check_count and check_count > 1):
            return set()

        return {(0, 0)}

    # Step 1.1: Sanitize d
    sf_d, y_scale = _squarefree_part_and_scale(d)
    if y_scale != 1:
        raw = decompose_number(
            n_int,
            sf_d,
            check_count=None,
            limited_checks=limited_checks,
            no_trivial_solutions=False,
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

    # The rest takes every prime without an element of its norm to be inert, which only holds with class number one
    if d not in _HEEGNER_D:
        return _decompose_by_ideals(Q, factors, check_count, no_trivial_solutions=no_trivial_solutions)

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

    # Each of a prime's k remaining factors is pi or conj(pi), and since multiplication commutes, their product only
    #   depends on how many of them are pi. So the 2**k ways to choose give exactly the same products as the k + 1
    #   powers pi**i * conj(pi)**(k - i), each just repeated many times over.
    #   When conj(pi) is a unit times pi (a ramified prime), those are all unit multiples of pi**k, and _orbit already
    #   tries every unit multiple of the total, so pi**k alone gives the same solutions.
    choices_by_prime: list[list[QuadInt]] = []
    for p, k in representable.items():
        pi, pi_bar = p_ring_pairs[p]
        if any(pi_bar == pi * u for u in pi.units):
            choices_by_prime.append([pi**k])
        else:
            choices_by_prime.append([pi**i * pi_bar ** (k - i) for i in range(k + 1)])

    found: set[tuple[int, int]] = set()

    for choices in product(*choices_by_prime):
        total = base_quad
        for choice in choices:
            total *= choice

        found |= _orbit(total, no_trivial_solutions=no_trivial_solutions)

    return found
