from __future__ import annotations

from sympy import factorint


def _is_squarefree(n: int | dict[int, int]) -> bool:
    if isinstance(n, int):
        n = abs(n)
        if n <= 1:
            return False

        facts = factorint(n)
    else:
        if not n:
            return False

        facts = n

    return all(i < 2 for i in facts.values())


def _sqrt_mod_prime(a: int, p: int) -> int | None:
    """
    Return a square root of a modulo the prime p, or None if a is not a square mod p.

    This is Tonelli-Shanks, in place of sympy's sqrt_mod. That one handles any modulus, which costs it a primality check
        and more on every call: about half the time decompose_number spent on numbers that were already factored.

    p is not checked for primality, for the same reason, but a composite p (which decompose_prime passes on if it is
        given one) still gets an answer: a root, or None, which then does not rule out a root.

    Returns:
        int | None: Some r with r*r % p == a % p (p - r is the other one).
    """
    a %= p
    if a == 0 or p == 2:
        return a

    if pow(a, (p - 1) // 2, p) != 1:
        return None  # Euler's criterion: a is not a square

    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)  # a**((p + 1)/2) == a * a**((p - 1)/2) == a

    # Write p - 1 == q * 2**s with q odd, and find some z that is not a square
    q, s = p - 1, 0
    while not q & 1:
        q >>= 1
        s += 1

    # Mod a prime, Euler's criterion makes every z**((p - 1)/2) 1 or -1, so anything else means p is composite. That
    #   turns up by p's smallest prime factor at the latest. It has to be caught, since a composite p need not have any
    #   z that gives -1 (25 and 65 have none), and then this would look for one forever.
    z = 2
    while True:
        euler = pow(z, (p - 1) // 2, p)
        if euler == p - 1:
            break

        if euler != 1:
            return None

        z += 1

    # Each step keeps r*r == a*t (mod p), with the order of t dividing 2**(m - 1), until t == 1. Then r is the root.
    m, c, t, r = s, pow(z, q, p), pow(a, q, p), pow(a, (q + 1) // 2, p)
    while t != 1:
        # The least i with t**(2**i) == 1, which is below m when p is prime
        i, t2 = 0, t
        while t2 != 1:
            t2 = t2 * t2 % p
            i += 1

        if i == m:
            return None  # only a composite p gets here (like 3277 for a == -7), and the next shift would be negative

        b = pow(c, 1 << (m - i - 1), p)
        m, c, t, r = i, b * b % p, t * b * b % p, r * b % p

    return r
