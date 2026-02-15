from typing import Callable, Iterator, Tuple, Union

OTHER_OP_TYPES = Union[complex, int, float]  # Types that QuadInt operations are compatible with (other than QuadInt)
_OTHER_OP_TYPES = (complex, int, float)  # I should be able to use the above with isinstance, but mypyc complains
OP_TYPES = Union['QuadInt', OTHER_OP_TYPES]


def _round_div_ties_away_from_zero(n: int, d: int) -> int:
    """Round n/d to nearest int; ties go away from 0. d must be > 0."""
    if d == 0:
        raise ZeroDivisionError
    if d < 0:
        n, d = -n, -d
    if n >= 0:
        return (n + d // 2) // d
    return -((-n + d // 2) // d)


def _split_uv(x: "QuadInt") -> Tuple[int, int]:
    """Return (u,v) for D=1 split-complex where u=(a+b)/den, v=(a-b)/den."""
    den = x.ring.den
    apb = x.a + x.b
    amb = x.a - x.b
    if apb % den or amb % den:
        # should be impossible if ring invariants hold
        raise ArithmeticError("Non-integral split-complex coordinates; check ring parameters/parity")
    return apb // den, amb // den


def _choose_best_in_neighborhood(
    *,
    A0: int,
    B0_for_A: Callable,
    score_for_AB: Callable,
    den: int,
) -> Tuple[int, int]:
    """
    Search (A0±1) * (B0(A)±1) and return best (A,B).

    Enforces den==2 parity constraint: A == B (mod 2).

    Returns:
        (bestA, bestB): The best options found for this search.
    """
    best_score: Union[Tuple[int, ...], None] = None
    bestA = bestB = 0

    for A in (A0 - 1, A0, A0 + 1):
        B0 = B0_for_A(A)
        for B in (B0 - 1, B0, B0 + 1):
            if den == 2 and ((A ^ B) & 1):
                continue

            s = score_for_AB(A, B)
            if best_score is None or s < best_score:
                best_score = s
                bestA, bestB = A, B

    return bestA, bestB


class QuadraticRing:
    """
    The quadratic integer ring (order) with basis (1, sqrt(D)) and fixed denominator den in {1,2}.

    Elements are represented as:
        (a + b*sqrt(D)) / den

    where a,b are integers stored as numerators.
    When den==2, integrality requires a ≡ b (mod 2).

    This object is *not* a type factory (no nested classes) — it just carries parameters.
    """

    __slots__ = ("D", "den")

    D: int
    den: int

    def __init__(self, D: int, den: Union[int, None] = None) -> None:
        """Initialize the ring settings"""
        self.D = int(D)

        if den is None:
            self.den = 2 if (self.D % 4) == 1 else 1
        else:
            self.den = den

    def __repr__(self) -> str:
        return f"QuadraticRing(D={self.D}, den={self.den})"

    def __call__(self, a: int = 0, b: int = 0) -> "QuadInt":
        """Create element (a + b*sqrt(D))/den with numerator coefficients a,b."""
        return QuadInt(self, int(a), int(b))

    def __contains__(self, x: object) -> bool:
        """Return True iff x is a QuadInt element of this ring (by parameters)."""
        if isinstance(x, int):
            return True

        if isinstance(x, float):
            return x.is_integer()

        if isinstance(x, complex):
            return self.D == -1 and self.den == 1

        if not isinstance(x, QuadInt):
            return False

        other = x.ring
        return (other.D == self.D) and (other.den == self.den)

    def from_obj(self, n: OP_TYPES) -> "QuadInt":
        """Embed integer (or float) n as (n*den + 0*sqrt(D))/den. Also supports complex if D==-1"""
        if isinstance(n, (int, float)):
            a = int(n)
            b = 0
        elif isinstance(n, complex):
            if self.D != -1:
                raise TypeError("Cannot mix QuadInt from different rings")

            a = int(n.real)
            b = int(n.imag)
        elif isinstance(n, QuadInt):
            return n
        else:
            return NotImplemented

        # The only time b is not 0 is if self.den is 1 anyway... No need to multiply
        return QuadInt(self, a * self.den, b)

    def from_ab(self, a: int, b: int) -> "QuadInt":
        """Create an integer keeping in mind the denominator"""
        da = int(a) * self.den
        db = int(b) * self.den
        return QuadInt(self, da, db)


class QuadInt:
    """
    Element of a specific QuadraticRing.

    Stored as numerators a,b for (a + b*sqrt(D)) / den.
    """
    __slots__ = ("ring", "a", "b")

    ring: QuadraticRing
    a: int
    b: int

    def __init__(self, ring: QuadraticRing, a: int = 0, b: int = 0) -> None:
        """Init and validate the integer works for this ring"""
        self.ring = ring
        self.a = int(a)
        self.b = int(b)

        den = self.ring.den
        if den == 2 and ((self.a ^ self.b) & 1):
            raise ValueError("For den=2, a and b must have the same parity")

    def _make(self, a: int, b: int):
        """Construct a new value of *this* conceptual type from internal numerators a,b."""
        return self.__class__(self.ring, a, b)

    def _from_obj(self, n: OP_TYPES):
        """Make a QuadInt on the current ring from a given object"""
        if isinstance(n, (int, float)):
            a = int(n)
            b = 0
        elif isinstance(n, QuadInt):
            return n
        else:
            return NotImplemented

        # The only time b is not 0 is if self.den is 1 anyway... No need to multiply
        return self._make(a * self.ring.den, b)

    def assert_same_ring(self, other: "QuadInt"):
        """Raise an error if other is not in the same ring as self"""
        if self.ring is not other.ring:
            # identity is fastest; if you want structural equality use (D,den)
            raise TypeError("Cannot mix QuadInt from different rings")

    def conjugate(self):
        """(a + b√D)/den -> (a - b√D)/den."""
        return self._make(self.a, -self.b)

    def __add__(self, other: OP_TYPES):
        if isinstance(other, _OTHER_OP_TYPES):
            other = self._from_obj(other)

        if isinstance(other, QuadInt):
            self.assert_same_ring(other)
            return self._make(self.a + other.a, self.b + other.b)

        return NotImplemented

    def __radd__(self, other: OTHER_OP_TYPES):
        return self.__add__(other)

    def __sub__(self, other: OP_TYPES):
        if isinstance(other, _OTHER_OP_TYPES):
            other = self._from_obj(other)

        if isinstance(other, QuadInt):
            self.assert_same_ring(other)
            return self._make(self.a - other.a, self.b - other.b)

        return NotImplemented

    def __rsub__(self, other: OTHER_OP_TYPES):
        return self.__neg__().__add__(other)

    def __neg__(self):
        return self._make(-self.a, -self.b)

    def __pos__(self):
        return self._make(self.a, self.b)

    def __mul__(self, other: OP_TYPES):
        if isinstance(other, _OTHER_OP_TYPES):
            other = self._from_obj(other)

        if isinstance(other, QuadInt):
            self.assert_same_ring(other)

            # Numerator algebra:
            # (a+b√D)(c+d√D) = (ac + bdD) + (ad+bc)√D
            # But values are /den. If both are (..)/den then product is (..)/den^2.
            # We keep representation /den by dividing numerator by den once:
            # (xy)/den^2 == (xy/den)/den  -> require xy divisible by den.
            D = self.ring.D
            den = self.ring.den

            a1, b1 = self.a, self.b
            a2, b2 = other.a, other.b

            A = a1 * a2 + b1 * b2 * D
            B = a1 * b2 + a2 * b1

            if den != 1:
                if (A % den) != 0 or (B % den) != 0:
                    raise ArithmeticError("Non-integral product; check ring parameters / parity")

                A //= den
                B //= den

            return self._make(A, B)

        return NotImplemented

    def __rmul__(self, other: OTHER_OP_TYPES):
        return self.__mul__(other)

    def __pow__(self, exp: float):
        e = int(exp)
        if e < 0:
            raise ValueError("Negative powers not supported in quadratic integer rings")

        # exponentiation by squaring
        result = self._make(self.ring.den, 0)
        base = self
        while e:
            if e & 1:
                result = result * base

            e >>= 1
            if e:
                base = base * base

        return result

    # region Euclidean-ish division (no Fraction; small neighborhood search in integer metric)
    def __divmod__(self, other: OP_TYPES):
        """
        Nearest-lattice division for D <= 0 (imaginary quadratic).

        Intended for Euclidean rings (e.g., D=-1, -2, -3, -7, -11 in the maximal order).

        Returns:
            tuple: The quotient and remainder of the division with other.

        Raises:
            ZeroDivisionError: For division by zero.
        """
        if isinstance(other, _OTHER_OP_TYPES):
            other = self._from_obj(other)

        if not isinstance(other, QuadInt):
            raise NotImplementedError

        self.assert_same_ring(other)

        D = self.ring.D
        den = self.ring.den

        if D > 1:
            raise NotImplementedError(
                "divmod implemented only for D<=1 (imaginary quadratic, dual numbers or split-complex)",
            )

        if D == 1:
            u1, v1 = _split_uv(self)
            u2, v2 = _split_uv(other)

            # Division by zero divisor (u2==0 or v2==0) is not well-defined.
            if u2 == 0 or v2 == 0:
                raise ZeroDivisionError("division by zero divisor in split-complex integers (a=±b)")

            qu0 = _round_div_ties_away_from_zero(u1, abs(u2))
            if u2 < 0:
                qu0 = -qu0

            qv0 = _round_div_ties_away_from_zero(v1, abs(v2))
            if v2 < 0:
                qv0 = -qv0

            def B0_for_A(A: int) -> int:  # noqa: ARG001
                return qv0

            def score_for_AB(A: int, B: int) -> Tuple[int, ...]:
                # remainder in (u,v)
                ru = u1 - A * u2
                rv = v1 - B * v2
                return (ru * ru + rv * rv, )

            # Enforce qu ≡ qv (mod 2) so (qu+qv)/2 and (qu-qv)/2 are integers.
            best_qu, best_qv = _choose_best_in_neighborhood(
                A0=qu0,
                B0_for_A=B0_for_A,
                score_for_AB=score_for_AB,
                den=2,  # re-use parity constraint check ((A^B)&1)==0
            )

            # Convert back: a = den*(qu+qv)/2, b = den*(qu-qv)/2
            s = best_qu + best_qv
            t = best_qu - best_qv
            # s,t are even because qu,qv same parity
            qa = (s * den) // 2
            qb = (t * den) // 2

            q = self._make(qa, qb)
            r = self - q * other
            return q, r

        if D == 0:
            # In dual numbers, (c + dε) is invertible iff c != 0.
            n = other.a
            num = self
        else:
            n = abs(other)
            num = self * other.conjugate()  # still in numerator-units for /den representation

        if n == 0:
            raise ZeroDivisionError

        A0 = _round_div_ties_away_from_zero(num.a, n)

        # Special case: dual numbers (D == 0)
        if D == 0:
            c, d = other.a, other.b

            def B0_for_A(A: int) -> int:
                return _round_div_ties_away_from_zero(self.b - A * d, c)

            # Lexicographic “small remainder”: minimize real remainder first, then ε remainder.
            def score_for_AB(A: int, B: int) -> Tuple[int, ...]:
                r0 = self.a - A * c
                r1 = self.b - A * d - B * c
                return (r0 * r0, r1 * r1)
        else:
            B0 = _round_div_ties_away_from_zero(num.b, n)
            absD = -D

            def B0_for_A(A: int) -> int:  # noqa: ARG001
                return B0

            def score_for_AB(A: int, B: int) -> Tuple[int, ...]:
                da = A * n - num.a
                db = B * n - num.b
                return (da * da + absD * (db * db), )

        bestA, bestB = _choose_best_in_neighborhood(
            A0=A0, B0_for_A=B0_for_A, score_for_AB=score_for_AB, den=den,
        )

        q = self._make(bestA, bestB)
        r = self - q * other
        return q, r

    def __truediv__(self, other: OP_TYPES):
        return self.__floordiv__(other)

    def __rtruediv__(self, other: OTHER_OP_TYPES):
        if isinstance(other, _OTHER_OP_TYPES):
            new_other = self._from_obj(other)
            return new_other.__truediv__(self)

        return NotImplemented

    def __floordiv__(self, other: OP_TYPES):
        q, _ = divmod(self, other)
        return q

    def __rfloordiv__(self, other: OTHER_OP_TYPES):
        if isinstance(other, _OTHER_OP_TYPES):
            new_other = self._from_obj(other)
            return new_other.__floordiv__(self)

        return NotImplemented

    def __mod__(self, other: "QuadInt"):
        _, r = divmod(self, other)
        return r
    # endregion

    def __abs__(self) -> int:
        """
        N((a+b√D)/den) = (a^2 - D*b^2) / den^2

        Always an integer for valid ring elements.

        Returns:
            int: The norm for the ring.

        Raises:
            ArithmeticError: If there is a non-integral norm for the ring.
        """
        D = self.ring.D
        den = self.ring.den
        num = self.a * self.a - D * self.b * self.b
        dd = den * den
        if (num % dd) != 0:
            raise ArithmeticError("Non-integral norm; check ring parameters / parity")
        return num // dd

    def __bool__(self) -> bool:
        return (self.a | self.b) != 0

    def __iter__(self) -> Iterator[int]:
        return iter((self.a, self.b))

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> int:
        if idx == 0:
            return self.a
        if idx == 1:
            return self.b
        raise IndexError("Quadratic integer index out of range (valid: 0 or 1)")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _OTHER_OP_TYPES):
            other = self._from_obj(other)

        if not isinstance(other, QuadInt):
            return False

        return self.ring.D == other.ring.D and self.ring.den == other.ring.den \
            and self.a == other.a and self.b == other.b

    def __ne__(self, other: object) -> bool:
        # This shouldn't be required but mypyc is really messing this up...
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.a, self.b, self.ring.D))

    def __repr__(self) -> str:
        D = self.ring.D
        den = self.ring.den

        parens = self.a != 0

        lead = "(" if parens else ""
        tail = ")" if parens else ""
        op = ("+" if parens else "") if self.b >= 0 else "-"
        a = self.a or ""
        b = abs(self.b)
        symbol = f"*sqrt({D})"

        core = f"{lead}{a}{op}{b}{symbol}{tail}"
        return f"{core}/{den}" if den != 1 else core


def make_quadint(D: int) -> QuadraticRing:
    """
    Returns a *ring instance* you can keep around:

        Q = make_quadint(-3)
        z = Q.from_ab(5, 2)   # (10 + 4*sqrt(-3))/2

    Returns:
        QuadraticRing: The ring instance which can be used almost as a type.
    """
    return QuadraticRing(D)
