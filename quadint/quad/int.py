from typing import TYPE_CHECKING, Iterator, Union  # noqa: UP035

if TYPE_CHECKING:
    from quadint.quad.rings import QuadraticRing

OTHER_OP_TYPES = Union[complex, int, float]  # Types that QuadInt operations are compatible with (other than QuadInt)
_OTHER_OP_TYPES = (complex, int, float)  # I should be able to use the above with isinstance, but mypyc complains
OP_TYPES = Union['QuadInt', OTHER_OP_TYPES]


class QuadInt:
    """
    Element of a specific QuadraticRing.

    Stored as numerators a,b for (a + b*sqrt(D)) / den.
    """
    __slots__ = ("ring", "a", "b")

    ring: "QuadraticRing"
    a: int
    b: int

    def __init__(self, ring: "QuadraticRing", a: int = 0, b: int = 0) -> None:
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

    @property
    def zero(self) -> "QuadInt":
        """Additive identity (0)."""
        return self._make(0, 0)

    @property
    def one(self) -> "QuadInt":
        """Multiplicative identity (1)."""
        return self._make(self.ring.den, 0)

    def _from_obj(self, n: OP_TYPES):
        """Make a QuadInt on the current ring from a given object"""
        if isinstance(n, (int, float)):
            a = int(n)
            b = 0
        elif isinstance(n, QuadInt):
            return n
        elif isinstance(n, complex):
            if self.ring.D != -1 or self.ring.den != 1:
                raise TypeError("Cannot mix QuadInt from different rings")

            a = int(n.real)
            b = int(n.imag)
        else:
            return NotImplemented

        # The only time b is not 0 is if self.den is 1 anyway... No need to multiply
        return self._make(a * self.ring.den, b)

    def assert_same_ring(self, other: "QuadInt"):
        """Raise an error if other is not in the same ring as self"""
        if self.ring is not other.ring:
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
        result = self.one
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
        """
        if isinstance(other, _OTHER_OP_TYPES):
            other = self._from_obj(other)

        if not isinstance(other, QuadInt):
            raise NotImplementedError

        self.assert_same_ring(other)
        return self.ring.divmod(self, other)

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

        return self.ring is other.ring and self.a == other.a and self.b == other.b

    def __ne__(self, other: object) -> bool:
        # This shouldn't be required but mypyc is really messing this up...
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.a, self.b, self.ring.D, self.ring.den))

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
