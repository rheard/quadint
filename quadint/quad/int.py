from __future__ import annotations

from math import gcd
from typing import TYPE_CHECKING, ClassVar, Iterator  # noqa: UP035

from sympy import factorint, isprime

if TYPE_CHECKING:
    from quadint.quad.rings import Factorization, QuadraticRing

_OTHER_OP_TYPES = (complex, int, float)  # I should be able to use the above with isinstance, but mypyc complains


def _key(w_: QuadInt) -> tuple[int, int, int, int, int]:
    """This is required (for now) as it appears that mypyc is having problems with sub-functions/lambdas"""
    return abs(abs(w_)), abs(w_.b), abs(w_.a), w_.a, w_.b


class QuadInt:
    """
    Element of a specific QuadraticRing.

    Stored as numerators a,b for (a + b*sqrt(D)) / den.
    """

    __slots__ = ("ring", "a", "b")

    ring: QuadraticRing
    a: int
    b: int

    SYMBOL: ClassVar[str] = "*sqrt({D})"
    DEFAULT_RING: ClassVar[QuadraticRing | None] = None

    # (a, b)^T = M * (x, y)^T
    BASIS_TO_INTERNAL: ClassVar[tuple[tuple[int, int], tuple[int, int]]] = ((1, 0), (0, 1))
    # (x, y)^T = (N * (a, b)^T) / INTERNAL_TO_BASIS_DEN
    INTERNAL_TO_BASIS: ClassVar[tuple[tuple[int, int], tuple[int, int]]] = ((1, 0), (0, 1))
    INTERNAL_TO_BASIS_DEN: ClassVar[int] = 1

    def __init__(self, a: int = 0, b: int = 0, ring: QuadraticRing | None = None, *, skip_basis: bool = False):
        """Init and validate the integer works for this ring"""
        ring = ring or self.DEFAULT_RING
        if ring is None:
            raise ValueError("A ring must be specified in some form to use a quadratic integer!")

        self.ring = ring
        if not skip_basis:
            a, b = self._basis_to_internal(int(a), int(b))
        else:
            a, b = int(a), int(b)

        self.a, self.b = a, b

        den = self.ring.den
        if den == 2 and ((self.a ^ self.b) & 1):
            raise ValueError("For den=2, a and b must have the same parity")

    def __init_subclass__(cls, *args: tuple, **kwargs: dict):
        """Register a subclass with the ring if it is defined"""
        super().__init_subclass__(*args, **kwargs)
        ring = getattr(cls, "DEFAULT_RING", None)
        if ring is not None:
            ring.DEFAULT_KLASS = cls

    def _make(self, a: int, b: int):
        """Construct a new value of *this* conceptual type from internal numerators a,b."""
        return self.__class__(a, b, self.ring, skip_basis=True)

    @property
    def zero(self) -> QuadInt:
        """Additive identity (0)."""
        return self._make(0, 0)

    @property
    def one(self) -> QuadInt:
        """Multiplicative identity (1)."""
        return self._make(self.ring.den, 0)

    @property
    def units(self) -> tuple[QuadInt, ...]:
        """
        Return the torsion units (roots of unity) in this order.

        Notes:
            This intentionally returns a finite subgroup only. In real quadratic rings
            (D >= 0), the full unit group is infinite; here we expose just the torsion
            part (typically `{±1}` except for D=1) because it is what canonical-associate and
            factorization normalization need.
        """
        one = self.one

        if self.ring.D == -1 and self.ring.den == 1:
            i = self._make(0, 1)
            return one, -one, i, -i

        if self.ring.D == 1:
            j_ = self._make(0, self.ring.den)
            return one, -one, j_, -j_

        if self.ring.D == -3 and self.ring.den == 2:
            w = self._make(-1, 1)
            return one, -one, w, -w, w * w, -(w * w)

        return one, -one

    def _canonical_associate(self) -> QuadInt:
        """Return canonical representative among associates for stable factor output."""

        best = self
        best_k = _key(self)
        for u in self.units[1:]:
            w = self * u
            kw = _key(w)
            if kw < best_k:
                best, best_k = w, kw

        return best

    def _from_obj(self, n: complex | int | float | QuadInt):
        """Make a QuadInt on the current ring from a given object"""
        if isinstance(n, (int, float)):
            a = int(n)
            b = 0
        elif isinstance(n, complex):
            if self.ring.D != -1 or self.ring.den != 1:
                raise TypeError("Cannot mix QuadInt from different rings")

            a = int(n.real)
            b = int(n.imag)
        elif isinstance(n, QuadInt):
            if n.ring is not self:
                raise TypeError("Cannot mix QuadInt from different rings")

            return n
        else:
            return NotImplemented

        # The only time b is not 0 is if self.den is 1 anyway... No need to multiply
        return self._make(a * self.ring.den, b)

    def assert_same_ring(self, other: QuadInt):
        """Raise an error if other is not in the same ring as self"""
        if self.ring is not other.ring:
            raise TypeError("Cannot mix QuadInt from different rings")

    def conjugate(self):
        """(a + b√D)/den -> (a - b√D)/den."""
        return self._make(self.a, -self.b)

    # region Basis vector operations
    @classmethod
    def _internal_to_basis(cls, a: int, b: int) -> tuple[int, int]:
        """Convert from internal a and b coords to basis coords"""
        (n00, n01), (n10, n11) = cls.INTERNAL_TO_BASIS
        den = cls.INTERNAL_TO_BASIS_DEN
        x_num = n00 * a + n01 * b
        y_num = n10 * a + n11 * b

        qX, rX = divmod(x_num, den)
        qY, rY = divmod(y_num, den)
        if rX or rY:
            raise ArithmeticError("Internal coordinates do not map cleanly to declared basis")

        return qX, qY

    @classmethod
    def _basis_to_internal(cls, x: int, y: int) -> tuple[int, int]:
        """Convert from basis coords to internal a and b coords"""
        (m00, m01), (m10, m11) = cls.BASIS_TO_INTERNAL
        return m00 * x + m01 * y, m10 * x + m11 * y

    @property
    def basis(self) -> tuple[int, int]:
        """The number in the basis vector"""
        return self._internal_to_basis(self.a, self.b)

    @property
    def basis_a(self) -> int:
        """a in the basis vector"""  # noqa: D403
        (n00, n01), _ = self.INTERNAL_TO_BASIS
        den = self.INTERNAL_TO_BASIS_DEN
        x_num = n00 * self.a + n01 * self.b

        return x_num // den

    @property
    def basis_b(self) -> int:
        """b in the basis vector"""  # noqa: D403
        _, (n10, n11) = self.INTERNAL_TO_BASIS
        den = self.INTERNAL_TO_BASIS_DEN
        y_num = n10 * self.a + n11 * self.b

        return y_num // den

    # endregion

    def __add__(self, other: complex | int | float | QuadInt):
        if isinstance(other, _OTHER_OP_TYPES):
            other = self._from_obj(other)

        if isinstance(other, QuadInt):
            self.assert_same_ring(other)
            return self._make(self.a + other.a, self.b + other.b)

        return NotImplemented

    def __radd__(self, other: int | float | complex):
        return self.__add__(other)

    def __sub__(self, other: complex | int | float | QuadInt):
        if isinstance(other, _OTHER_OP_TYPES):
            other = self._from_obj(other)

        if isinstance(other, QuadInt):
            self.assert_same_ring(other)
            return self._make(self.a - other.a, self.b - other.b)

        return NotImplemented

    def __rsub__(self, other: int | float | complex):
        return self.__neg__().__add__(other)

    def __neg__(self):
        return self._make(-self.a, -self.b)

    def __pos__(self):
        return self._make(self.a, self.b)

    def __mul__(self, other: complex | int | float | QuadInt):
        if isinstance(other, _OTHER_OP_TYPES):
            other = self._from_obj(other)

        if isinstance(other, QuadInt):
            self.assert_same_ring(other)

            # Numerator algebra:
            # (a+b√D)(c+d√D) = (ac + bdD) + (ad+bc)√D
            # But values are /den. If both are (..)/den then product is (..)/den^2.
            # We keep representation /den by dividing numerator by den once:
            # (xy)/den^2 == (xy/den)/den  -> require xy divisible by den.
            den = self.ring.den

            a1, b1 = self.a, self.b
            a2, b2 = other.a, other.b

            a_out = a1 * a2 + b1 * b2 * self.ring.D
            b_out = a1 * b2 + a2 * b1

            if den != 1:
                a_out, rA = divmod(a_out, den)
                b_out, rB = divmod(b_out, den)
                if rA != 0 or rB != 0:
                    raise ArithmeticError("Non-integral product; check ring parameters / parity")

            return self._make(a_out, b_out)

        return NotImplemented

    def __rmul__(self, other: int | float | complex):
        return self.__mul__(other)

    def __pow__(self, exp: float, mod: complex | int | float | QuadInt | None = None):
        if isinstance(mod, _OTHER_OP_TYPES):
            mod = self._from_obj(mod)

        if mod is not None:
            if not isinstance(mod, QuadInt):
                return NotImplemented

            self.assert_same_ring(mod)

            if mod == 0:
                raise ZeroDivisionError("pow() 3rd argument cannot be 0")

        e = int(exp)

        # Allow negative powers only in the modular case (like Python's pow()).
        if e < 0:
            if mod is None:
                raise ValueError("Negative powers not supported in quadratic integer rings without a modulus")

            # x^(-e) mod m == (x^{-1} mod m)^e mod m
            base = self.inv_mod(mod) % mod
            e = -e
        else:
            base = self % mod if mod is not None else self

        # exponentiation by squaring
        if mod is None:
            result = self.one
            while e:
                if e & 1:
                    result = result * base

                e >>= 1
                if e:
                    base = base * base

            return result

        result = self.one % mod
        while e:
            if e & 1:
                result = (result * base) % mod

            e >>= 1
            if e:
                base = (base * base) % mod

        return result

    # region Euclidean-ish division (no Fraction; small neighborhood search in integer metric)
    def __divmod__(self, other: complex | int | float | QuadInt):
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

    def __truediv__(self, other: complex | int | float | QuadInt):
        return self.__floordiv__(other)

    def __rtruediv__(self, other: int | float | complex):
        if isinstance(other, _OTHER_OP_TYPES):
            new_other = self._from_obj(other)
            return new_other.__truediv__(self)

        return NotImplemented

    def __floordiv__(self, other: complex | int | float | QuadInt):
        q, _ = divmod(self, other)
        return q

    def __rfloordiv__(self, other: int | float | complex):
        if isinstance(other, _OTHER_OP_TYPES):
            new_other = self._from_obj(other)
            return new_other.__floordiv__(self)

        return NotImplemented

    def __mod__(self, other: complex | int | float | QuadInt):
        _, r = divmod(self, other)
        return r

    def __rmod__(self, other: int | float | complex):
        if isinstance(other, _OTHER_OP_TYPES):
            new_other = self._from_obj(other)
            return new_other.__mod__(self)

        return NotImplemented

    def exact_div(self, divisor: complex | int | float | QuadInt) -> QuadInt | None:
        """Return q if y * q == self in this ring, else None."""
        if isinstance(divisor, _OTHER_OP_TYPES):
            divisor = self._from_obj(divisor)

        if not isinstance(divisor, QuadInt):
            raise NotImplementedError

        self.assert_same_ring(divisor)
        return self.ring.exact_div(self, divisor)

    def divides(self, x: complex | int | float | QuadInt) -> bool:
        """Return True iff self | x in this ring."""
        if isinstance(x, _OTHER_OP_TYPES):
            x = self._from_obj(x)

        if not isinstance(x, QuadInt):
            raise NotImplementedError

        self.assert_same_ring(x)
        return self.ring.divides(x, self)

    def xgcd(self, x: QuadInt) -> tuple[QuadInt, QuadInt, QuadInt]:
        """Extended gcd in Euclidean quadratic rings."""
        if isinstance(x, _OTHER_OP_TYPES):
            x = self._from_obj(x)

        if not isinstance(x, QuadInt):
            raise NotImplementedError

        self.assert_same_ring(x)
        return self.ring.xgcd(self, x)

    def gcd(self, x: QuadInt) -> QuadInt:
        """Greatest common divisor in Euclidean quadratic rings."""
        if isinstance(x, _OTHER_OP_TYPES):
            x = self._from_obj(x)

        if not isinstance(x, QuadInt):
            raise NotImplementedError

        self.assert_same_ring(x)
        return self.ring.gcd(self, x)

    def inv_mod(self, mod: complex | int | float | QuadInt) -> QuadInt:
        """Return the modular inverse of self modulo mod (if it exists)."""
        if isinstance(mod, _OTHER_OP_TYPES):
            mod = self._from_obj(mod)

        if not isinstance(mod, QuadInt):
            raise NotImplementedError

        self.assert_same_ring(mod)
        return self.ring.inv_mod(self, mod)

    # endregion

    def is_unit(self) -> bool:
        """Return True iff this element is a unit (invertible) in its ring."""
        return abs(abs(self)) == 1

    def is_irreducible(self) -> bool:
        """
        Return True iff this element is irreducible in its quadratic order.

        An element is irreducible if it is nonzero, not a unit,
            and every factorization `self = a*b` has either `a` or `b` a unit.

        For rings with explicit factorization support, this delegates to `factor_detail()`
            and checks whether the factorization contains exactly one irreducible factor.

        For rings without full factorization support, this still recognizes two
        standard norm-based certificates:

        * If `|N(self)|` is rational-prime, then `self` is irreducible.
        * If `|N(self)| == p**2` and the ring has no element of norm `p` or `-p`, then `self` is irreducible.

        Returns:
            bool: Is this quadratic integer irreducible?

        Raises:
            NotImplementedError: If irreducibility cannot be determined with the
                available ring algorithms.
        """
        if not self:
            return False

        abs_norm = abs(abs(self))
        # if self.is_unit():
        if abs_norm == 1:
            return False

        if isprime(abs_norm):
            return True

        if self.ring.supports_factorization():
            factorization = self.factor_detail()
            total_factors = sum(factorization.primes.values())
            return total_factors == 1

        norm_factors = factorint(abs_norm)
        if len(norm_factors) == 1:
            p, exponent = next(iter(norm_factors.items()))
            if exponent == 2 and not self.ring.has_element_with_norm(p) and not self.ring.has_element_with_norm(-p):
                return True

        raise NotImplementedError("Irreducibility is not implemented for this ring")

    def __abs__(self) -> int:
        """
        N((a+b√D)/den) = (a^2 - D*b^2) / den^2

        Always an integer for valid ring elements.

        Returns:
            int: The norm for the ring.

        Raises:
            ArithmeticError: If there is a non-integral norm for the ring.
        """
        den = self.ring.den
        num = self.a * self.a - self.ring.D * self.b * self.b
        dd = den * den
        d, m = divmod(num, dd)
        if m != 0:
            raise ArithmeticError("Non-integral norm; check ring parameters / parity")
        return d

    def factor(self) -> dict[QuadInt, int]:
        """Return a plain factor dict whose product is `self`."""
        return self.ring.factor(self)

    def factor_detail(self) -> Factorization:
        """Return structured `Factorization(unit, primes)` details."""
        return self.ring.factor_detail(self)

    def content(self) -> int:
        """Largest positive integer n such that x = n*y for some y in this ring."""
        if not self:
            return 0

        g = gcd(abs(self.a), abs(self.b))
        if g <= 1:
            return 1

        if self.ring.den == 1:
            return g

        # den == 2 rings require same parity for stored numerators.
        # If the maximal gcd quotient has mismatched parity, reducing by one factor of 2
        # is the unique maximal fix.
        qa = self.a // g
        qb = self.b // g
        if (qa ^ qb) & 1:
            return g // 2

        return g

    def __int__(self) -> int:
        """Return an int for pure integers (delegates to __index__)."""
        return self.__index__()

    def __index__(self) -> int:
        """Return an integer if this element is a plain integer (b == 0), else raise TypeError."""
        if self.b != 0:
            raise TypeError("cannot convert non-integer quadratic integer to int")
        return self.a // self.ring.den

    def __float__(self) -> float:
        """Return a float if this element is a plain integer (b == 0), else raise TypeError."""
        if self.b != 0:
            raise TypeError("cannot convert non-integer quadratic integer to float")
        return float(self.a // self.ring.den)

    def __complex__(self) -> complex:
        """Return a complex if this element is a Gaussian integer (D == -1, den == 1), else raise TypeError."""
        if self.ring.D != -1 or self.ring.den != 1:
            raise TypeError("cannot convert non-Gaussian quadratic integer to complex")
        return complex(self.a, self.b)

    def __bool__(self) -> bool:
        return (self.a | self.b) != 0

    def __iter__(self) -> Iterator[int]:
        return iter(self.basis)

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> int:
        if idx == 0:
            return self.basis_a
        if idx == 1:
            return self.basis_b
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
        den = self.ring.den

        parens = self.a != 0

        lead = "(" if parens else ""
        tail = ")" if parens else ""
        op = ("+" if parens else "") if self.b >= 0 else "-"
        a = self.a or ""
        b = abs(self.b)
        symbol = self.SYMBOL.format(D=self.ring.D)

        core = f"{lead}{a}{op}{b}{symbol}{tail}"
        return f"{core}/{den}" if den != 1 else core
