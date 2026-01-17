
from fractions import Fraction
from typing import Iterator, Tuple, Union

# Types that eisensteinint operations are compatible with (other than eisensteinint)
OTHER_OP_TYPES = Union[int, float]
_OTHER_OP_TYPES = (int, float)  # I should be able to use the above with isinstance, but mypyc complains
OP_TYPES = Union['eisensteinint', OTHER_OP_TYPES]


def _round_fraction_to_int(x: Fraction) -> int:
    """
    Round a rational to the nearest integer (ties away from 0).

    This is deterministic and avoids float precision issues.

    Returns:
        int: The nearest integer to the fraction.
    """
    n, d = x.numerator, x.denominator
    if n >= 0:
        return (n + d // 2) // d
    return -((-n + d // 2) // d)


def _nearest_eisenstein(a: Fraction, b: Fraction) -> Tuple[int, int]:
    """
    Given a rational Eisenstein element a + b*ω, return the nearest lattice point (u, v).

    We do an exact search over a small neighborhood around coefficient-wise rounding.

    Returns:
        tuple: The nearest lattice point (u, v)
    """
    u0 = _round_fraction_to_int(a)
    v0 = _round_fraction_to_int(b)

    # Embed a + b*ω into a 2D coordinate system (X, Y) that avoids sqrt/halves:
    target_X, target_Y = 2 * a - b, b

    best_u, best_v = u0, v0
    best_metric = None

    # Small exact neighborhood search (3x3 = 9 candidates).
    for u in (u0 - 1, u0, u0 + 1):
        for v in (v0 - 1, v0, v0 + 1):
            cand_X = Fraction(2 * u - v, 1)
            cand_Y = Fraction(v, 1)

            dx = cand_X - target_X
            dy = cand_Y - target_Y

            # proportional to true squared distance (scaled by 1/4); scaling irrelevant
            metric = dx * dx + 3 * dy * dy

            if best_metric is None or metric < best_metric:
                best_metric = metric
                best_u, best_v = u, v

    return best_u, best_v


class eisensteinint:
    """
    Represents an Eisenstein number with integer real and omega parts.

    Attributes:
        real (int): The real component.
        omega (int): The omega component.
    """

    __slots__ = ('real', 'omega')
    real: int
    omega: int

    def __init__(self, real: int = 0, omega: int = 0):
        """
        Initialize an eisensteinint instance.

        Args:
            real (int, optional): The real part of the number. Defaults to 0.
            omega (int, optional): The omega part of the number. Defaults to 0.
        """
        self.real = real
        self.omega = omega

    def __add__(self, other: OP_TYPES) -> 'eisensteinint':
        if isinstance(other, eisensteinint):
            return eisensteinint(self.real + other.real, self.omega + other.omega)

        if isinstance(other, int):
            return eisensteinint(self.real + other, self.omega)

        if isinstance(other, float):
            other = int(other)
            return eisensteinint(self.real + other, self.omega)

        return NotImplemented

    def __radd__(self, other: OTHER_OP_TYPES) -> 'eisensteinint':
        return self.__add__(other)

    def __sub__(self, other: OP_TYPES) -> 'eisensteinint':
        if isinstance(other, eisensteinint):
            return eisensteinint(self.real - other.real, self.omega - other.omega)

        if isinstance(other, int):
            return eisensteinint(self.real - other, self.omega)

        if isinstance(other, float):
            other = int(other)
            return eisensteinint(self.real - other, self.omega)

        return NotImplemented

    def __rsub__(self, other: OTHER_OP_TYPES) -> 'eisensteinint':
        return self.__neg__().__add__(other)

    def __neg__(self) -> 'eisensteinint':
        return eisensteinint(-self.real, -self.omega)

    def __pos__(self) -> 'eisensteinint':
        return eisensteinint(self.real, self.omega)

    def __mul__(self, other: OP_TYPES) -> 'eisensteinint':
        if isinstance(other, eisensteinint):
            a = self.real
            b = self.omega
            c = other.real
            d = other.omega

            # Here is the critical difference between Eisenstein integers and complex integers:
            #   For complex integers it would be:
            #       return complexint(a * c - b * d, a * d + b * c)
            #   so there is just an extra subtraction to make this work
            return eisensteinint(a * c - b * d, a * d + b * c - b * d)

        if isinstance(other, int):
            return eisensteinint(self.real * other, self.omega * other)

        if isinstance(other, float):
            other = int(other)
            return eisensteinint(self.real * other, self.omega * other)

        return NotImplemented

    def __rmul__(self, other: OTHER_OP_TYPES) -> 'eisensteinint':
        return self.__mul__(other)

    def conjugate(self) -> "eisensteinint":
        """
        Galois conjugation ω -> ω^2, where ω^2 = -1 - ω.

        conj(a + bω) = a + bω^2 = (a - b) + (-b)ω

        Returns:
            eisensteinint: The conjugate.
        """
        return eisensteinint(self.real - self.omega, -self.omega)

    def __divmod__(self, other: OP_TYPES) -> Tuple["eisensteinint", "eisensteinint"]:
        """
        Return q, r such that self = q*other + r and r is "small".

        This uses the complex-plane embedding and rounds to the nearest Eisenstein integer.
        It is the key primitive you'll want for an Euclidean algorithm in Z[ω].

        Returns:
            tuple: The quotient and remainder.
        """
        if isinstance(other, _OTHER_OP_TYPES):
            other = eisensteinint(int(other), 0)

        a, b = other.real, other.omega
        norm = a * a - a * b + b * b

        # Exact rational quotient in the (1, ω) basis:
        # q_exact = self * conj(other) / N(other)
        num = self * other.conjugate()
        a_frac = Fraction(num.real, norm)
        b_frac = Fraction(num.omega, norm)

        qa, qb = _nearest_eisenstein(a_frac, b_frac)
        q = eisensteinint(qa, qb)
        r = self - q * other
        return q, r

    def __floordiv__(self, other: OP_TYPES) -> "eisensteinint":
        q, _ = divmod(self, other)
        return q

    def __mod__(self, other: OP_TYPES) -> "eisensteinint":
        _, r = divmod(self, other)
        return r

    def __truediv__(self, other: OP_TYPES) -> 'eisensteinint':
        return self.__floordiv__(other)

    def __rtruediv__(self, other: OTHER_OP_TYPES) -> 'eisensteinint':
        if isinstance(other, _OTHER_OP_TYPES):
            new_other = eisensteinint(int(other), 0)
            return new_other.__truediv__(self)

        return NotImplemented

    def __rfloordiv__(self, other: OTHER_OP_TYPES) -> 'eisensteinint':
        return self.__rtruediv__(other)

    def __pow__(self, power, modulo: None = None) -> 'eisensteinint':  # noqa: ANN001
        if modulo is not None:
            raise TypeError("modulo argument not supported for this type")

        # accept only integers (or __index__-able); avoid float path
        try:
            e = power.__index__()  # avoids float; works for numpy ints too
        except AttributeError:
            if isinstance(power, eisensteinint):
                oreal = int(power.real)
                oomega = int(power.omega)

                if oreal == 0 and oomega == 0:
                    return eisensteinint(1, 0)

                if self.real == 0 and self.omega == 0:
                    if oomega != 0 or oreal < 0:
                        raise ZeroDivisionError('0.0 to a negative or Eisenstein power') from None

                    return eisensteinint(0, 0)

                # TODO: Add eisensteinint power

                return NotImplemented

            if isinstance(power, float):
                e = int(power)
            else:
                return NotImplemented

        if e == 0:
            return eisensteinint(1, 0)

        if e < 0:
            # 1/(a+bω) is not generally an Eisenstein integer.
            # TODO: If we want a rational-eisenstein type later, we can support it then.
            raise ValueError("negative exponents are not supported for eisensteinint")

        # exponentiation by squaring
        rr_a, rr_b = 1, 0           # result = 1
        pr_a, pr_b = self.real, self.omega  # base

        if e == 1:
            return eisensteinint(pr_a, pr_b)

        if e == 2:
            # (a+bω)^2 using multiplication rule
            a, b = pr_a, pr_b
            # (a + bω)(a + bω) = (a*a - b*b) + (a*b + b*a - b*b)ω
            return eisensteinint(a * a - b * b, 2 * a * b - b * b)

        while e:
            if e & 1:
                # result *= base
                a, b = rr_a, rr_b
                c, d = pr_a, pr_b
                rr_a, rr_b = (a * c - b * d), (a * d + b * c - b * d)

            e >>= 1
            if e:
                # base = base^2
                a, b = pr_a, pr_b
                pr_a, pr_b = (a * a - b * b), (2 * a * b - b * b)

        return eisensteinint(rr_a, rr_b)

    def __abs__(self) -> 'eisensteinint':
        return eisensteinint(abs(self.real), abs(self.omega))

    def __iter__(self) -> Iterator:
        return iter((self.real, self.omega))

    def __repr__(self) -> str:
        parens = self.real == 0

        lead = "(" if parens else ""
        tail = ")" if parens else ""

        op = "+" if self.omega >= 0 else "-"

        real = self.real or ""
        imag = abs(self.omega)

        return f"{lead}{real}{op}{imag}ω{tail}"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, eisensteinint):
            return self.real == other.real and self.omega == other.omega

        if isinstance(other, (float, int)):
            return self.omega == 0 and self.real == other

        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.real, self.omega))

    def __bool__(self) -> bool:
        return (self.real | self.omega) != 0


E0 = eisensteinint(0, 0)
E1 = eisensteinint(1, 0)
Ew = eisensteinint(0, 1)
