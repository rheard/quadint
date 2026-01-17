from __future__ import annotations

from typing import Iterator, Union

IntOrQuat = Union[int, "quatint"]


class quatint:
    """
    Integer (Hamilton) quaternion: w + x*i + y*j + z*k, with i^2=j^2=k^2=ijk=-1.

    Norm:
        N(q) = w*w + x*x + y*y + z*z

    Key property:
        N(p*q) = N(p) * N(q)
    """

    __slots__ = ("w", "x", "y", "z")

    def __init__(self, w: int = 0, x: int = 0, y: int = 0, z: int = 0) -> None:
        self.w = int(w)
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)

    def __repr__(self) -> str:
        return f"quatint({self.w}, {self.x}, {self.y}, {self.z})"

    def __iter__(self) -> Iterator[int]:
        yield self.w
        yield self.x
        yield self.y
        yield self.z

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.w, self.x, self.y, self.z)

    def __hash__(self) -> int:
        return hash((self.w, self.x, self.y, self.z))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, quatint):
            return False
        return (self.w, self.x, self.y, self.z) == (other.w, other.x, other.y, other.z)

    def __neg__(self) -> quatint:
        return quatint(-self.w, -self.x, -self.y, -self.z)

    def __add__(self, other: IntOrQuat) -> quatint:
        if isinstance(other, int):
            return quatint(self.w + other, self.x, self.y, self.z)
        return quatint(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)

    def __radd__(self, other: int) -> quatint:
        return quatint(self.w + other, self.x, self.y, self.z)

    def __sub__(self, other: IntOrQuat) -> quatint:
        if isinstance(other, int):
            return quatint(self.w - other, self.x, self.y, self.z)
        return quatint(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)

    def __rsub__(self, other: int) -> quatint:
        # other - self
        return quatint(other - self.w, -self.x, -self.y, -self.z)

    def __mul__(self, other: IntOrQuat) -> quatint:
        if isinstance(other, int):
            return quatint(self.w * other, self.x * other, self.y * other, self.z * other)

        # Hamilton product
        a1, b1, c1, d1 = self.w, self.x, self.y, self.z
        a2, b2, c2, d2 = other.w, other.x, other.y, other.z

        return quatint(
            a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
            a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
            a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
            a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
        )

    def __rmul__(self, other: int) -> quatint:
        return quatint(self.w * other, self.x * other, self.y * other, self.z * other)

    def conj(self) -> quatint:
        return quatint(self.w, -self.x, -self.y, -self.z)

    def norm(self) -> int:
        return self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z

    def __pow__(self, exp: int) -> quatint:
        if exp < 0:
            raise ValueError("quatint only supports non-negative integer powers (no division implemented).")
        result = quatint(1, 0, 0, 0)
        base = self
        e = exp
        while e:
            if e & 1:
                result = result * base
            base = base * base
            e >>= 1
        return result
