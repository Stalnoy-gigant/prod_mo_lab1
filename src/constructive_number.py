from __future__ import annotations
from fractions import Fraction
from typing import Union

Real = Union[int, float, "ConstructiveNumber"]


class ConstructiveNumber:
    def __init__(self, a: Union[Fraction, float, int], b: Union[Fraction, float, int]):
        self.a = Fraction(a)
        self.b = Fraction(b)
        if self.a > self.b:
            self.a, self.b = self.b, self.a

    @classmethod
    def from_real(cls, x: float, eps: float) -> "ConstructiveNumber":
        half = Fraction(eps) / 2
        center = Fraction(x)
        return cls(center - half, center + half)

    @classmethod
    def from_pair(cls, a: Union[Fraction, float, int], b: Union[Fraction, float, int]) -> "ConstructiveNumber":
        return cls(a, b)

    @property
    def eps(self) -> float:
        return float(self.b - self.a)

    def get(self, alpha: float = 0.5) -> float:
        alpha = max(0.0, min(1.0, alpha))
        return float(self.a + Fraction(alpha) * (self.b - self.a))

    def midpoint(self) -> float:
        return self.get(0.5)

    def __repr__(self) -> str:
        return f"CN([{float(self.a):.6g}, {float(self.b):.6g}], ε={self.eps:.2e})"

    def _coerce(self, other: Real) -> "ConstructiveNumber":
        if isinstance(other, ConstructiveNumber):
            return other
        return ConstructiveNumber(Fraction(other), Fraction(other))

    def __add__(self, other: Real) -> "ConstructiveNumber":
        o = self._coerce(other)
        return ConstructiveNumber(self.a + o.a, self.b + o.b)

    def __radd__(self, other: Real) -> "ConstructiveNumber":
        return self.__add__(other)

    def __sub__(self, other: Real) -> "ConstructiveNumber":
        o = self._coerce(other)
        return ConstructiveNumber(self.a - o.b, self.b - o.a)

    def __rsub__(self, other: Real) -> "ConstructiveNumber":
        o = self._coerce(other)
        return o.__sub__(self)

    def __mul__(self, other: Real) -> "ConstructiveNumber":
        o = self._coerce(other)
        products = [self.a * o.a, self.a * o.b, self.b * o.a, self.b * o.b]
        return ConstructiveNumber(min(products), max(products))

    def __rmul__(self, other: Real) -> "ConstructiveNumber":
        return self.__mul__(other)

    def __truediv__(self, other: Real) -> "ConstructiveNumber":
        o = self._coerce(other)
        if o.a <= 0 <= o.b:
            raise ZeroDivisionError(f"Деление на интервал, содержащий ноль: {o}")
        reciprocals = [Fraction(1, 1) / o.a, Fraction(1, 1) / o.b]
        inv = ConstructiveNumber(min(reciprocals), max(reciprocals))
        return self.__mul__(inv)

    def __rtruediv__(self, other: Real) -> "ConstructiveNumber":
        o = self._coerce(other)
        return o.__truediv__(self)

    def __neg__(self) -> "ConstructiveNumber":
        return ConstructiveNumber(-self.b, -self.a)

    def __pow__(self, n: int) -> "ConstructiveNumber":
        if not isinstance(n, int) or n < 0:
            raise ValueError("Поддерживаются только неотрицательные целые степени.")
        result = ConstructiveNumber(Fraction(1), Fraction(1))
        for _ in range(n):
            result = result * self
        return result

    def __eq__(self, other: Real) -> bool:
        o = self._coerce(other)
        return self.a == o.a and self.b == o.b

    def __lt__(self, other: Real) -> bool:
        o = self._coerce(other)
        return self.b < o.a

    def __le__(self, other: Real) -> bool:
        o = self._coerce(other)
        return self.b <= o.a

    def __gt__(self, other: Real) -> bool:
        o = self._coerce(other)
        return self.a > o.b

    def __ge__(self, other: Real) -> bool:
        o = self._coerce(other)
        return self.a >= o.b

    def __float__(self) -> float:
        return self.midpoint()

    def __abs__(self) -> "ConstructiveNumber":
        new_a = min(abs(self.a), abs(self.b))
        new_b = max(abs(self.a), abs(self.b))
        if self.a <= 0 <= self.b:
            new_a = Fraction(0)
        return ConstructiveNumber(new_a, new_b)