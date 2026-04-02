from __future__ import annotations
from typing import List, Callable, Optional, Tuple
import numpy as np


class BlackBox:

    def __init__(self, f: Callable, grad: Optional[Callable] = None, hess: Optional[Callable] = None, name: str = ""):
        self._f = f
        self._grad = grad
        self._hess = hess
        self.name = name
        self.f_calls = 0
        self.grad_calls = 0
        self.hess_calls = 0

    def __call__(self, x):
        self.f_calls += 1
        return self._f(x)

    def grad(self, x):
        if self._grad is None:
            raise NotImplementedError("Для этого черного ящика градиент не определен.")
        self.grad_calls += 1
        return self._grad(x)

    def hess(self, x):
        if self._hess is None:
            raise NotImplementedError("Для этого черного ящика не определена мешковина.")
        self.hess_calls += 1
        return self._hess(x)

    def reset_counters(self):
        self.f_calls = 0
        self.grad_calls = 0
        self.hess_calls = 0

    def __repr__(self):
        return (f"BlackBox('{self.name}', f_calls={self.f_calls}, "
                f"grad_calls={self.grad_calls}, hess_calls={self.hess_calls})")


def _make_quadratic(Q: np.ndarray, name: str) -> BlackBox:
    n = Q.shape[0]

    def f(x):
        result = x[0] * x[0] * Q[0, 0] * 0
        for i in range(n):
            for j in range(n):
                result = result + x[i] * x[j] * Q[i, j]
        return result

    def grad_f(x):
        g = []
        for i in range(n):
            val = x[0] * Q[i, 0] * 0 
            for j in range(n):
                val = val + x[j] * Q[i, j]
            g.append(val * 2)
        return g

    def hess_f(x):
        return 2 * Q

    return BlackBox(f, grad_f, hess_f, name=name)


def make_quadratic_good() -> BlackBox:
    Q = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    return _make_quadratic(Q, name="Quadratic-6D-good-cond")


def make_quadratic_bad() -> BlackBox:
    Q = np.diag([1.0, 10.0, 50.0, 100.0])
    return _make_quadratic(Q, name="Quadratic-4D-bad-cond")


def make_rosenbrock() -> BlackBox:

    def f(x):
        x0, x1, x2 = x[0], x[1], x[2]
        t0 = x1 - x0 * x0
        t1 = x2 - x1 * x1
        return (x0 - 1) * (x0 - 1) + t0 * t0 * 100 + \
               (x1 - 1) * (x1 - 1) + t1 * t1 * 100

    def grad_f(x):
        x0, x1, x2 = x[0], x[1], x[2]
        # df/dx0 = -2*(1-x0) + 100*2*(x1-x0^2)*(-2*x0)
        g0 = (x0 - 1) * (-2) + (x1 - x0 * x0) * x0 * (-400)
        g1 = (x1 - x0 * x0) * 200 + (x1 - 1) * (-2) + (x2 - x1 * x1) * x1 * (-400)
        g2 = (x2 - x1 * x1) * 200
        return [g0, g1, g2]

    def hess_f(x):
        x0, x1, x2 = float(x[0]), float(x[1]), float(x[2])
        H = np.zeros((3, 3))
        H[0, 0] = 2 + 400 * (3 * x0 ** 2 - x1)
        H[0, 1] = -400 * x0
        H[1, 0] = -400 * x0
        H[1, 1] = 202 + 400 * (3 * x1 ** 2 - x2)
        H[1, 2] = -400 * x1
        H[2, 1] = -400 * x1
        H[2, 2] = 200
        return H

    return BlackBox(f, grad_f, hess_f, name="Rosenbrock-3D")