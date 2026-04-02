from __future__ import annotations
from typing import List, Tuple, Optional, Callable
import numpy as np
from constructive_number import ConstructiveNumber


History = List[dict]


def _to_float_list(x) -> List[float]:
    return [float(xi) for xi in x]


def _cn_list(x_float: List[float], eps: float) -> List[ConstructiveNumber]:
    return [ConstructiveNumber.from_real(xi, eps) for xi in x_float]

class NelderMead:

    def __init__(
        self,
        alpha: float = 1.0,   # refl
        gamma: float = 2.0,   # exp
        rho: float = 0.5,     # cntr
        sigma: float = 0.5,   # shrnk
        tol: float = 1e-6,
        max_iter: int = 10000,
        cn_eps: Optional[float] = None,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma
        self.tol = tol
        self.max_iter = max_iter
        self.cn_eps = cn_eps

    def _call(self, f, x_float: List[float]):
        if self.cn_eps is not None:
            return float(f(_cn_list(x_float, self.cn_eps)))
        return float(f(x_float))

    def minimize(self, f, x0: List[float]) -> Tuple[List[float], History]:
        n = len(x0)
        history: History = []

        simplex = [np.array(x0, dtype=float)]
        for i in range(n):
            pt = np.array(x0, dtype=float)
            pt[i] += 0.05 if pt[i] != 0 else 0.00025
            simplex.append(pt)

        for iteration in range(self.max_iter):
            scores = [self._call(f, s.tolist()) for s in simplex]
            order = np.argsort(scores)
            simplex = [simplex[i] for i in order]
            scores = [scores[i] for i in order]

            best = simplex[0]
            history.append({
                "iter": iteration,
                "x": best.tolist(),
                "f": scores[0],
                "eps": self.cn_eps,
            })

            if np.std(scores) < self.tol:
                break

            centroid = np.mean(simplex[:-1], axis=0)

            xr = centroid + self.alpha * (centroid - simplex[-1])
            fr = self._call(f, xr.tolist())

            if scores[0] <= fr < scores[-2]:
                simplex[-1] = xr
            elif fr < scores[0]:                
                xe = centroid + self.gamma * (xr - centroid)
                fe = self._call(f, xe.tolist())
                simplex[-1] = xe if fe < fr else xr
            else:
                xc = centroid + self.rho * (simplex[-1] - centroid)
                fc = self._call(f, xc.tolist())
                if fc < scores[-1]:
                    simplex[-1] = xc
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + self.sigma * (simplex[i] - simplex[0])

        return simplex[0].tolist(), history



class GradientDescent:
    

    def __init__(
        self,
        lr: float = 0.01,
        tol: float = 1e-6,
        max_iter: int = 10000,
        line_search: bool = True,
        cn_eps: Optional[float] = None,
    ):
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.line_search = line_search
        self.cn_eps = cn_eps

    def _call_f(self, f, x_float: List[float]) -> float:
        if self.cn_eps is not None:
            return float(f(_cn_list(x_float, self.cn_eps)))
        return float(f(x_float))

    def _call_grad(self, f, x_float: List[float]) -> List[float]:
        if self.cn_eps is not None:
            g = f.grad(_cn_list(x_float, self.cn_eps))
        else:
            g = f.grad(x_float)
        return [float(gi) for gi in g]

    def _backtrack(self, f, x: np.ndarray, grad: np.ndarray, fx: float) -> float:
        lr = self.lr
        c, tau = 0.5, 0.5
        for _ in range(50):
            x_new = (x - lr * grad).tolist()
            if self._call_f(f, x_new) <= fx - c * lr * float(np.dot(grad, grad)):
                break
            lr *= tau
        return lr

    def minimize(self, f, x0: List[float]) -> Tuple[List[float], History]:
        x = np.array(x0, dtype=float)
        history: History = []

        for iteration in range(self.max_iter):
            fx = self._call_f(f, x.tolist())
            grad = np.array(self._call_grad(f, x.tolist()))

            history.append({
                "iter": iteration,
                "x": x.tolist(),
                "f": fx,
                "grad_norm": float(np.linalg.norm(grad)),
                "eps": self.cn_eps,
            })

            if np.linalg.norm(grad) < self.tol:
                break

            lr = self._backtrack(f, x, grad, fx) if self.line_search else self.lr
            x = x - lr * grad

        return x.tolist(), history



class NewtonMethod:

    def __init__(
        self,
        tol: float = 1e-6,
        max_iter: int = 1000,
        damping: bool = True,
        cn_eps: Optional[float] = None,
    ):
        self.tol = tol
        self.max_iter = max_iter
        self.damping = damping
        self.cn_eps = cn_eps

    def _call_f(self, f, x_float: List[float]) -> float:
        if self.cn_eps is not None:
            return float(f(_cn_list(x_float, self.cn_eps)))
        return float(f(x_float))

    def _call_grad(self, f, x_float: List[float]) -> np.ndarray:
        if self.cn_eps is not None:
            g = f.grad(_cn_list(x_float, self.cn_eps))
        else:
            g = f.grad(x_float)
        return np.array([float(gi) for gi in g])

    def _call_hess(self, f, x_float: List[float]) -> np.ndarray:
        return np.array(f.hess(x_float), dtype=float)

    def minimize(self, f, x0: List[float]) -> Tuple[List[float], History]:
        x = np.array(x0, dtype=float)
        history: History = []

        for iteration in range(self.max_iter):
            fx = self._call_f(f, x.tolist())
            grad = self._call_grad(f, x.tolist())
            H = self._call_hess(f, x.tolist())

            history.append({
                "iter": iteration,
                "x": x.tolist(),
                "f": fx,
                "grad_norm": float(np.linalg.norm(grad)),
                "eps": self.cn_eps,
            })

            if np.linalg.norm(grad) < self.tol:
                break

            try:
                direction = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                direction = grad 

            if self.damping:
                lr = 1.0
                c, tau = 0.5, 0.5
                for _ in range(50):
                    x_new = (x - lr * direction).tolist()
                    if self._call_f(f, x_new) < fx - c * lr * float(grad @ direction):
                        break
                    lr *= tau
            else:
                lr = 1.0

            x = x - lr * direction

        return x.tolist(), history