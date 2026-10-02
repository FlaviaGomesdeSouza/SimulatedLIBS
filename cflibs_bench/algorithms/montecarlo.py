"""MC-LIBS: otimização global Monte Carlo com caixas que encolhem
(Demidov et al.; Gornushkin & Völker, Sensors 2022, seção 2).

1. Sorteiam-se N_c configurações (T, R, n_i) uniformes no hipercubo D.
2. Escolhem-se as N_b de menor custo e constroem-se caixas menores em torno delas.
3. Na iteração seguinte, uma fração α das configurações vem de D e (1−α) das caixas.
4. A cada iteração as caixas encolhem por um fator; repete-se até convergir.

Em CPU usamos alguns milhares de configurações por iteração (o artigo usa
10^5–10^6 em GPU). Para acelerar com GPU basta trocar numpy por cupy no modelo.
"""
from __future__ import annotations

import numpy as np

from .base import Algorithm, FitResult
from .cost import CostFunction


class MonteCarloCF(Algorithm):
    name = "MC-LIBS"

    def __init__(self, model, n_configs=2000, n_boxes=5, shrink=0.9, alpha=0.1,
                 n_iter=40, cost="wcorr", bounds=None, polish=False, seed=None,
                 verbose=False):
        super().__init__(model)
        self.n_configs, self.n_boxes, self.shrink = n_configs, n_boxes, shrink
        self.alpha, self.n_iter, self.cost_kind = alpha, n_iter, cost
        self.bounds = model.default_bounds() if bounds is None else np.asarray(bounds, float)
        self.polish, self.seed, self.verbose = polish, seed, verbose

    def _fit(self, spectrum) -> FitResult:
        rng = np.random.default_rng(self.seed)
        cost = CostFunction(self.model, spectrum, self.cost_kind)
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        d = len(lo)
        X = rng.uniform(lo, hi, (self.n_configs, d))
        f = cost.theta_cost(X)
        elite_X, elite_f = X, f
        history = []
        half = (hi - lo) / 2.0
        for it in range(self.n_iter):
            idx = np.argsort(elite_f)[: self.n_boxes]
            centers, elite_X, elite_f = elite_X[idx], elite_X[idx], elite_f[idx]
            half = half * self.shrink
            n_glob = int(self.alpha * self.n_configs)
            n_box = self.n_configs - n_glob
            which = rng.integers(0, len(centers), n_box)
            Xb = centers[which] + rng.uniform(-1, 1, (n_box, d)) * half
            Xg = rng.uniform(lo, hi, (n_glob, d))
            X = np.clip(np.vstack([Xb, Xg]), lo, hi)
            f = cost.theta_cost(X)
            elite_X = np.vstack([elite_X, X])
            elite_f = np.concatenate([elite_f, f])
            history.append(float(elite_f.min()))
            if self.verbose:
                print(f"[MC] it {it:3d}  custo mínimo = {history[-1]:.3e}")
        best = elite_X[np.argmin(elite_f)]
        best_f = float(elite_f.min())
        if self.polish:
            from .local import polish_least_squares
            best, best_f = polish_least_squares(cost, best, self.bounds)
        res = self._result_from_theta(best, cost=best_f, n_evals=cost.n_calls,
                                      extra={"history": history})
        if self.polish:
            res.algorithm = "MC-LIBS + LS"
        return res
