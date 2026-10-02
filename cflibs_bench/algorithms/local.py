"""Otimização local determinística (mínimos quadrados não lineares).

Usa scipy.optimize.least_squares (Trust-Region-Reflective, a versão com limites
do Levenberg–Marquardt). Os resíduos são construídos de forma que
||r||² = 2·custo, logo minimiza-se exatamente a mesma função dos métodos globais.

Útil para mostrar o problema de mínimos locais citado no artigo do MC-LIBS
(partindo de pontos aleatórios) e como etapa de refinamento de métodos híbridos.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .base import Algorithm, FitResult
from .cost import CostFunction


def polish_least_squares(cost: CostFunction, x0, bounds, max_nfev=400):
    lo, hi = bounds[:, 0], bounds[:, 1]
    x0 = np.clip(x0, lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
    scale = hi - lo

    def fun(x):
        cost.n_calls += 1
        return cost.residuals(cost.model.synthesize_theta(x)[0])

    r = least_squares(fun, x0, bounds=(lo, hi), x_scale=scale, max_nfev=max_nfev,
                      diff_step=1e-4)
    return r.x, float(cost.theta_cost(r.x)[0])


class LocalLeastSquares(Algorithm):
    name = "Mínimos quadrados (TRF/LM)"

    def __init__(self, model, n_starts=1, x0=None, cost="wcorr", bounds=None,
                 max_nfev=400, seed=None):
        super().__init__(model)
        self.n_starts, self.x0, self.cost_kind = n_starts, x0, cost
        self.bounds = model.default_bounds() if bounds is None else np.asarray(bounds, float)
        self.max_nfev, self.seed = max_nfev, seed
        if n_starts > 1:
            self.name = f"Mínimos quadrados ({n_starts} partidas)"

    def _fit(self, spectrum) -> FitResult:
        rng = np.random.default_rng(self.seed)
        cost = CostFunction(self.model, spectrum, self.cost_kind)
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        starts = [self.x0] if self.x0 is not None else []
        while len(starts) < self.n_starts:
            starts.append(rng.uniform(lo, hi) if starts or self.n_starts > 1 else (lo + hi) / 2)
        best = (None, np.inf)
        for x0 in starts:
            x, f = polish_least_squares(cost, np.asarray(x0, float), self.bounds, self.max_nfev)
            if f < best[1]:
                best = (x, f)
        return self._result_from_theta(best[0], cost=best[1], n_evals=cost.n_calls,
                                       extra={"n_starts": len(starts)})
