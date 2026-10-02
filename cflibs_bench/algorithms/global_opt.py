"""Outros otimizadores globais estocásticos sobre a mesma função de custo.

* Recozimento simulado (dual annealing do SciPy) — variante usada por
  Gornushkin et al. e Herrera et al. no MC-LIBS original.
* Evolução diferencial — algoritmo genético populacional; aqui vetorizado
  (avalia a população inteira de uma vez no modelo direto).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution, dual_annealing

from .base import Algorithm, FitResult
from .cost import CostFunction


class SimulatedAnnealing(Algorithm):
    name = "Recozimento simulado"

    def __init__(self, model, maxfun=20000, cost="wcorr", bounds=None, seed=None):
        super().__init__(model)
        self.maxfun, self.cost_kind, self.seed = maxfun, cost, seed
        self.bounds = model.default_bounds() if bounds is None else np.asarray(bounds, float)

    def _fit(self, spectrum) -> FitResult:
        cost = CostFunction(self.model, spectrum, self.cost_kind)
        r = dual_annealing(lambda x: float(cost.theta_cost(x)[0]), self.bounds,
                           maxfun=self.maxfun, seed=self.seed,
                           minimizer_kwargs={"method": "L-BFGS-B",
                                             "options": {"maxfun": 300}})
        return self._result_from_theta(r.x, cost=float(r.fun), n_evals=cost.n_calls)


class DifferentialEvolution(Algorithm):
    name = "Evolução diferencial"

    def __init__(self, model, popsize=30, maxiter=300, cost="wcorr", bounds=None,
                 seed=None, tol=1e-8):
        super().__init__(model)
        self.popsize, self.maxiter, self.cost_kind = popsize, maxiter, cost
        self.seed, self.tol = seed, tol
        self.bounds = model.default_bounds() if bounds is None else np.asarray(bounds, float)

    def _fit(self, spectrum) -> FitResult:
        cost = CostFunction(self.model, spectrum, self.cost_kind)
        r = differential_evolution(lambda X: cost.theta_cost(np.atleast_2d(X.T)),
                                   self.bounds, popsize=self.popsize, maxiter=self.maxiter,
                                   seed=self.seed, tol=self.tol, vectorized=True,
                                   updating="deferred", polish=False, init="sobol")
        return self._result_from_theta(r.x, cost=float(r.fun), n_evals=cost.n_calls)
