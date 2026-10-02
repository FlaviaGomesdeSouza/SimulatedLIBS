"""Desmistura linear (problema direto linearizado), inspirada em Yaroshchyk et al.
(Spectrochim. Acta B 2006): o espectro é escrito como sistema de equações
lineares nas concentrações e resolvido por SVD ou NNLS.

Para cada temperatura T de uma grade, calcula-se o espectro-base b_e(T) de cada
elemento no limite opticamente fino (n_e unitário) e resolve-se

        y ≈ Σ_e c_e · b_e(T)       (c_e ≥ 0 no NNLS)

A temperatura escolhida é a de menor resíduo (refinada por busca 1-D). É muito
rápido, mas assume plasma opticamente fino: linhas autoabsorvidas enviesam o
resultado — um bom contraste com MC-LIBS.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar, nnls

from ..spectrum import estimate_noise_sigma
from .base import Algorithm, FitResult


class LinearUnmixing(Algorithm):
    def __init__(self, model, solver="nnls", T_grid=None, weighting="sqrt", R=0.1,
                 n_thin=1e8):
        super().__init__(model)
        self.solver, self.weighting = solver, weighting
        self.T_grid = np.linspace(5000, 20000, 31) if T_grid is None else np.asarray(T_grid)
        self.R, self.n_thin = R, n_thin
        self.name = {"nnls": "Desmistura linear (NNLS)", "svd": "Desmistura linear (SVD)"}[solver]

    def _basis(self, T):
        N = self.model.n_elements
        n = np.eye(N) * self.n_thin
        return self.model.synthesize(np.full(N, T), np.full(N, self.R), n)   # (N, M)

    def _solve(self, T, y, w):
        B = (self._basis(T) * w).T
        yw = y * w
        scale = np.linalg.norm(B, axis=0) + 1e-300
        Bn = B / scale
        if self.solver == "nnls":
            c, rnorm = nnls(Bn, yw, maxiter=50 * Bn.shape[1])
        else:
            c, *_ = np.linalg.lstsq(Bn, yw, rcond=1e-10)
            c = np.clip(c, 0, None)
            rnorm = np.linalg.norm(Bn @ c - yw)
        self._evals += 1
        return c / scale, rnorm

    def _fit(self, spectrum) -> FitResult:
        self._evals = 0
        y = spectrum.intensity
        sigma = estimate_noise_sigma(spectrum)
        w = 1.0 / np.sqrt(np.abs(y) + 3 * sigma) if self.weighting == "sqrt" else np.ones_like(y)
        r = np.array([self._solve(T, y, w)[1] for T in self.T_grid])
        i = int(np.argmin(r))
        lo, hi = self.T_grid[max(i - 1, 0)], self.T_grid[min(i + 1, len(r) - 1)]
        opt = minimize_scalar(lambda T: self._solve(T, y, w)[1], bounds=(lo, hi),
                              method="bounded", options={"xatol": 5.0})
        T = float(opt.x)
        c, rnorm = self._solve(T, y, w)
        return self._result_from_n(T, c * self.n_thin, cost=float(rnorm), n_evals=self._evals)
