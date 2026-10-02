"""Inferência bayesiana por MCMC (ensemble affine-invariant de Goodman & Weare,
o mesmo esquema do pacote emcee), vetorizada sobre os walkers.

Além do melhor ajuste, fornece a DISTRIBUIÇÃO a posteriori de T e das
concentrações — ou seja, incertezas e correlações (p.ex. a degenerescência n·R),
algo que os métodos de otimização não dão.

Verossimilhança gaussiana com σ do ruído estimado no espectro; a amplitude
absoluta é eliminada analiticamente (melhor fator de escala), como acontece
com espectros experimentais em unidades arbitrárias.
"""
from __future__ import annotations

import numpy as np

from ..composition import atomic_fractions_to_oxides
from ..spectrum import estimate_noise_sigma
from .base import Algorithm, FitResult
from .cost import CostFunction
from .local import polish_least_squares


class BayesianMCMC(Algorithm):
    name = "MCMC bayesiano"

    def __init__(self, model, n_walkers=48, n_steps=1500, burn_in=700, x0=None,
                 init_spread=0.02, bounds=None, seed=None, thin=5):
        super().__init__(model)
        self.n_walkers, self.n_steps, self.burn_in = n_walkers, n_steps, burn_in
        self.x0, self.init_spread, self.seed, self.thin = x0, init_spread, seed, thin
        self.bounds = model.default_bounds() if bounds is None else np.asarray(bounds, float)

    def _logp(self, X, y, sigma2):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        inside = np.all((X >= lo) & (X <= hi), axis=1)
        lp = np.full(len(X), -np.inf)
        if inside.any():
            S = self.model.synthesize_theta(X[inside])
            self._evals += int(inside.sum())
            a = (S @ y) / np.maximum(np.einsum("ij,ij->i", S, S), 1e-300)
            chi2 = np.sum((y - a[:, None] * S) ** 2, axis=1)
            lp[inside] = -0.5 * chi2 / sigma2
        return lp

    def _fit(self, spectrum) -> FitResult:
        rng = np.random.default_rng(self.seed)
        self._evals = 0
        y = spectrum.intensity
        sigma2 = estimate_noise_sigma(spectrum) ** 2
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        d = len(lo)
        x0 = self.x0
        if x0 is None:   # ponto inicial: mínimos quadrados a partir do centro do domínio
            cost = CostFunction(self.model, spectrum, "corr")
            x0, _ = polish_least_squares(cost, (lo + hi) / 2, self.bounds)
            self._evals += cost.n_calls
        X = np.clip(x0 + rng.normal(0, self.init_spread, (self.n_walkers, d)) * (hi - lo),
                    lo, hi)
        lp = self._logp(X, y, sigma2)
        half = self.n_walkers // 2
        chain, acc = [], 0
        for step in range(self.n_steps):
            for k in (0, 1):
                s = slice(0, half) if k == 0 else slice(half, None)
                c = slice(half, None) if k == 0 else slice(0, half)
                Xs, Xc = X[s], X[c]
                z = ((2.0 - 1) * rng.random(len(Xs)) + 1) ** 2 / 2.0     # a = 2
                Y = Xc[rng.integers(0, len(Xc), len(Xs))]
                prop = Y + z[:, None] * (Xs - Y)
                lp_new = self._logp(prop, y, sigma2)
                log_r = (d - 1) * np.log(z) + lp_new - lp[s]
                ok = np.log(rng.random(len(Xs))) < log_r
                Xs[ok], lps = prop[ok], lp[s]
                lps[ok] = lp_new[ok]
                X[s], lp[s] = Xs, lps
                acc += ok.sum()
            if step >= self.burn_in and step % self.thin == 0:
                chain.append(X.copy())
        chain = np.concatenate(chain)
        T, R, n = self.model.unpack(chain)
        fr = n / n.sum(1, keepdims=True)
        ox = np.array([list(atomic_fractions_to_oxides(dict(zip(self.model.elements, f))).values())
                       for f in fr[:: max(1, len(fr) // 2000)]])
        ox_names = list(atomic_fractions_to_oxides(dict(zip(self.model.elements, fr[0]))).keys())
        med = np.median(fr, 0)
        res = self._result_from_n(np.median(T), med, R=float(np.median(R)),
                                  theta=np.median(chain, 0), n_evals=self._evals)
        res.extra = {
            "T_std": float(np.std(T)),
            "oxide_p16": dict(zip(ox_names, np.percentile(ox, 16, 0))),
            "oxide_p84": dict(zip(ox_names, np.percentile(ox, 84, 0))),
            "acceptance": acc / (self.n_steps * self.n_walkers),
            "chain": chain,
        }
        return res
