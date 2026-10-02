"""Regressão por aprendizado de máquina treinada em espectros sintéticos
(na linha de Gąsior et al., Spectrochim. Acta B 199, 2023, que usaram o
SimulatedLIBS para gerar o conjunto de treino).

O modelo direto gera N espectros com composições geológicas aleatórias
(Dirichlet em torno de basalto/andesito/granito), temperaturas e ruído
aleatórios. Pré-processamento: normalização pelo máximo e log (Gąsior et al.
mostraram que o log melhora muito o desempenho). Alvos: T e a transformação
log-razão centrada (CLR) das frações atômicas — respeita o fechamento
composicional, que é revertido na predição (sempre soma 100 %).

Modelos: PLS (quimiometria clássica), Ridge+PCA, Random Forest, MLP (rede neural).
"""
from __future__ import annotations

import numpy as np

from ..composition import oxides_to_atomic_fractions, random_geological_composition
from ..spectrum import add_noise, Spectrum
from .base import Algorithm, FitResult


def _features(I: np.ndarray, floor=1e-4) -> np.ndarray:
    I = np.atleast_2d(I)
    I = I / np.max(I, axis=1, keepdims=True)
    return np.log10(np.maximum(I, floor))


class MLRegression(Algorithm):
    def __init__(self, model, kind="pls", n_train=1500, T_range=(7000.0, 14000.0),
                 noise=0.005, n_total=3e16, R=0.1, n_components=30, seed=0):
        super().__init__(model)
        self.kind, self.n_train, self.T_range = kind, n_train, T_range
        self.noise, self.n_total, self.R = noise, n_total, R
        self.n_components, self.seed = n_components, seed
        self.name = {"pls": "ML: PLS", "ridge": "ML: PCA+Ridge", "rf": "ML: Random Forest",
                     "mlp": "ML: Rede neural (MLP)"}[kind]
        self.estimator = None

    # ---------- dados de treino ----------
    def make_dataset(self, n, rng):
        m = self.model
        fr = np.empty((n, m.n_elements))
        for i in range(n):
            f = oxides_to_atomic_fractions(random_geological_composition(rng))
            fr[i] = [f.get(e, 0.0) for e in m.elements]
        fr = np.maximum(fr, 1e-7)
        fr /= fr.sum(1, keepdims=True)
        T = rng.uniform(*self.T_range, n)
        I = m.synthesize(T, np.full(n, self.R), fr * self.n_total)
        if self.noise:
            I = I + rng.normal(0, 1, I.shape) * self.noise * I.max(1, keepdims=True)
        return _features(I), fr, T

    @staticmethod
    def _clr(fr):
        lf = np.log(fr)
        return lf - lf.mean(1, keepdims=True)

    def _build(self):
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.decomposition import PCA
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import RidgeCV
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        if self.kind == "pls":
            return PLSRegression(n_components=min(self.n_components, 25), scale=True)
        if self.kind == "ridge":
            return make_pipeline(StandardScaler(), PCA(self.n_components),
                                 RidgeCV(alphas=np.logspace(-4, 3, 15)))
        if self.kind == "rf":
            return make_pipeline(PCA(self.n_components),
                                 RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1,
                                                       random_state=self.seed))
        if self.kind == "mlp":
            return make_pipeline(StandardScaler(), PCA(self.n_components),
                                 MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=3000,
                                              early_stopping=True, random_state=self.seed))
        raise ValueError(self.kind)

    def train(self):
        from sklearn.preprocessing import StandardScaler
        rng = np.random.default_rng(self.seed)
        X, fr, T = self.make_dataset(self.n_train, rng)
        Yraw = np.column_stack([T / 1e4, self._clr(fr)])
        self._yscaler = StandardScaler().fit(Yraw)
        self.estimator = self._build().fit(X, self._yscaler.transform(Yraw))
        return self

    def _fit(self, spectrum: Spectrum) -> FitResult:
        if self.estimator is None:
            self.train()
        Y = self._yscaler.inverse_transform(
            np.atleast_2d(self.estimator.predict(_features(spectrum.intensity))))[0]
        T = Y[0] * 1e4
        fr = np.exp(Y[1:])
        return self._result_from_n(T, fr / fr.sum(), n_evals=self.n_train)
