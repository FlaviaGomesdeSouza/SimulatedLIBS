"""Funções de custo entre espectro experimental e sintético.

* ``corr``     — eq. (4): 1 − coeficiente de correlação (Pearson).
* ``wcorr``    — eq. (5): correlação ponderada por fragmento, pesos de
                 Gornushkin & Völker (cada elemento contribui com W = 1/N,
                 repartido entre suas linhas pela intensidade integrada).
* ``wcorr_eq`` — variante de ``wcorr`` em que cada fragmento é ainda dividido
                 pela sua energia (Σ I²), de modo que linhas fracas e fortes
                 pesem de fato o mesmo (experimental: amplifica o ruído
                 de fragmentos sem sinal).

O padrão é ``wcorr``.

Todas são invariantes à escala do espectro (não é preciso calibrar a intensidade
absoluta) e aceitam lotes: ``cost(Isyn)`` com Isyn (Nc, M) devolve (Nc,).
"""
from __future__ import annotations

import numpy as np

from ..model import PlasmaModel
from ..spectrum import Spectrum, estimate_noise_sigma


def line_integrals(model: PlasmaModel, spectrum: Spectrum, half_width_factor=2.0) -> np.ndarray:
    """Intensidade integrada aproximada de cada linha do modelo (soma na janela)."""
    y = spectrum.intensity
    S = np.empty(len(model.lines))
    for l, (wl, seg) in enumerate(zip(model.lines.wl, model.line_segment)):
        h = half_width_factor * (model.instr_fwhm[seg] + model.lorentz_fwhm[l])
        m = np.abs(model.wl - wl) <= h
        S[l] = max(np.trapezoid(y[m], model.wl[m]) if m.sum() > 1 else 0.0, 0.0)
    return S


class CostFunction:
    def __init__(self, model: PlasmaModel, spectrum: Spectrum, kind: str = "wcorr"):
        self.model, self.kind = model, kind
        self.y = spectrum.intensity.astype(float)
        self.n_calls = 0
        w = np.ones_like(self.y)
        if kind in ("wcorr", "wcorr_eq"):
            S = line_integrals(model, spectrum) + 1e-30
            N = model.n_elements
            w_line = np.zeros(len(S))
            for e in range(N):
                m = model.el_index == e
                w_line[m] = (1.0 / N) * S[m] / S[m].sum()
            sigma2 = estimate_noise_sigma(spectrum) ** 2
            for k, seg in enumerate(model.segments):
                m = model.line_segment == k
                wk = np.sum(w_line[m] * S[m]) / np.sum(S[m])
                if kind == "wcorr_eq":
                    yk = self.y[seg]
                    wk /= np.sum((yk - yk.mean()) ** 2) + sigma2 * len(yk)
                w[seg] = wk
        elif kind != "corr":
            raise ValueError(kind)
        self.w = w / w.sum()
        self.yc = self.y - np.sum(self.w * self.y)
        self.ynorm = np.sqrt(np.sum(self.w * self.yc ** 2))

    def __call__(self, Isyn: np.ndarray) -> np.ndarray:
        Isyn = np.atleast_2d(Isyn)
        self.n_calls += len(Isyn)
        sc = Isyn - (Isyn @ self.w)[:, None]
        num = sc @ (self.w * self.yc)
        den = np.sqrt((sc ** 2) @ self.w) * self.ynorm + 1e-300
        return 1.0 - num / den

    def residuals(self, Isyn_1d: np.ndarray) -> np.ndarray:
        """Resíduos r com ||r||² = 2·custo (para mínimos quadrados locais)."""
        sc = Isyn_1d - np.sum(self.w * Isyn_1d)
        sn = np.sqrt(np.sum(self.w * sc ** 2)) + 1e-300
        return np.sqrt(self.w) * (sc / sn - self.yc / self.ynorm)

    def theta_cost(self, theta) -> np.ndarray:
        return self(self.model.synthesize_theta(theta))
