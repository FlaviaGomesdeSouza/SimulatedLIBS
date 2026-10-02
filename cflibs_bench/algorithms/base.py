"""Interface comum dos algoritmos."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..composition import atomic_fractions_to_oxides
from ..model import PlasmaModel
from ..spectrum import Spectrum


@dataclass
class FitResult:
    algorithm: str
    T: float
    atomic_fractions: dict
    oxide_wt: dict
    R: float | None = None
    cost: float | None = None
    runtime: float = 0.0
    n_evals: int = 0
    theta: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


class Algorithm:
    name = "base"

    def __init__(self, model: PlasmaModel):
        self.model = model

    def fit(self, spectrum: Spectrum) -> FitResult:
        if len(spectrum.wl) != len(self.model.wl) or not np.allclose(spectrum.wl, self.model.wl):
            spectrum = spectrum.resample(self.model.wl)
        t0 = time.perf_counter()
        res = self._fit(spectrum)
        res.runtime = time.perf_counter() - t0
        return res

    def _fit(self, spectrum: Spectrum) -> FitResult:   # pragma: no cover
        raise NotImplementedError

    # utilitário: densidades -> resultado
    def _result_from_n(self, T, n, **kw) -> FitResult:
        n = np.clip(np.asarray(n, float), 0, None)
        frac = dict(zip(self.model.elements, n / n.sum()))
        return FitResult(self.name, float(T), frac, atomic_fractions_to_oxides(frac), **kw)

    def _result_from_theta(self, theta, **kw) -> FitResult:
        T, R, n = self.model.unpack(theta)
        return self._result_from_n(T[0], n[0], R=float(R[0]), theta=np.asarray(theta), **kw)
