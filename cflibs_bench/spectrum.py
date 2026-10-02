"""Contêiner de espectro, ruído e E/S."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Spectrum:
    wl: np.ndarray                 # nm
    intensity: np.ndarray          # u.a.
    truth: dict = field(default_factory=dict)   # parâmetros verdadeiros (se sintético)
    noise_sigma: float | None = None

    def copy(self) -> "Spectrum":
        return Spectrum(self.wl.copy(), self.intensity.copy(), dict(self.truth), self.noise_sigma)

    def resample(self, wl_new: np.ndarray) -> "Spectrum":
        y = np.interp(wl_new, self.wl, self.intensity, left=0.0, right=0.0)
        return Spectrum(np.asarray(wl_new), y, dict(self.truth), self.noise_sigma)

    def to_csv(self, path: str) -> None:
        np.savetxt(path, np.column_stack([self.wl, self.intensity]), delimiter=",",
                   header="wavelength_nm,intensity", comments="")

    @classmethod
    def from_csv(cls, path: str, wl_col=0, int_col=1, skiprows=1) -> "Spectrum":
        data = np.loadtxt(path, delimiter=",", skiprows=skiprows)
        return cls(data[:, wl_col], data[:, int_col])


def add_noise(spec: Spectrum, gaussian=0.005, poisson=0.0, rng=None) -> Spectrum:
    """Ruído gaussiano (fração do máximo, como em Gornushkin & Völker: 0,5 %) e,
    opcionalmente, ruído tipo shot ∝ sqrt(I) (fração do máximo para I = Imax)."""
    rng = np.random.default_rng(rng)
    out = spec.copy()
    imax = np.max(np.abs(spec.intensity))
    sigma = gaussian * imax
    noise = rng.normal(0.0, sigma, spec.intensity.shape)
    if poisson > 0:
        noise += rng.normal(0.0, 1.0, spec.intensity.shape) * poisson * np.sqrt(
            np.abs(spec.intensity) * imax)
    out.intensity = spec.intensity + noise
    out.noise_sigma = float(sigma) if sigma > 0 else None
    out.truth["noise_gaussian"] = gaussian
    out.truth["noise_poisson"] = poisson
    return out


def estimate_noise_sigma(spec: Spectrum) -> float:
    """Estimativa robusta do desvio do ruído (MAD das diferenças de 1ª ordem)."""
    if spec.noise_sigma:
        return spec.noise_sigma
    d = np.diff(spec.intensity)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) or 1e-12
