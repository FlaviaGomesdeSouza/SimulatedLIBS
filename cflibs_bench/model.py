"""Modelo direto: plasma homogêneo, isotérmico, em LTE (Gornushkin & Völker 2022).

    I(λ) = B(λ, T) · [1 − exp(−τ(λ))]
    τ(λ) = Σ_linhas K_l(T, n, R) · P_l(λ)
    K_l  = π r_e λ0² f g_i exp(−E_i/kT)/U(T) · [1 − exp(−(E_k−E_i)/kT)] · n_s · R

As populações de neutros e íons de cada elemento vêm da equação de Saha com n_e
fixo. O perfil P_l é Lorentziano (Stark) e o espectro é convoluído com uma função
instrumental gaussiana (poder de resolução do Echelle). A autoabsorção e a
sobreposição de linhas são, portanto, incluídas automaticamente.

Toda a síntese é vetorizada: `synthesize` recebe lotes de configurações (Nc)
e devolve uma matriz (Nc, M) — é isso que torna o Monte Carlo viável em CPU.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .atomic_data import (IONIZATION_EV, STARK_FWHM_NM, load_lines,
                          partition_function)
from .composition import atomic_fractions_to_oxides, oxides_to_atomic_fractions
from .spectrum import Spectrum

K_B_EV = 8.617333262e-5          # eV/K
PI_RE_CM = np.pi * 2.8179403262e-13
SAHA_CONST = 4.8294e15           # 2·(2π m_e k T/h²)^{3/2} / T^{3/2}  [cm^-3 K^-3/2]
C2_NM_K = 1.438776877e7          # hc/k em nm·K


class PlasmaModel:
    """Modelo direto vetorizado.

    Parâmetros
    ----------
    elements : elementos a incluir (só os que possuem linhas no intervalo).
    wl_range : (λmin, λmax) em nm.
    ne : densidade eletrônica (cm^-3), fixa.
    step : passo da grade (nm).
    window : meia-largura (nm) da janela em torno de cada linha; janelas que se
        sobrepõem são fundidas em "fragmentos" (como no artigo do MC-LIBS).
    resolving_power : λ/Δλ do espectrômetro (Echelle típico: 5000–20000).
    """

    def __init__(self, elements=None, wl_range=(300.0, 780.0), ne=1e17, step=0.01,
                 window=0.5, resolving_power=10000.0, stark_scale=1.0):
        self.lines = load_lines(elements, wl_range)
        present = list(dict.fromkeys(self.lines.element.tolist()))
        if elements is None:
            elements = present
        missing = [e for e in elements if e not in present]
        if missing:
            raise ValueError(f"Sem linhas no intervalo para: {missing}")
        self.elements = list(elements)
        self.ne = float(ne)
        self.step = float(step)
        self.resolving_power = float(resolving_power)
        self.el_index = np.array([self.elements.index(e) for e in self.lines.element])
        self.ion_mask = self.lines.stage == 2
        self.chi = np.array([IONIZATION_EV[e] for e in self.elements])

        # ---- fragmentos espectrais ----
        order = np.argsort(self.lines.wl)
        segs = []
        for wl in self.lines.wl[order]:
            lo, hi = wl - window, wl + window
            if segs and lo <= segs[-1][1]:
                segs[-1][1] = max(segs[-1][1], hi)
            else:
                segs.append([lo, hi])
        grids, self.segments = [], []
        start = 0
        for lo, hi in segs:
            g = np.arange(lo, hi + 0.5 * step, step)
            grids.append(g)
            self.segments.append(slice(start, start + len(g)))
            start += len(g)
        self.wl = np.concatenate(grids)
        self.line_segment = np.array([
            next(i for i, (lo, hi) in enumerate(segs) if lo <= w <= hi) for w in self.lines.wl])

        # ---- perfis Lorentzianos normalizados (cm^-1) ----
        self.lorentz_fwhm = np.array([STARK_FWHM_NM[s] for s in self.lines.stage]) \
            * stark_scale * self.ne / 1e17
        g = self.lorentz_fwhm[:, None] / 2.0
        d = self.wl[None, :] - self.lines.wl[:, None]
        P = (g / np.pi) / (d ** 2 + g ** 2)                    # nm^-1
        P[np.abs(d) > window * 1.2] = 0.0
        self.P = P * 1e7                                       # cm^-1
        self.line_const = PI_RE_CM * (self.lines.wl * 1e-7) ** 2 * self.lines.f * self.lines.gi

        # ---- função instrumental (sigma em pixels por fragmento) ----
        self.instr_fwhm = np.array([self.wl[s].mean() / self.resolving_power for s in self.segments])
        self.instr_sigma_px = self.instr_fwhm / 2.3548 / self.step

    # ------------------------------------------------------------------
    @property
    def n_elements(self) -> int:
        return len(self.elements)

    def saha_ratio(self, T: np.ndarray) -> np.ndarray:
        """n_II / n_I para cada elemento. T: (Nc,) -> (Nc, Nel)."""
        T = np.atleast_1d(T)[:, None]
        U1 = np.stack([partition_function(e, 1, T[:, 0]) for e in self.elements], 1)
        U2 = np.stack([partition_function(e, 2, T[:, 0]) for e in self.elements], 1)
        return SAHA_CONST * T ** 1.5 * (U2 / U1) * np.exp(-self.chi / (K_B_EV * T)) / self.ne

    def species_densities(self, T, n):
        """Densidades de neutros e íons (Nc, Nel) a partir das densidades totais."""
        S = self.saha_ratio(T)
        n1 = n / (1.0 + S)
        return n1, n - n1

    def line_strengths(self, T, R, n) -> np.ndarray:
        """K_l (Nc, L) — profundidade óptica integrada de cada linha."""
        T = np.atleast_1d(np.asarray(T, float))
        R = np.atleast_1d(np.asarray(R, float))
        n = np.atleast_2d(np.asarray(n, float))
        n1, n2 = self.species_densities(T, n)
        n_sp = np.where(self.ion_mask[None, :], n2[:, self.el_index], n1[:, self.el_index])
        U = np.empty((len(T), len(self.lines)))
        for (el, st) in set(self.lines.species):
            m = (self.lines.element == el) & (self.lines.stage == st)
            U[:, m] = partition_function(el, st, T)[:, None]
        kT = K_B_EV * T[:, None]
        boltz = np.exp(-self.lines.Ei[None, :] / kT) / U
        stim = -np.expm1(-(self.lines.Ek - self.lines.Ei)[None, :] / kT)
        return self.line_const[None, :] * boltz * stim * n_sp * R[:, None]

    def planck(self, T) -> np.ndarray:
        T = np.atleast_1d(T)[:, None]
        lam = self.wl[None, :]
        return 1e15 / lam ** 5 / np.expm1(C2_NM_K / (lam * T))

    def synthesize(self, T, R, n, instrument=True, chunk=1024) -> np.ndarray:
        """Espectros sintéticos para lotes de (T, R, n). Retorna (Nc, M)."""
        T = np.atleast_1d(np.asarray(T, float))
        R = np.broadcast_to(np.atleast_1d(np.asarray(R, float)), T.shape)
        n = np.atleast_2d(np.asarray(n, float))
        out = np.empty((len(T), len(self.wl)))
        for a in range(0, len(T), chunk):
            b = slice(a, a + chunk)
            K = self.line_strengths(T[b], R[b], n[b])
            tau = K @ self.P
            I = self.planck(T[b]) * -np.expm1(-tau)
            if instrument:
                for s, sig in zip(self.segments, self.instr_sigma_px):
                    I[:, s] = gaussian_filter1d(I[:, s], sig, axis=1, mode="constant")
            out[b] = I
        return out

    def peak_optical_depth(self, T, R, n) -> np.ndarray:
        """τ no centro de cada linha (diagnóstico de autoabsorção)."""
        K = self.line_strengths(T, R, n)[0]
        return K * 2.0 / (np.pi * self.lorentz_fwhm) * 1e7

    # ------------------------------------------------------------------
    # Parametrização usada pelos otimizadores: θ = [T, log10 R, log10 n_1..n_N]
    def default_bounds(self, T=(5000.0, 20000.0), logR=(-3.0, -1.0), logn=(10.0, 19.0)):
        """Domínio D (eq. 3 do artigo; limite inferior de n ampliado para traços)."""
        return np.array([T, logR] + [logn] * self.n_elements, float)

    def unpack(self, theta):
        theta = np.atleast_2d(theta)
        return theta[:, 0], 10 ** theta[:, 1], 10 ** theta[:, 2:]

    def synthesize_theta(self, theta, **kw):
        return self.synthesize(*self.unpack(theta), **kw)

    # ------------------------------------------------------------------
    def simulate_sample(self, oxide_wt: dict, T=10000.0, R=0.1, n_total=3e16,
                        instrument=True) -> Spectrum:
        """Espectro "experimental" ideal (sem ruído) de uma amostra em % de óxidos."""
        frac = oxides_to_atomic_fractions(oxide_wt)
        frac = {e: frac.get(e, 0.0) for e in self.elements}
        n = np.array([frac[e] for e in self.elements]) * n_total
        I = self.synthesize([T], [R], n[None, :], instrument=instrument)[0]
        truth = {"T": T, "R": R, "n_total": n_total, "ne": self.ne,
                 "atomic_fractions": frac,
                 "oxide_wt": atomic_fractions_to_oxides(frac)}
        return Spectrum(self.wl.copy(), I, truth)
