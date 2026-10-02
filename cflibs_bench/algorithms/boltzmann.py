"""Métodos do problema inverso: CF-LIBS clássico (Saha–Boltzmann plot) e
calibração de ponto único.

CF-SBP (Ciucci et al. 1999; Tognoni et al. 2010)
    Para uma linha opticamente fina:
        ln(I λ³ / (g_i f)) = −E_k/kT + ln(n_s R / U_s) + const
    (forma equivalente a ln(I λ / g_k A_ki)). Linhas iônicas entram no mesmo
    gráfico deslocando E_k → E_k + χ e subtraindo ln(2·(2πm_e kT/h²)^{3/2}/n_e)
    (Saha–Boltzmann). Ajusta-se uma inclinação comum (−1/kT) e um intercepto por
    elemento; as concentrações saem do fechamento Σ c = 100 %.

Calibração de ponto único (Cavalcanti et al., Spectrochim. Acta B 87, 2013)
    Com UMA amostra de referência de matriz similar, calcula-se para cada linha
    o fator de correção entre o y medido e o y teórico (composição conhecida);
    esses fatores absorvem erros de A_ki/f, resposta instrumental e parte da
    autoabsorção e são aplicados ao espectro desconhecido.
"""
from __future__ import annotations

import numpy as np

from ..atomic_data import partition_function
from ..composition import oxides_to_atomic_fractions
from ..model import K_B_EV, SAHA_CONST
from ..spectrum import Spectrum, estimate_noise_sigma
from .base import Algorithm, FitResult


class SahaBoltzmannCF(Algorithm):
    name = "CF-LIBS (Saha-Boltzmann)"

    def __init__(self, model, exclude_resonance=False, exclude_blended=True,
                 min_snr=3.0, half_width_factor=2.0, n_iter=15, T0=10000.0):
        super().__init__(model)
        self.exclude_resonance, self.exclude_blended = exclude_resonance, exclude_blended
        self.min_snr, self.hw, self.n_iter, self.T0 = min_snr, half_width_factor, n_iter, T0
        if exclude_resonance:
            self.name = "CF-LIBS (SBP, sem ressonância)"

    # ---------- medição das intensidades integradas ----------
    def measure_lines(self, spectrum: Spectrum):
        m, L = self.model, self.model.lines
        wl, y = m.wl, spectrum.intensity
        sigma = estimate_noise_sigma(spectrum)
        width = np.array([m.instr_fwhm[s] for s in m.line_segment]) + m.lorentz_fwhm
        keep, I = [], []
        for l in range(len(L)):
            if self.exclude_resonance and L.Ei[l] < 0.15:
                continue
            if self.exclude_blended:
                d = np.abs(L.wl - L.wl[l])
                d[l] = np.inf
                if np.any(d < self.hw * (width[l] + width) / 2):
                    continue
            h = self.hw * width[l]
            sel = np.abs(wl - L.wl[l]) <= h
            if sel.sum() < 3:
                continue
            # linha de base linear entre as bordas da janela
            xs, ys = wl[sel], y[sel]
            base = np.interp(xs, [xs[0], xs[-1]], [ys[:2].mean(), ys[-2:].mean()])
            val = np.trapezoid(ys - base, xs)
            noise = sigma * np.sqrt(sel.sum()) * m.step
            if val > self.min_snr * noise:
                keep.append(l)
                I.append(val)
        return np.array(keep, int), np.array(I)

    def boltzmann_y(self, idx, I):
        L = self.model.lines
        return np.log(I * L.wl[idx] ** 3 / (L.gi[idx] * L.f[idx]))

    # ---------- ajuste do gráfico de Saha–Boltzmann ----------
    def solve(self, idx, y_raw, T=None):
        m, L = self.model, self.model.lines
        el = m.el_index[idx]
        ion = L.stage[idx] == 2
        present = np.unique(el)
        col = {e: i for i, e in enumerate(present)}
        T = self.T0 if T is None else T
        for _ in range(self.n_iter):
            x = L.Ek[idx] + np.where(ion, m.chi[el], 0.0)
            y = y_raw - np.where(ion, np.log(SAHA_CONST * T ** 1.5 / m.ne), 0.0)
            A = np.zeros((len(idx), 1 + len(present)))
            A[:, 0] = x
            A[np.arange(len(idx)), 1 + np.array([col[e] for e in el])] = 1.0
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            T_new = -1.0 / (K_B_EV * coef[0]) if coef[0] < 0 else 3e4
            T_new = float(np.clip(T_new, 2000.0, 3e4))
            if abs(T_new - T) < 0.1:
                T = T_new
                break
            T = T_new
        q = coef[1:]
        n = np.zeros(m.n_elements)
        S = m.saha_ratio(np.array([T]))[0]
        for e, i in col.items():
            U1 = partition_function(m.elements[e], 1, T)
            n[e] = U1 * np.exp(q[i] - q.max()) * (1.0 + S[e])
        resid = y - A @ coef
        return T, n, {"slope": coef[0], "intercepts": dict(zip([m.elements[e] for e in present], q)),
                      "residual_rms": float(np.sqrt(np.mean(resid ** 2))),
                      "plot": {"x": x, "y": y, "element": [m.elements[e] for e in el]}}

    def _fit(self, spectrum) -> FitResult:
        idx, I = self.measure_lines(spectrum)
        if len(idx) < 3:
            raise RuntimeError("Poucas linhas utilizáveis para o gráfico de Boltzmann")
        T, n, info = self.solve(idx, self.boltzmann_y(idx, I))
        missing = [e for i, e in enumerate(self.model.elements) if n[i] == 0]
        return self._result_from_n(T, n, n_evals=0,
                                   extra={"n_lines": len(idx), "missing": missing, **info})


class OnePointCalibration(SahaBoltzmannCF):
    """CF-LIBS com correção por um único padrão de referência (Cavalcanti et al. 2013)."""
    name = "Ponto único (Cavalcanti)"

    def __init__(self, model, reference: Spectrum, reference_oxides: dict | None = None,
                 reference_fractions: dict | None = None, **kw):
        """Composição do padrão: ``reference_oxides`` (% óxidos) ou
        ``reference_fractions`` (frações atômicas dos cátions)."""
        super().__init__(model, **kw)
        self.name = "Ponto único (Cavalcanti)"
        if reference_fractions is None:
            reference_fractions = oxides_to_atomic_fractions(reference_oxides)
        self._calibrate(reference.resample(model.wl) if len(reference.wl) != len(model.wl)
                        else reference, reference_fractions)

    def _calibrate(self, ref: Spectrum, frac: dict):
        m, L = self.model, self.model.lines
        idx, I = self.measure_lines(ref)
        y_raw = self.boltzmann_y(idx, I)
        T, _, _ = self.solve(idx, y_raw)
        c = np.array([frac.get(e, 0.0) for e in m.elements])
        S = m.saha_ratio(np.array([T]))[0]
        el = m.el_index[idx]
        ion = L.stage[idx] == 2
        n1 = c / (1.0 + S)
        U1 = np.array([partition_function(e, 1, T) for e in m.elements])
        ok = n1[el] > 0
        y_theo = -(L.Ek[idx] + np.where(ion, m.chi[el], 0.0)) / (K_B_EV * T) \
            + np.log(np.where(ok, n1[el] / U1[el], 1.0)) \
            + np.where(ion, np.log(SAHA_CONST * T ** 1.5 / m.ne), 0.0)
        offset = np.mean((y_raw - y_theo)[ok])
        self.correction = {int(l): float(y_theo[i] + offset - y_raw[i])
                           for i, l in enumerate(idx) if ok[i]}
        self.T_ref = T

    def _fit(self, spectrum) -> FitResult:
        idx, I = self.measure_lines(spectrum)
        sel = np.array([l in self.correction for l in idx], bool)
        idx, I = idx[sel], I[sel]
        y = self.boltzmann_y(idx, I) + np.array([self.correction[int(l)] for l in idx])
        T, n, info = self.solve(idx, y)
        return self._result_from_n(T, n, n_evals=0,
                                   extra={"n_lines": len(idx), "T_ref": self.T_ref, **info})
