"""Leitura e pré-processamento de espectros LIBS experimentais.

* ``read_xy``: lê arquivos de duas colunas (λ, I) com separador ``;``, ``,``,
  tabulação ou espaço, e vírgula decimal.
* ``load_folder``: agrupa os arquivos por amostra (ex.: "Dolomite 1 LIBS (3).xy"
  → "Dolomite 1") e devolve a média dos pontos de cada amostra.
* ``estimate_shift``: estima o deslocamento de calibração em λ por correlação
  com um "pente" das linhas do modelo.
* ``preprocess``: corrige o deslocamento, reamostra na grade do modelo, subtrai
  a linha de base por fragmento e normaliza.
"""
from __future__ import annotations

import glob
import os
import re
from collections import defaultdict

import numpy as np

from .model import PlasmaModel
from .spectrum import Spectrum


def read_xy(path: str) -> Spectrum:
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    rows = []
    for ln in lines:
        if ";" in ln:
            parts = ln.split(";")
        elif "\t" in ln:
            parts = ln.split("\t")
        elif ln.count(",") == 1 and "." in ln:
            parts = ln.split(",")
        else:
            parts = ln.replace(",", " ").split() if ln.count(",") == 1 and " " not in ln \
                else ln.split()
        try:
            rows.append([float(p.strip().replace(",", ".")) for p in parts[:2]])
        except ValueError:
            continue   # cabeçalho ou comentário
    data = np.array(rows)
    order = np.argsort(data[:, 0])
    return Spectrum(data[order, 0], data[order, 1])


def sample_name(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    base = re.sub(r"\s*\(\d+\)\s*$", "", base)          # "(3)"
    base = re.sub(r"[\s_-]+LIBS\b.*$", "", base, flags=re.I)
    base = re.sub(r"[_-]\d+$", "", base) if re.search(r"\D[_-]\d+$", base) and \
        not re.search(r"\s\d+[_-]\d+$", base) else base
    return base.strip()


def load_folder(folder: str, pattern="*.xy", average=True):
    """Dicionário amostra -> Spectrum (média dos pontos) ou lista de Spectrum."""
    groups = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(folder, pattern))):
        groups[sample_name(f)].append(read_xy(f))
    if not groups:
        raise FileNotFoundError(f"Nenhum arquivo {pattern} em {folder}")
    if not average:
        return dict(groups)
    out = {}
    for name, specs in groups.items():
        wl = specs[0].wl
        I = np.mean([np.interp(wl, s.wl, s.intensity) for s in specs], axis=0)
        sd = np.std([np.interp(wl, s.wl, s.intensity) for s in specs], axis=0)
        out[name] = Spectrum(wl, I, {"n_spots": len(specs), "spot_std": sd})
    return out


def estimate_shift(spec: Spectrum, model: PlasmaModel, max_shift=0.3, step=0.005) -> float:
    """Deslocamento s (nm) tal que λ_verdadeiro ≈ λ_medido − s."""
    wl, y = spec.wl, spec.intensity - np.median(spec.intensity)
    fw = np.median(model.instr_fwhm) + 0.02
    sig = fw / 2.3548
    best, best_s = -np.inf, 0.0
    for s in np.arange(-max_shift, max_shift + step / 2, step):
        tpl = np.zeros_like(wl)
        for l0 in model.lines.wl:
            m = np.abs(wl - (l0 + s)) < 4 * sig
            tpl[m] += np.exp(-0.5 * ((wl[m] - l0 - s) / sig) ** 2)
        score = np.dot(tpl, y) / (np.linalg.norm(tpl) + 1e-300)
        if score > best:
            best, best_s = score, s
    return float(best_s)


def preprocess(spec: Spectrum, model: PlasmaModel, shift: float | None = None,
               baseline_percentile=10.0) -> Spectrum:
    """Corrige λ, reamostra na grade do modelo, remove linha de base por fragmento."""
    if shift is None:
        shift = estimate_shift(spec, model)
    # ruído: MAD das diferenças no espectro original (antes de interpolar)
    d = np.diff(spec.intensity)
    sigma = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
    y = np.interp(model.wl, spec.wl - shift, spec.intensity)
    for seg in model.segments:
        y[seg] -= np.percentile(y[seg], baseline_percentile)
    scale = np.max(y)
    out = Spectrum(model.wl.copy(), y / scale, dict(spec.truth),
                   noise_sigma=max(sigma / scale, 1e-6))
    out.truth["shift_nm"] = shift
    return out


# Tabela 2 de Veneranda et al. (2023, Earth Space Sci., doi:10.1029/2023EA002829):
# composição catiônica (% massa, soma 100 %) por ICP-OES; <LOD -> 0.
VENERANDA_2023_ICP = {
    "Ankerite 1":  {"Ca": 56.21, "Fe": 14.08, "Mg": 27.03, "Mn": 1.47, "K": 1.13, "Na": 0.04, "Sr": 0, "Al": 0.05},
    "Aragonite 1": {"Ca": 98.00, "Fe": 0.13, "Mg": 0.14, "Mn": 0, "K": 0.27, "Na": 0.03, "Sr": 1.04, "Al": 0.12},
    "Aragonite 2": {"Ca": 97.11, "Fe": 0.02, "Mg": 0.01, "Mn": 0, "K": 0.72, "Na": 0.20, "Sr": 1.94, "Al": 0},
    "Aragonite 3": {"Ca": 97.31, "Fe": 0.12, "Mg": 0.28, "Mn": 0, "K": 0.32, "Na": 0.05, "Sr": 1.73, "Al": 0.20},
    "Calcite 1":   {"Ca": 99.53, "Fe": 0.01, "Mg": 0.06, "Mn": 0, "K": 0.38, "Na": 0, "Sr": 0, "Al": 0},
    "Calcite 2":   {"Ca": 99.59, "Fe": 0, "Mg": 0.38, "Mn": 0, "K": 0, "Na": 0, "Sr": 0.03, "Al": 0},
    "Calcite 3":   {"Ca": 99.93, "Fe": 0, "Mg": 0.07, "Mn": 0, "K": 0, "Na": 0, "Sr": 0, "Al": 0},
    "Dolomite 1":  {"Ca": 51.32, "Fe": 1.34, "Mg": 46.99, "Mn": 0.21, "K": 0, "Na": 0.08, "Sr": 0.02, "Al": 0.02},
    "Dolomite 2":  {"Ca": 51.25, "Fe": 0.43, "Mg": 47.34, "Mn": 0.06, "K": 0.92, "Na": 0, "Sr": 0, "Al": 0},
    "Dolomite 3":  {"Ca": 50.25, "Fe": 3.61, "Mg": 44.55, "Mn": 1.41, "K": 0.15, "Na": 0, "Sr": 0.03, "Al": 0},
    "Huntite 1":   {"Ca": 32.6, "Fe": 0, "Mg": 63.27, "Mn": 0, "K": 0.48, "Na": 1.15, "Sr": 2.49, "Al": 0},
    "Magnesite 1": {"Ca": 5.52, "Fe": 3.71, "Mg": 89.08, "Mn": 0.21, "K": 1.48, "Na": 0, "Sr": 0, "Al": 0},
    "Magnesite 2": {"Ca": 6.20, "Fe": 0, "Mg": 93.8, "Mn": 0, "K": 0, "Na": 0, "Sr": 0, "Al": 0},
    "Magnesite 3": {"Ca": 2.30, "Fe": 0, "Mg": 97.70, "Mn": 0, "K": 0, "Na": 0, "Sr": 0, "Al": 0},
    "Siderite 1":  {"Ca": 2.32, "Fe": 88.53, "Mg": 5.50, "Mn": 1.86, "K": 1.71, "Na": 0.09, "Sr": 0, "Al": 0},
    "Siderite 2":  {"Ca": 3.66, "Fe": 83.53, "Mg": 3.57, "Mn": 9.09, "K": 0.10, "Na": 0.04, "Sr": 0, "Al": 0},
    "Siderite 3":  {"Ca": 2.62, "Fe": 86.79, "Mg": 7.55, "Mn": 1.91, "K": 0.09, "Na": 1.03, "Sr": 0, "Al": 0},
}


# ----------------------------------------------------------------------------
# Calibração de resposta por um padrão (versão "modelo direto" do ponto único)
# ----------------------------------------------------------------------------
def fit_reference_response(model: PlasmaModel, ref: Spectrum, ref_fractions: dict,
                           T_bounds=(3000.0, 15000.0), min_rel_signal=0.02, clip=(0.05, 20.0)):
    """Ajusta o padrão com composição CONHECIDA (só T, R e n_total livres) e
    devolve um ganho por fragmento g_k = <y, s>/<s, s> (normalizado pela mediana).

    Os ganhos absorvem a resposta espectral não corrigida do instrumento, erros
    de dados atômicos e parte da autoabsorção — como na calibração de ponto
    único de Cavalcanti et al., mas aplicada ao espectro inteiro. Fragmentos
    sem sinal suficiente no padrão ficam com ganho 1.
    """
    from scipy.optimize import least_squares

    from .algorithms.cost import CostFunction

    c = np.array([max(ref_fractions.get(e, 0.0), 1e-6) for e in model.elements])
    c /= c.sum()
    cost = CostFunction(model, ref, "corr")

    def synth(p):
        return model.synthesize([p[0]], [10 ** p[1]], (c * 10 ** p[2])[None, :])[0]

    lo = np.array([T_bounds[0], -3.0, 14.0])
    hi = np.array([T_bounds[1], -1.0, 19.0])
    best = None
    for T0 in np.linspace(T_bounds[0] + 500, T_bounds[1] - 500, 5):
        r = least_squares(lambda p: cost.residuals(synth(p)), [T0, -2.0, 16.5], bounds=(lo, hi),
                          x_scale=hi - lo, diff_step=1e-4, max_nfev=200)
        if best is None or r.cost < best.cost:
            best = r
    s = synth(best.x)
    y = ref.intensity
    energy = np.array([np.sum(s[seg] ** 2) for seg in model.segments])
    gains = np.ones(len(model.segments))
    ok = energy > (min_rel_signal ** 2) * energy.max()
    for k, seg in enumerate(model.segments):
        if ok[k]:
            gains[k] = np.dot(y[seg], s[seg]) / energy[k]
    gains[ok] /= np.median(gains[ok])
    gains = np.clip(gains, *clip)
    return gains, {"T_ref": float(best.x[0]), "theta_ref": best.x, "valid": ok}


def apply_response(spec: Spectrum, model: PlasmaModel, gains: np.ndarray) -> Spectrum:
    out = spec.copy()
    for g, seg in zip(gains, model.segments):
        out.intensity[seg] = spec.intensity[seg] / g
    scale = np.max(out.intensity)
    out.intensity /= scale
    if out.noise_sigma:
        out.noise_sigma /= scale
    return out
