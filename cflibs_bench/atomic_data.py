"""Dados atômicos usados pelo modelo direto.

As 74 linhas de Ca, Al, Mg, Si, Fe, Mn, Ti e Cr foram transcritas da Tabela 2 de
Gornushkin & Völker, "Intrinsic Performance of Monte Carlo Calibration-Free
Algorithm for LIBS" (Sensors 2022). Foram acrescentadas as linhas de ressonância
de Na I e K I (relevantes para amostras geológicas), com dados do NIST ASD.

As funções de partição e as larguras Stark são APROXIMAÇÕES tabeladas para
permitir um benchmark autocontido (sem acesso ao NIST). Como o espectro "teste" e
os algoritmos usam o mesmo modelo, isso não afeta a comparação entre algoritmos;
para trabalhar com espectros reais, substitua-as por dados do NIST/Griem.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Energias em cm^-1, comprimento de onda em nm (ar), f = força de oscilador.
# Colunas: elemento, estágio (1 = neutro, 2 = íon), lambda, Ei, Ek, f_ik, gi, gk
_LINES = """
Ca 1 428.301 15210 38552 0.1990 3 5
Ca 1 428.936 15158 38465 0.500 1 3
Ca 1 429.899 15210 38465 0.129 3 3
Ca 1 430.253 15316 38552 0.378 5 5
Ca 1 430.774 15210 38418 0.1849 3 1
Ca 1 431.865 15316 38465 0.1200 5 3
Ca 1 558.876 20371 38259 0.2317 7 7
Ca 1 559.849 20335 38192 0.2009 3 3
Ca 1 560.285 20349 38192 0.0399 5 3
Ca 2 315.887 25192 56839 0.847 2 4
Ca 2 317.933 25414 56858 0.8200 4 6
Ca 2 318.128 25414 56839 0.088 4 4
Ca 2 370.602 25192 52167 0.172 2 2
Ca 2 373.690 25414 52167 0.164 4 2
Ca 2 393.366 0 25414 0.682 2 4
Ca 2 396.847 0 25192 0.330 2 2
Al 1 308.215 0 32435 0.1670 2 4
Al 1 309.271 112 32437 0.1570 4 6
Al 1 394.401 0 25348 0.1160 2 2
Al 1 396.152 112 25348 0.1160 4 2
Mg 1 516.732 21850 41197 0.1350 1 3
Mg 1 517.268 21870 41197 0.1350 3 3
Mg 1 518.360 21911 41197 0.1360 5 3
Si 1 390.552 15394 40992 0.091 1 3
Fe 1 303.739 888 33802 0.0671 3 5
Fe 1 304.760 704 33507 0.0533 5 7
Fe 1 305.745 6928 39626 0.0359 11 9
Fe 1 305.909 416 33096 0.0294 7 9
Fe 1 306.724 7376 39970 0.0342 9 7
Fe 1 371.993 0 26875 0.0411 9 11
Fe 1 372.256 704 27560 0.0103 5 5
Fe 1 374.336 7986 34692 0.0328 5 3
Fe 1 374.556 704 27395 0.0339 5 7
Fe 1 374.826 888 27560 0.0321 3 5
Fe 1 374.948 7377 34040 0.1610 9 9
Fe 1 375.823 7728 34329 0.1340 7 7
Fe 1 376.719 8155 34692 0.1360 3 3
Fe 1 404.581 11976 36686 0.2120 9 9
Fe 1 406.359 12561 37163 0.1650 5 5
Fe 1 407.174 12969 37521 0.1900 5 5
Mn 1 403.075 0 24802 0.0550 6 8
Mn 1 403.307 0 24788 0.0403 6 6
Mn 1 403.449 0 24779 0.0257 6 4
Mn 1 403.575 17282 42054 0.0800 8 6
Mn 1 404.136 17052 41789 0.193 10 10
Mn 1 472.746 23549 44696 0.057 6 6
Mn 1 473.909 23720 44815 0.081 4 4
Mn 1 475.404 18402 39431 0.137 6 8
Mn 1 475.585 23720 44696 0.210 4 6
Mn 1 476.151 23819 44815 0.364 2 4
Mn 1 476.237 23297 44289 0.333 8 10
Mn 1 476.642 23549 44523 0.210 6 8
Mn 1 478.343 18531 39431 0.138 8 8
Mn 1 482.352 18705 39431 0.139 10 8
Mn 2 344.199 14326 43371 0.0500 9 7
Mn 2 346.031 14594 43485 0.0334 7 5
Mn 2 347.413 14781 43557 0.0179 5 3
Mn 2 348.290 14781 43485 0.0291 5 5
Mn 2 348.868 14901 43557 0.0385 3 3
Ti 1 498.173 6843 26911 0.2900 11 13
Ti 1 499.107 6743 26773 0.2670 9 11
Ti 1 499.950 6661 26657 0.254 7 9
Ti 2 323.452 393 31301 0.2685 10 10
Ti 2 323.657 226 31114 0.2150 8 8
Ti 2 323.904 94 30959 0.198 10 10
Ti 2 324.199 0 30836 0.2317 4 4
Ti 2 324.860 10025 40798 0.388 6 8
Ti 2 336.121 226 29968 0.3350 8 10
Ti 2 337.279 94 29735 0.3210 6 8
Ti 2 338.376 0 29544 0.3580 4 6
Cr 1 357.869 0 27935 0.3660 7 9
Cr 1 359.349 0 27820 0.2910 7 7
Cr 1 425.434 0 23499 0.1100 7 8
Cr 1 427.480 0 23386 0.0842 7 7
Na 1 588.995 0 16973 0.641 2 4
Na 1 589.592 0 16956 0.320 2 2
K 1 766.490 0 13043 0.682 2 4
K 1 769.896 0 12985 0.340 2 2
"""

# Linhas EXTRAS (opt-in: ``load_lines(..., extra=True)``), úteis para carbonatos e
# outras matrizes ricas em Mg/Sr. f calculado de A_ki/log gf do NIST ASD — conferir
# antes de uso quantitativo. Não fazem parte do benchmark de referência do README.
_EXTRA_LINES = """
Ca 1 422.673 0 23652 1.75 1 3
Ca 1 610.272 15158 31539 0.161 1 3
Ca 1 612.222 15210 31539 0.161 3 3
Ca 1 616.217 15316 31539 0.163 5 3
Mg 1 382.935 21850 47957 0.62 1 3
Mg 1 383.231 21870 47957 0.46 3 5
Mg 1 383.829 21911 47957 0.52 5 7
Sr 1 460.733 0 21698 1.92 1 3
Sr 2 407.771 0 24517 0.707 2 4
Sr 2 421.552 0 23715 0.338 2 2
"""

# Energia de ionização (eV) — NIST.
IONIZATION_EV = {
    "Ca": 6.1132, "Al": 5.9858, "Mg": 7.6462, "Si": 8.1517, "Fe": 7.9024,
    "Mn": 7.4340, "Ti": 6.8281, "Cr": 6.7665, "Na": 5.1391, "K": 4.3407,
    "Sr": 5.6949,
}

# Massa atômica (u).
ATOMIC_MASS = {
    "Ca": 40.078, "Al": 26.982, "Mg": 24.305, "Si": 28.085, "Fe": 55.845,
    "Mn": 54.938, "Ti": 47.867, "Cr": 51.996, "Na": 22.990, "K": 39.098, "Sr": 87.62,
    "O": 15.999,
}

# Funções de partição APROXIMADAS em T = 5000, 10000, 15000, 20000 K.
_U_T = np.array([5000.0, 10000.0, 15000.0, 20000.0])
_U_TABLE = {
    ("Ca", 1): (1.13, 3.0, 8.0, 18.0), ("Ca", 2): (2.0, 2.4, 3.2, 4.2),
    ("Al", 1): (5.8, 6.2, 7.5, 10.0), ("Al", 2): (1.0, 1.0, 1.1, 1.3),
    ("Mg", 1): (1.0, 1.1, 1.6, 3.0), ("Mg", 2): (2.0, 2.0, 2.0, 2.1),
    ("Si", 1): (9.3, 10.5, 13.0, 17.0), ("Si", 2): (5.6, 5.9, 6.3, 7.0),
    ("Fe", 1): (26.0, 45.0, 90.0, 180.0), ("Fe", 2): (40.0, 55.0, 75.0, 100.0),
    ("Mn", 1): (6.0, 8.5, 18.0, 35.0), ("Mn", 2): (7.1, 8.5, 12.0, 17.0),
    ("Ti", 1): (28.0, 50.0, 100.0, 190.0), ("Ti", 2): (52.0, 75.0, 105.0, 140.0),
    ("Cr", 1): (7.4, 15.0, 35.0, 70.0), ("Cr", 2): (6.2, 9.0, 14.0, 21.0),
    ("Na", 1): (2.0, 2.4, 6.0, 15.0), ("Na", 2): (1.0, 1.0, 1.0, 1.0),
    ("K", 1): (2.0, 3.0, 10.0, 30.0), ("K", 2): (1.0, 1.0, 1.0, 1.0),
    ("Sr", 1): (1.0, 1.3, 3.0, 8.0), ("Sr", 2): (2.0, 2.2, 2.6, 3.2),
}

# Larguras Lorentzianas (FWHM, nm) a ne = 1e17 cm^-3 — aproximação uniforme por estágio.
STARK_FWHM_NM = {1: 0.010, 2: 0.012}

CM1_TO_EV = 1.239841984e-4


def partition_function(element: str, stage: int, T) -> np.ndarray:
    """U(T) por interpolação linear de ln U em T (extrapolação constante nas bordas)."""
    T = np.asarray(T, dtype=float)
    logU = np.log(np.asarray(_U_TABLE[(element, stage)]))
    return np.exp(np.interp(T, _U_T, logU))


@dataclass(frozen=True)
class LineList:
    element: np.ndarray   # str
    stage: np.ndarray     # int
    wl: np.ndarray        # nm
    Ei: np.ndarray        # eV
    Ek: np.ndarray        # eV
    f: np.ndarray
    gi: np.ndarray
    gk: np.ndarray

    def __len__(self) -> int:
        return len(self.wl)

    def select(self, mask) -> "LineList":
        return LineList(*(getattr(self, k)[mask] for k in self.__dataclass_fields__))

    @property
    def species(self) -> list[tuple[str, int]]:
        return list(zip(self.element.tolist(), self.stage.tolist()))


def load_lines(elements=None, wl_range=None, extra=False) -> LineList:
    text = _LINES.strip() + ("\n" + _EXTRA_LINES.strip() if extra else "")
    rows = [r.split() for r in text.splitlines()]
    el = np.array([r[0] for r in rows])
    stage = np.array([int(r[1]) for r in rows])
    num = np.array([[float(x) for x in r[2:]] for r in rows])
    lines = LineList(
        element=el, stage=stage, wl=num[:, 0],
        Ei=num[:, 1] * CM1_TO_EV, Ek=num[:, 2] * CM1_TO_EV,
        f=num[:, 3], gi=num[:, 4], gk=num[:, 5],
    )
    mask = np.ones(len(lines), bool)
    if elements is not None:
        mask &= np.isin(lines.element, list(elements))
    if wl_range is not None:
        mask &= (lines.wl >= wl_range[0]) & (lines.wl <= wl_range[1])
    return lines.select(mask)


ALL_ELEMENTS = ["Si", "Al", "Fe", "Ca", "Mg", "Na", "K", "Ti", "Mn", "Cr"]
