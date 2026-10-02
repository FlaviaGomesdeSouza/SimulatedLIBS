"""Conversões entre % em massa de óxidos (forma usual em geoquímica) e frações atômicas.

As amostras de referência abaixo têm composições APROXIMADAS de materiais de
referência geológicos (USGS BCR-2, AGV-2, G-2) e da escória sintética do artigo
de Gornushkin & Völker (2022). Servem para testes; confira os valores
certificados antes de usá-los como verdade em trabalho experimental.
"""
from __future__ import annotations

import numpy as np

from .atomic_data import ATOMIC_MASS

# elemento -> (fórmula do óxido, átomos do elemento, átomos de O)
OXIDES = {
    "Si": ("SiO2", 1, 2), "Ti": ("TiO2", 1, 2), "Al": ("Al2O3", 2, 3),
    "Fe": ("Fe2O3", 2, 3), "Mn": ("MnO", 1, 1), "Mg": ("MgO", 1, 1),
    "Ca": ("CaO", 1, 1), "Na": ("Na2O", 2, 1), "K": ("K2O", 2, 1),
    "Cr": ("Cr2O3", 2, 3),
}
OXIDE_TO_ELEMENT = {v[0]: k for k, v in OXIDES.items()}


def _oxide_mass_per_cation(el: str) -> float:
    _, a, b = OXIDES[el]
    return (a * ATOMIC_MASS[el] + b * ATOMIC_MASS["O"]) / a


def oxides_to_atomic_fractions(oxide_wt: dict[str, float]) -> dict[str, float]:
    """% massa de óxidos -> fração atômica dos cátions (soma 1; O não incluído)."""
    moles = {}
    for ox, wt in oxide_wt.items():
        el = OXIDE_TO_ELEMENT[ox]
        moles[el] = wt / _oxide_mass_per_cation(el)
    tot = sum(moles.values())
    return {el: m / tot for el, m in moles.items()}


def atomic_fractions_to_oxides(frac: dict[str, float]) -> dict[str, float]:
    """Frações atômicas dos cátions -> % massa de óxidos normalizada a 100 %."""
    mass = {OXIDES[el][0]: x * _oxide_mass_per_cation(el) for el, x in frac.items()}
    tot = sum(mass.values())
    return {ox: 100.0 * m / tot for ox, m in mass.items()}


def normalize(d: dict[str, float]) -> dict[str, float]:
    tot = sum(d.values())
    return {k: v / tot for k, v in d.items()}


SAMPLES: dict[str, dict[str, float]] = {
    # Basalto (aprox. BCR-2)
    "basalto": {"SiO2": 54.1, "TiO2": 2.26, "Al2O3": 13.5, "Fe2O3": 13.8, "MnO": 0.196,
                "MgO": 3.59, "CaO": 7.12, "Na2O": 3.16, "K2O": 1.79, "Cr2O3": 0.0026},
    # Andesito (aprox. AGV-2)
    "andesito": {"SiO2": 59.3, "TiO2": 1.05, "Al2O3": 16.9, "Fe2O3": 6.69, "MnO": 0.099,
                 "MgO": 1.79, "CaO": 5.20, "Na2O": 4.19, "K2O": 2.88, "Cr2O3": 0.0025},
    # Granito (aprox. G-2)
    "granito": {"SiO2": 69.1, "TiO2": 0.48, "Al2O3": 15.4, "Fe2O3": 2.66, "MnO": 0.03,
                "MgO": 0.75, "CaO": 1.96, "Na2O": 4.08, "K2O": 4.48, "Cr2O3": 0.0013},
    # Escória metalúrgica (Gornushkin & Völker 2022, Tabela 1; FeO -> Fe2O3)
    "escoria": {"Al2O3": 19.7, "CaO": 27.0, "Cr2O3": 0.5, "Fe2O3": 26.7, "MgO": 11.5,
                "MnO": 8.0, "TiO2": 0.3, "SiO2": 9.0},
}


def random_geological_composition(rng: np.random.Generator, spread=0.5) -> dict[str, float]:
    """Composição aleatória (% óxidos): rocha-base (basalto/andesito/granito) com
    cada óxido multiplicado por um fator log-normal (σ = ``spread``) e renormalizada.

    A perturbação multiplicativa preserva a ordem de grandeza dos traços
    (uma Dirichlet faria MnO/Cr2O3 colapsarem para ~0). Útil para treino de ML.
    """
    base = SAMPLES[rng.choice(["basalto", "andesito", "granito"])]
    keys = list(base)
    x = np.array([base[k] for k in keys]) * np.exp(rng.normal(0.0, spread, len(keys)))
    x = 100.0 * x / x.sum()
    return dict(zip(keys, x))
