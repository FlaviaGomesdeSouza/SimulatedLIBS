"""Ponte com o pacote SimulatedLIBS (Kastek & Gąsior, PyPI ``SimulatedLIBS``).

O SimulatedLIBS gera espectros consultando o formulário LIBS do NIST ASD
(precisa de internet). Aqui ele serve como gerador INDEPENDENTE do "espectro
experimental": outro banco de linhas, perfis e premissas (plasma opticamente
fino). Inverter esses espectros com o ``PlasmaModel`` testa a robustez dos
algoritmos a ERRO DE MODELO — situação mais próxima de dados reais.

Exemplo::

    from cflibs_bench.adapters.simulatedlibs import spectrum_from_simulatedlibs
    from cflibs_bench.composition import SAMPLES, oxides_to_atomic_fractions

    frac = oxides_to_atomic_fractions(SAMPLES["basalto"])
    spec = spectrum_from_simulatedlibs(frac, Te_eV=0.86, Ne=1e17,
                                       resolution=10000, low_w=300, upper_w=780,
                                       cache="basalto_nist.csv")

Observação: a composição enviada ao NIST é a fração relativa de cada elemento
(o O dos óxidos não é incluído, pois não é medido pelo modelo).
"""
from __future__ import annotations

import os

from ..composition import atomic_fractions_to_oxides
from ..spectrum import Spectrum

EV_TO_K = 11604.518


def spectrum_from_simulatedlibs(fractions: dict[str, float], Te_eV=1.0, Ne=1e17,
                                resolution=1000, low_w=200, upper_w=1000, max_ion_charge=2,
                                step=None, cache: str | None = None,
                                min_percentage=1e-3) -> Spectrum:
    """Gera (ou lê do cache CSV) um espectro do SimulatedLIBS.

    ``fractions``: frações relativas por elemento (normalizadas para 100 %).
    Elementos abaixo de ``min_percentage`` % são omitidos (o NIST rejeita zeros).
    """
    truth = {"T": Te_eV * EV_TO_K, "ne": Ne, "atomic_fractions": dict(fractions),
             "oxide_wt": atomic_fractions_to_oxides(fractions), "source": "SimulatedLIBS/NIST"}
    if cache and os.path.exists(cache):
        s = Spectrum.from_csv(cache)
        s.truth = truth
        return s
    try:
        from SimulatedLIBS import simulation
    except ImportError as e:   # pragma: no cover
        raise ImportError("Instale com: pip install SimulatedLIBS") from e
    tot = sum(fractions.values())
    els = [e for e, v in fractions.items() if 100 * v / tot >= min_percentage]
    pct = [round(100 * fractions[e] / tot, 4) for e in els]
    libs = simulation.SimulatedLIBS(Te=Te_eV, Ne=Ne, elements=els, percentages=pct,
                                    resolution=resolution, low_w=low_w, upper_w=upper_w,
                                    max_ion_charge=max_ion_charge, webscraping="static")
    if step:
        libs.interpolate(step)
    df = libs.get_interpolated_spectrum()
    s = Spectrum(df["wavelength"].to_numpy(float), df["intensity"].to_numpy(float), truth)
    if cache:
        s.to_csv(cache)
    return s


def dataset_from_simulatedlibs(input_csv: str, output_csv="dataset_simulatedlibs.csv", size=100,
                               Te_min=0.7, Te_max=2.0, Ne_min=1e17, Ne_max=2e17):
    """Gera um conjunto de treino com ``SimulatedLIBS.create_dataset`` (como em Gąsior et al.).

    ``input_csv``: tabela de composições, uma coluna por elemento e a última
    coluna 'name' (formato exigido pelo SimulatedLIBS).
    """
    from SimulatedLIBS import simulation
    return simulation.SimulatedLIBS.create_dataset(
        input_csv, output_csv, size=size, Te_min=Te_min, Te_max=Te_max,
        Ne_min=Ne_min, Ne_max=Ne_max)
