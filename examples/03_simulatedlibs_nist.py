"""Espectro gerado pelo SimulatedLIBS (NIST) invertido com o modelo próprio.

Requer internet e ``pip install SimulatedLIBS``. O SimulatedLIBS usa outro banco
de linhas e assume plasma opticamente fino, então este exemplo mede a robustez
dos algoritmos a ERRO DE MODELO (mais próximo de um espectro real).

    python examples/03_simulatedlibs_nist.py
"""
from cflibs_bench import SAMPLES, PlasmaModel, add_noise
from cflibs_bench.adapters.simulatedlibs import spectrum_from_simulatedlibs
from cflibs_bench.algorithms import LinearUnmixing, LocalLeastSquares, SahaBoltzmannCF
from cflibs_bench.composition import oxides_to_atomic_fractions

frac = oxides_to_atomic_fractions(SAMPLES["basalto"])
nist = spectrum_from_simulatedlibs(frac, Te_eV=0.86, Ne=1e17, resolution=5000,
                                   low_w=300, upper_w=780, step=0.01,
                                   cache="basalto_simulatedlibs.csv")
nist = add_noise(nist, 0.005, rng=0)

# Mesmo poder de resolução do espectro gerado; a grade do modelo é reamostrada.
model = PlasmaModel(resolving_power=5000)
for alg in [LocalLeastSquares(model, n_starts=3, seed=0), SahaBoltzmannCF(model),
            LinearUnmixing(model)]:
    r = alg.fit(nist)
    print(f"\n{r.algorithm}: T = {r.T:.0f} K (verdadeira {nist.truth['T']:.0f} K)")
    for ox, true in nist.truth["oxide_wt"].items():
        print(f"  {ox:6s} {true:8.3f} {r.oxide_wt[ox]:8.3f}")
