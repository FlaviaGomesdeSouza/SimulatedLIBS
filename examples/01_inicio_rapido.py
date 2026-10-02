"""Exemplo mínimo: espectro sintético de basalto + três algoritmos.

    python examples/01_inicio_rapido.py
"""
import numpy as np

from cflibs_bench import SAMPLES, PlasmaModel, add_noise
from cflibs_bench.algorithms import MonteCarloCF, OnePointCalibration, SahaBoltzmannCF

model = PlasmaModel(wl_range=(300, 780), ne=1e17, resolving_power=10000)  # Echelle típico

# "Experimental" sintético: basalto a 10 000 K, ruído gaussiano de 0,5 % do máximo
spec = add_noise(model.simulate_sample(SAMPLES["basalto"], T=10000), 0.005, rng=1)
ref = add_noise(model.simulate_sample(SAMPLES["andesito"], T=10000), 0.005, rng=2)

tau = model.peak_optical_depth(10000, 0.1, np.array(
    [spec.truth["atomic_fractions"][e] for e in model.elements])[None] * 3e16)
print("Linhas mais autoabsorvidas (τ no centro):")
for i in np.argsort(-tau)[:5]:
    print(f"  {model.lines.element[i]} {'I' * model.lines.stage[i]:3s} {model.lines.wl[i]:.3f} nm  τ={tau[i]:.1f}")

for alg in [MonteCarloCF(model, n_iter=20, polish=True, seed=0),
            SahaBoltzmannCF(model),
            OnePointCalibration(model, ref, SAMPLES["andesito"])]:
    r = alg.fit(spec)
    print(f"\n{r.algorithm}: T = {r.T:.0f} K  ({r.runtime:.1f} s)")
    print(f"  {'óxido':6s} {'verdadeiro':>10s} {'estimado':>10s} {'erro':>8s}")
    for ox, true in spec.truth["oxide_wt"].items():
        est = r.oxide_wt[ox]
        print(f"  {ox:6s} {true:10.4f} {est:10.4f} {100 * (est - true) / true:+7.1f}%")
