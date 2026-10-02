"""Robustez ao ruído: erro mediano dos óxidos maiores vs. nível de ruído.

    python examples/02_estudo_ruido.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cflibs_bench.benchmark import run

levels = [0.001, 0.003, 0.01, 0.03]
algs = ["ls", "mcmc", "sbp", "one_point", "nnls", "ml_ridge"]
curves = {}
for lv in levels:
    df, _, _ = run(noise=lv, seeds=2, algorithms=algs, plot=False, verbose=False)
    maj = df[df["classe"].str.startswith("maior")]
    for alg, d in maj.groupby("algoritmo"):
        curves.setdefault(alg, []).append(np.median(np.abs(d["erro_rel_%"])))
    print(f"ruído {lv:.3f} concluído")

fig, ax = plt.subplots(figsize=(7, 4.5))
for alg, ys in curves.items():
    ax.plot(levels, ys, "o-", label=alg)
ax.set(xscale="log", yscale="log", xlabel="σ do ruído / máximo do espectro",
       ylabel="|erro| mediano, óxidos > 1 % (%)", title="Robustez ao ruído (basalto, 10 000 K)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("estudo_ruido.png", dpi=150)
print("Figura salva em estudo_ruido.png")
