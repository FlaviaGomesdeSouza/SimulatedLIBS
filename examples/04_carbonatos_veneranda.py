"""Teste com espectros LIBS REAIS: carbonatos de Veneranda et al. (2023).

Dados: https://doi.org/10.5281/zenodo.7803300 (pasta "analytical data/LIBS",
85 arquivos .xy = 17 amostras × 5 pontos, emulador SimulCam/SuperCam, Echelle,
255–800 nm, atmosfera terrestre). Composição de referência: ICP-OES (Tabela 2
do artigo), em % massa dos cátions.

Métodos comparados (todos sem curva de calibração):
  1. CF-LIBS Saha–Boltzmann (sem padrão)
  2. Ponto único de Cavalcanti (1 padrão, correção por linha)
  3. Mínimos quadrados com o modelo direto (sem padrão)
  4. Mínimos quadrados + calibração de resposta por 1 padrão (ganho por fragmento)

Uso:
    python examples/04_carbonatos_veneranda.py --data "C:/caminho/analytical data/LIBS"
    python examples/04_carbonatos_veneranda.py --data ... --reference "Dolomite 3"
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cflibs_bench.algorithms import LocalLeastSquares, OnePointCalibration, SahaBoltzmannCF
from cflibs_bench.composition import atomic_fractions_to_cation_wt, cation_wt_to_atomic_fractions
from cflibs_bench.model import PlasmaModel
from cflibs_bench.real_data import (VENERANDA_2023_ICP, apply_response, fit_reference_response,
                                    load_folder, preprocess)

ELEMENTS = ["Ca", "Mg", "Fe", "Mn", "Na", "K", "Sr", "Al"]
MAIN = ["Ca", "Mg", "Fe", "Mn"]

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help='pasta com os arquivos .xy (ex.: "analytical data/LIBS")')
p.add_argument("--reference", default="Ankerite 1", help="amostra usada como padrão único")
p.add_argument("--resolving-power", type=float, default=3500.0)
p.add_argument("--ne", type=float, default=1e17)
p.add_argument("--out", default="resultados_carbonatos")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

model = PlasmaModel(elements=ELEMENTS, resolving_power=a.resolving_power, ne=a.ne,
                    extra_lines=True, step=0.02)
print(f"Lendo {a.data} ...")
raw = load_folder(a.data)
data = {k: preprocess(v, model) for k, v in raw.items()}
print(f"{len(data)} amostras; deslocamento médio de λ = "
      f"{np.mean([s.truth['shift_nm'] for s in data.values()]):+.3f} nm")

ref_frac = cation_wt_to_atomic_fractions(VENERANDA_2023_ICP[a.reference])
gains, info = fit_reference_response(model, data[a.reference], ref_frac)
print(f"Padrão: {a.reference} (T ajustada = {info['T_ref']:.0f} K, "
      f"{info['valid'].sum()}/{len(gains)} fragmentos calibrados)")
bounds = model.default_bounds(T=(3000.0, 15000.0))
one_point = OnePointCalibration(model, data[a.reference], reference_fractions=ref_frac)
methods = {
    "SBP (sem padrão)": lambda s: SahaBoltzmannCF(model).fit(s),
    "SBP + ponto único": lambda s: one_point.fit(s),
    "LS (sem padrão)": lambda s: LocalLeastSquares(model, n_starts=3, seed=0, bounds=bounds).fit(s),
    "LS + resposta (1 padrão)": lambda s: LocalLeastSquares(
        model, n_starts=3, seed=0, bounds=bounds).fit(apply_response(s, model, gains)),
}

rows = []
for name, spec in data.items():
    if name == a.reference:
        continue
    icp = VENERANDA_2023_ICP.get(name)
    if icp is None:
        print(f"  {name}: sem composição de referência, pulando")
        continue
    t0 = time.perf_counter()
    for meth, fn in methods.items():
        r = fn(spec)
        wt = atomic_fractions_to_cation_wt(r.atomic_fractions)
        for el in ELEMENTS:
            rows.append({"amostra": name, "metodo": meth, "elemento": el, "T_K": r.T,
                         "estimado_wt": wt.get(el, 0.0), "icp_wt": icp.get(el, 0.0)})
    print(f"  {name:12s} ok ({time.perf_counter() - t0:.1f} s)")

df = pd.DataFrame(rows)
df["erro_abs_wt"] = (df["estimado_wt"] - df["icp_wt"]).abs()
df.to_csv(os.path.join(a.out, "resultados.csv"), index=False)
mae = df[df.elemento.isin(MAIN)].pivot_table(index="metodo", columns="elemento",
                                             values="erro_abs_wt", aggfunc="mean")[MAIN]
mae["média"] = mae.mean(axis=1)
mae = mae.sort_values("média").round(1)
mae.to_csv(os.path.join(a.out, "erro_medio_absoluto.csv"))
print("\nErro médio absoluto vs ICP-OES (pontos de % massa catiônica):")
print(mae.to_string())

fig, axes = plt.subplots(1, len(MAIN), figsize=(16, 4.2))
for ax, el in zip(axes, MAIN):
    d = df[df.elemento == el]
    for i, (meth, g) in enumerate(d.groupby("metodo", sort=False)):
        ax.plot(g.icp_wt, g.estimado_wt, "osD^"[i % 4], ms=5, alpha=0.8, label=meth)
    lim = max(d.icp_wt.max(), d.estimado_wt.max()) * 1.05 + 0.5
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set(xlim=(0, lim), ylim=(0, lim), title=el, xlabel="ICP-OES (% massa)",
           ylabel="LIBS sem curva (% massa)")
axes[0].legend(fontsize=7)
fig.suptitle(f"Carbonatos de Veneranda et al. (2023) — padrão único: {a.reference}")
fig.tight_layout()
fig.savefig(os.path.join(a.out, "estimado_vs_icp.png"), dpi=150)
print(f"\nArquivos salvos em {a.out}/")
