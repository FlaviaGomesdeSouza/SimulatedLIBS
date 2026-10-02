"""Teste com espectros LIBS REAIS: carbonatos de Veneranda et al. (2023).

COMO USAR
---------
1. Edite o bloco "CONFIGURAÇÃO" logo abaixo (caminhos e padrão).
2. Salve o arquivo (Ctrl+S).
3. Clique em ▶ (Run Python File), no canto superior direito do VS Code.

Dados: https://doi.org/10.5281/zenodo.7803300 (pasta "analytical data/LIBS",
85 arquivos .xy = 17 amostras × 5 pontos). Composição de referência: ICP-OES
(Tabela 2 do artigo), em % massa dos cátions.

Métodos comparados (todos sem curva de calibração):
  1. CF-LIBS Saha–Boltzmann (sem padrão)
  2. Ponto único de Cavalcanti (1 padrão, correção por linha)
  3. Mínimos quadrados com o modelo direto (sem padrão)
  4. Mínimos quadrados + calibração de resposta por 1 padrão (ganho por fragmento)
"""

# =============================================================================
#                              CONFIGURAÇÃO
#        (edite apenas esta parte; mantenha o r antes das aspas)
# =============================================================================

# Pasta onde estão os arquivos .xy (ex.: "Dolomite 1 LIBS (1).xy").
# Dica: no Explorador de Arquivos, clique na barra de endereço da pasta,
# copie (Ctrl+C) e cole aqui entre as aspas.
PASTA_DADOS = r"D:\FLAVIA\CETEM\03_PROJETO LASER\LIBS\dados_veneranda\LIBS"

# Pasta onde os resultados (tabelas .csv e figura .png) serão salvos.
# Se não existir, será criada.
PASTA_RESULTADOS = r"D:\FLAVIA\CETEM\03_PROJETO LASER\LIBS\resultados_carbonatos"

# Amostra(s) usada(s) como padrão único. Para testar vários padrões de uma vez,
# coloque mais de um nome na lista; cada um gera uma subpasta de resultados.
# Nomes válidos: Ankerite 1, Aragonite 1, Aragonite 2, Aragonite 3, Calcite 1,
# Calcite 2, Calcite 3, Dolomite 1, Dolomite 2, Dolomite 3, Huntite 1,
# Magnesite 1, Magnesite 2, Magnesite 3, Siderite 1, Siderite 2, Siderite 3
PADROES = ["Ankerite 1"]
# Exemplo com três padrões:
# PADROES = ["Ankerite 1", "Dolomite 3", "Siderite 2"]

# Parâmetros do espectrômetro/plasma (normalmente não precisa mudar)
PODER_DE_RESOLUCAO = 3500     # λ/Δλ (medido pela largura das linhas desses dados)
DENSIDADE_ELETRONICA = 1e17   # cm^-3

# =============================================================================
#                 (daqui para baixo não é preciso editar)
# =============================================================================
import argparse
import glob
import os
import sys
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


def erro(msg):
    print("\n*** ERRO ***\n" + msg + "\n")
    sys.exit(1)


def verificar_configuracao(pasta_dados, padroes):
    if not os.path.isdir(pasta_dados):
        erro(f"A pasta de dados não foi encontrada:\n  {pasta_dados}\n"
             "Confira o caminho em PASTA_DADOS (copie da barra de endereço do Explorador).")
    if not glob.glob(os.path.join(pasta_dados, "*.xy")):
        sub = [d for d in glob.glob(os.path.join(pasta_dados, "*")) if os.path.isdir(d)]
        dica = ("\nSubpastas encontradas: " + ", ".join(os.path.basename(s) for s in sub)
                + "\nTalvez os arquivos estejam numa delas.") if sub else ""
        erro(f"Nenhum arquivo .xy dentro de:\n  {pasta_dados}{dica}")
    invalidos = [p for p in padroes if p not in VENERANDA_2023_ICP]
    if invalidos:
        erro(f"Padrão(ões) desconhecido(s): {invalidos}\n"
             f"Use um destes nomes (exatamente assim): {', '.join(VENERANDA_2023_ICP)}")


def rodar(data, model, padrao, pasta_saida):
    os.makedirs(pasta_saida, exist_ok=True)
    ref_frac = cation_wt_to_atomic_fractions(VENERANDA_2023_ICP[padrao])
    gains, info = fit_reference_response(model, data[padrao], ref_frac)
    print(f"\n=== Padrão: {padrao} (T ajustada = {info['T_ref']:.0f} K, "
          f"{info['valid'].sum()}/{len(gains)} fragmentos calibrados) ===")
    bounds = model.default_bounds(T=(3000.0, 15000.0))
    one_point = OnePointCalibration(model, data[padrao], reference_fractions=ref_frac)
    methods = {
        "SBP (sem padrão)": lambda s: SahaBoltzmannCF(model).fit(s),
        "SBP + ponto único": lambda s: one_point.fit(s),
        "LS (sem padrão)": lambda s: LocalLeastSquares(model, n_starts=3, seed=0,
                                                       bounds=bounds).fit(s),
        "LS + resposta (1 padrão)": lambda s: LocalLeastSquares(
            model, n_starts=3, seed=0, bounds=bounds).fit(apply_response(s, model, gains)),
    }

    rows = []
    for name, spec in data.items():
        if name == padrao:
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
    df.to_csv(os.path.join(pasta_saida, "resultados.csv"), index=False)
    mae = df[df.elemento.isin(MAIN)].pivot_table(index="metodo", columns="elemento",
                                                 values="erro_abs_wt", aggfunc="mean")[MAIN]
    mae["média"] = mae.mean(axis=1)
    mae = mae.sort_values("média").round(1)
    mae.to_csv(os.path.join(pasta_saida, "erro_medio_absoluto.csv"))
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
    fig.suptitle(f"Carbonatos de Veneranda et al. (2023) — padrão único: {padrao}")
    fig.tight_layout()
    fig.savefig(os.path.join(pasta_saida, "estimado_vs_icp.png"), dpi=150)
    plt.close(fig)
    return mae["média"].rename(padrao)


def main():
    # Opcional: os valores da CONFIGURAÇÃO podem ser substituídos pelo terminal
    p = argparse.ArgumentParser(description="Carbonatos de Veneranda et al. (2023)")
    p.add_argument("--data", default=PASTA_DADOS)
    p.add_argument("--out", default=PASTA_RESULTADOS)
    p.add_argument("--reference", nargs="+", default=PADROES)
    p.add_argument("--resolving-power", type=float, default=PODER_DE_RESOLUCAO)
    p.add_argument("--ne", type=float, default=DENSIDADE_ELETRONICA)
    a = p.parse_args()

    verificar_configuracao(a.data, a.reference)
    model = PlasmaModel(elements=ELEMENTS, resolving_power=a.resolving_power, ne=a.ne,
                        extra_lines=True, step=0.02)
    print(f"Lendo {a.data} ...")
    raw = load_folder(a.data)
    data = {k: preprocess(v, model) for k, v in raw.items()}
    print(f"{len(data)} amostras; deslocamento médio de λ = "
          f"{np.mean([s.truth['shift_nm'] for s in data.values()]):+.3f} nm")

    resumo = []
    for padrao in a.reference:
        saida = a.out if len(a.reference) == 1 else os.path.join(a.out, padrao.replace(" ", "_"))
        resumo.append(rodar(data, model, padrao, saida))
        print(f"Resultados salvos em: {os.path.abspath(saida)}")

    if len(resumo) > 1:
        tab = pd.concat(resumo, axis=1).round(1)
        os.makedirs(a.out, exist_ok=True)
        tab.to_csv(os.path.join(a.out, "comparacao_padroes.csv"))
        print("\nErro médio (Ca, Mg, Fe, Mn) por padrão — menor é melhor:")
        print(tab.to_string())
        print(f"Tabela salva em: {os.path.abspath(os.path.join(a.out, 'comparacao_padroes.csv'))}")


if __name__ == "__main__":
    main()
