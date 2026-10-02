"""Bancada de comparação de algoritmos CF-LIBS em espectros sintéticos.

Uso (linha de comando)::

    python -m cflibs_bench.benchmark --sample basalto --noise 0.005 --seeds 3
    python -m cflibs_bench.benchmark --algorithms mc,mc_ls,sbp,one_point --T 9000
    python -m cflibs_bench.benchmark --list

Gera em ``--out``: resultados.csv (formato longo), resumo.csv e figuras.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from . import algorithms as A
from .composition import SAMPLES
from .model import PlasmaModel
from .spectrum import add_noise

QUICK = os.environ.get("CFLIBS_QUICK") == "1"


def build_algorithms(model, names, reference=None, reference_oxides=None, seed=0):
    """Fábrica de algoritmos com configurações padrão razoáveis em CPU."""
    q = QUICK
    factory = {
        "mc": lambda: A.MonteCarloCF(model, n_iter=15 if q else 40, seed=seed),
        "mc_ls": lambda: A.MonteCarloCF(model, n_iter=8 if q else 20, polish=True, seed=seed),
        "sa": lambda: A.SimulatedAnnealing(model, maxfun=2000 if q else 15000, seed=seed),
        "de": lambda: A.DifferentialEvolution(model, popsize=10 if q else 20,
                                              maxiter=40 if q else 150, seed=seed),
        "ls": lambda: A.LocalLeastSquares(model, n_starts=1),
        "ls_multi": lambda: A.LocalLeastSquares(model, n_starts=5, seed=seed),
        "mcmc": lambda: A.BayesianMCMC(model, n_steps=200 if q else 1500,
                                       burn_in=100 if q else 700, seed=seed),
        "nnls": lambda: A.LinearUnmixing(model, "nnls"),
        "svd": lambda: A.LinearUnmixing(model, "svd"),
        "sbp": lambda: A.SahaBoltzmannCF(model),
        "sbp_nores": lambda: A.SahaBoltzmannCF(model, exclude_resonance=True),
        "one_point": lambda: A.OnePointCalibration(model, reference, reference_oxides),
        "ml_pls": lambda: A.MLRegression(model, "pls", n_train=300 if q else 1500, seed=seed),
        "ml_ridge": lambda: A.MLRegression(model, "ridge", n_train=300 if q else 1500, seed=seed),
        "ml_rf": lambda: A.MLRegression(model, "rf", n_train=300 if q else 1500, seed=seed),
        "ml_mlp": lambda: A.MLRegression(model, "mlp", n_train=300 if q else 1500, seed=seed),
    }
    if names == ["all"]:
        names = list(factory)
    return {n: factory[n]() for n in names}


ALL_NAMES = ["mc", "mc_ls", "sa", "de", "ls", "ls_multi", "mcmc", "nnls", "svd", "sbp",
             "sbp_nores", "one_point", "ml_pls", "ml_ridge", "ml_rf", "ml_mlp"]


def concentration_class(wt):
    return "maior (>1%)" if wt > 1 else ("menor (0,1–1%)" if wt >= 0.1 else "traço (<0,1%)")


def run(sample="basalto", reference="andesito", T=10000.0, noise=0.005, seeds=1,
        algorithms=("mc_ls", "de", "mcmc", "nnls", "sbp", "one_point", "ml_ridge"),
        resolving_power=10000.0, ne=1e17, n_total=3e16, out=None, plot=True, verbose=True):
    model = PlasmaModel(ne=ne, resolving_power=resolving_power)
    ref_clean = model.simulate_sample(SAMPLES[reference], T=T, n_total=n_total)
    rows, fits = [], {}
    for seed in range(seeds):
        clean = model.simulate_sample(SAMPLES[sample], T=T, n_total=n_total)
        spec = add_noise(clean, noise, rng=1000 + seed)
        ref = add_noise(ref_clean, noise, rng=2000 + seed)
        algs = build_algorithms(model, list(algorithms), ref, SAMPLES[reference], seed)
        truth = spec.truth["oxide_wt"]
        for key, alg in algs.items():
            t0 = time.perf_counter()
            try:
                r = alg.fit(spec)
            except Exception as e:   # noqa: BLE001 — registra a falha e segue
                print(f"  [{key}] falhou: {e}")
                continue
            fits[(key, seed)] = r
            if verbose:
                print(f"  seed {seed}  {r.algorithm:32s} T={r.T:8.0f} K  "
                      f"{time.perf_counter() - t0:7.1f} s")
            for ox, true in truth.items():
                est = r.oxide_wt.get(ox, 0.0)
                rows.append({"seed": seed, "algoritmo": r.algorithm, "chave": key, "oxido": ox,
                             "verdadeiro_wt": true, "estimado_wt": est,
                             "erro_rel_%": 100 * (est - true) / true,
                             "classe": concentration_class(true), "T_verdadeira": T,
                             "T_estimada": r.T, "tempo_s": r.runtime, "avaliacoes": r.n_evals})
    df = pd.DataFrame(rows)
    summary = summarize(df)
    if out:
        os.makedirs(out, exist_ok=True)
        df.to_csv(os.path.join(out, "resultados.csv"), index=False)
        summary.to_csv(os.path.join(out, "resumo.csv"))
        if plot:
            make_plots(df, model, spec, fits, out)
    return df, summary, fits


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(abs_err=df["erro_rel_%"].abs())
    tab = df.pivot_table(index="algoritmo", columns="classe", values="abs_err", aggfunc="median")
    tab.columns = [f"|erro| mediano {c} (%)" for c in tab.columns]
    g = df.groupby("algoritmo")
    tab["|ΔT| (K)"] = g.apply(lambda d: (d["T_estimada"] - d["T_verdadeira"]).abs().mean())
    tab["tempo (s)"] = g["tempo_s"].mean()
    tab["avaliações do modelo"] = g["avaliacoes"].mean()
    return tab.sort_values(tab.columns[0]).round(2)


def make_plots(df, model, spec, fits, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # 1) mapa de calor: |erro relativo| mediano por algoritmo × óxido
    tab = df.assign(e=df["erro_rel_%"].abs()).pivot_table(
        index="algoritmo", columns="oxido", values="e", aggfunc="median")
    conc = df.groupby("oxido")["verdadeiro_wt"].first().sort_values(ascending=False)
    tab = tab[conc.index]
    tab = tab.loc[tab[[c for c in conc.index if conc[c] > 1]].median(1).sort_values().index]
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(tab) + 1.8))
    im = ax.imshow(np.clip(tab.values, 0.1, 1e3), cmap="RdYlGn_r",
                   norm=LogNorm(0.1, 1e3), aspect="auto")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.values[i, j]
            ax.text(j, i, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", va="center",
                    fontsize=7)
    ax.set_xticks(range(len(conc)), [f"{o}\n{conc[o]:.3g}%" for o in conc.index], fontsize=8)
    ax.set_yticks(range(len(tab)), tab.index, fontsize=8)
    ax.set_title("|erro relativo| mediano (%) — colunas por concentração decrescente")
    fig.colorbar(im, ax=ax, label="|erro| (%)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "erro_vs_concentracao.png"), dpi=150)
    plt.close(fig)

    # 2) espectro e ajustes dos métodos com modelo direto (último seed): visão geral + zoom
    seed = max(s for _, s in fits)
    ynorm = spec.intensity.max()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [2, 1]})
    for ax, (lo, hi) in zip(axes, [(model.wl.min(), model.wl.max()), (370.5, 377.5)]):
        sel = (spec.wl >= lo) & (spec.wl <= hi)
        ax.plot(spec.wl[sel], spec.intensity[sel] / ynorm, "k.", ms=2, label="teste")
        for (key, s), r in fits.items():
            if s == seed and r.theta is not None:
                I = model.synthesize_theta(r.theta)[0]
                a = (I @ spec.intensity) / (I @ I)
                ax.plot(model.wl[sel], a * I[sel] / ynorm, lw=0.9, label=r.algorithm)
        ax.set(xlabel="λ (nm)", xlim=(lo, hi))
    axes[0].set(ylabel="Intensidade normalizada", title="Ajustes (fragmentos concatenados)")
    axes[1].set_title("Zoom: linhas de Fe I")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ajustes.png"), dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", default="basalto", choices=list(SAMPLES))
    p.add_argument("--reference", default="andesito", choices=list(SAMPLES),
                   help="padrão usado na calibração de ponto único")
    p.add_argument("--T", type=float, default=10000.0)
    p.add_argument("--noise", type=float, default=0.005, help="σ do ruído / máximo do espectro")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--algorithms", default="mc_ls,de,mcmc,nnls,sbp,one_point,ml_ridge",
                   help="lista separada por vírgulas ou 'all'")
    p.add_argument("--resolving-power", type=float, default=10000.0)
    p.add_argument("--out", default="resultados")
    p.add_argument("--list", action="store_true", help="lista os algoritmos disponíveis")
    a = p.parse_args(argv)
    if a.list:
        print("\n".join(ALL_NAMES))
        return
    algs = ["all"] if a.algorithms == "all" else a.algorithms.split(",")
    _, summary, _ = run(a.sample, a.reference, a.T, a.noise, a.seeds, algs,
                        a.resolving_power, out=a.out)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary)
    print(f"\nArquivos salvos em {a.out}/")


if __name__ == "__main__":
    main()
