"""Algoritmos de análise sem calibração (CF-LIBS) com interface comum.

Todos recebem um :class:`~cflibs_bench.model.PlasmaModel` e expõem
``fit(spectrum) -> FitResult``.
"""
from .base import Algorithm, FitResult
from .boltzmann import OnePointCalibration, SahaBoltzmannCF
from .cost import CostFunction
from .global_opt import DifferentialEvolution, SimulatedAnnealing
from .linear import LinearUnmixing
from .local import LocalLeastSquares
from .mcmc import BayesianMCMC
from .ml import MLRegression
from .montecarlo import MonteCarloCF

__all__ = ["Algorithm", "FitResult", "CostFunction", "MonteCarloCF", "SimulatedAnnealing",
           "DifferentialEvolution", "LocalLeastSquares", "BayesianMCMC", "LinearUnmixing",
           "SahaBoltzmannCF", "OnePointCalibration", "MLRegression"]
