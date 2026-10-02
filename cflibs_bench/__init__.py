"""cflibs_bench — bancada de testes de algoritmos de LIBS sem calibração
(CF-LIBS) em espectros sintéticos."""
from .composition import SAMPLES, atomic_fractions_to_oxides, oxides_to_atomic_fractions
from .model import PlasmaModel
from .spectrum import Spectrum, add_noise

__all__ = ["PlasmaModel", "Spectrum", "add_noise", "SAMPLES",
           "oxides_to_atomic_fractions", "atomic_fractions_to_oxides"]
__version__ = "0.1.0"
