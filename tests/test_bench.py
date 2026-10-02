import numpy as np
import pytest

from cflibs_bench import SAMPLES, PlasmaModel, add_noise
from cflibs_bench.algorithms import (BayesianMCMC, LinearUnmixing, LocalLeastSquares,
                                     MonteCarloCF, OnePointCalibration, SahaBoltzmannCF,
                                     CostFunction)
from cflibs_bench.composition import atomic_fractions_to_oxides, oxides_to_atomic_fractions

MAJORS = ["SiO2", "Al2O3", "Fe2O3", "CaO"]


@pytest.fixture(scope="module")
def model():
    return PlasmaModel()


def rel_err(res, truth, keys):
    return np.array([abs(res.oxide_wt[k] - truth[k]) / truth[k] for k in keys])


def test_oxide_roundtrip():
    ox = SAMPLES["basalto"]
    back = atomic_fractions_to_oxides(oxides_to_atomic_fractions(ox))
    tot = sum(ox.values())
    for k, v in ox.items():
        assert back[k] == pytest.approx(100 * v / tot, rel=1e-9)


def test_forward_shapes_and_selfabsorption(model):
    n = np.full((3, model.n_elements), 1e15)
    I = model.synthesize([8000, 10000, 12000], [0.1] * 3, n)
    assert I.shape == (3, len(model.wl)) and np.all(np.isfinite(I)) and I.min() >= -1e-12
    # intensidade não cresce linearmente com n quando a linha é opticamente espessa
    I1 = model.synthesize([10000], [0.1], n[:1] * 1.0, instrument=False)[0]
    I2 = model.synthesize([10000], [0.1], n[:1] * 100.0, instrument=False)[0]
    assert I2.max() / I1.max() < 100


def test_cost_minimum_at_truth(model):
    s = model.simulate_sample(SAMPLES["basalto"])
    n = np.array([s.truth["atomic_fractions"][e] for e in model.elements]) * 3e16
    th = np.r_[10000, -1, np.log10(np.maximum(n, 1e10))]
    c = CostFunction(model, s)
    assert c.theta_cost(th)[0] < 1e-8
    th2 = th.copy(); th2[0] = 12000
    assert c.theta_cost(th2)[0] > 1e-4


def test_sbp_thin_plasma(model):
    s = model.simulate_sample(SAMPLES["basalto"], n_total=1e13)
    r = SahaBoltzmannCF(model).fit(s)
    assert r.T == pytest.approx(10000, rel=0.02)
    assert np.all(rel_err(r, s.truth["oxide_wt"], MAJORS) < 0.1)


def test_linear_unmixing_thin(model):
    s = model.simulate_sample(SAMPLES["granito"], n_total=1e13)
    r = LinearUnmixing(model).fit(s)
    assert r.T == pytest.approx(10000, rel=0.01)
    assert np.all(rel_err(r, s.truth["oxide_wt"], MAJORS) < 0.02)


def test_one_point_runs(model):
    s = add_noise(model.simulate_sample(SAMPLES["basalto"]), 0.005, rng=0)
    ref = add_noise(model.simulate_sample(SAMPLES["andesito"]), 0.005, rng=1)
    r = OnePointCalibration(model, ref, SAMPLES["andesito"]).fit(s)
    assert abs(sum(r.oxide_wt.values()) - 100) < 1e-6


def test_local_least_squares_recovers_majors(model):
    s = add_noise(model.simulate_sample(SAMPLES["basalto"]), 0.002, rng=0)
    r = LocalLeastSquares(model).fit(s)
    assert r.T == pytest.approx(10000, rel=0.01)
    assert np.all(rel_err(r, s.truth["oxide_wt"], MAJORS) < 0.05)


def test_mc_cost_decreases(model):
    s = add_noise(model.simulate_sample(SAMPLES["basalto"]), 0.005, rng=0)
    r = MonteCarloCF(model, n_configs=300, n_iter=5, seed=0).fit(s)
    h = r.extra["history"]
    assert h[-1] <= h[0]


def test_mcmc_quick(model):
    s = add_noise(model.simulate_sample(SAMPLES["basalto"]), 0.005, rng=0)
    r = BayesianMCMC(model, n_walkers=24, n_steps=40, burn_in=20, seed=0).fit(s)
    assert "oxide_p16" in r.extra and 0 < r.extra["acceptance"] < 1
