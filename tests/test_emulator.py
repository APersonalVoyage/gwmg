"""Tests for the emulator support code that doesn't need hi_class or cosmopower:
the LHS sampler and the validation error statistics."""
import numpy as np

from gwmg.emulator import (latin_hypercube, DEFAULT_RANGES,
                           fractional_errors, summarise_errors, write_report)
from gwmg.emulator.validate import cosmic_variance_sigma


def test_latin_hypercube_shape_and_bounds():
    n = 50
    s = latin_hypercube(DEFAULT_RANGES, n, seed=3)
    assert set(s) == set(DEFAULT_RANGES)
    for k, (lo, hi) in DEFAULT_RANGES.items():
        assert s[k].shape == (n,)
        assert s[k].min() >= lo and s[k].max() <= hi


def test_fractional_errors_and_summary():
    true = np.array([[1.0, 2.0, 4.0]])
    pred = np.array([[1.1, 2.0, 3.6]])          # 10%, 0%, 10% errors
    fe = fractional_errors(pred, true)
    assert np.allclose(fe, [[0.1, 0.0, 0.1]])
    s = summarise_errors(fe)
    assert 0.0 <= s["median"] <= s["p95"] <= s["max"]
    assert np.isclose(s["max"], 0.1)


def test_cosmic_variance_sigma():
    ell = np.array([2.0, 100.0, 1000.0])
    tt = np.array([[1e-9, 1e-11, 1e-13]])
    ee = np.array([[1e-10, 1e-12, 1e-14]])
    te = np.array([[0.0, -1e-12, 1e-14]])       # TE crosses zero
    s_te = cosmic_variance_sigma(ell, tt, ee, te, "te")
    assert s_te.shape == tt.shape
    assert np.all(s_te > 0)                      # never vanishes, even where TE=0
    s_tt = cosmic_variance_sigma(ell, tt, ee, te, "tt")
    assert np.allclose(s_tt, np.sqrt(2.0 / (2 * ell + 1)) * tt)


def test_write_report(tmp_path):
    summ = {"median": 0.001, "p68": 0.002, "p95": 0.005, "p99": 0.01, "max": 0.03}
    report = {"logpk": {"phys": summ},
              "cl_te": {"phys": summ, "cv": {"median": 0.2, "p68": 0.3,
                                             "p95": 0.6, "p99": 0.9, "max": 2.0}}}
    p = tmp_path / "rep.txt"
    write_report(report, str(p))
    text = p.read_text()
    assert "cl_te" in text and "logpk" in text and "cv" in text


def test_sigma8_from_pk():
    """sigma_8 integral: a power law with a known normalisation is positive and
    scales as sqrt(amplitude)."""
    from gwmg.emulator.predict import _sigma8_from_pk
    k = np.logspace(-4, 0, 400)
    pk = 2e4 * k ** 0.96 / (1 + (k / 0.02) ** 2) ** 2
    s1 = _sigma8_from_pk(k, pk)
    s2 = _sigma8_from_pk(k, 4 * pk)
    assert s1 > 0
    assert np.isclose(s2 / s1, 2.0, rtol=1e-6)


def test_emulator_box_check():
    """The training-box guard is pure logic, so it is testable without cosmopower."""
    from gwmg.emulator.predict import Emulator, BOX, PARAMS8
    inside = {k: 0.5 * (lo + hi) for k, (lo, hi) in BOX.items()}
    assert set(inside) == set(PARAMS8)
    assert Emulator.in_box(None, **inside)
    outside = dict(inside, omega_m=0.9)
    assert not Emulator.in_box(None, **outside)


def test_parameter_bias():
    """Fisher-bias formalism, with a fake likelihood (no Planck data / cosmopower):
    a perfect emulator biases nothing; a wrong one biases something."""
    from gwmg.emulator.chi2 import parameter_bias
    rng = np.random.default_rng(0)
    nmode, ndata = 60, 20
    ell = np.arange(2, 2 + nmode).astype(float)
    W = rng.normal(size=(ndata, nmode))           # fake linear binning

    class FakeLite:
        fisher = np.eye(ndata)

        def mean(self, dl, ellmin=2):
            return np.atleast_2d(dl) @ W.T

    lite = FakeLite()
    fid = np.abs(rng.normal(size=nmode)) * 1e-10 + 1e-11
    steps = np.array([0.01, 0.02])
    da, db = rng.normal(size=nmode) * 1e-12, rng.normal(size=nmode) * 1e-12
    deriv = {"ell": ell, "keys": np.array(["a", "b"]), "steps": steps, "fid": fid,
             "a_p": fid + steps[0] * da, "a_m": fid - steps[0] * da,
             "b_p": fid + steps[1] * db, "b_m": fid - steps[1] * db}

    perfect = parameter_bias(lite, deriv, fid, ell)          # emulator == truth
    assert perfect["keys"] == ["a", "b"]
    assert perfect["bias_over_sigma"].shape == (2,)
    assert perfect["max_abs_bias_over_sigma"] < 1e-9

    wrong = parameter_bias(lite, deriv, fid * 1.001, ell)    # 0.1% off
    assert wrong["max_abs_bias_over_sigma"] > 0
