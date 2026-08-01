"""Tests for the pure GW likelihood core. Needs only numpy+scipy (+ gwmg installed):
    pip install -e .[test]  &&  pytest
"""
import numpy as np
from scipy import interpolate

from gwmg.likelihood import gw_log_likelihood


def _original_reference(z_model, dl_model, h_model, dgw_ratio_model,
                        d_gw_obs, z_obs, sigma_dgw, sigma_z, v_rms):
    """Verbatim re-implementation of the original thesis dgw.likelihood() — used
    to prove the packaged version matches on the normal (ascending-z) path."""
    z = np.asarray(z_model, float); dl = np.asarray(dl_model, float)
    h = np.asarray(h_model, float); ratio = np.asarray(dgw_ratio_model, float)
    if z[1] < z[0]:
        z = z[::-1]; dl = dl[::-1]                       # original reversed ONLY z, dl
    dgw = ratio * dl
    Hz = interpolate.UnivariateSpline(z, 3e5 * h, k=1, s=0)(z_obs)
    dd = interpolate.UnivariateSpline(z, dl, k=1, s=0).derivative()(z_obs)
    dth = interpolate.UnivariateSpline(z, dgw, k=1, s=0)(z_obs)
    sn = np.sqrt(sigma_dgw ** 2 + (dd * sigma_z) ** 2)
    sl = d_gw_obs * 0.066 * ((1 - (1 + z_obs) ** -0.25) / 0.25) ** 1.8
    sv = d_gw_obs * (1 + 3e5 * (1 + z_obs) / (Hz * d_gw_obs)) * (v_rms / 3e5)
    return -0.5 * ((d_gw_obs - dth) ** 2 / (sn ** 2 + sl ** 2 + sv ** 2)).sum()


def _toy(alpha_m=0.0, n=400, zmax=2.0):
    z = np.linspace(1e-4, zmax, n)
    Ez = np.sqrt(0.3 * (1 + z) ** 3 + 0.7)
    h = 70.0 * Ez / 3e5
    integrand = 3e5 / (70.0 * Ez)
    dc = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(z))])
    dl = (1 + z) * dc
    ratio = np.exp(alpha_m / 2.0 * np.log(1 + z))
    return z, dl, h, ratio


EVENTS = dict(d_gw_obs=np.array([40.0, 5300.0]), z_obs=np.array([0.0099, 0.82]),
              sigma_dgw=np.array([11.0, 2500.0]), sigma_z=np.array([1e-4, 1e-4]),
              v_rms=np.array([500.0, 500.0]))


def test_matches_original_ascending():
    for am in (0.0, -0.5, 0.3, 1.0):
        z, dl, h, r = _toy(am)
        assert np.isclose(gw_log_likelihood(z, dl, h, r, **EVENTS),
                          _original_reference(z, dl, h, r, **EVENTS), rtol=1e-12)


def test_reversal_invariance_is_fixed():
    """A table and its row-reversed copy describe the same model -> same likelihood."""
    for am in (-0.4, 0.2, 0.9):
        z, dl, h, r = _toy(am)
        asc = gw_log_likelihood(z, dl, h, r, **EVENTS)
        desc = gw_log_likelihood(z[::-1], dl[::-1], h[::-1], r[::-1], **EVENTS)
        assert np.isclose(asc, desc, rtol=1e-10)


def test_finite_and_nonpositive():
    z, dl, h, r = _toy(0.0)
    v = gw_log_likelihood(z, dl, h, r, **EVENTS)
    assert np.isfinite(v) and v <= 0.0
