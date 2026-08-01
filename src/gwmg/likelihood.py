"""GW standard-siren likelihood (numpy/scipy only).

Depends only on numpy/scipy so it can be imported and tested without CosmoSIS.
The CosmoSIS module gwmg.pipeline.modules.dgw wraps this and reads the datablock.

Physics (Baker & Harrison 2020, arXiv:2007.13791):
    d_GW(z) = d_L(z) * exp( 0.5 * integral_0^z alpha_M(z') / (1+z') dz' )
Error budget, their eqs. (3.9)-(3.11):
    sigma^2 = sigma_meas^2 + sigma_lens^2 + sigma_v^2
    sigma_lens = 0.066 d_L [(1-(1+z)^-0.25)/0.25]^1.8       (weak lensing)
    sigma_v    = d_L [1 + c(1+z)/(H d_L)] sqrt(<v^2>)/c     (peculiar velocity)
"""
import numpy as np
from scipy import interpolate

C_KM_S = 3.0e5  # speed of light [km/s], matching the thesis/reference convention


def gw_log_likelihood(z_model, dl_model, h_model, dgw_ratio_model,
                      d_gw_obs, z_obs, sigma_dgw, sigma_z, v_rms,
                      c_km_s=C_KM_S):
    """Return the summed GW standard-siren Gaussian log-likelihood (-0.5 chi^2).

    Parameters
    ----------
    z_model, dl_model, h_model, dgw_ratio_model : array_like
        Theory tables from hi_class: redshift, EM luminosity distance d_L(z) [Mpc],
        H(z)/c [1/Mpc], and d_L^GW/d_L^EM(z).
    d_gw_obs, z_obs, sigma_dgw, sigma_z, v_rms : array_like
        Per-event observations (distances in Mpc, v_rms in km/s).
    c_km_s : float
        Speed of light in km/s.
    """
    z_model = np.asarray(z_model, dtype=float)
    dl_model = np.asarray(dl_model, dtype=float)
    h_model = np.asarray(h_model, dtype=float)
    dgw_ratio_model = np.asarray(dgw_ratio_model, dtype=float)

    # Splines need ascending x; reverse all columns together so the ratio and
    # H(z) stay paired with z. (The original code reversed only z and d_l.)
    if z_model[1] < z_model[0]:
        z_model = z_model[::-1]
        dl_model = dl_model[::-1]
        h_model = h_model[::-1]
        dgw_ratio_model = dgw_ratio_model[::-1]

    dgw_model = dgw_ratio_model * dl_model

    Hz_spl = interpolate.UnivariateSpline(z_model, c_km_s * h_model, k=1, s=0)
    Hz = Hz_spl(z_obs)

    # Redshift-error term uses d(d_L)/dz, not d(d_L^GW)/dz, matching the thesis.
    dl_spl = interpolate.UnivariateSpline(z_model, dl_model, k=1, s=0)
    dd_gwdz = dl_spl.derivative()(z_obs)

    dgw_spl = interpolate.UnivariateSpline(z_model, dgw_model, k=1, s=0)
    d_gw_theory = dgw_spl(z_obs)

    sigma_n = np.sqrt(sigma_dgw ** 2. + (dd_gwdz * sigma_z) ** 2.)
    sigma_lens = d_gw_obs * 0.066 * ((1. - (1 + z_obs) ** -0.25) / 0.25) ** 1.8
    sigma_v = d_gw_obs * (1 + c_km_s * (1. + z_obs) / (Hz * d_gw_obs)) * (v_rms / c_km_s)

    chisquare = (d_gw_obs - d_gw_theory) ** 2. / (sigma_n ** 2. + sigma_lens ** 2. + sigma_v ** 2.)
    return -0.5 * chisquare.sum()

# NOTE (preserved from the thesis): the redshift-error term uses d(d_L)/dz, not
# d(d_L^GW)/dz. Kept identical so results reproduce the thesis; differentiate
# dgw_spl instead if you deliberately want the GW-distance derivative.


def load_events(path):
    """Load an events file: columns d_gw_obs[Mpc] z sigma_dgw[Mpc] sigma_z v_rms[km/s].
    Returns a tuple of 1-D arrays (works for 1 or N events)."""
    cols = np.loadtxt(path, unpack=True, comments="#")
    return tuple(np.atleast_1d(a) for a in cols)
