"""Emulator drop-in replacement for hi_class_interface.

Populates the CosmoSIS datablock from trained CosmoPower emulators (CMB TT,
matter power / sigma_8, growth f-sigma_8) plus an exact LCDM background and the
GW luminosity-distance integral, so the downstream likelihoods (Planck, RSD,
BAO, GW sirens) see the same datablock they would from hi_class. It runs in the
combined `gw-e2e` environment (cosmosis + cosmopower); no classy is needed.

Because the expansion history is LCDM (w = -1) the background is analytic, and
propto_omega gives alpha_M(a) = alpha_M0 * Omega_DE(a)/Omega_DE(1) = alpha_M0 /
E(z)^2, so the GW distance ratio is a cheap integral. The emulators are valid
only inside their training box; out-of-box proposals are rejected.

Config (options in the module's ini section):
    cmb_model, logpk_model, growth_model : restore paths for the three emulators
    zmax   (default 3.0), dz (default 0.01), kmax (default 1.0)
"""
import numpy as np
from cosmosis.datablock import names, option_section

try:
    from scipy.integrate import cumulative_trapezoid as _cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz as _cumtrapz

cosmo = names.cosmological_parameters
distances = names.distances
cmb_cl = names.cmb_cl
growthparams = names.growth_parameters
horndeski = "horndeski_parameters"

C_KMS = 299792.458
OMNUH2 = 0.00083                      # matches the training-set generation

PARAMS8 = ["omega_m", "h0", "omega_b", "n_s", "A_s", "tau", "alpha_B0", "alpha_M0"]
PARAMS6 = PARAMS8[:6]
# Training box (must match generate_lcdm_tt.py / generate_growth.py).
BOX = {"omega_m": (0.24, 0.36), "h0": (0.61, 0.76), "omega_b": (0.041, 0.054),
       "n_s": (0.93, 1.00), "A_s": (1.7e-9, 2.5e-9), "tau": (0.02, 0.12),
       "alpha_B0": (-1.0, 3.0), "alpha_M0": (-1.0, 6.0)}


def setup(options):
    from cosmopower import cosmopower_NN
    cfg = {
        "cmb": cosmopower_NN(restore=True,
                             restore_filename=options.get_string(option_section, "cmb_model")),
        "logpk": cosmopower_NN(restore=True,
                               restore_filename=options.get_string(option_section, "logpk_model")),
        "growth": cosmopower_NN(restore=True,
                                restore_filename=options.get_string(option_section, "growth_model")),
        "zmax": options.get_double(option_section, "zmax", default=3.0),
        "dz": options.get_double(option_section, "dz", default=0.01),
    }
    cfg["ell"] = cfg["cmb"].modes.astype(float)
    # The CMB emulator may be LCDM (6 params) or MG-aware (8 params, i.e. including
    # alpha_B0/alpha_M0). Use whichever list the trained model declares: the MG-aware
    # one captures the alpha-lensing response that actually constrains alpha_M.
    cfg["cmb_params"] = [str(s) for s in cfg["cmb"].parameters]
    cfg["kh"] = cfg["logpk"].modes.astype(float)     # h/Mpc
    cfg["zg"] = cfg["growth"].modes.astype(float)    # growth redshift grid
    return cfg


def _params_from_block(block):
    h0 = block[cosmo, "h0"]
    h2 = h0 ** 2
    ombh2 = block[cosmo, "ombh2"]
    omch2 = block[cosmo, "omch2"]
    if block.has_value(cosmo, "A_s"):
        a_s = block[cosmo, "A_s"]
    else:
        a_s = np.exp(block[cosmo, "logA"]) / 1e10
    return {
        "omega_m": (omch2 + ombh2 + OMNUH2) / h2,
        "h0": h0,
        "omega_b": ombh2 / h2,
        "n_s": block[cosmo, "n_s"],
        "A_s": a_s,
        "tau": block[cosmo, "tau"],
        "alpha_B0": block[horndeski, "parameters_smg__2"],
        "alpha_M0": block[horndeski, "parameters_smg__3"],
    }


def _in_box(p):
    return all(BOX[k][0] <= p[k] <= BOX[k][1] for k in BOX)


def _sigma8(kh, pk):
    """sigma_8 from P(k) [(Mpc/h)^3] on the kh grid [h/Mpc], top-hat R = 8 Mpc/h."""
    x = kh * 8.0
    w = 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3
    integ = pk * w ** 2 * kh ** 2
    return np.sqrt(np.trapz(integ, kh) / (2.0 * np.pi ** 2))


def _rs_drag(ombh2, omch2):
    """Sound horizon at the drag epoch, Aubourg et al. 2015 (arXiv:1411.1074)
    eq. 16 fitting formula (~0.02% for standard cosmologies), massless nu."""
    omega_cb = ombh2 + omch2
    return 55.154 * np.exp(-72.3 * (0.0006) ** 2) / (omega_cb ** 0.25351 * ombh2 ** 0.12807)


def execute(block, cfg):
    p = _params_from_block(block)
    if not _in_box(p):
        return 1                       # reject: outside the emulator's validity box

    h0 = p["h0"]
    om = p["omega_m"]
    # radiation density (photons + N_ur massless neutrinos)
    Tcmb = 2.726
    og = 2.469e-5 / h0 ** 2 * (Tcmb / 2.725) ** 4          # photons
    orad = og * (1.0 + 0.2271 * 3.046)                     # + neutrinos
    ol = 1.0 - om - orad                                   # flat

    z = np.arange(0.0, cfg["zmax"] + cfg["dz"], cfg["dz"])
    E = np.sqrt(om * (1 + z) ** 3 + orad * (1 + z) ** 4 + ol)

    # --- CMB TT (6 standard parameters) ---
    cl_tt = (10.0 ** cfg["cmb"].predictions_np({k: [p[k]] for k in cfg["cmb_params"]}))[0]
    ell = cfg["ell"]
    fac = ell * (ell + 1.0) / (2 * np.pi) * (Tcmb * 1e6) ** 2
    block[cmb_cl, "ell"] = ell.astype(int)   # planck_py uses ell as slice indices
    block[cmb_cl, "tt"] = cl_tt * fac
    zeros = np.zeros_like(ell)
    for s in ("ee", "te", "bb", "pp", "tp"):
        block[cmb_cl, s] = zeros        # TT-only likelihood; keys must exist

    # --- P(k) and sigma_8 (8 parameters) ---
    pk = 10.0 ** cfg["logpk"].predictions_np({k: [p[k]] for k in PARAMS8})[0]
    sigma8 = _sigma8(cfg["kh"], pk)
    block[cosmo, "sigma_8"] = sigma8
    block[cosmo, "omega_m"] = om

    # --- background: distances, H, sound horizon ---
    dc = (C_KMS / (100.0 * h0)) * _cumtrapz(1.0 / E, z, initial=0.0)   # comoving [Mpc]
    d_a = dc / (1 + z)
    d_l = dc * (1 + z)
    block[distances, "z"] = z
    block[distances, "nz"] = len(z)
    block[distances, "d_a"] = d_a
    block[distances, "d_l"] = d_l
    block[distances, "d_m"] = dc
    block[distances, "mu"] = 5.0 * np.log10(d_l + 1e-100) + 25.0
    block[distances, "H"] = (100.0 * h0 * E) / C_KMS      # H(z)/c in 1/Mpc (as classy)
    block[distances, "a"] = 1.0 / (1 + z)
    block[distances, "rs_zdrag"] = _rs_drag(p["omega_b"] * h0 ** 2, om * h0 ** 2 - p["omega_b"] * h0 ** 2 - OMNUH2)

    # --- GW luminosity distance: alpha_M(z) = alpha_M0 * Omega_DE(z),
    #     with Omega_DE(z) = Omega_Lambda / E(z)^2 (propto_omega, LCDM) ---
    alpha_mz = p["alpha_M0"] * ol / E ** 2
    ratio = np.exp(0.5 * _cumtrapz(alpha_mz / (1 + z), z, initial=0.0))
    block[distances, "alpha_mz"] = alpha_mz
    block[distances, "d_l_gw_on_d_l_em"] = ratio
    block[distances, "d_l_gw"] = d_l * ratio

    # --- growth f-sigma_8 (8 parameters) ---
    # The RSD likelihoods want growth_parameters d_z = D(z) and f_z = f(z), and
    # use them for both fsigma8 and sigma_8(z) = sigma_8 * D(z)/D(0). The emulator
    # gives fsigma8(z) only, so recover D and f from the scale-independent growth
    # relation fsigma8 = -sigma_8 (1+z) dD/dz with D(0) = 1, then f = fsigma8/(sigma_8 D).
    zg = cfg["zg"]
    fs8 = cfg["growth"].predictions_np({k: [p[k]] for k in PARAMS8})[0]
    D = 1.0 - _cumtrapz(fs8 / (sigma8 * (1 + zg)), zg, initial=0.0)
    f = fs8 / (sigma8 * D)
    block[growthparams, "z"] = zg
    block[growthparams, "d_z"] = D
    block[growthparams, "f_z"] = f
    block[growthparams, "fsigma_8"] = fs8
    block[growthparams, "sigma_8"] = sigma8
    return 0


def cleanup(config):
    return 0
