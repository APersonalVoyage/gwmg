"""hi_class to CosmoSIS interface.

Runs hi_class through its classy wrapper and writes the datablock outputs the
downstream likelihoods use, including the GW distance ratio
distances/d_l_gw_on_d_l_em read by dgw.py.

Requires `import classy` to resolve to a hi_class build (see docs/install.md),
not a stock CLASS install.
"""
import sys
import traceback

import numpy as np
from cosmosis.datablock import names, option_section

import classy

# cumtrapz was renamed to cumulative_trapezoid and removed in scipy >= 1.14.
try:
    from scipy.integrate import cumulative_trapezoid as _cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz as _cumtrapz

cosmo = names.cosmological_parameters
distances = names.distances
cmb_cl = names.cmb_cl
growthparams = names.growth_parameters
horndeski = "horndeski_parameters"


def setup(options):
    config = {
        "lmax": options.get_int(option_section, "lmax", default=2500),
        "zmax": options.get_double(option_section, "zmax", default=3.0),
        "kmax": options.get_double(option_section, "kmax", default=1.0),
        "debug": options.get_bool(option_section, "debug", default=False),
        "lensing": options.get_string(option_section, "lensing", default="yes"),
        "expansion_model": options.get_string(option_section, "expansion_model", default="lcdm"),
        "gravity_model": options.get_string(option_section, "gravity_model", default="propto_omega"),
        "modes": options.get_string(option_section, "modes", default="s"),
        "output": options.get_string(option_section, "output", default="tCl,lCl,pCl,mPk"),
        "sBBN file": options.get_string(option_section, "sBBN_file"),
        "k_output_values": options.get_string(option_section, "k_output_values", default=""),
        # GW-distance integrand (see get_class_outputs):
        #   False (default): d_gw/d_em = exp(0.5 * int alpha_M/(1+z) dz)  [standard]
        #   True: divides alpha_M by Omega_smg(0), reproducing the older thesis code.
        "alpha_m_over_omega_smg": options.get_bool(
            option_section, "alpha_m_over_omega_smg", default=False),
    }
    config["cosmo"] = classy.Class()
    return config


def smg_params(block):
    """Concatenate parameters_smg__1..N into the comma string classy expects.
    For 'propto_omega': x_k, x_b, x_m, x_t, M*^2_ini."""
    snl = []
    for i in range(1, 20):
        if block.has_value(horndeski, "parameters_smg__%i" % i):
            snl.append(block[horndeski, "parameters_smg__%i" % i])
        else:
            break
    return ",".join(map(str, snl))


def get_class_inputs(block, config):
    params = {
        "output": config["output"],
        "modes": config["modes"],
        "l_max_scalars": config["lmax"],
        "P_k_max_h/Mpc": config["kmax"],
        "lensing": config["lensing"],
        "z_max_pk": config["zmax"],
        "n_s": block[cosmo, "n_s"],
        # omega_b/omega_cdm as physical densities: the `consistency` module,
        # which must run before this one, derives ombh2/omch2 from omega_b,
        # omega_m and h0 in the values file.
        "omega_b": block[cosmo, "ombh2"],
        "omega_cdm": block[cosmo, "omch2"],
        "tau_reio": block[cosmo, "tau"],
        "T_cmb": block.get_double(cosmo, "t_cmb", default=2.726),
        "N_ur": block.get_double(cosmo, "N_ur", default=3.046),
        "k_pivot": block.get_double(cosmo, "k_pivot", default=0.05),
        "sBBN file": config["sBBN file"],
    }

    if config["k_output_values"]:
        params["k_output_values"] = config["k_output_values"]

    if block.has_value(cosmo, "100*theta_s"):
        params["100*theta_s"] = block[cosmo, "100*theta_s"]
    if block.has_value(cosmo, "h0"):
        params["H0"] = 100 * block[cosmo, "h0"]
    if block.has_value(cosmo, "A_s"):
        params["A_s"] = block[cosmo, "A_s"]
    if block.has_value(cosmo, "logA"):
        params["ln10^{10}A_s"] = block[cosmo, "logA"]

    if block.has_value(horndeski, "omega_fld"):
        # omega_fld conflicts with cosmosis' default Omega_Lambda handling.
        raise ValueError("omega_fld is not supported by this interface; "
                         "specify omega_smg instead.")

    # --- Horndeski / modified-gravity setup ------------------------------- #
    if not block.has_value(horndeski, "omega_smg") or block[horndeski, "omega_smg"] == 0.0:
        # standard CLASS (GR) case: no smg parameters
        pass
    else:
        params["gravity_model"] = config["gravity_model"]
        params["expansion_model"] = config["expansion_model"]
        params["parameters_smg"] = block.get_string(
            horndeski, "parameters_smg", default=smg_params(block))
        params["kineticity_safe_smg"] = block.get_double(
            horndeski, "kineticity_safe_smg", default=0.0)

        omega_smg = block[horndeski, "omega_smg"]
        if omega_smg < 0.0:
            # scalar-field branch: Omega_smg inferred from the closure equation.
            if config["expansion_model"] == "lcdm":
                params["expansion_smg"] = 0.5
            elif config["expansion_model"] == "wowa":
                params["expansion_smg"] = "{0}, {1}, {2}".format(
                    0.5, block.get_double(cosmo, "w"), block.get_double(cosmo, "wa"))
            params["Omega_Lambda"] = block.get_double(horndeski, "omega_lambda_smg", default=0.0)
            params["Omega_fld"] = 0.0
            params["Omega_smg"] = block.get_int(horndeski, "omega_smg", default=-1)
        elif 0.0 < omega_smg < 1.0:
            if config["expansion_model"] == "lcdm":
                params["expansion_smg"] = omega_smg
            params["Omega_smg"] = omega_smg

    # --- massive neutrinos ----------------------------------------------- #
    if block.has_value(cosmo, "N_ur") and block[cosmo, "N_ur"] != 3.046:
        params["N_ncdm"] = block[cosmo, "N_ncdm"]
        if block[cosmo, "N_ncdm"] == 1:
            if block.has_value(cosmo, "m_ncdm"):
                params["m_ncdm"] = block[cosmo, "m_ncdm"]
            if block.has_value(cosmo, "omega_ncdm"):
                params["omega_ncdm"] = block[cosmo, "omega_ncdm"]
            params["T_ncdm"] = block[cosmo, "T_ncdm"]
        elif block[cosmo, "N_ncdm"] > 1:
            m_nu, o_nu, T_nu = [], [], []
            for i in range(1, 4):
                if block.has_value(cosmo, "m_ncdm__%i" % i):
                    m_nu.append(block[cosmo, "m_ncdm__%i" % i])
                    T_nu.append(block[cosmo, "T_ncdm__%i" % i])
                if block.has_value(cosmo, "omega_ncdm__%i" % i):
                    o_nu.append(block[cosmo, "omega_ncdm__%i" % i])
                    T_nu.append(block[cosmo, "T_ncdm__%i" % i])
            if m_nu:
                params["m_ncdm"] = ",".join(map(str, m_nu))  # MODERNISED (py3 print removed)
            if o_nu:
                params["omega_ncdm"] = ",".join(map(str, o_nu))
            params["T_ncdm"] = ",".join(map(str, T_nu))

    return params


def get_background_ascending(c):
    # This classy build lacks the Omega_smg()/alpha_m_at_z()/sigma8_at_z()/
    # growthrate_at_z()/linear_growth_factor() methods, so read the equivalent
    # quantities from get_background(): M2_running_smg = alpha_M,
    # 'gr.fac. D'/'gr.fac. f' = D(z)/f(z), (.)rho_smg/(.)rho_crit at z=0 = Omega_smg.
    # The table is returned with z descending; sort ascending for np.interp.
    bg = c.get_background()
    order = np.argsort(bg["z"])
    return {key: val[order] for key, val in bg.items()}


def get_class_outputs(block, c, config):
    # Derived parameters
    block[cosmo, "sigma_8"] = c.sigma8()
    h0 = block[cosmo, "h0"]
    block[cosmo, "omega_m"] = c.Omega_m()

    bg = get_background_ascending(c)  # also used by the growth block below
    is_mg = block.has_value(horndeski, "omega_smg") and block[horndeski, "omega_smg"] != 0.0
    if is_mg:
        omega_smg = bg["(.)rho_smg"][0] / bg["(.)rho_crit"][0]  # at z=0
        block[cosmo, "omega_lambda_smg"] = omega_smg

    # --- matter power spectrum (linear) ---------------------------------- #
    dz = 0.01
    kmin, nk = 1e-5, 200
    kmax = config["kmax"] * h0
    z = np.arange(0.0, config["zmax"] + dz, dz)
    k = np.logspace(np.log10(kmin), np.log10(kmax), nk)
    nz = len(z)

    P = np.zeros((nk, nz))
    for i, ki in enumerate(k):
        for j, zj in enumerate(z):
            P[i, j] = c.pk_lin(ki, zj)
    block.put_grid("matter_power_lin", "k_h", k / h0, "z", z, "p_k", P * h0 ** 3)

    # --- distances -------------------------------------------------------- #
    block[distances, "z"] = z
    block[distances, "nz"] = nz
    d_l = np.array([c.luminosity_distance(zi) for zi in z])
    d_a = np.array([c.angular_distance(zi) for zi in z])
    block[distances, "d_l"] = d_l
    block[distances, "d_a"] = d_a
    block[distances, "d_m"] = d_a * (1 + z)
    # classy Hubble(z) returns H(z)/c in 1/Mpc -> dgw.py multiplies by c(km/s).
    block[distances, "H"] = np.array([c.Hubble(zi) for zi in z])
    block[distances, "mu"] = 5.0 * np.log10(d_l + 1e-100) + 25.0

    # --- GW luminosity distance ------------------------------------------ #
    if not is_mg:
        # GR: GW and EM distances coincide.
        block[distances, "alpha_mz"] = np.zeros_like(d_l)
        block[distances, "d_l_gw_on_d_l_em"] = np.ones_like(d_l)
        block[distances, "d_l_gw"] = d_l
    else:
        # d_L^GW/d_L^EM = exp(0.5 * int_0^z alpha_M(z')/(1+z') dz')
        alpha_mz = np.interp(z, bg["z"], bg["M2_running_smg"])
        # Standard relation uses alpha_M directly. The alpha_m_over_omega_smg
        # option divides by Omega_smg(0) to reproduce the older thesis code.
        if config["alpha_m_over_omega_smg"]:
            alpha_mz = alpha_mz / omega_smg
        ratio = np.exp(0.5 * _cumtrapz(alpha_mz / (1.0 + z), z, initial=0))
        block[distances, "alpha_mz"] = alpha_mz
        block[distances, "d_l_gw_on_d_l_em"] = ratio
        block[distances, "d_l_gw"] = d_l * ratio

    block[distances, "age"] = c.age()
    block[distances, "rs_zdrag"] = c.rs_drag()
    block[distances, "a"] = 1.0 / (1.0 + z)

    # --- growth ----------------------------------------------------------- #
    d_z = np.interp(z, bg["z"], bg["gr.fac. D"])  # D(0) == 1, so sigma8(z) = sigma8() * D(z)
    grr = np.interp(z, bg["z"], bg["gr.fac. f"])
    s8 = c.sigma8() * d_z
    block[growthparams, "z"] = z
    block[growthparams, "s8_z"] = s8
    block[growthparams, "grr_z"] = grr
    block[growthparams, "f_z"] = grr
    block[growthparams, "fsigma_8"] = grr * s8       # read by BOSS/6dFGS
    block[growthparams, "sigma_8"] = s8
    block[growthparams, "D_z"] = d_z

    # --- CMB C_ell -------------------------------------------------------- #
    c_ell = c.lensed_cl() if config["lensing"] == "yes" else c.raw_cl()
    ell = c_ell["ell"][2:]
    block[cmb_cl, "ell"] = ell
    tcmb_muk = block.get_double(cosmo, "t_cmb", default=2.726) * 1e6
    f = ell * (ell + 1.0) / (2 * np.pi) * tcmb_muk ** 2
    f1 = ell * (ell + 1.0) / (2 * np.pi)
    for s in ("tt", "ee", "te", "bb", "tp"):
        block[cmb_cl, s] = c_ell[s][2:] * f
    block[cmb_cl, "pp"] = c_ell["pp"][2:] * f1


def execute(block, config):
    c = config["cosmo"]
    try:
        c.set(get_class_inputs(block, config))
        c.compute()
        get_class_outputs(block, c, config)
    except classy.CosmoError as error:
        if config["debug"]:
            sys.stderr.write("Error in hi_class (debug=T):\n")
            traceback.print_exc(file=sys.stderr)
        else:
            sys.stderr.write("Error in hi_class. Set debug=T for info: {}\n".format(error))
        return 1
    finally:
        c.struct_cleanup()
    return 0


def cleanup(config):
    config["cosmo"].empty()
