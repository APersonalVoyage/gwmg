"""CosmoSIS module: GW standard-siren likelihood.

The numerical core is in gwmg.likelihood.gw_log_likelihood so it can be tested
without CosmoSIS. Requires gwmg to be installed so this file (loaded by CosmoSIS
via its path) can import it.
"""
import os
import sys

import numpy as np
from cosmosis.datablock import names, option_section

from gwmg.likelihood import gw_log_likelihood, load_events

likes = names.likelihoods
distances = names.distances
run_dir = os.path.split(__file__)[0]


def setup(options):
    datadir = options.get_string(option_section, "dirname", default=run_dir)
    datafile = options.get_string(option_section, "filename")
    return load_events(os.path.join(datadir, datafile))


def execute(block, config):
    d_gw_obs, z_obs, sigma_dgw, sigma_z, v_rms = config
    log_like = gw_log_likelihood(
        z_model=block[distances, "z"],
        dl_model=block[distances, "d_l"],
        h_model=block[distances, "h"],
        dgw_ratio_model=block[distances, "d_l_gw_on_d_l_em"],
        d_gw_obs=d_gw_obs, z_obs=z_obs,
        sigma_dgw=sigma_dgw, sigma_z=sigma_z, v_rms=v_rms,
    )
    if not np.isfinite(log_like):
        sys.stderr.write("Non-finite LogLike in d_gw_like\n")
    block[likes, "DGW_LIKE"] = float(log_like)
    return 0
