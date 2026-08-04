"""Generate the hi_class TT derivative spectra for the emulator Fisher-bias test.

Runs hi_class (gw-hiclass env) at the Planck 2018 best-fit cosmology and at
best-fit +/- one step in each of the 6 standard parameters, alpha=0, with the
same settings the pipeline and the LCDM training set use (N_ur=3.046 massless,
propto_omega). The output feeds gwmg.emulator.chi2.parameter_bias, which turns
these into the emulator-induced parameter bias in units of the Planck-TT sigma.

    python fisher_deriv_tt.py --out fisher_deriv.npz
"""
import argparse
import os

import numpy as np
import classy

LMAX = 2600
OMNUH2 = 0.00083
SBBN = os.environ.get("HICLASS_DIR",
        "/Users/abhishekkarkola/MSc_Thesis/hi_class_public") + "/external/bbn/sBBN.dat"

# Planck 2018 best-fit in the pipeline's parameter convention (Omega_m, Omega_b
# as fractions). Steps are ~1 sigma, only used for the finite-difference slope.
FID = dict(omega_m=0.31560, h0=0.6736, omega_b=0.049302, n_s=0.9649, A_s=2.099e-9, tau=0.0544)
STEP = dict(omega_m=0.003, h0=0.005, omega_b=0.00033, n_s=0.0042, A_s=0.03e-9, tau=0.0073)
KEYS = list(FID)


def _inputs(p):
    h2 = p["h0"] ** 2
    return {"output": "tCl,pCl,lCl", "lensing": "yes", "modes": "s",
            "l_max_scalars": LMAX, "T_cmb": 2.726, "N_ur": 3.046, "sBBN file": SBBN,
            "gravity_model": "propto_omega", "expansion_model": "lcdm",
            "parameters_smg": "1.0, 0., 0., 0., 1.",
            "kineticity_safe_smg": 0.0, "expansion_smg": 0.5,
            "Omega_Lambda": 0.0, "Omega_fld": 0.0, "Omega_smg": -1,
            "H0": 100 * p["h0"], "omega_b": p["omega_b"] * h2,
            "omega_cdm": p["omega_m"] * h2 - p["omega_b"] * h2 - OMNUH2,
            "n_s": p["n_s"], "A_s": p["A_s"], "tau_reio": p["tau"]}


def _run(p):
    c = classy.Class(); c.set(_inputs(p)); c.compute()
    tt = c.lensed_cl(LMAX)["tt"][2:].copy(); c.struct_cleanup(); c.empty()
    return tt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")   # hi_class is not thread-safe

    spec = {"fid": _run(FID)}
    print("fid done", flush=True)
    for k in KEYS:
        for sgn, tag in ((+1, "p"), (-1, "m")):
            d = dict(FID); d[k] = FID[k] + sgn * STEP[k]
            spec["%s_%s" % (k, tag)] = _run(d)
            print("%s_%s done" % (k, tag), flush=True)

    np.savez(a.out, ell=np.arange(2, LMAX + 1).astype(float),
             fid_params=np.array([FID[k] for k in KEYS]),
             steps=np.array([STEP[k] for k in KEYS]), keys=np.array(KEYS), **spec)
    print("saved ->", a.out)


if __name__ == "__main__":
    main()
