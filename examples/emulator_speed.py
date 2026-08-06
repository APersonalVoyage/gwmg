"""Compare the emulator against exact hi_class: accuracy and speed.

Uses the emulators shipped in pretrained/ by default, so it runs straight away.
hi_class is optional: with it you get the head-to-head comparison, without it you
still see the emulator's predictions and timing.

    python examples/emulator_speed.py
"""
import argparse
import time

import numpy as np

from gwmg.emulator import Emulator

FIDUCIAL = dict(omega_m=0.315, h0=0.674, omega_b=0.049, n_s=0.965,
                A_s=2.1e-9, tau=0.054, alpha_B0=1.0, alpha_M0=0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None,
                    help="directory holding emu_cl_tt, emu_logpk, emu_fsigma8 "
                         "(default: the models shipped in pretrained/)")
    a = ap.parse_args()

    emu = Emulator(a.model_dir)
    out = emu.predict(**FIDUCIAL)                    # warm up (first call builds graphs)
    t = time.time()
    for _ in range(20):
        out = emu.predict(**FIDUCIAL)
    ms = (time.time() - t) / 20 * 1e3

    ell = out.ell
    print("emulator: %.1f ms per evaluation" % ms)
    print("  D_ell(220)   = %.0f uK^2" % out.dl_tt[np.argmin(abs(ell - 220))])
    print("  sigma_8      = %.4f" % out.sigma8)
    print("  f-sigma_8(0) = %.4f" % out.fsigma8[0])
    print("  d_GW/d_EM at z=1 = %.4f" % np.interp(1.0, out.z_bg, out.dgw_ratio))

    # How the modified-gravity parameters move the observables.
    print("\nvarying alpha_M0 (alpha_B0 = 0):")
    print("   alpha_M0   f-sigma_8(0)   d_GW/d_EM(z=1)")
    for aM in (0.0, 1.0, 3.0):
        o = emu.predict(**dict(FIDUCIAL, alpha_B0=0.0, alpha_M0=aM))
        print("   %6.1f     %.4f         %.4f"
              % (aM, o.fsigma8[0], np.interp(1.0, o.z_bg, o.dgw_ratio)))

    try:
        import classy
    except ImportError:
        print("\n(hi_class not importable; skipping the exact comparison)")
        return

    print("\nrunning exact hi_class for comparison ...")
    import os
    sbbn = os.path.join(os.environ.get("HICLASS_DIR", ""), "external/bbn/sBBN.dat")
    if not os.path.exists(sbbn):
        print("HICLASS_DIR not set correctly; skipping.")
        return
    h2 = FIDUCIAL["h0"] ** 2
    c = classy.Class()
    c.set({"output": "tCl,pCl,lCl,mPk", "lensing": "yes", "l_max_scalars": 2600,
           "P_k_max_h/Mpc": 1.0, "T_cmb": 2.726, "N_ur": 3.046, "sBBN file": sbbn,
           "gravity_model": "propto_omega", "expansion_model": "lcdm",
           "parameters_smg": "1.0, %g, %g, 0., 1." % (FIDUCIAL["alpha_B0"], FIDUCIAL["alpha_M0"]),
           "kineticity_safe_smg": 0.0, "expansion_smg": 0.5,
           "Omega_Lambda": 0.0, "Omega_fld": 0.0, "Omega_smg": -1,
           "H0": 100 * FIDUCIAL["h0"], "omega_b": FIDUCIAL["omega_b"] * h2,
           "omega_cdm": FIDUCIAL["omega_m"] * h2 - FIDUCIAL["omega_b"] * h2 - 0.00083,
           "n_s": FIDUCIAL["n_s"], "A_s": FIDUCIAL["A_s"], "tau_reio": FIDUCIAL["tau"]})
    t = time.time(); c.compute(); secs = time.time() - t
    cl = c.lensed_cl(2600)["tt"][2:]
    l = np.arange(2, 2601)
    dl = cl * l * (l + 1) / (2 * np.pi) * (2.726e6) ** 2
    s8 = c.sigma8()
    c.struct_cleanup()

    i = np.argmin(abs(l - 220))
    print("  hi_class: %.2f s per evaluation  ->  emulator is %.0fx faster"
          % (secs, secs / (ms / 1e3)))
    print("  D_ell(220): emulator %.0f  exact %.0f  (%.2f%%)"
          % (out.dl_tt[np.argmin(abs(ell - 220))], dl[i],
             100 * abs(out.dl_tt[np.argmin(abs(ell - 220))] / dl[i] - 1)))
    print("  sigma_8   : emulator %.4f  exact %.4f  (%.2f%%)"
          % (out.sigma8, s8, 100 * abs(out.sigma8 / s8 - 1)))


if __name__ == "__main__":
    main()
