"""One-call prediction API for the trained emulators.

Give it the six standard cosmological parameters plus the two Horndeski
functions, and it returns the CMB temperature spectrum, the linear matter power
spectrum, the growth rate combination f-sigma_8(z), and the gravitational-wave
luminosity-distance ratio, in a few milliseconds instead of seconds of hi_class:

    from gwmg.emulator import Emulator

    emu = Emulator("emulators")
    out = emu.predict(omega_m=0.315, h0=0.674, omega_b=0.049, n_s=0.965,
                      A_s=2.1e-9, tau=0.054, alpha_B0=1.0, alpha_M0=0.5)
    out.ell, out.cl_tt        # CMB TT (dimensionless C_ell) and out.dl_tt in muK^2
    out.k_h, out.pk           # linear P(k) [(Mpc/h)^3] at z=0, and out.sigma8
    out.z, out.fsigma8        # growth
    out.z_bg, out.dgw_ratio   # d_L^GW / d_L^EM

The spectra come from the networks; the background and the GW distance ratio are
computed exactly (the LCDM expansion makes them analytic and cheap). Parameters
outside the training box raise ValueError unless ``check_box=False``.
"""
import os

import numpy as np

try:
    from scipy.integrate import cumulative_trapezoid as _cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz as _cumtrapz

PARAMS8 = ["omega_m", "h0", "omega_b", "n_s", "A_s", "tau", "alpha_B0", "alpha_M0"]
# Training box; must match the generation scripts.
BOX = {"omega_m": (0.24, 0.36), "h0": (0.61, 0.76), "omega_b": (0.041, 0.054),
       "n_s": (0.93, 1.00), "A_s": (1.7e-9, 2.5e-9), "tau": (0.02, 0.12),
       "alpha_B0": (-1.0, 3.0), "alpha_M0": (-1.0, 6.0)}
TCMB = 2.726
C_KMS = 299792.458


class Prediction(object):
    """Container for one emulator evaluation (attributes listed in Emulator.predict)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return ("Prediction(ell[%d], k_h[%d], z[%d], sigma8=%.4f)"
                % (len(self.ell), len(self.k_h), len(self.z), self.sigma8))


def _sigma8_from_pk(k_h, pk):
    """sigma_8 from P(k) [(Mpc/h)^3] on k_h [h/Mpc], top-hat R = 8 Mpc/h."""
    x = k_h * 8.0
    w = 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3
    return np.sqrt(np.trapz(pk * w ** 2 * k_h ** 2, k_h) / (2.0 * np.pi ** 2))


class Emulator(object):
    """The trained gwmg emulators behind a single ``predict`` call.

    ``model_dir`` may hold the three models directly (``emu_cl_tt``,
    ``emu_logpk``, ``emu_fsigma8``), or the individual paths can be given.
    """

    def __init__(self, model_dir=None, cmb=None, logpk=None, growth=None):
        from cosmopower import cosmopower_NN

        def _p(explicit, name):
            if explicit:
                return explicit
            if not model_dir:
                raise ValueError("give model_dir or an explicit path for %s" % name)
            return os.path.join(model_dir, name)

        self.cmb = cosmopower_NN(restore=True, restore_filename=_p(cmb, "emu_cl_tt"))
        self.logpk = cosmopower_NN(restore=True, restore_filename=_p(logpk, "emu_logpk"))
        self.growth = cosmopower_NN(restore=True, restore_filename=_p(growth, "emu_fsigma8"))
        self.cmb_params = [str(s) for s in self.cmb.parameters]
        self.ell = self.cmb.modes.astype(float)
        self.k_h = self.logpk.modes.astype(float)
        self.z = self.growth.modes.astype(float)

    def in_box(self, **p):
        """True if every parameter lies inside the training box."""
        return all(BOX[k][0] <= p[k] <= BOX[k][1] for k in BOX)

    def predict(self, zmax=3.0, dz=0.01, check_box=True, **p):
        """Emulate the observables for one cosmology.

        Keyword parameters: omega_m, h0, omega_b, n_s, A_s, tau, alpha_B0,
        alpha_M0. Returns a Prediction with ell, cl_tt, dl_tt, k_h, pk, sigma8,
        z, fsigma8, z_bg, dgw_ratio, alpha_mz, H_z, d_l.
        """
        missing = [k for k in PARAMS8 if k not in p]
        if missing:
            raise TypeError("missing parameters: %s" % ", ".join(missing))
        if check_box and not self.in_box(**p):
            bad = ["%s=%g not in [%g, %g]" % (k, p[k], BOX[k][0], BOX[k][1])
                   for k in BOX if not (BOX[k][0] <= p[k] <= BOX[k][1])]
            raise ValueError("outside the emulator's training box: " + "; ".join(bad))

        one = {k: [float(p[k])] for k in PARAMS8}
        cl_tt = (10.0 ** self.cmb.predictions_np({k: one[k] for k in self.cmb_params}))[0]
        pk = (10.0 ** self.logpk.predictions_np(one))[0]
        fs8 = self.growth.predictions_np(one)[0]

        # exact LCDM background + GW distance ratio
        h0, om = p["h0"], p["omega_m"]
        og = 2.469e-5 / h0 ** 2 * (TCMB / 2.725) ** 4
        orad = og * (1.0 + 0.2271 * 3.046)
        ol = 1.0 - om - orad
        zb = np.arange(0.0, zmax + dz, dz)
        E = np.sqrt(om * (1 + zb) ** 3 + orad * (1 + zb) ** 4 + ol)
        d_c = (C_KMS / (100.0 * h0)) * _cumtrapz(1.0 / E, zb, initial=0.0)
        # propto_omega: alpha_M(z) = alpha_M0 * Omega_DE(z) = alpha_M0 * Omega_L / E^2
        alpha_mz = p["alpha_M0"] * ol / E ** 2
        ratio = np.exp(0.5 * _cumtrapz(alpha_mz / (1 + zb), zb, initial=0.0))

        return Prediction(
            ell=self.ell, cl_tt=cl_tt,
            dl_tt=cl_tt * self.ell * (self.ell + 1) / (2 * np.pi) * (TCMB * 1e6) ** 2,
            k_h=self.k_h, pk=pk, sigma8=float(_sigma8_from_pk(self.k_h, pk)),
            z=self.z, fsigma8=fs8,
            z_bg=zb, dgw_ratio=ratio, alpha_mz=alpha_mz,
            H_z=100.0 * h0 * E, d_l=d_c * (1 + zb), params=dict(p),
        )
