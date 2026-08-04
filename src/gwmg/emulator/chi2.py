"""Likelihood-level validation: does the emulated CMB give the same Planck chi2?

Per-mode cosmic-variance accuracy (validate.py) is a proxy. What the inference
actually depends on is the Planck TT chi2, summed over all bandpowers. This
module recomputes the Plik-lite TT chi2 from the emulated C_ell and from the
exact (test-set) C_ell and reports the shift, Dchi2 = chi2_emu - chi2_exact.

If |Dchi2| stays well under ~1 across the box -- and especially for the
best-fitting samples, where the posterior actually lives -- the emulator does
not bias the inference and the bespoke CMB emulator is good enough. If the tail
biases chi2, that is the signal to switch the CMB to a pre-trained LCDM emulator.

The Plik-lite binning is reimplemented here in pure numpy (validated to machine
precision against cosmosis' planck_lite_py golden reference, loglike =
-104.59579619576277 for 2015 TT + low-ell) so this runs in the emulator env
without astropy or cosmosis on the path.
"""
import os

import numpy as np

from .train import PARAM_NAMES, _sanity_mask

# C_ell (dimensionless, from classy lensed_cl) -> D_ell [muK^2], matching
# hi_class_interface.py: f = ell(ell+1)/(2 pi) * (T_cmb * 1e6)^2, T_cmb=2.726 K.
_TCMB_MUK = 2.726e6

_COLS = ["spectrum", "index", "band_ell_min", "band_ell_max",
         "weight_row_min", "weight_row_max", "band_ell_nominal",
         "bandpower", "bandpower_sigma"]


def _read_table(path):
    d = np.genfromtxt(path, dtype=None, encoding=None, names=_COLS)
    spec = np.array([s.decode() if isinstance(s, bytes) else str(s)
                     for s in np.atleast_1d(d["spectrum"])])
    return np.atleast_1d(d), spec


def default_planck_data_dir():
    """The Plik-lite data dir shipped with the cosmosis-standard-library."""
    root = os.environ.get("COSMOSIS_SRC_DIR", "")
    return os.path.join(root, "cosmosis-standard-library", "likelihood",
                        "planck_py", "data")


class PlanckLiteTT:
    """Plik-lite TT Gaussian likelihood (numpy-only reimplementation).

    Faithful to cosmosis' planck_lite_py: TT bandpowers only, optional low-ell
    bins prepended, chi2 = (data - mu) . fisher . (data - mu).
    """

    def __init__(self, data_directory=None, year=2015, use_low_ell=True):
        import scipy.linalg
        data_directory = data_directory or default_planck_data_dir()
        version = "18" if str(year) == "2015" else "22"
        base = os.path.join(data_directory, str(year),
                            "planck_lite_%s_v%s" % (year, version))
        d, spec = _read_table(base + ".dat")
        mask = spec == "TT"
        self.hi = d[mask]
        self.w_hi = np.loadtxt(base + "_weights.dat")
        cov = np.load(base + "_cov.npz")["cov"][mask][:, mask]
        self.use_low_ell = use_low_ell
        if use_low_ell:
            lbase = os.path.join(data_directory, "%s_low_ell" % year,
                                 "planck_lite_%s_v%s" % (year, version))
            dl, _ = _read_table(lbase + ".dat")   # low-ell table is all TT
            self.lo = dl
            self.w_lo = np.loadtxt(lbase + "_weights.dat")
            cov_lo = np.load(lbase + "_cov.npz")["cov"]
            cov = scipy.linalg.block_diag(cov_lo, cov)
            self.data_vector = np.concatenate([self.lo["bandpower"], self.hi["bandpower"]])
        else:
            self.data_vector = np.array(self.hi["bandpower"])
        self.fisher = scipy.linalg.cho_solve(
            scipy.linalg.cho_factor(cov), np.identity(len(cov))).transpose()

    @staticmethod
    def _bin_batch(tab, weight, cltt, ellmin):
        """Bin per-ell C_ell (nsamp, nmode) into bandpowers (nsamp, nbin)."""
        mu = np.empty((cltt.shape[0], len(tab)))
        for i, row in enumerate(tab):
            b1 = int(row["band_ell_min"]) - ellmin
            b2 = int(row["band_ell_max"]) - ellmin
            w1, w2 = int(row["weight_row_min"]), int(row["weight_row_max"])
            mu[:, i] = cltt[:, b1:b2] @ weight[w1:w2]
        return mu

    def mean(self, Dltt, ellmin=2):
        """Binned theory vector mu (nsamp, ndata) for D_ell^TT [muK^2]."""
        Dltt = np.atleast_2d(Dltt)
        ls = np.arange(Dltt.shape[1]) + ellmin
        cltt = Dltt / (ls * (ls + 1.0) / (2 * np.pi))
        mu = self._bin_batch(self.hi, self.w_hi, cltt, ellmin)
        if self.use_low_ell:
            mu = np.concatenate([self._bin_batch(self.lo, self.w_lo, cltt, ellmin), mu], axis=1)
        return mu

    def chi2(self, Dltt, ellmin=2):
        """chi2 for a batch of theory D_ell^TT, shape (nsamp, nmode) [muK^2]."""
        d = self.data_vector[None, :] - self.mean(Dltt, ellmin)
        return np.einsum("ij,jk,ik->i", d, self.fisher, d)


def _cl_to_dl(cl, ell):
    """Dimensionless C_ell -> D_ell [muK^2]."""
    return cl * (ell * (ell + 1.0) / (2 * np.pi)) * _TCMB_MUK ** 2


def _summ(x):
    x = np.asarray(x)
    return {"median": float(np.median(x)), "p68": float(np.percentile(x, 68)),
            "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)),
            "max": float(np.max(x))}


def chi2_validation(model_dir, test_dir, planck_data_dir=None, year=2015,
                    use_low_ell=True, best_frac=0.1, verbose=True):
    """Compare Planck TT chi2 from emulated vs exact C_ell over the test set.

    Returns a dict with the Dchi2 = chi2_emu - chi2_exact distribution, both
    over all samples and over the best-fitting ``best_frac`` (lowest exact chi2,
    i.e. the region the posterior actually samples), plus the best-fit sample.
    """
    from cosmopower import cosmopower_NN

    ell = dict(np.load(os.path.join(test_dir, "grids.npz")))["ell"].astype(float)
    ellmin = int(ell[0])
    raw = np.load(os.path.join(test_dir, "features_cl_tt.npy"))     # dimensionless C_ell
    ok = _sanity_mask(raw, 1e-6)
    true_cl = raw[ok]
    P = np.load(os.path.join(test_dir, "parameters.npz"))
    params = {k: P[k][ok] for k in P.files}

    model = os.path.join(model_dir, "emu_cl_tt")
    cp = cosmopower_NN(restore=True, restore_filename=model)
    emu_cl = 10.0 ** cp.predictions_np(params)                     # log10 -> C_ell

    lite = PlanckLiteTT(planck_data_dir, year=year, use_low_ell=use_low_ell)
    dl_true = _cl_to_dl(true_cl, ell)
    dl_emu = _cl_to_dl(emu_cl, ell)
    chi2_true = lite.chi2(dl_true, ellmin=ellmin)
    chi2_emu = lite.chi2(dl_emu, ellmin=ellmin)
    dchi2 = chi2_emu - chi2_true
    adchi2 = np.abs(dchi2)

    order = np.argsort(chi2_true)
    nbest = max(1, int(best_frac * len(chi2_true)))
    best = order[:nbest]
    imin = int(order[0])

    report = {
        "n": int(len(chi2_true)),
        "chi2_true_min": float(chi2_true[imin]),
        "chi2_true_median": float(np.median(chi2_true)),
        "abs_dchi2_all": _summ(adchi2),
        "abs_dchi2_best": _summ(adchi2[best]),
        "best_frac": best_frac,
        "best_fit_dchi2": float(dchi2[imin]),
        "best_fit_params": {k: float(v[imin]) for k, v in params.items()},
    }

    if verbose:
        a, b = report["abs_dchi2_all"], report["abs_dchi2_best"]
        print("n=%d   exact chi2: min %.1f  median %.1f  (ndata=%d)"
              % (report["n"], report["chi2_true_min"], report["chi2_true_median"],
                 len(lite.data_vector)))
        print("|Dchi2| all         : med %.3f  p95 %.3f  p99 %.3f  max %.3f"
              % (a["median"], a["p95"], a["p99"], a["max"]))
        print("|Dchi2| best %2d%%    : med %.3f  p95 %.3f  p99 %.3f  max %.3f"
              % (int(best_frac * 100), b["median"], b["p95"], b["p99"], b["max"]))
        print("best-fit sample Dchi2: %+.3f  (exact chi2 = %.2f)"
              % (report["best_fit_dchi2"], report["chi2_true_min"]))
    return report


def write_chi2_report(report, path):
    a, b = report["abs_dchi2_all"], report["abs_dchi2_best"]
    lines = [
        "# gwmg emulator likelihood-level validation (Planck TT chi2)",
        "# Dchi2 = chi2(emulated C_ell) - chi2(exact C_ell)",
        "# good enough if |Dchi2| << 1 for the best-fitting samples",
        "n_test_samples        %d" % report["n"],
        "exact_chi2_min        %.4f" % report["chi2_true_min"],
        "exact_chi2_median     %.4f" % report["chi2_true_median"],
        "",
        "# region                median      p95        p99        max",
        "abs_dchi2_all         %-11.4f %-10.4f %-10.4f %-10.4f"
        % (a["median"], a["p95"], a["p99"], a["max"]),
        "abs_dchi2_best_%02d%%     %-11.4f %-10.4f %-10.4f %-10.4f"
        % (int(report["best_frac"] * 100), b["median"], b["p95"], b["p99"], b["max"]),
        "",
        "best_fit_dchi2        %+.4f" % report["best_fit_dchi2"],
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def parameter_bias(lite, deriv, emu_cl, ell_emu):
    """Emulator-induced parameter bias via the Fisher formalism.

    This is the decisive test for a CMB emulator: per-multipole accuracy is a
    proxy, and even the summed chi2 mixes a benign constant offset with the part
    that actually shifts parameters. Only the parameter-dependent part biases the
    posterior. Working at the Planck best-fit, where the data residual is minimal,
    the induced shift is Delta_theta = -Cov . B^T . F . delta, with delta the
    emulator error, B the binned theory derivatives, F the Planck Fisher matrix,
    and Cov = (B^T F B)^-1 the parameter covariance.

    ``deriv`` is an npz-like mapping (from ``scripts/fisher_deriv_tt.py``) with:
    ``ell`` (theory multipoles), ``keys`` (parameter names), ``steps`` (finite-
    difference step per parameter), ``fid`` (best-fit C_ell), and ``<key>_p`` /
    ``<key>_m`` (C_ell at +/- one step). ``emu_dl``/``ell_emu`` is the emulator's
    C_ell at the best-fit and its multipole grid. Returns bias in units of the
    Planck-TT sigma per parameter, plus the max over parameters.
    """
    keys = [str(k) for k in deriv["keys"]]
    steps = np.asarray(deriv["steps"], float)
    ell = np.asarray(deriv["ell"], float)

    def mu(cl, l):
        return lite.mean(_cl_to_dl(cl, l))[0]

    mu_fid = mu(deriv["fid"], ell)
    B = np.column_stack([
        (mu(deriv["%s_p" % k], ell) - mu(deriv["%s_m" % k], ell)) / (2 * steps[i])
        for i, k in enumerate(keys)])
    F = lite.fisher
    cov = np.linalg.inv(B.T @ F @ B)
    sigma = np.sqrt(np.diag(cov))
    delta = mu(emu_cl, ell_emu) - mu_fid          # emulator error, binned
    bias = -cov @ (B.T @ F @ delta)
    bos = bias / sigma
    return {
        "keys": keys,
        "sigma": sigma,
        "bias": bias,
        "bias_over_sigma": bos,
        "dchi2_fixed": float(delta @ F @ delta),
        "max_abs_bias_over_sigma": float(np.max(np.abs(bos))),
    }
