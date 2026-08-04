"""Validate trained CosmoPower emulators against exact hi_class.

Loads emulators (from ``gwmg emu-train``) and a held-out test set (from
``gwmg emu-gen`` with a different --seed), predicts each spectrum, and reports
accuracy in two forms:

  * physical fractional error, |C_emu / C_true - 1|, on the actual spectrum
    (not the log-transformed feature) so the numbers reflect real accuracy;
  * for the CMB spectra, the error in units of cosmic variance,
    |C_emu - C_true| / sigma_cv. This is well-defined even where TE crosses zero
    and is the physically meaningful "is it good enough" measure: values well
    below 1 mean the emulator error is smaller than the irreducible cosmic
    variance.

The predictions need cosmopower; the error statistics are plain numpy.
"""
import os

import numpy as np

from .train import PARAM_NAMES, SPECTRA, _sanity_mask

# spectrum name -> CMB type for the cosmic-variance calculation
CMB = {"cl_tt": "tt", "cl_ee": "ee", "cl_te": "te"}


def fractional_errors(pred, true, eps=1e-30):
    """|pred - true| / |true| element-wise (guards tiny/zero true values)."""
    return np.abs(pred - true) / np.maximum(np.abs(true), eps)


def summarise_errors(x):
    """Percentiles of a flattened error array."""
    f = np.asarray(x).ravel()
    return {
        "median": float(np.median(f)),
        "p68": float(np.percentile(f, 68)),
        "p95": float(np.percentile(f, 95)),
        "p99": float(np.percentile(f, 99)),
        "max": float(np.max(f)),
    }


def cosmic_variance_sigma(ell, c_tt, c_ee, c_te, which):
    """Idealised full-sky cosmic-variance sigma on the estimated C_ell.

    ell has shape (nmode,); c_* have shape (nsamp, nmode). Returns (nsamp, nmode).
    For TE, sigma^2 = (C_TT C_EE + C_TE^2) / (2l+1), which never vanishes, so the
    TE accuracy stays well-defined at TE's zero-crossings.
    """
    twolp1 = (2.0 * ell + 1.0)[None, :]
    if which == "tt":
        return np.sqrt(2.0 / twolp1) * np.abs(c_tt)
    if which == "ee":
        return np.sqrt(2.0 / twolp1) * np.abs(c_ee)
    if which == "te":
        # clip to >=0: tiny negative C_TT/C_EE from numerical noise at high ell
        # can otherwise make the variance argument negative -> sqrt gives nan.
        var = (np.clip(c_tt, 0, None) * np.clip(c_ee, 0, None) + c_te ** 2) / twolp1
        return np.sqrt(np.maximum(var, 0.0))
    raise ValueError("unknown spectrum: %s" % which)


def _spectra_by_name(spectra):
    d = {}
    for fname, gk, log10, cap, n_pcas in spectra:
        d[os.path.splitext(fname)[0].replace("features_", "")] = (fname, gk, log10, cap, n_pcas)
    return d


def _restore(model_path, n_pcas):
    """Restore the right emulator class (NN or PCA+NN, per the 5th SPECTRA field)."""
    if n_pcas:
        from cosmopower import cosmopower_PCAplusNN
        return cosmopower_PCAplusNN(restore=True, restore_filename=model_path)
    from cosmopower import cosmopower_NN
    return cosmopower_NN(restore=True, restore_filename=model_path)


def _predict_physical(cp, params, log10):
    """Emulator prediction in physical (spectrum) space."""
    pred = cp.predictions_np(params)
    return 10.0 ** pred if log10 else pred


def validate_emulators(model_dir, test_dir, spectra=SPECTRA, verbose=True):
    """Compare each trained emulator against the exact test set.

    Returns {name: {"phys": summary, "cv": summary}} where "cv" is present only
    for the CMB spectra.
    """
    P = np.load(os.path.join(test_dir, "parameters.npz"))
    params_all = {k: P[k] for k in P.files}
    grids = dict(np.load(os.path.join(test_dir, "grids.npz")))
    byname = _spectra_by_name(spectra)

    def _model(name):
        return os.path.join(model_dir, "emu_" + name)

    def _available(name):
        return (name in byname
                and os.path.exists(os.path.join(test_dir, byname[name][0]))
                and os.path.exists(_model(name) + ".pkl"))

    report = {}

    # --- physical fractional error, per spectrum (each with its own mask) ---
    for name, (fname, _gk, log10, cap, _npcas) in byname.items():
        if not _available(name):
            if verbose:
                print("skip %s (missing test features or model)" % name)
            continue
        raw = np.load(os.path.join(test_dir, fname))
        ok = _sanity_mask(raw, cap)
        true_phys = raw[ok]
        p = {k: v[ok] for k, v in params_all.items()}
        cp = _restore(_model(name), byname[name][4])
        pred_phys = _predict_physical(cp, p, log10)
        report[name] = {"phys": summarise_errors(fractional_errors(pred_phys, true_phys))}

    # --- cosmic-variance-normalised error for whichever CMB spectra exist ---
    # TT needs only C_TT, EE only C_EE, but the TE variance needs all three.
    present = [n for n in CMB if _available(n)]
    cv_needs = {"tt": {"cl_tt"}, "ee": {"cl_ee"}, "te": {"cl_tt", "cl_ee", "cl_te"}}
    if present:
        raws = {n: np.load(os.path.join(test_dir, byname[n][0])) for n in present}
        common = None
        for n in present:
            m = _sanity_mask(raws[n], byname[n][3])
            common = m if common is None else (common & m)
        ell = grids["ell"]
        p = {k: v[common] for k, v in params_all.items()}
        true = {n: raws[n][common] for n in present}
        pred = {n: _predict_physical(_restore(_model(n), byname[n][4]), p, byname[n][2])
                for n in present}
        for n in present:
            which = CMB[n]
            if not cv_needs[which].issubset(present):
                continue  # e.g. TE without TT/EE available
            sig = cosmic_variance_sigma(ell, true.get("cl_tt"), true.get("cl_ee"),
                                        true.get("cl_te"), which)
            cv = np.abs(pred[n] - true[n]) / np.maximum(sig, 1e-300)
            report[n]["cv"] = summarise_errors(cv)

    if verbose:
        for name, r in report.items():
            line = "%-8s  phys%%: med %6.2f  p95 %7.2f  p99 %8.2f" % (
                name, 100 * r["phys"]["median"], 100 * r["phys"]["p95"],
                100 * r["phys"]["p99"])
            if "cv" in r:
                line += "   |  CV: med %.3f  p95 %.3f  p99 %.3f" % (
                    r["cv"]["median"], r["cv"]["p95"], r["cv"]["p99"])
            print(line)
    return report


def write_report(report, path):
    """Write a human-readable accuracy report.

    Physical fractional error is shown as a percentage; the CMB cosmic-variance
    error is shown as a fraction of cosmic variance (values << 1 are good).
    """
    lines = [
        "# gwmg emulator validation",
        "# phys = physical fractional error on the spectrum (%)",
        "# cv   = error in units of cosmic variance (CMB only; <1 is good)",
        "# spectrum   phys_median  phys_p95   phys_p99   cv_median  cv_p95    cv_p99",
    ]
    for name, r in report.items():
        p = r["phys"]
        cv = r.get("cv")
        cvcols = ("%-10.4f %-9.4f %-9.4f" % (cv["median"], cv["p95"], cv["p99"])
                  if cv else "%-10s %-9s %-9s" % ("-", "-", "-"))
        lines.append("%-11s %-12.4f %-10.4f %-10.4f %s" % (
            name, 100 * p["median"], 100 * p["p95"], 100 * p["p99"], cvcols))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
