"""Generate a CosmoPower-format training set from hi_class.

For each Latin-hypercube sample of the cosmological + Horndeski parameters, run
hi_class and store the expensive observables (CMB Cl, matter P(k)) plus the
cheap modified-gravity background signal (alpha_M(z), d_GW/d_EM). The output is
laid out exactly as CosmoPower's ``train`` expects:

    training_parameters : dict {param_name: array[n_ok]}
    training_features   : array[n_ok, n_features]     (one file per spectrum)

Run hi_class stability failures are expected across the wide MG prior; those
samples are skipped and counted, never crash the run.

Requires the hi_class ``classy`` build (same as the pipeline) and HICLASS_DIR
set (for external/bbn/sBBN.dat).
"""
import os

import numpy as np

# Sampling ranges: (min, max). Cover the pipeline priors so the emulator is
# valid across the whole posterior. Keys are the gwmg/thesis parameter names.
DEFAULT_RANGES = {
    "omega_m":  (0.1, 0.9),
    "h0":       (0.55, 0.91),
    "omega_b":  (0.03, 0.12),
    "n_s":      (0.87, 1.07),
    "A_s":      (0.5e-9, 5.0e-9),
    "tau":      (0.04, 0.125),
    "alpha_B0": (-1.0, 3.0),   # parameters_smg__2
    "alpha_M0": (-1.0, 6.0),   # parameters_smg__3
}
# "Smart box": tight, generous ranges around the well-measured standard
# parameters (they are pinned by the data, so the chain never leaves this
# region), while keeping the FULL range for the modified-gravity parameters we
# actually want to constrain. Training over this box instead of the full prior
# gives far better emulator accuracy for the same number of samples. The box is
# many sigma wide on the standard parameters, so it comfortably contains the
# posterior; when the emulator is used in the MCMC, proposals outside it should
# be rejected (the posterior is negligible there anyway).
SMART_RANGES = {
    "omega_m":  (0.24, 0.36),
    "h0":       (0.61, 0.76),
    "omega_b":  (0.041, 0.054),
    "n_s":      (0.93, 1.00),
    "A_s":      (1.7e-9, 2.5e-9),
    "tau":      (0.02, 0.12),
    "alpha_B0": (-1.0, 3.0),   # full range (this is what we constrain)
    "alpha_M0": (-1.0, 6.0),   # full range
}
OMNUH2 = 0.00083  # fixed, as in values_horndeski.ini


def latin_hypercube(ranges, n, seed=0):
    """Return dict {name: array[n]} of LHS samples over the given ranges."""
    from scipy.stats import qmc
    names = list(ranges)
    lo = np.array([ranges[k][0] for k in names])
    hi = np.array([ranges[k][1] for k in names])
    unit = qmc.LatinHypercube(d=len(names), seed=seed).random(n)
    scaled = qmc.scale(unit, lo, hi)
    return {k: scaled[:, i] for i, k in enumerate(names)}


def _class_inputs(p, lmax, kmax, zmax, sbbn):
    """Build a classy parameter dict from one sampled point p (dict of scalars)."""
    h2 = p["h0"] ** 2
    ombh2 = p["omega_b"] * h2
    omch2 = p["omega_m"] * h2 - ombh2 - OMNUH2
    return {
        "output": "tCl,pCl,lCl,mPk", "lensing": "yes", "modes": "s",
        "l_max_scalars": lmax, "P_k_max_h/Mpc": kmax, "z_max_pk": zmax,
        "H0": 100 * p["h0"], "omega_b": ombh2, "omega_cdm": omch2,
        "n_s": p["n_s"], "A_s": p["A_s"], "tau_reio": p["tau"],
        "T_cmb": 2.726, "N_ur": 3.046, "sBBN file": sbbn,
        "gravity_model": "propto_omega", "expansion_model": "lcdm",
        "parameters_smg": "1.0,%r,%r,0.,1." % (p["alpha_B0"], p["alpha_M0"]),
        "kineticity_safe_smg": 0.0, "expansion_smg": 0.5,
        "Omega_Lambda": 0.0, "Omega_fld": 0.0, "Omega_smg": -1,
    }


def generate_training_set(n_samples, outdir, lmax=2600, kmax=1.0, zmax=3.0,
                          seed=0, hiclass_dir=None, nk=200, verbose=True, label="",
                          ranges=None):
    """Sample the parameter space, run hi_class, and save a CosmoPower-format set.

    ``ranges`` is the sampling box (default DEFAULT_RANGES; pass SMART_RANGES for
    the tight-standard-params box). Writes into ``outdir``: parameters.npz,
    features_cl_{tt,te,ee}.npy, features_logpk.npy, features_mg.npz, grids.npz.
    Returns (n_ok, n_fail).
    """
    import classy

    hiclass_dir = hiclass_dir or os.environ.get("HICLASS_DIR")
    if not hiclass_dir:
        raise RuntimeError("Set HICLASS_DIR (holds external/bbn/sBBN.dat).")
    sbbn = os.path.join(hiclass_dir, "external", "bbn", "sBBN.dat")
    os.makedirs(outdir, exist_ok=True)

    samples = latin_hypercube(ranges or DEFAULT_RANGES, n_samples, seed=seed)
    names = list(samples)
    ell = np.arange(2, lmax + 1)
    kgrid = np.logspace(-4, np.log10(kmax), nk)          # h/Mpc
    zgrid = np.arange(0.0, zmax + 0.01, 0.01)

    kept = {k: [] for k in names}
    cl_tt, cl_te, cl_ee, logpk, alpha_mz, dgw_ratio = [], [], [], [], [], []
    n_ok = n_fail = 0

    cosmo = classy.Class()
    for i in range(n_samples):
        p = {k: float(samples[k][i]) for k in names}
        try:
            cosmo.set(_class_inputs(p, lmax, kmax, zmax, sbbn))
            cosmo.compute()
            cl = cosmo.lensed_cl(lmax)
            tt, te, ee = cl["tt"][2:], cl["te"][2:], cl["ee"][2:]
            h = p["h0"]
            pk = np.array([cosmo.pk_lin(kk * h, 0.0) * h ** 3 for kk in kgrid])
            bg = cosmo.get_background()
            order = np.argsort(bg["z"])
            zb, am = bg["z"][order], bg["M2_running_smg"][order]
            am_z = np.interp(zgrid, zb, am)
            from scipy.integrate import cumulative_trapezoid as ct
            ratio = np.exp(0.5 * ct(am_z / (1 + zgrid), zgrid, initial=0))
            # Some near-unstable models compute without raising but return
            # non-finite spectra; treat those as failures too.
            if not all(np.isfinite(a).all() for a in (tt, te, ee, pk)):
                raise ValueError("non-finite spectrum")
        except (classy.CosmoError, Exception):
            n_fail += 1
            cosmo.struct_cleanup()
            continue
        finally:
            cosmo.struct_cleanup()

        for k in names:
            kept[k].append(p[k])
        cl_tt.append(tt); cl_te.append(te); cl_ee.append(ee)
        logpk.append(np.log10(pk))
        alpha_mz.append(am_z); dgw_ratio.append(ratio)
        n_ok += 1
        if verbose and (i + 1) % max(1, n_samples // 20) == 0:
            print("%s%d/%d  (ok=%d fail=%d)" % (label, i + 1, n_samples, n_ok, n_fail),
                  flush=True)

    np.savez(os.path.join(outdir, "parameters.npz"),
             **{k: np.array(v) for k, v in kept.items()})
    np.save(os.path.join(outdir, "features_cl_tt.npy"), np.array(cl_tt))
    np.save(os.path.join(outdir, "features_cl_te.npy"), np.array(cl_te))
    np.save(os.path.join(outdir, "features_cl_ee.npy"), np.array(cl_ee))
    np.save(os.path.join(outdir, "features_logpk.npy"), np.array(logpk))
    np.savez(os.path.join(outdir, "features_mg.npz"),
             alpha_mz=np.array(alpha_mz), dgw_on_dem=np.array(dgw_ratio))
    np.savez(os.path.join(outdir, "grids.npz"), ell=ell, k_h=kgrid, z=zgrid)
    if verbose:
        print("Saved %d samples (%d hi_class failures) to %s" % (n_ok, n_fail, outdir))
    return n_ok, n_fail


# --------------------------------------------------------------------------- #
# Parallel generation (embarrassingly parallel: one LHS per worker, then merge)
# --------------------------------------------------------------------------- #
_FEATURE_NPY = ("features_cl_tt", "features_cl_te", "features_cl_ee", "features_logpk")


def merge_training_sets(indirs, outdir):
    """Concatenate several training-set directories into one. Grids must match."""
    os.makedirs(outdir, exist_ok=True)
    params, feats = {}, {k: [] for k in _FEATURE_NPY}
    mg = {"alpha_mz": [], "dgw_on_dem": []}
    grids_ref = None
    for d in indirs:
        g = dict(np.load(os.path.join(d, "grids.npz")))
        if grids_ref is None:
            grids_ref = g
        else:
            for k in grids_ref:
                if not np.array_equal(grids_ref[k], g[k]):
                    raise ValueError("grid '%s' differs in %s — cannot merge" % (k, d))
        P = np.load(os.path.join(d, "parameters.npz"))
        for k in P.files:
            params.setdefault(k, []).append(P[k])
        for k in _FEATURE_NPY:
            feats[k].append(np.load(os.path.join(d, "%s.npy" % k)))
        M = np.load(os.path.join(d, "features_mg.npz"))
        mg["alpha_mz"].append(M["alpha_mz"]); mg["dgw_on_dem"].append(M["dgw_on_dem"])

    np.savez(os.path.join(outdir, "parameters.npz"),
             **{k: np.concatenate(v) for k, v in params.items()})
    for k in _FEATURE_NPY:
        np.save(os.path.join(outdir, "%s.npy" % k), np.concatenate(feats[k]))
    np.savez(os.path.join(outdir, "features_mg.npz"),
             alpha_mz=np.concatenate(mg["alpha_mz"]),
             dgw_on_dem=np.concatenate(mg["dgw_on_dem"]))
    np.savez(os.path.join(outdir, "grids.npz"), **grids_ref)
    n = sum(len(x) for x in params[next(iter(params))]) if params else 0
    print("Merged %d dirs -> %d samples in %s" % (len(indirs), n, outdir))
    return n


def _gen_worker(kw):
    """Top-level worker for multiprocessing (spawn-safe): generate one chunk."""
    return generate_training_set(**kw)


def generate_parallel(n_samples, outdir, workers=8, seed=0, verbose=True, **kw):
    """Run `workers` independent hi_class generators (each its own process +
    classy instance + LHS seed), then merge into `outdir`. Splits n_samples
    across workers. Returns (n_ok, n_fail)."""
    import multiprocessing as mp

    os.makedirs(outdir, exist_ok=True)
    per = [n_samples // workers] * workers
    for i in range(n_samples % workers):
        per[i] += 1
    subdirs = [os.path.join(outdir, "_part%02d" % i) for i in range(workers)]
    jobs = [dict(n_samples=per[i], outdir=subdirs[i], seed=seed + i,
                 verbose=True, label="[w%d] " % i, **kw)
            for i in range(workers) if per[i] > 0]

    ctx = mp.get_context("spawn")
    with ctx.Pool(len(jobs)) as pool:
        results = pool.map(_gen_worker, jobs)
    n_ok = sum(r[0] for r in results)
    n_fail = sum(r[1] for r in results)
    merge_training_sets([s for s, p in zip(subdirs, per) if p > 0], outdir)
    if verbose:
        print("Parallel: %d ok, %d hi_class failures across %d workers"
              % (n_ok, n_fail, len(jobs)))
    return n_ok, n_fail
