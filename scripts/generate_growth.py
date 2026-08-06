"""Parallel hi_class generator for the growth emulator: f-sigma_8(z) over the
full 8-parameter box (standard parameters + alpha_M, alpha_B).

Unlike the CMB set (LCDM, temperature only), growth depends strongly on the
modified-gravity parameters, so alpha_M and alpha_B vary here. f-sigma_8 is read
from hi_class the same way the exact interface does: fsigma8(z) = f(z) sigma_8
D(z), with f and D from get_background()'s 'gr.fac. f' / 'gr.fac. D'. Output is
matter power only (needed for sigma_8) -- no C_ell -- so each evaluation is cheap.

Same robust design and on-disk layout as generate_lcdm_tt.py. Run in gw-hiclass:
    python generate_growth.py -n 8000 --outdir training_set_growth --seed 3 --workers 6
"""
import argparse
import glob
import os
from functools import lru_cache
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

PARAMS = ["omega_m", "h0", "omega_b", "n_s", "A_s", "tau", "alpha_B0", "alpha_M0"]
RANGES = {
    "omega_m": (0.24, 0.36), "h0": (0.61, 0.76), "omega_b": (0.041, 0.054),
    "n_s": (0.93, 1.00), "A_s": (1.7e-9, 2.5e-9), "tau": (0.02, 0.12),
    "alpha_B0": (-1.0, 3.0), "alpha_M0": (-1.0, 6.0),
}
OMNUH2 = 0.00083
ZGRID = np.linspace(0.0, 1.5, 31)          # covers the RSD survey redshifts
KMAX = 1.0
@lru_cache(maxsize=1)
def _sbbn():
    """Path to hi_class's sBBN table. HICLASS_DIR must point at the hi_class
    checkout; there is no sensible default, so fail loudly rather than silently
    using a path that only exists on one machine."""
    d = os.environ.get("HICLASS_DIR")
    if not d:
        raise SystemExit("HICLASS_DIR is not set. Point it at your hi_class "
                         "checkout, e.g. export HICLASS_DIR=/path/to/hi_class_public")
    f = os.path.join(d, "external", "bbn", "sBBN.dat")
    if not os.path.exists(f):
        raise SystemExit("no sBBN table at %s -- is HICLASS_DIR correct?" % f)
    return f



def latin_hypercube(ranges, n, seed=0):
    rng = np.random.default_rng(seed)
    cut = np.linspace(0.0, 1.0, n + 1)
    out = {}
    for k in ranges:
        lo, hi = ranges[k]
        u = rng.uniform(cut[:-1], cut[1:])
        rng.shuffle(u)
        out[k] = lo + (hi - lo) * u
    return out


def _inputs(p):
    h2 = p["h0"] ** 2
    return {
        "output": "mPk", "z_max_pk": float(ZGRID[-1] + 0.1), "P_k_max_h/Mpc": KMAX,
        "T_cmb": 2.726, "N_ur": 3.046, "sBBN file": _sbbn(),
        "gravity_model": "propto_omega", "expansion_model": "lcdm",
        "parameters_smg": "1.0, %g, %g, 0., 1." % (p["alpha_B0"], p["alpha_M0"]),
        "kineticity_safe_smg": 0.0, "expansion_smg": 0.5,
        "Omega_Lambda": 0.0, "Omega_fld": 0.0, "Omega_smg": -1,
        "H0": 100 * p["h0"], "omega_b": p["omega_b"] * h2,
        "omega_cdm": p["omega_m"] * h2 - p["omega_b"] * h2 - OMNUH2,
        "n_s": p["n_s"], "A_s": p["A_s"], "tau_reio": p["tau"],
    }


def _fsigma8(cosmo):
    bg = cosmo.get_background()
    order = np.argsort(bg["z"])
    z, D, f = bg["z"][order], bg["gr.fac. D"][order], bg["gr.fac. f"][order]
    Dz = np.interp(ZGRID, z, D)
    fz = np.interp(ZGRID, z, f)
    s8z = cosmo.sigma8() * Dz              # D(0) = 1, so sigma8(z) = sigma8 * D(z)
    return fz * s8z


def _run_batch(args):
    bi, sub = args
    import classy
    cosmo = classy.Class()
    n = len(sub[PARAMS[0]])
    kp = {k: [] for k in PARAMS}
    fs8 = []
    for i in range(n):
        p = {k: float(sub[k][i]) for k in PARAMS}
        try:
            cosmo.set(_inputs(p)); cosmo.compute()
            v = _fsigma8(cosmo)
            if not np.isfinite(v).all():
                raise ValueError("non-finite")
            for k in PARAMS:
                kp[k].append(p[k])
            fs8.append(v)
        except Exception:
            pass
        finally:
            cosmo.struct_cleanup()
    return bi, kp, fs8


def _merge(partdir, outdir):
    parts = sorted(glob.glob(os.path.join(partdir, "part_*.npz")))
    allp = {k: [] for k in PARAMS}
    allf = []
    for pf in parts:
        z = np.load(pf)
        allf.append(z["fsigma8"])
        for k in PARAMS:
            allp[k].append(z[k])
    if not allf:
        print("no parts to merge"); return 0
    fs8 = np.concatenate(allf, axis=0)
    np.savez(os.path.join(outdir, "parameters.npz"),
             **{k: np.concatenate(allp[k]) for k in PARAMS})
    np.save(os.path.join(outdir, "features_fsigma8.npy"), fs8)
    np.savez(os.path.join(outdir, "grids.npz"), z=ZGRID)
    print("merged %d parts -> %d samples in %s" % (len(parts), len(fs8), outdir))
    return len(fs8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args()
    partdir = os.path.join(a.outdir, "_parts")
    os.makedirs(partdir, exist_ok=True)
    if a.merge_only:
        _merge(partdir, a.outdir); return

    samples = latin_hypercube(RANGES, a.n, seed=a.seed)
    batches = np.array_split(np.arange(a.n), int(np.ceil(a.n / a.batch)))
    jobs = [(bi, {k: samples[k][idx] for k in PARAMS})
            for bi, idx in enumerate(batches)
            if not os.path.exists(os.path.join(partdir, "part_%05d.npz" % bi))]
    print("total batches %d, to run %d, workers %d" % (len(batches), len(jobs), a.workers))

    from concurrent.futures import ProcessPoolExecutor, as_completed
    done = nok = 0
    try:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(_run_batch, j): j[0] for j in jobs}
            for fut in as_completed(futs):
                bi, kp, fs8 = fut.result()
                if fs8:
                    np.savez(os.path.join(partdir, "part_%05d.npz" % bi),
                             fsigma8=np.array(fs8), **{k: np.array(v) for k, v in kp.items()})
                    nok += len(fs8)
                done += 1
                if done % 5 == 0 or done == len(jobs):
                    print("batches %d/%d  new_samples=%d" % (done, len(jobs), nok), flush=True)
    finally:
        _merge(partdir, a.outdir)


if __name__ == "__main__":
    main()
