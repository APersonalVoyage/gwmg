"""Parallel hi_class LCDM TT training-set generator (6 standard params, alpha=0).

Robust, low-memory design:
  * small batches streamed to per-batch part files on disk, so the main process
    holds almost nothing (the previous all-in-RAM version was SIGKILLed);
  * ProcessPoolExecutor so a dead worker raises instead of hanging silently;
  * persistent workers (no max_tasks_per_child: recycling under the spawn start
    method deadlocked at the first recycle boundary; classy's struct_cleanup
    keeps per-worker memory flat anyway);
  * parts persist, so an interrupted run can be merged / resumed.

Same on-disk layout as `gwmg emu-gen` (parameters.npz, features_cl_tt.npy,
grids.npz). Run in the gw-hiclass env:
    python generate_lcdm_tt.py -n 24000 --outdir training_set_lcdm --seed 1 --workers 6
"""
import argparse
import glob
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

PARAMS = ["omega_m", "h0", "omega_b", "n_s", "A_s", "tau"]
RANGES = {
    "omega_m": (0.24, 0.36), "h0": (0.61, 0.76), "omega_b": (0.041, 0.054),
    "n_s": (0.93, 1.00), "A_s": (1.7e-9, 2.5e-9), "tau": (0.02, 0.12),
}
OMNUH2 = 0.00083
LMAX = 2600
SBBN = os.environ.get("HICLASS_DIR",
        "/Users/abhishekkarkola/MSc_Thesis/hi_class_public") + "/external/bbn/sBBN.dat"


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
        "output": "tCl,pCl,lCl", "lensing": "yes", "modes": "s",
        "l_max_scalars": LMAX, "T_cmb": 2.726, "N_ur": 3.046, "sBBN file": SBBN,
        "gravity_model": "propto_omega", "expansion_model": "lcdm",
        "parameters_smg": "1.0, 0., 0., 0., 1.",
        "kineticity_safe_smg": 0.0, "expansion_smg": 0.5,
        "Omega_Lambda": 0.0, "Omega_fld": 0.0, "Omega_smg": -1,
        "H0": 100 * p["h0"], "omega_b": p["omega_b"] * h2,
        "omega_cdm": p["omega_m"] * h2 - p["omega_b"] * h2 - OMNUH2,
        "n_s": p["n_s"], "A_s": p["A_s"], "tau_reio": p["tau"],
    }


def _run_batch(args):
    """Compute one small batch; return only the successful samples."""
    bi, sub = args
    import classy
    cosmo = classy.Class()
    n = len(sub[PARAMS[0]])
    kp = {k: [] for k in PARAMS}
    tts = []
    for i in range(n):
        p = {k: float(sub[k][i]) for k in PARAMS}
        try:
            cosmo.set(_inputs(p)); cosmo.compute()
            tt = cosmo.lensed_cl(LMAX)["tt"][2:].copy()
            if not np.isfinite(tt).all():
                raise ValueError("non-finite")
            for k in PARAMS:
                kp[k].append(p[k])
            tts.append(tt)
        except Exception:
            pass
        finally:
            cosmo.struct_cleanup()
    return bi, kp, tts


def _merge(partdir, outdir):
    parts = sorted(glob.glob(os.path.join(partdir, "part_*.npz")))
    allp = {k: [] for k in PARAMS}
    alltt = []
    for pf in parts:
        z = np.load(pf)
        alltt.append(z["cl_tt"])
        for k in PARAMS:
            allp[k].append(z[k])
    if not alltt:
        print("no parts to merge"); return 0
    tt = np.concatenate(alltt, axis=0)
    np.savez(os.path.join(outdir, "parameters.npz"),
             **{k: np.concatenate(allp[k]) for k in PARAMS})
    np.save(os.path.join(outdir, "features_cl_tt.npy"), tt)
    np.savez(os.path.join(outdir, "grids.npz"), ell=np.arange(2, LMAX + 1))
    print("merged %d parts -> %d samples in %s" % (len(parts), len(tt), outdir))
    return len(tt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--merge-only", action="store_true", help="just merge existing parts")
    a = ap.parse_args()
    partdir = os.path.join(a.outdir, "_parts")
    os.makedirs(partdir, exist_ok=True)

    if a.merge_only:
        _merge(partdir, a.outdir); return

    samples = latin_hypercube(RANGES, a.n, seed=a.seed)
    batches = np.array_split(np.arange(a.n), int(np.ceil(a.n / a.batch)))
    # skip batches whose part file already exists (resume)
    jobs = [(bi, {k: samples[k][idx] for k in PARAMS})
            for bi, idx in enumerate(batches)
            if not os.path.exists(os.path.join(partdir, "part_%05d.npz" % bi))]
    print("total batches %d, to run %d, workers %d, batch %d"
          % (len(batches), len(jobs), a.workers, a.batch), flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    done = nok = 0
    try:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(_run_batch, j): j[0] for j in jobs}
            for fut in as_completed(futs):
                bi, kp, tts = fut.result()
                if tts:
                    np.savez(os.path.join(partdir, "part_%05d.npz" % bi),
                             cl_tt=np.array(tts), **{k: np.array(v) for k, v in kp.items()})
                    nok += len(tts)
                done += 1
                if done % 5 == 0 or done == len(jobs):
                    print("batches %d/%d  new_samples=%d" % (done, len(jobs), nok), flush=True)
    finally:
        _merge(partdir, a.outdir)


if __name__ == "__main__":
    main()
