"""Train the growth (f-sigma_8) emulator: 8 params -> f-sigma_8 on the z grid.

Trained on RAW f-sigma_8 (not log): it is O(1) with a small dynamic range, so the
bridge reads predictions_np directly (no 10**). A magnitude cap drops the rare
large-alpha cosmologies where hi_class growth blows up (finite but ~1e128).
"""
import argparse
import os

import numpy as np

PARAMS = ["omega_m", "h0", "omega_b", "n_s", "A_s", "tau", "alpha_B0", "alpha_M0"]


def _legacy_adam():
    import tensorflow as tf
    try:
        return tf.keras.optimizers.legacy.Adam()
    except AttributeError:
        return tf.keras.optimizers.Adam()


def _load(d, cap):
    P = np.load(os.path.join(d, "parameters.npz"))
    f = np.load(os.path.join(d, "features_fsigma8.npy"))
    z = np.load(os.path.join(d, "grids.npz"))["z"]
    ok = np.isfinite(f).all(axis=1) & (np.abs(f).max(axis=1) < cap)
    return {k: P[k][ok] for k in PARAMS}, f[ok], z, int((~ok).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("training_dir")
    ap.add_argument("--test-dir", default=None)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--cap", type=float, default=1.5)
    # lr=1e-2: lr=1e-3 freezes on the initial plateau for this cosmopower setup
    # (same trap as cl_tt), regardless of the target's dynamic range.
    ap.add_argument("--lr", type=float, nargs="+", default=[1e-2])
    ap.add_argument("--patience", type=int, default=150)
    ap.add_argument("--max-epochs", type=int, default=1500)
    a = ap.parse_args()
    os.makedirs(a.model_dir, exist_ok=True)

    params, feat, z, dropped = _load(a.training_dir, a.cap)
    print("training growth  n=%d (dropped %d)  modes=%d" % (len(feat), dropped, len(z)))
    ns = len(a.lr)
    kw = dict(validation_split=0.1, learning_rates=list(a.lr),
              batch_sizes=[1024] * ns, gradient_accumulation_steps=[1] * ns,
              patience_values=[a.patience] * ns, max_epochs=[a.max_epochs] * ns)

    from cosmopower import cosmopower_NN
    cp = cosmopower_NN(parameters=PARAMS, modes=z, n_hidden=[512] * 4,
                       optimizer=_legacy_adam(), verbose=True)
    cp.train(training_parameters=params, training_features=feat,
             filename_saved_model=os.path.join(a.model_dir, "emu_fsigma8"), **kw)
    print("saved -> %s/emu_fsigma8" % a.model_dir)

    if a.test_dir:
        tp, tf_, _, _ = _load(a.test_dir, a.cap)
        pred = cp.predictions_np(tp)
        err = np.abs(pred / tf_ - 1.0)
        print("=== growth per-mode fractional error (test) ===")
        print("median %.4f%%  p95 %.4f%%  p99 %.4f%%" %
              (100 * np.median(err), 100 * np.percentile(err, 95), 100 * np.percentile(err, 99)))


if __name__ == "__main__":
    main()
