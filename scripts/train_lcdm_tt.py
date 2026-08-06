"""Train a self-consistent 6-parameter LCDM TT emulator on hi_class (gw-emu env).

Reads a training set from generate_lcdm_tt.py and trains one cosmopower_NN mapping
the 6 standard parameters -> log10(C_l^TT). Single-stage schedule (multi-stage
degrades and cosmopower saves the last stage; see emulator/train.py notes).

    python train_lcdm_tt.py training_set_lcdm --model-dir emulators_lcdm
"""
import argparse
import os

import numpy as np

PARAMS6 = ["omega_m", "h0", "omega_b", "n_s", "A_s", "tau"]
# MG-aware: alpha_M/alpha_B change the CMB through lensing, and that lensing
# response is what actually constrains alpha_M -- a 6-param (LCDM) CMB emulator
# is blind to it and loosens the alpha_M posterior by ~5x.
PARAMS8 = PARAMS6 + ["alpha_B0", "alpha_M0"]


def _legacy_adam():
    import tensorflow as tf
    try:
        return tf.keras.optimizers.legacy.Adam()
    except AttributeError:
        return tf.keras.optimizers.Adam()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("training_dir")
    ap.add_argument("--model-dir", required=True)
    # lr=1e-2: cl_tt has a huge dynamic range and lr=1e-3 freezes on the initial
    # plateau (loss stuck at 0.075, 11% error). 1e-2 breaks through to ~0.0009.
    ap.add_argument("--lr", type=float, nargs="+", default=[1e-2])
    ap.add_argument("--patience", type=int, default=150)
    ap.add_argument("--max-epochs", type=int, default=1500)
    ap.add_argument("--nodes", type=int, default=512)
    ap.add_argument("--mg", action="store_true",
                    help="MG-aware: train on all 8 params (incl. alpha_B0/alpha_M0)")
    a = ap.parse_args()
    os.makedirs(a.model_dir, exist_ok=True)
    PARAMS = PARAMS8 if a.mg else PARAMS6

    P = np.load(os.path.join(a.training_dir, "parameters.npz"))
    params = {k: P[k] for k in PARAMS}
    ell = np.load(os.path.join(a.training_dir, "grids.npz"))["ell"]
    raw = np.load(os.path.join(a.training_dir, "features_cl_tt.npy"))

    # Filter both tails. The upper cut catches numerically blown-up runs; the
    # lower cut catches unstable runs with zero/negative C_l, which log10 maps to
    # ~-300 and which then dominate the MSE loss (0.1% of the MG set did this and
    # wrecked training: loss 7.35 vs 0.0009). Real C_l bottom out near 1e-17.
    ok = (np.isfinite(raw).all(axis=1)
          & (np.abs(raw).max(axis=1) < 1e-6)
          & (raw.min(axis=1) > 1e-20))
    if not ok.all():
        print("dropped %d non-finite/outlier samples" % int((~ok).sum()))
        raw = raw[ok]
        params = {k: v[ok] for k, v in params.items()}
    features = np.log10(np.clip(raw, 1e-300, None))
    print("training %s cl_tt  n=%d  modes=%d  params=%s"
          % ("MG-aware" if a.mg else "LCDM", len(features), len(ell), PARAMS))

    nstage = len(a.lr)
    kw = dict(validation_split=0.1, learning_rates=list(a.lr),
              batch_sizes=[1024] * nstage, gradient_accumulation_steps=[1] * nstage,
              patience_values=[a.patience] * nstage, max_epochs=[a.max_epochs] * nstage)

    from cosmopower import cosmopower_NN
    cp = cosmopower_NN(parameters=PARAMS, modes=ell, n_hidden=[a.nodes] * 4,
                       optimizer=_legacy_adam(), verbose=True)
    cp.train(training_parameters=params, training_features=features,
             filename_saved_model=os.path.join(a.model_dir, "emu_cl_tt"), **kw)
    print("saved -> %s/emu_cl_tt" % a.model_dir)


if __name__ == "__main__":
    main()
