"""Train CosmoPower emulators on a hi_class training set.

Loads a directory produced by ``gwmg emu-gen`` and trains one CosmoPower NN
emulator per observable (mapping the 8 cosmological+Horndeski parameters onto
the spectrum). Trained models are saved into ``model_dir``.

Requires ``cosmopower`` (which pulls in TensorFlow) — install separately:
    pip install cosmopower

NOTE: this module follows CosmoPower's documented ``cosmopower_NN.train`` API.
CosmoPower's interface has shifted across versions; if training errors on an
argument name, check it against your installed cosmopower and adjust
``_TRAIN_KW`` below. The data-loading half is covered by tests.
"""
import os

import numpy as np

# The 8 input parameters, in a fixed order (must match at inference time).
PARAM_NAMES = ["omega_m", "h0", "omega_b", "n_s", "A_s", "tau", "alpha_B0", "alpha_M0"]

# Which stored features to train, and on which mode grid. Each entry:
#   (feature_file, grid_key_in_grids.npz, log10_transform?, raw_abs_max_or_None)
#
# raw_abs_max guards against rare hi_class runs that go numerically unstable
# without raising an exception, so isfinite() alone doesn't catch them (seen:
# raw C_l ~1e286). log10() happens to compress such an outlier down to a
# "finite" value (log10(1e286) = 286), so tt/ee limp through training noisily;
# but for the unlogged TE spectrum, squaring a ~1e31 raw value for the MSE
# loss overflows float32 straight to inf on the very first batch, permanently.
# Real classy C_l's are ~1e-8..1e-16 (empirically: tt/ee 99th pct ~3e-9, te
# genuine tail runs up to ~1e-7 before a clean 3-order-of-magnitude jump into
# obvious garbage) -- 1e-6 sits in that gap and drops <0.2% of samples.
# logpk is already log10-space (O(1-5)), so no raw-magnitude check applies.
# TT-only CMB (matching Baker & Harrison's Plik-lite TT-only). We drop the EE
# and TE emulators: EE is hard and TE (sign-changing, zero-crossing) is not
# reliably emulable as a raw spectrum, and the Planck likelihood is set to TT.
# 5th field n_pcas: None = plain cosmopower_NN; an int = cosmopower_PCAplusNN
# with that many PCA components. The PCA+NN path is kept for reference, but on
# this data it validated *worse* than the plain NN for cl_tt (CV ~45 vs ~1.9),
# so both spectra currently use plain NN.
SPECTRA = [
    ("features_logpk.npy", "k_h", False, None, None),   # NN
    ("features_cl_tt.npy", "ell", True, 1e-6, None),    # NN (PCA+NN did worse here)
]

# Default: high-accuracy schedule (5 stages, low learning rates, high max_epochs).
# This is what a full overnight training run uses. For a quick first pass, pass
# train_kw=_TRAIN_KW_FAST to train_emulators (3 stages, early stopping).
_TRAIN_KW = dict(
    validation_split=0.1,
    learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    batch_sizes=[1024, 1024, 1024, 1024, 1024],
    gradient_accumulation_steps=[1, 1, 1, 1, 1],
    patience_values=[100, 100, 100, 100, 100],
    max_epochs=[1000, 1000, 1000, 1000, 1000],
)

# Single-stage schedule. CosmoPower saves the *last* stage's end weights, and
# empirically the low-LR later stages degrade rather than refine here (both logpk
# and cl_tt got worse at 1e-3 and 1e-4), so a multi-stage run saves the worst
# model. One moderate-LR stage with generous patience trains until it plateaus
# and saves that plateau -- no later stage to overwrite it with a worse one.
_TRAIN_KW_FAST = dict(
    validation_split=0.1,
    learning_rates=[1e-3],
    batch_sizes=[1024],
    gradient_accumulation_steps=[1],
    patience_values=[150],
    max_epochs=[1500],
)


def load_training_set(training_dir):
    """Return (params_dict, grids_dict). params_dict is {name: array[n]}."""
    P = np.load(os.path.join(training_dir, "parameters.npz"))
    params = {k: P[k] for k in P.files}
    grids = dict(np.load(os.path.join(training_dir, "grids.npz")))
    return params, grids


def _load_raw(training_dir, fname):
    return np.load(os.path.join(training_dir, fname))


def _sanity_mask(raw, raw_abs_max):
    """Finite AND (if raw_abs_max given) not a numerical-blowup outlier."""
    mask = np.isfinite(raw).all(axis=1)
    if raw_abs_max is not None:
        mask &= np.abs(raw).max(axis=1) < raw_abs_max
    return mask


def _make_optimizer():
    """Legacy Adam. CosmoPower sets per-stage learning rates via optimizer.lr,
    which the v2.11+ Keras optimizer does not honour correctly (and which is
    flagged as broken on Apple Silicon), causing low-LR stages to degrade rather
    than refine. The legacy optimizer respects .lr and trains stably."""
    import tensorflow as tf
    try:
        return tf.keras.optimizers.legacy.Adam()
    except AttributeError:
        return tf.keras.optimizers.Adam()


def _train_nn(train_params, features, modes, model_path, n_hidden, kw, verbose):
    from cosmopower import cosmopower_NN
    cp = cosmopower_NN(parameters=PARAM_NAMES, modes=modes, n_hidden=list(n_hidden),
                       optimizer=_make_optimizer(), verbose=verbose)
    cp.train(training_parameters=train_params, training_features=features,
             filename_saved_model=model_path, **kw)


def _train_pca_nn(train_params, features, modes, n_pcas, model_path, n_hidden, kw, verbose):
    """cosmopower_PCAplusNN. The PCA step reads training data from .npz files
    (features under the key 'features'), so write them to a temp dir first."""
    import tempfile
    from cosmopower import cosmopower_PCA, cosmopower_PCAplusNN
    with tempfile.TemporaryDirectory() as td:
        pfile = os.path.join(td, "params")     # cosmopower appends ".npz"
        ffile = os.path.join(td, "features")
        np.savez(pfile + ".npz", **train_params)
        np.savez(ffile + ".npz", features=features)
        cp_pca = cosmopower_PCA(parameters=PARAM_NAMES, modes=modes, n_pcas=n_pcas,
                                parameters_filenames=[pfile],
                                features_filenames=[ffile], verbose=verbose)
        cp_pca.transform_and_stack_training_data(filename=os.path.join(td, "pca"))
        cp = cosmopower_PCAplusNN(cp_pca=cp_pca, n_hidden=list(n_hidden),
                                  optimizer=_make_optimizer(), verbose=verbose)
        cp.train(filename_saved_model=model_path, **kw)


def train_emulators(training_dir, model_dir, spectra=SPECTRA,
                    n_hidden=(512, 512, 512, 512), train_kw=None, verbose=True):
    """Train and save one CosmoPower emulator per spectrum (NN or PCA+NN,
    per the 5th SPECTRA field). Returns the list of saved model paths."""
    os.makedirs(model_dir, exist_ok=True)
    params, grids = load_training_set(training_dir)
    n = len(params[PARAM_NAMES[0]])
    kw = dict(_TRAIN_KW)
    if train_kw:
        kw.update(train_kw)

    saved = []
    for fname, grid_key, log10, raw_abs_max, n_pcas in spectra:
        path = os.path.join(training_dir, fname)
        if not os.path.exists(path):
            if verbose:
                print("skip %s (not found)" % fname)
            continue
        modes = grids[grid_key]
        raw = _load_raw(training_dir, fname)
        assert raw.shape == (n, len(modes)), \
            "%s shape %s != (%d, %d)" % (fname, raw.shape, n, len(modes))

        # Drop non-finite / numerical-blowup-outlier samples (see SPECTRA note).
        ok = _sanity_mask(raw, raw_abs_max)
        train_params = params
        if not ok.all():
            if verbose:
                print("  dropped %d non-finite/outlier samples" % int((~ok).sum()))
            raw = raw[ok]
            train_params = {k: v[ok] for k, v in params.items()}

        features = np.log10(np.clip(raw, 1e-300, None)) if log10 else raw

        name = os.path.splitext(fname)[0].replace("features_", "")   # e.g. cl_tt
        model_path = os.path.join(model_dir, "emu_" + name)
        kind = "PCA+NN(%d)" % n_pcas if n_pcas else "NN"
        if verbose:
            print("training %-10s  n=%d  modes=%d  %s  ->  %s"
                  % (name, len(train_params[PARAM_NAMES[0]]), len(modes), kind, model_path))

        if n_pcas:
            _train_pca_nn(train_params, features, modes, n_pcas, model_path,
                          n_hidden, kw, verbose)
        else:
            _train_nn(train_params, features, modes, model_path, n_hidden, kw, verbose)
        saved.append(model_path)
    if verbose:
        print("Saved %d emulators to %s" % (len(saved), model_dir))
    return saved
