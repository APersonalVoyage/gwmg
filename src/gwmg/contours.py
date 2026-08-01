"""Plotting: turn CosmoSIS emcee output into corner / alpha_B-alpha_M contours.

Importable engine used by ``gwmg plot``. Requires the ``plot`` extra
(chainconsumer, pandas, matplotlib). Tolerates a still-growing chain file.
"""
import os
import warnings

import numpy as np

# CosmoSIS column name (lower-cased) -> LaTeX label.
LABELS = {
    "cosmological_parameters--omega_m": r"$\Omega_m$",
    "cosmological_parameters--h0": r"$h_0$",
    "cosmological_parameters--omega_b": r"$\Omega_b$",
    "cosmological_parameters--n_s": r"$n_s$",
    "cosmological_parameters--a_s": r"$10^{9} A_s$",
    "cosmological_parameters--tau": r"$\tau$",
    "cosmological_parameters--sigma_8": r"$\sigma_8$",
    "horndeski_parameters--parameters_smg__2": r"$\alpha_{B0}$",
    "horndeski_parameters--parameters_smg__3": r"$\alpha_{M0}$",
}
A_S_KEY = "cosmological_parameters--a_s"
ALPHA_B, ALPHA_M = r"$\alpha_{B0}$", r"$\alpha_{M0}$"
DROP = {"prior", "post", "weight", "like", "importance"}
ALPHA_EXTENTS = {ALPHA_B: (-1.0, 3.0), ALPHA_M: (-1.0, 6.0)}


class ChainNotReady(Exception):
    """A CosmoSIS output file exists but has no usable samples yet (emcee only
    flushes every `nsteps` steps)."""
    def __init__(self, path, why):
        self.path, self.why = path, why
        super().__init__("%s: %s" % (path, why))


def load_chain(path, burn_in_frac=0.3):
    """Load a CosmoSIS chain into a (DataFrame, posterior_column) pair.

    Raises :class:`ChainNotReady` if the file has no usable samples yet.
    """
    import pandas as pd

    with open(path) as fh:
        header = fh.readline().lstrip("#").split()
    if not header:
        raise ChainNotReady(path, "file is empty")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = np.genfromtxt(path, comments="#", invalid_raise=False)
    if data.size == 0 or data.shape[0] == 0:
        raise ChainNotReady(path, "header written but no sample rows yet")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != len(header):
        data = data[:, : len(header)]
    if len(data) < 10:
        raise ChainNotReady(path, "only %d sample(s) so far" % len(data))

    raw = pd.DataFrame(data, columns=[h.lower() for h in header])
    raw = raw.iloc[int(burn_in_frac * len(raw)):].reset_index(drop=True)
    if A_S_KEY in raw.columns:
        raw[A_S_KEY] = raw[A_S_KEY] * 1e9

    post = raw["post"] if "post" in raw.columns else None
    keep = [c for c in raw.columns if c not in DROP]
    df = raw[keep].rename(columns={c: LABELS.get(c, c) for c in keep})
    if post is not None:
        df["post"] = post.values
    return df, ("post" if post is not None else None)


def plot_contours(chains, outdir=".", burn_in=0.3, usetex=False):
    """chains: list of (path, label, color|None). Writes corner_all.png and,
    if present, alpha_B_alpha_M.png. Returns the list of files written."""
    from chainconsumer import Chain, ChainConsumer, PlotConfig

    os.makedirs(outdir, exist_ok=True)
    c = ChainConsumer()
    seen, n = set(), 0
    for path, label, color in chains:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        df, post_col = load_chain(path, burn_in_frac=burn_in)   # may raise ChainNotReady
        seen.update(x for x in df.columns if x != "post")
        kw = dict(samples=df, name=label)
        if post_col:
            kw["posterior_column"] = post_col
        if color:
            kw["color"] = color
        c.add_chain(Chain(**kw))
        n += 1
    if n == 0:
        raise ChainNotReady("(none)", "no chains had usable samples")

    written = []
    c.set_plot_config(PlotConfig(usetex=usetex, max_ticks=4, summarise=True))
    full = os.path.join(outdir, "corner_all.png")
    c.plotter.plot(filename=full)
    written.append(full)

    if ALPHA_B in seen and ALPHA_M in seen:
        c.set_plot_config(PlotConfig(usetex=usetex, max_ticks=4,
                                     extents=ALPHA_EXTENTS, summarise=True))
        alphas = os.path.join(outdir, "alpha_B_alpha_M.png")
        c.plotter.plot(columns=[ALPHA_B, ALPHA_M], filename=alphas)
        written.append(alphas)
    return written
