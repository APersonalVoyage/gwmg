"""Constrain the Planck-mass run rate alpha_M from GW standard sirens.

A complete, self-contained analysis that runs in a few seconds and needs only
numpy and scipy -- no CosmoSIS, no hi_class. It uses the real GW170817 and
GW190521 measurements shipped with the package.

    python examples/constrain_alpha_M.py

The point of the example is to show what the pipeline does in miniature. The full
pipeline solves the cosmology with hi_class and samples eight parameters against
CMB, RSD and BAO data as well; here the background is a fixed LCDM and we scan
the single parameter alpha_M on a grid.

Physics: in modified gravity, gravitational waves lose amplitude as they
propagate, so they look further away than they are. The GW luminosity distance is

    d_GW(z) = d_EM(z) * exp( 0.5 * int_0^z alpha_M(z') / (1 + z') dz' )

With the propto_omega parametrisation, alpha_M(z) = alpha_M0 * Omega_DE(z).
alpha_M0 = 0 is General Relativity.
"""
import os

import numpy as np
from scipy.integrate import cumulative_trapezoid

from gwmg import gw_log_likelihood, load_events, pipeline_dir

C_KM_S = 299792.458
H0, OMEGA_M = 67.4, 0.315          # fixed LCDM background for this example


def background(z):
    """Exact LCDM distances and H(z). Cheap, so no emulator needed."""
    E = np.sqrt(OMEGA_M * (1 + z) ** 3 + (1 - OMEGA_M))
    d_c = (C_KM_S / H0) * cumulative_trapezoid(1.0 / E, z, initial=0.0)
    return d_c * (1 + z), H0 * E / C_KM_S      # d_L [Mpc], H(z)/c [1/Mpc]


def dgw_ratio(z, alpha_M0):
    """d_GW / d_EM for propto_omega: alpha_M(z) = alpha_M0 * Omega_DE(z)."""
    E2 = OMEGA_M * (1 + z) ** 3 + (1 - OMEGA_M)
    alpha_M = alpha_M0 * (1 - OMEGA_M) / E2
    return np.exp(0.5 * cumulative_trapezoid(alpha_M / (1 + z), z, initial=0.0))


def main():
    # The two events, read from the file the pipeline itself uses.
    d_gw_obs, z_obs, sigma_dgw, sigma_z, v_rms = load_events(
        os.path.join(pipeline_dir(), "data", "gw", "ligo_data.txt"))
    events = dict(d_gw_obs=d_gw_obs, z_obs=z_obs, sigma_dgw=sigma_dgw,
                  sigma_z=sigma_z, v_rms=v_rms)
    print("events (d_GW [Mpc], z, sigma_d [Mpc]):")
    for d, zo, s in zip(d_gw_obs, z_obs, sigma_dgw):
        print("   %8.1f  %.4f  %8.1f" % (d, zo, s))

    z = np.linspace(0.0, 1.0, 2000)
    d_l, h = background(z)

    # Scan alpha_M0 and evaluate the log-likelihood at each value. This is what
    # the MCMC does, only in 8 dimensions and with more data.
    grid = np.linspace(-2.0, 6.0, 400)
    logL = np.array([
        gw_log_likelihood(z, d_l, h, dgw_ratio(z, a), **events) for a in grid])

    # Posterior for a flat prior, then summarise it.
    post = np.exp(logL - logL.max())
    post /= np.trapezoid(post, grid) if hasattr(np, "trapezoid") else np.trapz(post, grid)
    mean = np.trapezoid(grid * post, grid) if hasattr(np, "trapezoid") else np.trapz(grid * post, grid)
    var = (np.trapezoid(grid ** 2 * post, grid) if hasattr(np, "trapezoid")
           else np.trapz(grid ** 2 * post, grid)) - mean ** 2
    cdf = np.concatenate([[0], np.cumsum(0.5 * (post[1:] + post[:-1]) * np.diff(grid))])
    lo, hi = np.interp([0.16, 0.84], cdf / cdf[-1], grid)

    print("\nalpha_M0 = %.2f +/- %.2f      (68%% interval %.2f to %.2f)"
          % (mean, np.sqrt(var), lo, hi))
    print("GR (alpha_M0 = 0) is %s at 68%%" %
          ("consistent" if lo <= 0 <= hi else "disfavoured"))
    print("""
Note the constraint is weak and the posterior keeps rising towards the edge of
the scan: two sirens on their own barely pin alpha_M down. That is precisely why
the full pipeline adds CMB, RSD and BAO data, which brings it to
alpha_M0 = 0.4 +/- 0.8. Run `gwmg run gw_lss_emcee` for that version.""")

    # A text histogram, so the example needs no plotting library.
    print("\nposterior:")
    for i in range(0, len(grid), 10):
        bar = "#" * int(60 * post[i] / post.max())
        print("  %+5.2f | %s" % (grid[i], bar))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(install matplotlib to also get alpha_M_posterior.png)")
        return
    plt.figure(figsize=(6, 3.5))
    plt.plot(grid, post, lw=2)
    plt.axvline(0, ls=":", c="grey")
    plt.text(0.05, 0.9 * post.max(), "GR", color="grey")
    plt.xlabel(r"$\alpha_{M0}$"); plt.ylabel("posterior")
    plt.title("GW170817 + GW190521 standard sirens")
    plt.tight_layout(); plt.savefig("alpha_M_posterior.png", dpi=150)
    print("\nwrote alpha_M_posterior.png")


if __name__ == "__main__":
    main()
