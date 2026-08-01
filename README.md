# gwmg

A Python 3 / CosmoSIS 3 pipeline for constraining modified gravity with
gravitational-wave standard sirens and large-scale-structure data.

gwmg implements the framework of Baker & Harrison (2020), arXiv:2007.13791.
hi_class computes the Horndeski cosmology and the modified gravitational-wave
luminosity distance; CosmoSIS combines a GW standard-siren likelihood with CMB,
RSD and BAO data to constrain the Horndeski functions alpha_M and alpha_B.

It began as a rebuild of an MSc thesis pipeline (Python 2.7, a lost interface
module, hardcoded cluster paths) and is now a modernised, tested, installable
version.

## Contents

- `gwmg.gw_log_likelihood`: the GW siren log-likelihood (numpy/scipy only)
- a CosmoSIS `dgw` module and a hi_class-to-CosmoSIS interface
- pipeline configs (a quick test run and the full GW+LSS chain) and the
  GW170817 + GW190521 data
- a `gwmg` command line: `info`, `run`, `plot`, `validate`
- corner/contour plotting with ChainConsumer

## Install

The Python package installs with pip:

    pip install -e .[dev]
    gwmg validate

CosmoSIS 3 and hi_class are installed separately (they are not pip-installable).
See `docs/install.md`.

## Usage

    gwmg info
    gwmg run hi_class_test --test
    gwmg run gw_lss_emcee --mpi 8
    gwmg plot output/gw_lss_horndeski.txt --outdir plots

## Validation

The pipeline is checked against Baker & Harrison (2020): the GW luminosity
distance (their eq. 2.12), the alpha_i proportional-to-Omega_Lambda ansatz
(eq. 2.14), and the lensing and peculiar-velocity error model (eqs. 3.9-3.11).
See `docs/validation.md`.

## Status

Work in progress. The pipeline runs and reproduces the physics of the source
paper. A machine-learning emulator to speed up the analysis is in development
and not yet part of this repository.

## License

MIT. See `LICENSE`.
