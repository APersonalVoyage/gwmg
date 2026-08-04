# gwmg

[![CI](https://github.com/APersonalVoyage/gwmg/actions/workflows/ci.yml/badge.svg)](https://github.com/APersonalVoyage/gwmg/actions/workflows/ci.yml)

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
- a CosmoPower neural-network emulator of the expensive physics, with a full
  validation chain (`emu-gen`, `emu-train`, `emu-validate`, `emu-chi2`,
  `emu-bias`); see `docs/emulator.md`

## Install

gwmg has two parts: the Python package (the likelihood, CLI and plotting, which
install with pip) and the heavier CosmoSIS 3 + hi_class stack it drives (which
is installed separately, because it is not pip-installable).

Install the package straight from GitHub:

    pip install git+https://github.com/APersonalVoyage/gwmg.git

Or clone it first, which also gives you the configs, data and tests locally and
lets you modify the code:

    git clone https://github.com/APersonalVoyage/gwmg.git
    cd gwmg
    pip install -e .[dev]

Check the core works (this needs no CosmoSIS):

    gwmg validate

That gives you the GW likelihood, the `gwmg` command line and plotting. To run
the full pipeline you also need CosmoSIS 3 and hi_class; see `docs/install.md`
for those.

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

## Emulator

Running the exact pipeline is slow because every step solves the full
cosmological equations. gwmg includes a CosmoPower neural-network emulator
(arXiv:2106.03846) that learns the expensive physics so the same analysis runs
in minutes. A self-consistent CMB emulator trained on hi_class biases the Planck
parameters by at most 0.19 sigma, beating an off-the-shelf pre-trained emulator
(0.53 sigma) by removing the code mismatch between the two. The emulator code and
the validation methodology (why per-multipole metrics mislead, and the
Fisher-matrix parameter-bias test that replaces them) are documented in
`docs/emulator.md`.

The emulator depends on CosmoPower and TensorFlow, whose stack conflicts with
classy's, so it runs in its own environment; install its extras with
`pip install -e .[emulator]`.

## Status

Work in progress. The pipeline runs and reproduces the physics of the source
paper, and the emulator is validated as a component. The next step is the
end-to-end demonstration: the full accelerated inference reproducing the exact
constraints. See the roadmap in `docs/emulator.md`.

## License

MIT. See `LICENSE`.
