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
- CosmoPower neural-network emulators of the CMB, matter power and growth, an
  `Emulator` Python API, a drop-in CosmoSIS module that replaces hi_class, and a
  three-level validation suite (`emu-gen`, `emu-train`, `emu-validate`,
  `emu-chi2`, `emu-bias`) — see the [Emulator](#emulator) section below and
  `docs/emulator.md`

## How it works

gwmg is a thin layer over [CosmoSIS](https://cosmosis.readthedocs.io), which runs
Bayesian inference as a **pipeline of modules**. Three ideas are enough to
understand everything else here.

**1. A module is a Python file with `setup()` and `execute()`.** Modules run in
order and communicate through a *datablock*: a typed key-value store, namespaced
by section. One module writes `distances/d_l`, a later one reads it. Modules do
not call each other.

**2. One likelihood evaluation looks like this:**

```
       emcee proposes 8 parameters
                 │   omega_m, h0, omega_b, n_s, A_s, tau, alpha_B0, alpha_M0
                 ▼
        ┌─────────────────┐
        │ consistency     │  derive ombh2, omch2, ...
        │ hi_class        │  solve the cosmology  <-- the slow step (seconds)
        └────────┬────────┘     (or `emulator`, ~7 ms)
                 │ writes cmb_cl, matter_power_lin, distances, growth_parameters
                 ▼
           [ datablock ]
                 │ read by
                 ▼
    planck_py, boss, 6dfgs, wigglez_bao, mgs_bao, dgw
                 │ each writes likelihoods/*_LIKE
                 ▼
        total log-likelihood ──> back to emcee
```

The theory module is the expensive one and the only one the emulator replaces.
Everything downstream is unchanged, which is why the exact and emulated runs are
directly comparable.

**3. A run is defined by two ini files and a data file**, and nothing else:

| File | Controls |
|---|---|
| `configs/gw_lss_emcee.ini` | which modules run, which likelihoods count, sampler settings |
| `configs/values_horndeski.ini` | the parameters: fixed values, or `min start max` to sample |
| `data/gw/ligo_data.txt` | the gravitational-wave events |

So "changing the analysis" means editing those three, not touching code. That is
what the next section covers.

**What gwmg itself contributes**, as opposed to CosmoSIS and hi_class: the GW
standard-siren likelihood (`gwmg.gw_log_likelihood` and the `dgw` module), the
`hi_class_interface` module that translates between hi_class and the datablock,
the emulator and its drop-in module, the ready-made configs and data, and the CLI
that wires the paths together.

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
the full pipeline you also need CosmoSIS 3 and hi_class, which live in a single
conda environment together with the emulator stack — see the Quick start below
and `docs/install.md`.

## Quick start (from a clean machine)

This is the shortest path from nothing to a plot. It assumes conda/mamba
(Miniforge) and takes roughly an hour, most of it building hi_class.

```bash
# 1. get the code
git clone https://github.com/APersonalVoyage/gwmg.git
cd gwmg

# 2. one environment for everything (see docs/install.md for the detail)
conda env create -f environment.yml
conda activate gwmg
cosmosis-build-standard-library
source cosmosis-configure

# hi_class, built against this environment's numpy
git clone https://github.com/miguelzuma/hi_class_public.git
cd hi_class_public && make ; cd python
CC=clang python setup.py build_ext --inplace      # CC=gcc on Linux
pip install --no-build-isolation .
cd ../..

# 3. tell gwmg where things are
export COSMOSIS_SRC_DIR=$PWD            # the parent of cosmosis-standard-library
export HICLASS_DIR=$PWD/hi_class_public
export OMP_NUM_THREADS=1                # hi_class is not thread-safe

# 4. check it all resolves
gwmg info
gwmg run hi_class_test --test           # one evaluation, ~10 s

# 5. run a short chain (minutes, not converged -- just to see it work)
gwmg run gw_lss_emcee --mpi 4

# 6. plot it
gwmg plot output/gw_lss_horndeski.txt --outdir plots
```

Step 6 writes `plots/corner_all.png` (all eight parameters) and
`plots/alpha_B_alpha_M.png` (the modified-gravity constraint). Both are corner
plots: 1D histograms on the diagonal, 2D contours off it.

A converged chain takes days on six cores. For a quick look, edit `samples` in
the `[emcee]` block of the config (see below) or use the emulator, which does the
same run in about ten minutes.

## Changing the analysis

The bundled configs live inside the installed package, so copy them somewhere
writable, edit, and point `gwmg run` at your copy:

```bash
# copy the bundled pipeline (configs + modules + data) into ./myrun
python -c "import gwmg, shutil; shutil.copytree(gwmg.pipeline_dir(), 'myrun')"

# edit myrun/configs/*.ini as you like, then run it
gwmg run myrun/configs/gw_lss_emcee.ini --pipeline-dir myrun
```

`--pipeline-dir` sets `PIPELINE_DIR`, which the configs use to resolve their own
module and data paths, so keep the directory structure intact. Everything below
refers to paths inside that copy.

**Priors and starting points** — `configs/values_horndeski.ini`. Each line is
`min  start  max`; a single value fixes the parameter. The two Horndeski
parameters are:

```ini
[horndeski_parameters]
parameters_smg__2  = -1.   0.41   3.     ; alpha_B0, prior U(-1, 3)
parameters_smg__3  = -1.   0.01   6.     ; alpha_M0, prior U(-1, 6)
```

To fix alpha_M at zero (a GR run), replace that line with
`parameters_smg__3 = 0.`. To widen a prior, change the min/max. The standard
cosmological parameters are in the `[cosmological_parameters]` block above it.

**Sampler and data** — `configs/gw_lss_emcee.ini`. `walkers` and `samples` in the
`[emcee]` block set the chain length; `modules` and `likelihoods` in `[pipeline]`
set which datasets are used (drop `dgw` for an LSS-only run, for instance).

**Gravitational-wave events** — `data/gw/ligo_data.txt`, one event per row:

```
# d_gw_obs[Mpc]  z_obs   sigma_dgw[Mpc]  sigma_z  v_rms[km/s]
40.0            0.0099   11              0.0001   500      # GW170817
5300            0.438    2500            0.0001   500      # GW190521
```

Add a row to include a new detection. Distances are in **Mpc**, not Gpc.

## Outputs

`gwmg run` writes a plain-text chain to the `output/` directory, one row per
sample, with a header naming the columns. The last two columns are the prior and
posterior; `cosmological_parameters--*` and `horndeski_parameters--*` are the
sampled parameters. It is readable with `numpy.loadtxt` or `pandas`, so you can
make your own histograms:

```python
import numpy as np, matplotlib.pyplot as plt
d = np.loadtxt("output/gw_lss_horndeski.txt")
names = open("output/gw_lss_horndeski.txt").readline().lstrip("#").split()
aM = d[len(d)//2:, names.index("horndeski_parameters--parameters_smg__3")]  # drop burn-in
plt.hist(aM, bins=40, density=True)
print("alpha_M0 = %.2f +/- %.2f" % (aM.mean(), aM.std()))
```

CosmoSIS flushes the file periodically rather than continuously, so the chain
appears in chunks while running; `gwmg plot` says so if you plot too early.

## Usage reference

    gwmg info                                   # show paths and what resolves
    gwmg validate                               # self-check, needs no CosmoSIS
    gwmg run hi_class_test --test               # single evaluation
    gwmg run gw_lss_emcee --mpi 8               # exact chain
    gwmg run gw_lss_emulated                    # emulated chain (minutes)
    gwmg plot output/chain.txt --outdir plots   # corner plots
    gwmg plot a.txt:Exact:black b.txt:Emulated:red --outdir plots   # overlay two

## Validation

The pipeline is checked against Baker & Harrison (2020): the GW luminosity
distance (their eq. 2.12), the alpha_i proportional-to-Omega_Lambda ansatz
(eq. 2.14), and the lensing and peculiar-velocity error model (eqs. 3.9-3.11).
See `docs/validation.md`.

## Emulator

Running the exact pipeline is slow because every step solves the full
cosmological equations, so a full chain takes days. gwmg includes CosmoPower
neural-network emulators (arXiv:2106.03846) of the expensive physics, and the
same inference runs in about ten minutes:

![emulated vs exact](docs/emulator_vs_exact.png)

| | alpha_B0 | alpha_M0 | wall time |
|---|---|---|---|
| exact hi_class | 1.00 +/- 0.44 | 0.41 +/- 0.81 | days, 6 cores |
| **emulated** | **1.08 +/- 0.44** | **0.32 +/- 0.82** | **10 min, 1 core** |

The emulated pipeline recovers the exact Horndeski constraints to a few per cent
in the marginal widths. Getting there needed one design choice that is easy to
get wrong: the **CMB emulator has to include alpha_M and alpha_B as inputs**.
Emulating the CMB in LCDM only, on the argument that modified gravity barely
changes the temperature spectrum, degrades alpha_M by more than a factor of two
and biases alpha_B — the small effect that argument dismisses is exactly the
likelihood curvature that constrains alpha. The full methodology, including why
per-multipole accuracy metrics mislead, is in `docs/emulator.md` and
`docs/analysis_and_results.md`.

### What is emulated

| Quantity | Inputs | Accuracy |
|---|---|---|
| CMB TT C_ell (2 <= l <= 2600) | 8 params, **including alpha_M0, alpha_B0** | 0.17% median |
| Linear P(k), sigma_8 | 8 params | 0.1% median |
| Growth f-sigma_8(z), 0 < z < 1.5 | 8 params | 1.2% median |
| Distances, H(z), sound horizon, GW distance ratio | computed exactly, not emulated | 0.02-0.2% vs hi_class |

The background is analytic for an LCDM expansion, so only the perturbation
spectra are worth emulating.

### Setup

The emulator needs no separate environment: `environment.yml` already includes
CosmoPower, and hi_class is built against the same `numpy<1.25` that TensorFlow
requires, so `classy`, CosmoPower and CosmoSIS all coexist. Just point gwmg at
your trained models:

    export GWMG_EMU=/path/to/where/your/emulators/live

### Using the emulators directly

    from gwmg.emulator import Emulator

    emu = Emulator("emulators")     # dir with emu_cl_tt, emu_logpk, emu_fsigma8
    out = emu.predict(omega_m=0.315, h0=0.674, omega_b=0.049, n_s=0.965,
                      A_s=2.1e-9, tau=0.054, alpha_B0=1.0, alpha_M0=0.5)

    out.ell, out.cl_tt, out.dl_tt   # CMB TT
    out.k_h, out.pk, out.sigma8     # linear P(k) and sigma_8
    out.z, out.fsigma8              # growth
    out.z_bg, out.dgw_ratio         # d_L^GW / d_L^EM (the siren observable)

About 7 ms per call, against ~6.4 s for the equivalent hi_class evaluation.
Parameters outside the training box raise `ValueError`.

### Running the accelerated chain

    gwmg run gw_lss_emulated

`configs/gw_lss_emulated.ini` is identical to the exact `gw_lss_emcee.ini` except
that the hi_class module is swapped for `modules/emulator_interface.py`, so the
two are directly comparable.

### Training your own

Trained weights are not distributed (they are large and tied to a specific
hi_class build; an emulator trained on a different Boltzmann code introduces a
measurable bias). Three networks go into one model directory — `emu_cl_tt`,
`emu_logpk`, `emu_fsigma8`:

    # 1. training data (hours on ~6 cores). --smart-box keeps the full alpha
    #    priors but restricts the standard parameters to the posterior region.
    gwmg emu-gen -n 8000 --outdir training_set --smart-box --workers 6 --seed 1
    gwmg emu-gen -n 2000 --outdir test_set     --smart-box --workers 6 --seed 2
    python scripts/generate_growth.py -n 8000 --outdir training_growth --seed 3 --workers 6

    # 2. train (minutes each)
    gwmg emu-train training_set --model-dir emulators        # CMB + matter power
    python scripts/train_growth.py training_growth --model-dir emulators

    # 3. check accuracy per multipole
    gwmg emu-validate test_set --model-dir emulators --report accuracy.txt

    # 4. check the induced parameter bias
    python scripts/fisher_deriv_tt.py --out fisher_deriv.npz
    gwmg emu-bias --deriv fisher_deriv.npz --model-dir emulators

### Scope

Valid for the `propto_omega` ansatz with an LCDM expansion (w = -1),
alpha_T = alpha_H = 0, alpha_K fixed, TT only, massless neutrinos, and inside the
trained box (alpha_B0 in [-1, 3], alpha_M0 in [-1, 6], standard parameters a few
sigma around Planck). Anything beyond that needs retraining, which the scripts
above support. Full detail and limitations: `docs/emulator.md`.

## Status

The pipeline runs and reproduces the physics of the source paper, and the
emulator is validated end to end against it. Trained emulator weights are not
committed (they are large and tied to a specific hi_class build); the generation,
training and validation tooling to reproduce them is in `scripts/` and the
`gwmg emu-*` commands. See `docs/emulator.md` for scope and limitations.

## License

MIT. See `LICENSE`.
