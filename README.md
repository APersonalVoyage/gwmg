# gwmg

[![CI](https://github.com/APersonalVoyage/gwmg/actions/workflows/ci.yml/badge.svg)](https://github.com/APersonalVoyage/gwmg/actions/workflows/ci.yml)
[![Docs](https://github.com/APersonalVoyage/gwmg/actions/workflows/docs.yml/badge.svg)](https://apersonalvoyage.github.io/gwmg/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/APersonalVoyage/gwmg/blob/main/notebooks/getting_started.ipynb)

**[Documentation](https://apersonalvoyage.github.io/gwmg/)** •
[Overview](#overview) •
[Installation](#installation) •
[Getting Started](#getting-started) •
[How It Works](#how-it-works) •
[Physics](https://apersonalvoyage.github.io/gwmg/physics/) •
[Running the Pipeline](#running-the-pipeline) •
[Emulator](#emulator) •
[Trained Models](#trained-models) •
[Training](#training) •
[Validation](#validation) •
[Support](#contributing-and-support) •
[Citation](#citation)

## Overview

**gwmg** constrains modified gravity on cosmological scales using
gravitational-wave standard sirens together with large-scale-structure and CMB
data. It measures the Horndeski functions `alpha_M` and `alpha_B`, which
parametrise deviations from General Relativity in the propagation of
gravitational waves and in the growth of cosmic structure, implementing the
framework of [Baker & Harrison (2020)](https://arxiv.org/abs/2007.13791).

The package supplies the layer that is specific to this measurement and is
usually the part not released: the GW standard-siren likelihood, the interface
between `hi_class` and `CosmoSIS`, the GW luminosity-distance model, and
ready-to-run configurations for GW170817 and GW190521.

It also provides **neural-network emulators** of the expensive physics, built
with [CosmoPower](https://arxiv.org/abs/2106.03846), which reduce a multi-day
inference to about ten minutes. Trained models are included, so the emulator
works with no training step.

![emulated vs exact](docs/emulator_vs_exact.png)

| | alpha_B0 | alpha_M0 | wall time |
|---|---|---|---|
| exact `hi_class` | 1.00 +/- 0.44 | 0.41 +/- 0.81 | days, 6 cores |
| **emulated** | **1.08 +/- 0.44** | **0.32 +/- 0.82** | **10 min, 1 core** |

gwmg began as a rebuild of an MSc thesis pipeline (Python 2.7, a lost interface
module, hardcoded cluster paths) and is now a modernised, tested, installable
version. Two errors in the original analysis were identified and corrected along
the way: a factor in the GW luminosity distance, and a units error in the input
catalogue.

## Installation

gwmg has two layers: the pure Python package, which installs with pip, and the
CosmoSIS 3 + hi_class stack it drives, which is built separately.

**The package alone** — enough for the GW likelihood, the CLI, plotting, the
emulator API and the examples:

```bash
pip install git+https://github.com/APersonalVoyage/gwmg.git
```

or from a clone, which also gives you the configs, data, tests and trained
emulators:

```bash
git clone https://github.com/APersonalVoyage/gwmg.git
cd gwmg
pip install -e .[dev]
gwmg validate          # self-check, needs no CosmoSIS
pytest
```

**The full stack** lives in one conda environment. Everything — CosmoSIS,
CosmoPower and hi_class — coexists provided hi_class is built against the same
`numpy<1.25` that TensorFlow requires:

```bash
conda env create -f environment.yml
conda activate gwmg
cosmosis-build-standard-library
source cosmosis-configure

git clone https://github.com/miguelzuma/hi_class_public.git
cd hi_class_public && make ; cd python
CC=clang python setup.py build_ext --inplace      # CC=gcc on Linux
pip install --no-build-isolation .
cd ../..

export COSMOSIS_SRC_DIR=$PWD          # parent of cosmosis-standard-library
export HICLASS_DIR=$PWD/hi_class_public
export OMP_NUM_THREADS=1              # hi_class is not thread-safe
```

See [`docs/install.md`](docs/install.md) for the detail and a troubleshooting
section. Check it resolved with `gwmg info`.

## Getting Started

Minimal working examples. The first two need only the pip install; the third
needs the full stack.

**Emulate the observables for a modified-gravity cosmology** (~7 ms, against
~6.4 s for the equivalent hi_class call):

```python
from gwmg.emulator import Emulator

emu = Emulator()                     # uses the trained models in pretrained/
out = emu.predict(omega_m=0.315, h0=0.674, omega_b=0.049, n_s=0.965,
                  A_s=2.1e-9, tau=0.054, alpha_B0=1.0, alpha_M0=0.5)

out.ell, out.cl_tt, out.dl_tt   # CMB TT: C_ell, and D_ell in muK^2
out.k_h, out.pk, out.sigma8     # linear P(k) at z=0, and sigma_8
out.z, out.fsigma8              # growth rate combination f-sigma_8(z)
out.z_bg, out.dgw_ratio         # d_L^GW / d_L^EM, the standard-siren observable
out.alpha_mz, out.H_z, out.d_l  # alpha_M(z), H(z), luminosity distance
```

Parameters outside the training box raise `ValueError`; pass `check_box=False`
to override.

**Evaluate the GW standard-siren likelihood** (pure numpy/scipy):

```python
import numpy as np, os
from gwmg import gw_log_likelihood, load_events, pipeline_dir

events = load_events(os.path.join(pipeline_dir(), "data", "gw", "ligo_data.txt"))
d_gw_obs, z_obs, sigma_dgw, sigma_z, v_rms = events

z = np.linspace(0, 1, 2000)
out = emu.predict(omega_m=0.315, h0=0.674, omega_b=0.049, n_s=0.965,
                  A_s=2.1e-9, tau=0.054, alpha_B0=0.0, alpha_M0=0.5)
logL = gw_log_likelihood(
    out.z_bg, out.d_l, out.H_z / 299792.458, out.dgw_ratio,
    d_gw_obs=d_gw_obs, z_obs=z_obs, sigma_dgw=sigma_dgw,
    sigma_z=sigma_z, v_rms=v_rms)
```

**Run the full inference:**

```bash
gwmg run gw_lss_emulated                     # emulated, ~10 minutes
gwmg run gw_lss_emcee --mpi 8                # exact, days
gwmg plot output/gw_lss_emulated.txt --outdir plots
```

**A demo notebook** walks through the whole idea — the data, the model, the
likelihood, and the resulting constraint — and runs in the browser with no
install: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/APersonalVoyage/gwmg/blob/main/notebooks/getting_started.ipynb)

Two runnable scripts live in [`examples/`](examples):
`constrain_alpha_M.py` (a complete miniature analysis of the real GW170817 and
GW190521 data, no CosmoSIS needed) and `emulator_speed.py` (emulator versus exact
hi_class, accuracy and timing).

## How It Works

gwmg is a thin layer over [CosmoSIS](https://cosmosis.readthedocs.io), which runs
Bayesian inference as a **pipeline of modules**. Modules run in order and
communicate through a *datablock*, a typed key-value store namespaced by section:
one module writes `distances/d_l`, a later one reads it. Modules never call each
other.

One likelihood evaluation:

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

The theory module is the expensive one and the only one the emulator replaces,
which is why the exact and emulated runs are directly comparable.

A **likelihood** is a number saying how well one parameter set reproduces the
data. A **chain** is what the sampler produces by proposing thousands of
parameter sets and keeping them in proportion to their likelihood, so the density
of samples is the posterior.

For what each parameter means, how the likelihood and posterior are computed, and
how to read the chains and corner plots, see
[Physics and method](https://apersonalvoyage.github.io/gwmg/physics/) in the
documentation.

## Running the Pipeline

A run is defined by two ini files and a data file, and nothing else:

| File | Controls |
|---|---|
| `configs/gw_lss_emcee.ini` | which modules run, which likelihoods count, sampler settings |
| `configs/values_horndeski.ini` | the parameters: fixed values, or `min start max` to sample |
| `data/gw/ligo_data.txt` | the gravitational-wave events |

The bundled copies live inside the installed package, so copy them somewhere
writable and point `gwmg run` at your copy:

```bash
python -c "import gwmg, shutil; shutil.copytree(gwmg.pipeline_dir(), 'myrun')"
gwmg run myrun/configs/gw_lss_emcee.ini --pipeline-dir myrun
```

**Priors** — `configs/values_horndeski.ini`. Each line is `min start max`; a
single value fixes the parameter.

```ini
[horndeski_parameters]
parameters_smg__2  = -1.   0.41   3.     ; alpha_B0, prior U(-1, 3)
parameters_smg__3  = -1.   0.01   6.     ; alpha_M0, prior U(-1, 6)
```

Replacing the last line with `parameters_smg__3 = 0.` fixes alpha_M at zero, a GR
run. The standard cosmological parameters are in `[cosmological_parameters]`.

**Sampler and datasets** — `configs/gw_lss_emcee.ini`: `walkers` and `samples`
set the chain length; `modules` and `likelihoods` set which datasets are used
(drop `dgw` for an LSS-only run).

**GW events** — `data/gw/ligo_data.txt`, one per row. Add a row for a new
detection; distances are in **Mpc**, not Gpc.

```
# d_gw_obs[Mpc]  z_obs   sigma_dgw[Mpc]  sigma_z  v_rms[km/s]
40.0            0.0099   11              0.0001   500      # GW170817
5300            0.438    2500            0.0001   500      # GW190521
```

**Outputs** — a plain-text chain in `output/`, one row per sample, with a header
naming the columns. Readable with `numpy.loadtxt` or `pandas`:

```python
import numpy as np
f = "output/gw_lss_horndeski.txt"
d = np.loadtxt(f)
names = open(f).readline().lstrip("#").split()
aM = d[len(d)//2:, names.index("horndeski_parameters--parameters_smg__3")]
print("alpha_M0 = %.2f +/- %.2f" % (aM.mean(), aM.std()))
```

`gwmg plot` writes corner plots (`corner_all.png`, `alpha_B_alpha_M.png`) and can
overlay chains: `gwmg plot a.txt:Exact:black b.txt:Emulated:red --outdir plots`.
CosmoSIS flushes the chain in chunks, so it appears gradually while running.

## Emulator

Every likelihood evaluation of the exact pipeline calls hi_class, so a converged
chain takes days. The emulator replaces that step with neural networks and the
same inference runs in about ten minutes, recovering the exact Horndeski
constraints to a few per cent in the marginal widths (see the table in
[Overview](#overview)).

| Quantity | Source | Median accuracy |
|---|---|---|
| CMB TT `C_ell`, 2 <= l <= 2600 | network, 8 params **incl. alpha_M0, alpha_B0** | 0.17% |
| Linear `P(k)`, sigma_8 | network, 8 params | 0.1% |
| Growth `f-sigma_8(z)` | network, 8 params | 1.2% |
| Distances, H(z), sound horizon, GW distance ratio | computed exactly | 0.02–0.2% |

The background is analytic for a LCDM expansion, so only the perturbation spectra
are worth emulating.

**The CMB emulator must include alpha_M and alpha_B.** It is tempting to emulate
the CMB in LCDM only, since modified gravity barely changes the primary
temperature spectrum. That is wrong: the small effect it discards is the lensing
response, which carries much of the likelihood curvature in the alpha directions.
Doing so broadens alpha_M0 by more than a factor of two and biases alpha_B0. This
could not be caught by component-level validation — the LCDM emulator passes at
0.19 sigma with alpha held fixed — and only appears in the full inference.

Full methodology, including why per-multipole accuracy metrics mislead, is in
[`docs/emulator.md`](docs/emulator.md) and
[`docs/analysis_and_results.md`](docs/analysis_and_results.md).

## Trained Models

Trained emulators ship in [`pretrained/`](pretrained) (16 MB), so no training is
needed to use them. `Emulator()` loads them by default, and
`configs/gw_lss_emulated.ini` points at them.

Provenance matters here: an emulator trained against a *different* Boltzmann code
leaves a 0.53 sigma bias in H0 in this pipeline. See
[`pretrained/README.md`](pretrained/README.md) for the exact hi_class version and
settings they were trained under, their validity box, and how to verify them
against your own build:

```bash
python scripts/fisher_deriv_tt.py --out fisher_deriv.npz
gwmg emu-bias --deriv fisher_deriv.npz --model-dir pretrained
```

Ours report `max |bias/sigma| = 0.19`.

## Training

Only needed if you change the hi_class build or settings, widen the parameter
box, or move to a different gravity parametrisation.

```bash
# 1. training data (hours on ~6 cores). --smart-box keeps the full alpha priors
#    but restricts the standard parameters to the posterior region.
gwmg emu-gen -n 8000 --outdir training_set --smart-box --workers 6 --seed 1
gwmg emu-gen -n 2000 --outdir test_set     --smart-box --workers 6 --seed 2
python scripts/generate_growth.py -n 8000 --outdir training_growth --seed 3 --workers 6

# 2. train (minutes each)
gwmg emu-train training_set --model-dir emulators        # CMB + matter power
python scripts/train_growth.py training_growth --model-dir emulators

# 3. per-multipole accuracy
gwmg emu-validate test_set --model-dir emulators --report accuracy.txt

# 4. induced parameter bias
python scripts/fisher_deriv_tt.py --out fisher_deriv.npz
gwmg emu-bias --deriv fisher_deriv.npz --model-dir emulators
```

**Scope.** The emulators are valid for the `propto_omega` ansatz with a LCDM
expansion (`w = -1`), `alpha_T = alpha_H = 0`, `alpha_K` fixed, TT only, massless
neutrinos, and inside the trained box (`alpha_B0` in [-1, 3], `alpha_M0` in
[-1, 6], standard parameters a few sigma around Planck).

## Validation

The pipeline is checked against Baker & Harrison (2020): the GW luminosity
distance (their eq. 2.12), the `alpha_i` proportional-to-Omega_Lambda ansatz
(eq. 2.14), and the lensing and peculiar-velocity error model (eqs. 3.9–3.11).
See [`docs/validation.md`](docs/validation.md).

The emulator is validated at three levels — per-multipole accuracy, the
Fisher-matrix parameter bias (`gwmg emu-bias`), and end-to-end against the exact
pipeline. See [`docs/emulator.md`](docs/emulator.md).

## Contributing and Support

Bug reports and feature requests are welcome via the
[issue tracker](https://github.com/APersonalVoyage/gwmg/issues); contributions
via pull requests are most welcome.

## Citation

If you use gwmg, please cite the software paper (in review) along with the
underlying codes: [CosmoSIS](https://arxiv.org/abs/1409.3409),
[hi_class](https://arxiv.org/abs/1605.06102),
[CLASS](https://arxiv.org/abs/1104.2933),
[CosmoPower](https://arxiv.org/abs/2106.03846),
[emcee](https://arxiv.org/abs/1202.3665), and the analysis framework of
[Baker & Harrison (2020)](https://arxiv.org/abs/2007.13791).

```bibtex
@software{gwmg,
  author = {Karkola, Abhishek},
  title  = {gwmg: a reproducible pipeline and emulator for testing modified
            gravity with gravitational-wave standard sirens},
  year   = {2026},
  url    = {https://github.com/APersonalVoyage/gwmg}
}
```

## License

MIT. See [LICENSE](LICENSE).
