# Install

`gwmg` has two layers:

1. the **pure Python package** (API + CLI + plotting) — installs with pip;
2. the **heavy stack** it wraps — CosmoSIS 3, hi_class, and optionally CosmoPower
   — which is not pip-installable and is built separately.

Everything lives in **one conda environment**. The only real constraint is numpy:
CosmoPower pulls in TensorFlow, which pins `numpy<1.25`, and hi_class's `classy`
wrapper is a compiled extension that must be built against the same numpy. Build
classy inside this environment and the whole stack coexists.

## 1. The gwmg package alone

If you only want the GW likelihood, CLI and plotting (no CosmoSIS needed):

```bash
pip install -e .[dev]
gwmg validate     # self-check
pytest            # test suite
```

## 2. The full stack

Use conda/mamba (recommended: Miniforge).

```bash
# environment: CosmoSIS, CosmoPower, build tools, numpy<1.25
conda env create -f environment.yml
conda activate gwmg

# CosmoSIS standard library
cosmosis-build-standard-library
source cosmosis-configure

# hi_class, built against THIS environment's numpy
git clone https://github.com/miguelzuma/hi_class_public.git
cd hi_class_public
make                                        # builds the C library
cd python
CC=clang python setup.py build_ext --inplace   # use CC=gcc on Linux
pip install --no-build-isolation .          # isolation breaks the old setup.py
cd ../..
```

Notes on the hi_class build:

- `make` will fail at its final "classy" step if it cannot find Cython on the
  bare `python` it invokes; that is harmless, since the C library is already
  built by then and the two commands above build the wrapper properly.
- If the Cython build errors on `ctypedef np.int_t DTYPE_i` in
  `python/classy.pyx`, delete that line — it is dead and invalid on newer numpy.

Check it worked:

```bash
python -c "import classy, cosmopower, cosmosis; print('all three OK')"
```

## 3. Environment variables

```bash
export COSMOSIS_SRC_DIR=/parent/of/cosmosis-standard-library   # the *parent* dir
export HICLASS_DIR=/path/to/hi_class_public                    # holds external/bbn/sBBN.dat
export OMP_NUM_THREADS=1                                       # hi_class is not thread-safe
export GWMG_EMU=/path/to/your/emulators                        # emulator runs only
# PIPELINE_DIR is set automatically by `gwmg run`
```

Check everything resolves:

```bash
gwmg info
gwmg run hi_class_test --test
```

## 4. Troubleshooting

**`ModuleNotFoundError: No module named 'classy'` after building** — the wrapper
was probably installed into a different Python. Build and install it with the
same interpreter as the environment (`which python` should point inside it).

**`pip install .` fails in `hi_class_public/python`** — pip's build isolation
does not see numpy/Cython. Use `pip install --no-build-isolation .`.

**`gwmg plot` fails with `cannot import name 'Sentinel' from 'typing_extensions'`**
— ChainConsumer pulls in pydantic, which needs a recent version:
`pip install -U typing_extensions`.

**`ModuleNotFoundError: No module named 'camb'`** — the CosmoSIS consistency
module imports camb even when unused: `conda install -c conda-forge camb`.

**Chain output looks empty while running** — CosmoSIS flushes every `nsteps`
samples, so the file appears in chunks. Wait for the first flush.

**Segfaults or hangs when generating training data** — hi_class is not
thread-safe. Keep `OMP_NUM_THREADS=1` and parallelise over processes.

## 5. Trained emulator weights

Trained networks **are** included, in `pretrained/` (16 MB for the three
models), so the emulator works with no training step. See `pretrained/README.md`
for what they were trained on, their validity box, and how to verify them against
your own hi_class build with `gwmg emu-bias`.

Retrain only if you change the hi_class build or its settings, widen the
parameter box, or move to a different gravity parametrisation; the commands are
in the README and `docs/emulator.md`. Regenerating the training sets takes a few
hours on six cores, and training a few minutes each.
