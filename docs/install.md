# Install

`gwmg` has two layers:

1. the **pure Python package** (API + CLI + plotting) — installs with pip;
2. the **heavy stack** it wraps — CosmoSIS 3 and hi_class — which is not
   pip-installable and is built separately.

## 1. The gwmg package

```bash
cd gwmg_pkg
pip install -e .[dev]
gwmg validate     # self-check (needs no CosmoSIS)
pytest            # run the test suite
```

## 2. CosmoSIS 3 + hi_class (the compute stack)

Use conda/mamba (recommended: Miniforge). CosmoSIS is on conda-forge; hi_class is
built from source (it is a CLASS fork with a compiled `classy` wrapper).

```bash
conda env create -f environment-hiclass.yml     # or the explicit command below
conda activate gw-hiclass
cosmosis-build-standard-library          # fetch + build the standard library
source cosmosis-configure

# hi_class
git clone https://github.com/miguelzuma/hi_class_public.git
cd hi_class_public && make && python setup.py build && pip install .
# note: with numpy>=2, delete the dead `ctypedef np.int_t DTYPE_i` line in
# python/classy.pyx before `make` if the Cython build fails.
export HICLASS_DIR=$PWD
```

<details>
<summary>Equivalent explicit conda command</summary>

```bash
conda create -n gw-hiclass -c conda-forge python=3.11 numpy scipy cython gsl \
    compilers cosmosis cosmosis-build-standard-library camb emcee getdist
```
</details>

## 3. The emulator environment (optional)

The emulator needs CosmoPower, which pulls in TensorFlow and pins `numpy<1.25`,
while a modern `classy` build wants numpy 2.x. The two therefore cannot share an
environment. This is not a problem in practice: when the emulator is in use
hi_class is not needed, and training data is passed between the two environments
as files on disk.

```bash
conda env create -f environment-emulator.yml
conda activate gw-emu
```

This environment contains CosmoSIS as well, so the accelerated pipeline
(`gwmg run gw_lss_emulated`) runs here too.

## 4. Environment variables

```bash
export COSMOSIS_SRC_DIR=/parent/of/cosmosis-standard-library   # the *parent* dir
export HICLASS_DIR=/path/to/hi_class_public                    # holds external/bbn/sBBN.dat
export OMP_NUM_THREADS=1                                       # hi_class is not thread-safe
# emulator only: parent directory of your trained model directories
export GWMG_EMU=/path/to/where/your/emulators/live
# PIPELINE_DIR is set automatically by `gwmg run`
```

Check everything resolves:

```bash
gwmg info
gwmg run hi_class_test --test
```

## 5. Trained emulator weights

Trained networks are **not** distributed with the repository: they are large and
are tied to a specific hi_class build, and an emulator trained against a
different Boltzmann code introduces a measurable bias (see `docs/emulator.md`).
Regenerate them with the scripts in `scripts/`; the commands are listed in
`docs/emulator.md`. Generating the training sets takes a few hours on six cores
and training a few minutes each.
