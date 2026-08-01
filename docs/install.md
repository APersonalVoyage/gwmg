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
conda create -n gw-hiclass -c conda-forge python=3.11 numpy scipy cython gsl \
    compilers cosmosis cosmosis-build-standard-library emcee getdist
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

## 3. Environment variables used by `gwmg run`

```bash
export COSMOSIS_SRC_DIR=/parent/of/cosmosis-standard-library   # the *parent* dir
export HICLASS_DIR=/path/to/hi_class_public                    # holds external/bbn/sBBN.dat
# PIPELINE_DIR is set automatically by `gwmg run`
```

Check everything resolves:

```bash
gwmg info
gwmg run hi_class_test --test
```
