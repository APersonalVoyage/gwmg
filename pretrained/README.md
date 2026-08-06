# Pre-trained emulators

Ready to use — no training required:

```python
from gwmg.emulator import Emulator
emu = Emulator()                     # loads these models by default
out = emu.predict(omega_m=0.315, h0=0.674, omega_b=0.049, n_s=0.965,
                  A_s=2.1e-9, tau=0.054, alpha_B0=1.0, alpha_M0=0.5)
```

or for the accelerated pipeline:

```bash
export GWMG_EMU=$(python -c "import gwmg,os;print(os.path.dirname(gwmg.__file__))")
gwmg run gw_lss_emulated
```

| File | Emulates | Inputs | Median accuracy |
|---|---|---|---|
| `emu_cl_tt.pkl` | lensed CMB TT `C_ell`, 2 <= l <= 2600 | 8 | 0.17% (0.055 sigma_CV) |
| `emu_logpk.pkl` | linear `log10 P(k)`, 200 modes to k = 1 h/Mpc, z = 0 | 8 | 0.1% |
| `emu_fsigma8.pkl` | `f-sigma_8(z)`, 31 points over 0 <= z <= 1.5 | 8 | 1.2% |

Inputs in all cases, in this order:
`omega_m, h0, omega_b, n_s, A_s, tau, alpha_B0, alpha_M0`.

## What they were trained on

Provenance matters here more than usual. We measured that a CMB emulator trained
against a *different* Boltzmann code leaves a 0.53 sigma bias in H0 in this
pipeline, purely from code mismatch (see `docs/emulator.md`). These models were
trained on:

- **hi_class** (Zumalacarregui et al. 2017), the public v2 release, built from
  https://github.com/miguelzuma/hi_class_public
- **Gravity model** `propto_omega`: `alpha_i(a) = alpha_i0 * Omega_DE(a)`, with a
  LCDM expansion history (`w = -1`), `alpha_T = alpha_H = 0`, `alpha_K` fixed
- **Settings**: `T_cmb = 2.726`, `N_ur = 3.046` (massless neutrinos),
  `omega_nu h^2 = 0.00083` held fixed, lensed `C_ell`, `l_max = 2600`
- **Training set**: ~29,000 hi_class evaluations (CMB and P(k)), ~7,200 for
  growth, Latin-hypercube sampled

## Validity box

Predictions outside this box raise `ValueError`. The emulators are *not* valid
beyond it, and the pipeline module rejects such proposals.

| Parameter | Range |
|---|---|
| `omega_m` | 0.24 – 0.36 |
| `h0` | 0.61 – 0.76 |
| `omega_b` | 0.041 – 0.054 |
| `n_s` | 0.93 – 1.00 |
| `A_s` | 1.7e-9 – 2.5e-9 |
| `tau` | 0.02 – 0.12 |
| `alpha_B0` | -1.0 – 3.0 |
| `alpha_M0` | -1.0 – 6.0 |

The standard parameters are deliberately restricted to a few sigma around the
Planck values — that is where the posterior lives, and concentrating the training
budget there is what makes the accuracy achievable.

## Verifying them against your own build

If your hi_class differs from ours, check before trusting these weights. It takes
about ten minutes: 13 hi_class evaluations for the derivatives, then a Fisher
analysis of the induced parameter bias.

```bash
python scripts/fisher_deriv_tt.py --out fisher_deriv.npz
gwmg emu-bias --deriv fisher_deriv.npz --model-dir pretrained
```

Ours report `max |bias/sigma| = 0.19`. Anything comparable is fine; anything much
larger means your build differs enough that you should retrain (see the README).

## Reproducing them

The generation and training commands are in the README under "Training your own".
Regenerating the training sets takes a few hours on six cores; training takes
minutes.
