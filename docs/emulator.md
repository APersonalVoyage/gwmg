# The gwmg emulator: accelerating the inference

The exact `gwmg` pipeline is slow because every likelihood evaluation calls
`hi_class` to solve the full cosmological equations, so a full MCMC runs for days.
The emulator replaces that expensive physics with neural networks
([CosmoPower](https://arxiv.org/abs/2106.03846)), and the same inference runs in
**about ten minutes**.

![emulated vs exact](emulator_vs_exact.png)

| | alpha_B0 | alpha_M0 | wall time |
|---|---|---|---|
| exact `hi_class` | 1.00 +/- 0.44 | 0.41 +/- 0.81 | days, 6 cores |
| **emulated (MG-aware CMB)** | **1.08 +/- 0.44** | **0.32 +/- 0.82** | **10 min, 1 core** |
| emulated (LCDM CMB) | 0.09 +/- 0.84 | 1.24 +/- 1.74 | 10 min, 1 core |

The emulated pipeline recovers the exact constraints on both Horndeski parameters,
with marginal widths agreeing to a few per cent.

## Using it directly

Parameters in, observables out, in about 7 ms (`hi_class` takes ~6.4 s for the
same call, so roughly 900x faster):

```python
from gwmg.emulator import Emulator

emu = Emulator("emulators")            # dir holding emu_cl_tt, emu_logpk, emu_fsigma8
out = emu.predict(omega_m=0.315, h0=0.674, omega_b=0.049, n_s=0.965,
                  A_s=2.1e-9, tau=0.054, alpha_B0=1.0, alpha_M0=0.5)

out.ell, out.cl_tt, out.dl_tt   # CMB TT: dimensionless C_ell, and D_ell in muK^2
out.k_h, out.pk, out.sigma8     # linear P(k) at z=0 [(Mpc/h)^3], and sigma_8
out.z, out.fsigma8              # growth rate combination f-sigma_8(z)
out.z_bg, out.dgw_ratio         # d_L^GW / d_L^EM, the standard-siren observable
out.alpha_mz, out.H_z, out.d_l  # alpha_M(z), H(z), luminosity distance
```

Accuracy against `hi_class` at that cosmology: D_ell(220) to 0.05%, sigma_8 to
0.35%, f-sigma_8(0) to 0.25%. Parameters outside the training box raise
`ValueError` (pass `check_box=False` to override, at your own risk). Models can
also be given individually with `Emulator(cmb=..., logpk=..., growth=...)`.

## Architecture: a hybrid, not a single black box

Each quantity is routed to the cheapest source that is accurate enough:

| Quantity | Source | Accuracy |
|---|---|---|
| CMB TT C_ell | neural network, **8 parameters incl. alpha_M, alpha_B** | 0.17% median |
| Matter P(k), sigma_8 | neural network, 8 parameters | 0.1% median |
| Growth f-sigma_8(z) | neural network, 8 parameters | 1.2% median |
| Distances, H(z), sound horizon | computed exactly, inline | 0.02-0.2% vs hi_class |
| GW luminosity-distance ratio | computed exactly, inline | 0.08% vs hi_class |

Because the expansion history is LCDM the background is analytic and costs
microseconds, so only the perturbation quantities are worth emulating. The
emulators are valid only inside their training box; the pipeline module rejects
out-of-box proposals, which is safe because the posterior has no mass there.

## The CMB emulator must be modified-gravity aware

The single most important design lesson. It is tempting to emulate the CMB in
LCDM only, on the argument that alpha_M and alpha_B barely change the primary
temperature spectrum (at the best fit the shift is only Delta chi^2 ~ 12 for
alpha = 0.5) and that the alpha constraints come from the sirens and from growth.

**That argument is wrong, and the end-to-end run is what exposes it.** That
Delta chi^2 *is* the constraint: it is the curvature of the likelihood in the
alpha direction, coming from how modified gravity lenses the temperature
spectrum. Emulating the CMB in LCDM throws it away, and the alpha posteriors
degrade badly (blue dashed curve above): alpha_M widens by more than a factor of
two and develops a tail to the prior edge, while alpha_B is biased from ~1.1 to
~0.1. Training the CMB emulator on all eight parameters recovers the exact result.

Note that this could not have been caught by component-level validation. The
emulator's parameter bias was measured at *fixed* alpha, where it looked
excellent (0.19 sigma). Only running the full inference, where alpha is
marginalised over, revealed the loss.

## Validating an emulator: which metric?

Choosing the accuracy metric is itself a modelling decision, and the obvious
choice is too lenient:

1. **Per-multipole cosmic variance is not enough.** The natural metric is the
   error per multipole in units of sigma_CV(l) = sqrt(2/(2l+1)) C_l. An emulator
   sitting at that floor can still shift the Planck chi^2 by hundreds, because
   the likelihood uses binned bandpowers whose errors are far tighter, and
   emulator error is correlated across scale so it does not average down within a
   band.
2. **The summed chi^2 conflates two effects.**
   Delta chi^2 = 2 (d - t)^T F delta + delta^T F delta. Away from the best fit
   the first term dominates and flips sign; only at the best fit, where the data
   residual is minimal, is delta^T F delta meaningful.
3. **The decisive metric is the induced parameter bias**,
   Delta theta = -Cov B^T F delta with Cov = (B^T F B)^-1, reported in units of
   the marginal uncertainty. Implemented as `gwmg.emulator.chi2.parameter_bias`
   and exposed as `gwmg emu-bias`.
4. **And even that is not sufficient on its own** — see the section above. A
   parameter you marginalise over must be validated by running the inference.

## Practical lessons

- **`hi_class` is not thread-safe.** Run with `OMP_NUM_THREADS=1` (~8.6 s per
  TT evaluation); multithreading crashes it. Parallelise over processes.
- **Stream training data to disk.** Accumulating tens of thousands of spectra in
  memory gets the generator OOM-killed. The generators write per-batch part files
  and merge at the end, which also makes them resumable.
- **Do not recycle pool workers.** `ProcessPoolExecutor(max_tasks_per_child=...)`
  deadlocks at the recycle boundary under the spawn start method.
- **CMB C_l needs a high learning rate.** A single-stage schedule at lr = 1e-3
  freezes on the initial plateau (loss stuck near 0.075, 11% error); lr = 1e-2
  breaks through to ~0.001. This was not specific to the CMB: the growth emulator
  froze the same way.
- **Filter both tails of the training data.** Unstable `hi_class` runs produce
  C_l that are finite but enormous (~1e286), *and* runs with zero or negative
  C_l which log10 maps to -300. Either kind dominates the MSE loss. In one case
  41 contaminated samples out of 29,000 (0.1%) pushed the validation loss from
  0.0035 to 7.35 and made an emulator look impossible to train.
- **Multi-stage learning-rate schedules degraded** rather than refined here, and
  CosmoPower saves the last stage, so a multi-stage run saves the worst model.

## Reproducing it

The emulator half needs CosmoPower and TensorFlow, whose dependency stack
conflicts with `classy`'s. Use two environments: one with `hi_class` to generate
training data, and one with CosmoSIS + CosmoPower (no `classy`) to train and run.
See `docs/install.md`.

```bash
# 1. training data (hi_class env). CMB uses the 8-parameter box; alpha varies.
python scripts/generate_lcdm_tt.py -n 8000 --outdir training_set --seed 1 --workers 6
python scripts/generate_growth.py  -n 8000 --outdir training_growth --seed 3 --workers 6

# 2. train (emulator env). --mg trains the 8-parameter, MG-aware CMB emulator.
python scripts/train_lcdm_tt.py training_set --model-dir emulators --mg --lr 0.01
python scripts/train_growth.py  training_growth --model-dir emulators_growth

# 3. component accuracy
gwmg emu-validate test_set --model-dir emulators --report accuracy.txt
python scripts/fisher_deriv_tt.py --out fisher_deriv.npz     # hi_class env
gwmg emu-bias --deriv fisher_deriv.npz --model-dir emulators # emulator env

# 4. the accelerated inference
gwmg run gw_lss_emulated
```

`configs/gw_lss_emulated.ini` is identical to the exact `gw_lss_emcee.ini` except
that the `hi_class` module is replaced by `modules/emulator_interface.py`, so the
two are directly comparable. Point `cmb_model`, `logpk_model` and `growth_model`
at your trained emulators; the module reads each network's declared parameter
list, so it works with either a 6-parameter (LCDM) or 8-parameter (MG-aware) CMB
emulator.

## Scope and limitations

The trained emulators are valid for: the `propto_omega` ansatz with an LCDM
expansion (w = -1), alpha_T = alpha_H = 0 and alpha_K fixed; TT only (no TE, EE
or lensing reconstruction); massless neutrinos (N_ur = 3.046); alpha_B0 in
[-1, 3] and alpha_M0 in [-1, 6]; and standard parameters in a box a few sigma
wide around the Planck values. They are trained against one `hi_class` build, and
self-consistency with that build matters: an emulator trained on a *different*
Boltzmann code left a ~0.5 sigma bias in H0 purely from the code mismatch.

Retraining is what generalises this to a wider box, more of Horndeski, or another
solver; the generation, training and validation tooling is the reusable part.
