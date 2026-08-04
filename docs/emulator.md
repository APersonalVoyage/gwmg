# The gwmg emulator: accelerating the inference

The exact `gwmg` pipeline is slow because every likelihood evaluation calls
`hi_class` to solve the full set of cosmological equations (order 1 to 10 seconds
per step), so a full MCMC runs for hours to days. The emulator replaces the
expensive physics with neural networks so the same analysis runs in minutes,
using the [CosmoPower](https://arxiv.org/abs/2106.03846) framework.

This document records the design, the validation methodology, and the results.
It is deliberately detailed on validation, because the central lesson of this
work is that the obvious accuracy metric is misleading and the honest test lives
further downstream than one would like.

## Architecture: a hybrid, not a single black box

Each quantity is routed to the cheapest source that is accurate enough:

| Quantity | Source | Why |
|---|---|---|
| CMB TT C_ell | Neural-network emulator | Expensive, and the demanding accuracy target |
| Matter P(k), sigma_8 | Neural-network emulator (`logpk`) | Expensive; 0.1% accuracy, far below RSD/BAO error bars |
| Distances, H(z) | Computed exactly, inline | An LCDM Friedmann integral; microseconds |
| GW luminosity-distance ratio | Computed exactly, inline | One cheap integral of alpha_M |
| Growth f-sigma_8 | Open (emulate or compute) | Cost sits between the two |

The modified-gravity dials (alpha_M, alpha_B) enter through growth, the
background, and the GW distance, all of which are handled exactly or by the
`logpk` emulator. They have a negligible effect on the primary CMB TT spectrum
(see below), so the CMB emulator is trained on LCDM.

## The accuracy problem

A CMB emulator for a Planck likelihood needs to be accurate to a small fraction
of a percent. The subtlety is how you measure "accurate enough".

**Per-multipole cosmic variance is too lenient.** The natural metric is the
emulator error per multipole in units of cosmic variance, sigma_CV(l) =
sqrt(2/(2l+1)) C_l. An early emulator sat right at that floor (CV of order 1)
and looked fine. But the Planck TT chi-squared computed from that emulator was
off by hundreds. Cosmic variance is defined per single multipole; Planck
measures *binned bandpowers*, each averaging tens of multipoles, so its error
bars are far tighter than single-multipole cosmic variance. And emulator error
is *correlated across scale*, so it does not average down inside a bin the way
random noise would. A per-multipole metric cannot see either effect.

**Even the summed chi-squared is not the whole story.** Write
Delta_chi2 = 2 (data - theory) . F . delta + delta . F . delta, where delta is
the emulator error and F the Planck Fisher matrix. The first term scales with
the data residual and flips sign, so away from the best fit it dominates and is
misleading. Evaluated *at* the best fit, where the residual is minimal, the
penalty delta . F . delta is the honest number.

**The arbiter: parameter bias.** A constant chi-squared offset biases nothing;
it shifts every point on the likelihood equally. Only the parameter dependence
of the error matters. The induced shift is

    Delta_theta = - Cov . B^T . F . delta

with B the binned theory derivatives and Cov = (B^T F B)^-1 the parameter
covariance. Reported in units of the Planck-TT sigma, this is the quantity that
decides whether an emulator is usable. It is implemented in
`gwmg.emulator.chi2.parameter_bias` and exposed as `gwmg emu-bias`.

## Results

Three emulators, evaluated by the same Fisher-bias test at the Planck best-fit:

| Emulator | Per-multipole error (median) | Max parameter bias |
|---|---|---|
| Bespoke, under-trained (lr=1e-3) | 11.7% | 11.9 sigma |
| Public pre-trained LCDM (CosmoPower CP_paper) | -- | 0.53 sigma |
| **Self-consistent, this work** | **0.17%** | **0.192 sigma** |

The public pre-trained emulator is accurate but leaves a ~0.5 sigma bias in the
Hubble constant. That bias is *not* the network: it is a mismatch between the
Boltzmann code the public emulator was trained on and the slightly different
`hi_class` in this pipeline (a difference of the same size shows up from the
neutrino settings alone). Training the emulator on `hi_class` itself removes the
mismatch by construction and brings the bias down to 0.19 sigma, concentrated in
the A_s / tau amplitude direction. That is below the ~0.2 sigma systematic
tolerance typical analyses accept, and it is sample-limited, so more training
data pushes it further.

### Practical lessons

- **hi_class is not thread-safe.** It must be run with `OMP_NUM_THREADS=1`
  (about 8.6 s per TT evaluation); multithreading crashes it.
- **Stream training data to disk.** Accumulating tens of thousands of spectra in
  memory gets the generator OOM-killed; the generator writes per-batch part
  files and merges at the end (also making it resumable).
- **Do not recycle pool workers.** `ProcessPoolExecutor(max_tasks_per_child=...)`
  deadlocks at the recycle boundary under the spawn start method.
- **CMB C_l needs a high learning rate.** Because C_l spans many orders of
  magnitude, a single-stage schedule at lr=1e-3 freezes on the initial plateau
  (loss stuck near 0.075, 11% error); lr=1e-2 breaks through to loss ~0.0009.

## Reproducing it

The emulator half runs in a separate environment from `hi_class` (the CosmoPower
and classy dependency stacks conflict), communicating through files on disk.

    # 1. training + test sets (gw-hiclass env): 6 standard params, alpha=0
    python scripts/generate_lcdm_tt.py -n 8000 --outdir training_set_lcdm --seed 1 --workers 6
    python scripts/generate_lcdm_tt.py -n 2000 --outdir test_set_lcdm     --seed 2 --workers 6

    # 2. train the emulator (gw-emu env)
    python scripts/train_lcdm_tt.py training_set_lcdm --model-dir emulators_lcdm

    # 3. per-multipole accuracy (gw-emu env)
    gwmg emu-validate test_set_lcdm --model-dir emulators_lcdm --report accuracy.txt

    # 4. Fisher-bias: derivative spectra (gw-hiclass), then the bias (gw-emu)
    python scripts/fisher_deriv_tt.py --out fisher_deriv.npz
    gwmg emu-bias --deriv fisher_deriv.npz --model-dir emulators_lcdm

## Limitations and next steps

- The CMB emulator is validated as a component. The natural next step is the
  end-to-end demonstration: run the full MCMC with the emulator in place, show
  it reproduces the exact-pipeline constraints on alpha_M and alpha_B, and
  quantify the speedup. That is the result a paper turns on.
- Decide the growth (f-sigma_8) route: emulate or compute directly.
- Wire the emulator into the sampler as a `hi_class` drop-in, with out-of-box
  proposals rejected (the emulator is only valid inside its training box).
- Pushing the bias below 0.1 sigma is a matter of more training data.
