# gwmg emulator: analysis and results

The longer methods-and-results companion to the short software paper
(`paper.md`). See `emulator.md` for the practical guide.

## Analysis

**Emulation strategy.** The pipeline is accelerated by a hybrid scheme rather
than a single emulator. The expensive perturbation quantities (the CMB
temperature spectrum C_l^TT, the matter power spectrum P(k), and the growth rate
combination f-sigma_8) are emulated with neural networks; the cheap background
(distances, H(z), the sound horizon) and the gravitational-wave
luminosity-distance modification are evaluated exactly at each step, since the
LCDM expansion history makes them analytic. The exact background agrees with
`hi_class` to 0.02-0.2%, and the GW distance ratio to 0.08%.

**Training data.** Parameters are drawn by Latin-hypercube sampling over a box
that encloses the posterior: standard parameters a few sigma wide around the
Planck values, and the full priors on alpha_B0 and alpha_M0. Restricting the
standard parameters concentrates the training points where the chain actually
goes, and as a side effect raises the `hi_class` stability success rate from
about 50% to 90%, since near-fiducial backgrounds keep far more alpha
combinations stable. Spectra are computed with `hi_class` run single-threaded
(it is not thread-safe) and parallelised across processes.

**Emulators.** Each is a fully-connected network (4 x 512) mapping parameters to
the spectrum, trained with CosmoPower. Single-stage learning-rate schedules are
used, because multi-stage schedules degraded the later low-learning-rate stages
and CosmoPower retains the last stage. The CMB emulator is trained on all eight
parameters, including alpha_M0 and alpha_B0 (see Results).

**Validation.** Three levels, in increasing order of stringency:

1. *Per-multipole*, in units of cosmic variance sigma_CV(l) = sqrt(2/(2l+1)) C_l.
   Too lenient: the Planck likelihood uses binned bandpowers with much tighter
   errors, and emulator error is correlated across scale so it does not average
   down within a band.
2. *Likelihood-level*, the induced parameter bias
   Delta theta = -Cov B^T F delta, with B the binned theory derivatives,
   F the Planck Fisher matrix and Cov = (B^T F B)^-1, evaluated at the best fit
   and reported in units of the marginal uncertainty (`gwmg emu-bias`).
   Necessary, but measured at fixed alpha.
3. *End-to-end*, running the full MCMC with the emulator in place and comparing
   the marginal posteriors to the exact pipeline. This is the only level that
   tests a parameter which is marginalised over.

## Results

**Component accuracy.** The CMB emulator reaches a median per-multipole error of
0.17% (0.055 in units of cosmic variance) and, in the Fisher analysis at the
Planck best fit, biases the six standard parameters by at most 0.19 sigma. The
matter-power emulator reaches 0.1% and the growth emulator 1.2%, both well below
the few-percent RSD and BAO uncertainties they feed.

For comparison, a public pre-trained LCDM CMB emulator, accurate on its own
training code, biases H0 by 0.53 sigma in this pipeline. That is not a network
error but a code-mismatch systematic: it was trained on a different Boltzmann
code, and matching the neutrino treatment alone changes the best-fit chi^2 by 18.
Training on the same `hi_class` build removes the mismatch by construction.

**End-to-end.** Running the full inference (32 walkers, identical sampler
settings and likelihoods) with the emulator in place:

| | alpha_B0 | alpha_M0 | wall time |
|---|---|---|---|
| exact `hi_class` | 1.00 +/- 0.44 | 0.41 +/- 0.81 | days, 6 cores |
| emulated, MG-aware CMB | 1.08 +/- 0.44 | 0.32 +/- 0.82 | 10 min, 1 core |
| emulated, LCDM CMB | 0.09 +/- 0.84 | 1.24 +/- 1.74 | 10 min, 1 core |

The MG-aware emulated pipeline recovers the exact marginal widths to a few per
cent on both Horndeski parameters, and reproduces the shape of the alpha_M
posterior including its tail. Both results are consistent with General
Relativity.

**The CMB carries the modified-gravity constraint.** The third row is the
scientifically interesting one. Emulating the CMB in LCDM only -- on the
reasonable-sounding argument that alpha_M and alpha_B barely change the primary
temperature spectrum -- degrades alpha_M by more than a factor of two and biases
alpha_B from ~1.1 to ~0.1. The small shift that argument dismisses
(Delta chi^2 ~ 12 at alpha = 0.5, from the modified-gravity lensing of the
temperature spectrum) *is* the likelihood curvature that constrains alpha.

This has a direct implication for emulator-accelerated modified-gravity
analyses: the published modified-gravity emulators are for the matter power
spectrum, and a pipeline that emulates P(k) and growth but treats the CMB as
LCDM will report substantially weaker and biased constraints on the Horndeski
functions.

It is also a lesson about validation. The CMB emulator's parameter bias was
measured at fixed alpha, where it was 0.19 sigma and looked excellent.
Component-level validation cannot certify a parameter that is marginalised over;
only the end-to-end run can.

## Caveats

The exact chain used for the comparison is a partial run (~360 steps of 32
walkers). Its marginal widths had plateaued -- the alpha_M width rose through
burn-in and then oscillated around 0.8 with no trend over the final ~100 steps --
so the comparison is meaningful, but a fully converged exact chain would sharpen
it, and the visible wiggles in the exact posterior are sampling noise. The
speed comparison is wall-clock under different parallelisation (emulated on one
core, exact on six) and is intended as an order-of-magnitude statement.
