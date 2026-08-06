---
title: 'gwmg: a reproducible pipeline and emulator for testing modified gravity with gravitational-wave standard sirens'
tags:
  - Python
  - cosmology
  - gravitational waves
  - modified gravity
  - machine learning
  - Bayesian inference
authors:
  - name: Abhishek Karkola
    orcid: 0009-0001-8035-0195
    affiliation: 1
affiliations:
  - name: Independent researcher
    index: 1
date: 6 August 2026 
bibliography: paper.bib
---

# Summary

`gwmg` is a Python package for testing General Relativity on cosmological scales,
using gravitational-wave (GW) standard sirens together with large-scale-structure
and cosmic microwave background (CMB) data. It constrains the Horndeski functions
$\alpha_{\rm M}$ and $\alpha_{\rm B}$, which parametrise deviations from Einstein
gravity in the propagation of gravitational waves and in the growth of cosmic
structure, following the framework of @BakerHarrison2020.

The package drives the modified-gravity Einstein–Boltzmann solver `hi_class`
[@hiclass; @CLASS] through the `CosmoSIS` inference framework [@CosmoSIS], and
supplies the layer specific to this measurement: the GW standard-siren
likelihood, the interface between the two libraries, the GW luminosity-distance
model, and ready-to-run configurations for the GW170817 and GW190521 events. It
additionally provides neural-network emulators of the expensive physics, built
with `CosmoPower` [@CosmoPower], which reduce a multi-day inference to about ten
minutes, together with the tooling needed to train and validate them.

![Marginal posteriors on the Horndeski parameters. The emulated pipeline (red)
recovers the constraints of the exact `hi_class` pipeline (black) at a fraction of
the cost. Emulating the CMB in $\Lambda$CDM only (blue dashed) broadens
$\alpha_{\rm M0}$ and biases $\alpha_{\rm B0}$.](docs/emulator_vs_exact.png)

# Statement of need

Analyses of this kind couple several independent community codes. The layer that
ties them into a specific measurement — the likelihood, the interfaces between
codes, and the run configurations — is analysis-specific, rarely released, and is
precisely the part most likely to be lost when a project ends. `gwmg` packages
that layer as installable, tested software, so the analysis can be reproduced,
applied to new GW detections, or extended with newer survey data rather than
reconstructed from scratch. The package originated as a reconstruction of an
unmaintained thesis pipeline; in rebuilding it, two errors in the original
analysis were identified and corrected, in the GW luminosity-distance integrand
and in the units of the input catalogue.

The second need is speed. Every likelihood evaluation calls `hi_class` to solve
the linear cosmological equations, taking seconds, so a converged Markov chain
Monte Carlo run [@emcee] takes days. This limits how often such analyses can be
rerun as new events are detected. Emulators are the established remedy
[@CosmoPower; @Capse], and emulators for the modified-gravity matter power
spectrum are available [@OrjuelaQuintana2024; @Brando2022], but assembling a
complete accelerated pipeline requires emulating the CMB and growth as well, and
requires deciding when an emulator is accurate enough to leave the inference
unbiased. `gwmg` provides both the exact pipeline and the tooling to build and
validate emulators against it.

# Functionality

`gwmg` exposes a command-line interface (`info`, `run`, `plot`, `validate`) for
the exact pipeline, and an emulator workflow (`emu-gen`, `emu-train`,
`emu-validate`, `emu-chi2`, `emu-bias`) covering training-set generation,
training, and three levels of validation. Trained emulators act as a drop-in
replacement for `hi_class` within CosmoSIS, so the exact and accelerated
pipelines share the same configuration and likelihoods and are directly
comparable. A Python API returns the observables directly:

```python
from gwmg.emulator import Emulator
emu = Emulator("emulators")
out = emu.predict(omega_m=0.315, h0=0.674, omega_b=0.049, n_s=0.965,
                  A_s=2.1e-9, tau=0.054, alpha_B0=1.0, alpha_M0=0.5)
out.cl_tt, out.pk, out.sigma8, out.fsigma8, out.dgw_ratio
```

This evaluates in roughly 7 ms against 6.4 s for the corresponding `hi_class`
call, and agrees with it to better than 0.5 per cent on the quantities entering
the likelihoods.

The validation tooling implements a hierarchy that we found necessary in
practice. The accuracy metric usually quoted for CMB emulators, the per-multipole
error in units of cosmic variance, is too lenient here: the likelihood uses
binned bandpowers with much tighter errors, and emulator error is correlated
across scale so it does not average down within a band. `gwmg emu-bias` therefore
implements a Fisher-matrix analysis of the emulator-induced parameter bias, which
is directly interpretable in units of the marginal uncertainty. Even that proves
insufficient for a parameter that is marginalised over: a CMB emulator trained in
$\Lambda$CDM passes at $0.19\,\sigma$ yet, in the full inference, broadens
$\alpha_{\rm M0}$ by more than a factor of two and biases $\alpha_{\rm B0}$,
because it discards the lensing response of the temperature spectrum to modified
gravity. Training the CMB emulator on the Horndeski parameters as well recovers
the exact posteriors.

# Availability and testing

`gwmg` is available on GitHub, is `pip`-installable, and is released under the
MIT license with a test suite and continuous integration. The core — the GW
likelihood, command line and plotting — installs with `numpy` and `scipy` only;
the heavier `CosmoSIS` and `hi_class` stack it drives is installed separately,
and the emulator, which depends on `CosmoPower` and TensorFlow, is an optional
extra. The documentation covers installation,
validation of the pipeline against the source paper, and the emulator
methodology and its limitations.

# Acknowledgements

This work grew out of an MSc research project supervised by Ian Harrison. It
builds on the `CLASS`, `hi_class`, `CosmoSIS`, `CosmoPower` and `emcee`
open-source projects, and we thank their developers.

# References
