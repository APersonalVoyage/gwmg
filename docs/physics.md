# Physics and method

What the analysis measures, what each parameter means, and how to read the
results. No prior cosmology assumed.

## The question

General Relativity is tested to exquisite precision in the Solar System and in
binary pulsars. Whether it still holds across billions of light years is a
separate question, and one of the ways the accelerating expansion of the universe
might be explained is that gravity itself behaves differently on those scales.

This pipeline allows gravity to deviate in two specific, physically motivated
ways and asks the data how large those deviations are permitted to be.

## The parameters

Eight numbers are sampled. Six describe a standard universe; two describe the
deviation from General Relativity.

| Parameter | Symbol | Meaning |
|---|---|---|
| `omega_m` | $\Omega_{\rm m}$ | fraction of the universe's energy in matter |
| `h0` | $h$ | expansion rate today, $H_0 = 100h$ km/s/Mpc |
| `omega_b` | $\Omega_{\rm b}$ | fraction in ordinary atoms rather than dark matter |
| `n_s` | $n_{\rm s}$ | tilt of the primordial fluctuations left by inflation |
| `A_s` | $A_{\rm s}$ | amplitude of those fluctuations |
| `tau` | $\tau$ | optical depth: when the first stars reionised the gas |
| **`alpha_B0`** | $\alpha_{\rm B0}$ | **braiding**: mixing of the scalar field with gravity, which changes how fast structure grows |
| **`alpha_M0`** | $\alpha_{\rm M0}$ | **Planck-mass run rate**: gravity's strength changes with time, so gravitational waves lose amplitude as they travel |

$\alpha_{\rm M0} = \alpha_{\rm B0} = 0$ is exactly General Relativity. The whole
analysis asks how far from zero the data allow them to be.

### The Horndeski framework

These two come from [Horndeski gravity](https://arxiv.org/abs/1605.06102), the
most general scalar–tensor theory with second-order equations of motion. Rather
than pick a specific model, it describes linear perturbations through a small set
of free functions of time. We use the `propto_omega` parametrisation, in which
each function tracks the dark-energy density,

$$\alpha_i(a) = \alpha_{i0}\,\frac{\Omega_{\rm DE}(a)}{\Omega_{\rm DE}(a=1)},$$

so the deviation switches on as dark energy comes to dominate, and vanishes in the
early universe where General Relativity is well tested. We fix
$\alpha_{\rm T} = 0$, required at the per-mille level by GW170817 arriving with
its gamma-ray burst, and hold the kineticity $\alpha_{\rm K}$ fixed, since the
data barely constrain it.

### Why gravitational waves are the right probe

If $\alpha_{\rm M} \neq 0$, gravitational waves are damped as they propagate, so a
merger appears **further away** in gravitational waves than the same event does in
light:

$$\frac{d_L^{\rm GW}(z)}{d_L^{\rm EM}(z)} = \exp\left[\frac{1}{2}\int_0^z \frac{\alpha_{\rm M}(z')}{1+z'}\,dz'\right]$$

Measure a merger's distance from the gravitational-wave signal, get its redshift
independently from an electromagnetic counterpart, and the mismatch constrains
$\alpha_{\rm M}$ directly. That is the standard-siren test.

## The simulation

For any set of those eight numbers, `hi_class` solves the linear
Einstein–Boltzmann equations from shortly after the Big Bang to today, and outputs
what we would actually observe:

- the CMB temperature power spectrum $C_\ell^{TT}$
- the matter power spectrum $P(k)$, i.e. how galaxies cluster
- the growth rate combination $f\sigma_8(z)$
- distances and $H(z)$
- the GW/EM distance ratio above

One call takes a few seconds. That cost is the reason the emulator exists.

## The data

| Dataset | What it constrains |
|---|---|
| Planck 2015 CMB (TT) | the standard parameters, plus $\alpha$ through lensing |
| BOSS, 6dFGS ($f\sigma_8$) | growth of structure, sensitive to $\alpha_{\rm B}$ |
| WiggleZ, SDSS MGS (BAO) | distances |
| GW170817, GW190521 | $\alpha_{\rm M}$ via the distance ratio |

GW170817 is a neutron-star merger whose counterpart in NGC 4993 gives a redshift.
GW190521 is a black-hole merger with a candidate counterpart.

## Likelihood, prior, posterior

**Likelihood** answers: *if these eight numbers were true, how probable is the
data we actually saw?* For each dataset,

$$\chi^2 = \sum_i \frac{(\text{predicted}_i - \text{observed}_i)^2}{\sigma_i^2},
\qquad \log \mathcal{L} = -\tfrac{1}{2}\chi^2$$

which is just "how many error bars off am I, squared, summed". A perfect fit gives
$\chi^2 = 0$ and the maximum log-likelihood of zero; poor fits go sharply
negative. Each likelihood module reports one number and they are summed, so the
`Likelihood = -107.58` printed by a test run is the total across all six datasets.
**Higher, meaning less negative, is a better fit.**

**Prior** is what you believed beforehand. Ours are flat:
$\alpha_{\rm M0} \in [-1, 6]$ and $\alpha_{\rm B0} \in [-1, 3]$, all values
equally plausible a priori. These are the `min start max` entries in
`values_horndeski.ini`.

**Posterior** is what you believe after seeing the data:

$$P(\theta \mid \text{data}) \propto \mathcal{L}(\text{data} \mid \theta)\, P(\theta)$$

It is not a single number but a *distribution* over the eight-dimensional
parameter space: high where parameters fit, low where they do not. **The posterior
is the answer.**

## Chains and MCMC

You cannot map eight dimensions on a grid: even 100 points per axis is $10^{16}$
simulations. Instead you wander. Propose a nearby parameter set, evaluate its
posterior, and step there preferentially when it is better. Do this long enough
and **the places visited most often are the high-posterior places**.

The record of where you went is the **chain**, and the density of samples *is* the
posterior. You never compute the distribution; you sample it. We use
[emcee](https://arxiv.org/abs/1202.3665) with 32 walkers exploring in parallel, so
every step appends 32 rows.

Two practical points:

- **Burn-in.** Walkers start somewhere arbitrary and take time to find the good
  region. Those early samples are discarded; we drop the first half.
- **Convergence.** A chain is converged when running it longer stops changing the
  answer. Ours plateaued: the $\alpha_{\rm M0}$ width rose through burn-in to
  about 0.8 and then stopped moving.

## Reading the results

The posterior lives in 8D, but results are quoted one parameter at a time.
**Marginalising** means collapsing the other seven: across everything else the
data allow, what values can this parameter take? That gives the familiar 1D
histogram, whose mean is the central value and whose standard deviation is the
error bar.

$$\alpha_{\rm M0} = 0.41 \pm 0.81$$

means the data centre it near 0.4 but anywhere from roughly $-0.4$ to $1.2$ is
plausible. **Zero sits comfortably inside, so General Relativity is consistent
with the data.** A wide error bar means weakly constrained; a narrow one means the
data pin it down.

The chain file is plain text, one row per sample, with a header naming the
columns:

```python
import numpy as np
f = "output/gw_lss_horndeski.txt"
d = np.loadtxt(f)
names = open(f).readline().lstrip("#").split()
aM = d[len(d)//2:, names.index("horndeski_parameters--parameters_smg__3")]  # drop burn-in
print(f"alpha_M0 = {aM.mean():.2f} +/- {aM.std():.2f}")
```

Note that `parameters_smg__2` is $\alpha_{\rm B0}$ and `parameters_smg__3` is
$\alpha_{\rm M0}$; the last two columns are the prior and posterior.

`gwmg plot` renders this as a **corner plot**: the diagonal holds each parameter's
1D histogram, and the off-diagonal panels show 2D contours for each pair. A
tilted, elongated contour means the two parameters are degenerate, so the data
constrain a combination of them rather than each independently.

## What the emulator changes

Nothing conceptual. The sampler, the likelihoods and the posterior are identical.
The emulator replaces only the simulation step: instead of solving the equations
in seconds, neural networks trained on tens of thousands of solved examples
reproduce the same outputs in about 7 ms. Days become minutes, and the answer is
the same to within a few per cent on the error bars.

The one subtlety, described in [the emulator page](emulator.md), is that the CMB
emulator must take $\alpha_{\rm M0}$ and $\alpha_{\rm B0}$ as inputs. Training it
without them passes every component-level check yet corrupts the final constraints,
because the CMB's sensitivity to modified gravity through lensing carries much of
the information.

## The result

$$\alpha_{\rm B0} = 1.00 \pm 0.44, \qquad \alpha_{\rm M0} = 0.41 \pm 0.81$$

Both consistent with zero within their uncertainties, so there is **no evidence
for a deviation from General Relativity**. With only two standard sirens the
constraints are loose; more detections will tighten them, which is exactly why a
pipeline that reruns in ten minutes is worth having.
