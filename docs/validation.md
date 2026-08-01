# Validation against Baker & Harrison (2020)

`gwmg` implements the framework of **Baker & Harrison (2020)**, *"Constraining
Scalar-Tensor Modified Gravity with Gravitational Waves and Large Scale Structure
Surveys"*, JCAP ([arXiv:2007.13791](https://arxiv.org/abs/2007.13791)). Each core
piece is checked directly against the paper's equations.

## Modified GW luminosity distance — their eq. (2.12)

> d_GW = C⁻¹ d_L = d_L · exp[ −½ ∫₀ˣ α_M(x) dx ],   x = ln a

Changing variable to redshift (dx = −dz/(1+z)) gives the form `gwmg` integrates:

> d_GW(z) = d_L(z) · exp[ +½ ∫₀ᶻ α_M(z′)/(1+z′) dz′ ]

There is **no division by Ω_Λ** in the integrand — it is the physical α_M(z),
which `hi_class` provides directly as the `M2_running_smg` background column.
(An earlier version of the code divided α_M by Ω_smg(0); this was a bug, and is
disabled by default. Set `alpha_m_over_omega_smg = T` only to reproduce those
specific legacy numbers.)

## The α_i ansatz — their eq. (2.14)

> α_i(a) = α_i0 · Ω_Λ(a),   with parameters of interest {α_M0, α_B0}

This is exactly `hi_class`'s `propto_omega` (α_i ∝ Ω_DE), so `parameters_smg__3`
= α_M0 and `parameters_smg__2` = α_B0. The sampling priors on these parameters
are the priors on α_M0 / α_B0.

## GW distance error model — their eqs. (3.9)–(3.11)

> σ²_dGW = σ²_meas + σ²_lens + σ²_v
> σ_lens = 0.066 d_L [(1−(1+z)⁻⁰·²⁵)/0.25]¹·⁸
> σ_v    = d_L [1 + c(1+z)/(H d_L)] √⟨v²⟩ / c,   ⟨v²⟩ = 500 km/s

Implemented line-for-line in `gwmg.likelihood.gw_log_likelihood`.

## Large-scale-structure data

Matches the paper's choices: Planck 2015 CMB, RSD f σ₈ from BOSS DR12 + 6dFGS at
k_fid = 0.05 h/Mpc, BAO from WiggleZ + SDSS MGS + BOSS DR12. (The paper uses
Plik-lite **TT-only** at high-ℓ; `gwmg` currently uses Planck-lite TTTEEE — set
`[planck_py] spectra = TT` to match exactly.)

## Scope note — GW190521

The paper's *LIGO* error model (their eq. 4.4, σ ≃ 5.63×10⁻⁴ d²) is calibrated
for simulated binary-neutron-star sirens at z < 0.2, d < 400 Mpc. **GW190521 is a
5.3 Gpc black-hole merger**, outside that regime, so its uncertainty is taken from
the actual LVK measurement (d ≈ 5300 Mpc, σ ≈ 2500 Mpc) with the EM-counterpart
redshift z = 0.438 (ZTF19abanrhr). Applying this pipeline to GW190521 therefore
*extends* Baker & Harrison to a real high-redshift event.
