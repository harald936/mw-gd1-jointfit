# Roadmap

## V0 (this week)
- Plot a literature Milky Way rotation curve (RC) table (Eilers+2019).
- Overlay a baseline model curve from a standard potential (MWPotential2014).
- Plot a literature GD-1 member list to sanity-check sky track & proper motions.

## V1
- Fit RC-only with a gNFW halo (free γ, flattening q), disk+bulge priors from literature.
- Add a Gaussian Process over R to absorb RC residual systematics.
- Posterior predictive checks.

## V2
- Add a GD-1 likelihood (positions & proper motions in stream frame) with mixture-model membership.
- Model PM zero-point drift & distance gradient with GPs.

## V3
- Rebuild a probabilistic GD-1 catalog from raw Gaia DR3 (CMD matched filter, RUWE/parallax cuts).
- Ablations, Bayesian bootstrap on selection, SBC, mock injection–recovery.
