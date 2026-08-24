"""Models (ROADMAP phases 2-3).

Planned:
- qfm_constellation: small shared-weight QFMs (qml-essentials Model, analytic expval,
  vmap over batch x edges) inside message passing; ansatz arms XY_Brickwork / Matchgate /
  Circuit_19
- front_end: elementwise residual MLP (per-feature 1->16->1, zero-init output) and fixed
  isotropic whitening
- classical_gnn: Flax/Optax message-passing baseline (parameter-matched + unconstrained)
"""
