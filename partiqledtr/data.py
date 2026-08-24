"""Decay event generation and LCAG dataset construction (ROADMAP phase 1).

Planned nodes:
- generate_decays: phasespace-based topology sampling and event generation
- to_lcag: decay tree -> LCAG matrix, leaf shuffling, -1 padding/ignore
- featurize: momentum direction angles (theta, phi) + energy; Cartesian variant for ablation
"""
