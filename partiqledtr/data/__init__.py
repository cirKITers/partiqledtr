"""Decay generation and LCAG dataset construction (ROADMAP phase 1).

Modules are imported directly (``from partiqledtr.data.lcag import topology_to_lcag``);
this package deliberately re-exports nothing so that importing any data module does
not pull in :mod:`partiqledtr.data.generation`, the only module that imports
TensorFlow (via phasespace).
"""
