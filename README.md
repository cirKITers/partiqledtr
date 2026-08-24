# PartiqleDTR

This repo facilitates to revive the former partiqlegan project.
The goal is to reconstruct intermediate decay products based on simulated decay events as reference data.

Tech stack:
- qml-essentials : for quantum Fourier models and JAX based simulation
- JAX : array computation and autodiff
- Flax : neural network modules
- Optax : optimizer and training loop
- Fluksio : data science pipeline and experiment tracking

References (`./references/`):
- baumbauen: classical gnn based approach with message passing
- partiqlegan: hybrid quantum-classical approach
- reconstructing-paper & improving-paper: paper corresponding to the hybrid approach
- fourier-fingerprints: correlations between frequency components (FCC) of QFMs as an inductive-bias descriptor
- unflattening: latest research concerning input distribution dependence for quantum circuits

Idea:
Create a new approach for the decay tree reconstruction problem by using quantum Fourier models (through `Model` in qml-essentials) in combination with a classical MLP.
This should serve as an application scenario for the `unflattening` paper, where we showed that the input distribution for a data re-uploading model get's whitened for some ansaetze when being combined with an MLP (and whitening is actually preferrable for these ansaetze).
Problem back then was that the simulation of a quantum model took an awful long time and therefore a purely quantum based gnn wasn't feasible.
However we showed that the qnn seems to be beneficial for the training.
While there is a low chance that we can acutally show an advantage compared to the purely classical case, it would already be interesting to research if the trigonometric properties of a QFM fit in the context of this given problem.
This formulates the main hypothesis: a constallation of QFMs acting as a GNN with an MLP as the feature map to adjust frequency components (trainable frequencies) can solve the problem of prediciting the LCAG of particle decay events. The input distribution flattening can be observed in this scenario. 

Note:
Fluksio is a relatively new framework (developed by myself).
Documentation is available here: https://docs.fluksio.com/getting-started/data-science/
If we hit any limitations or encounter problems, we should stop and flag them in `NOTEPAD.md` instead of trying workarounds.
Then Fluksio will be fixed and we can continue.
The same holds true for any limitations/issues with qml-essentials.

## Theoretical Motivation

A QFM realizes a truncated Fourier series in its encoded features, so its hypothesis
class is a trigonometric polynomial with a spectrum fixed by the encoding. Decay
kinematics fit this class naturally: momentum directions are periodic quantities, and
decay angular distributions are low-order expansions in spherical harmonics. Encoding
direction angles directly aligns the model's basis with the structure of the data — the
inductive-bias question this project probes.

Two prior results shape the design. First, the unflattening work shows that for
floor-free, polynomial-DLA ansaetze the input angle distribution decides trainability:
clustered angles annihilate the loss signal, while a classical front end rescues it
through the encoder channel. Kinematic features cluster encoding angles naturally (soft
particles yield near-zero angles), so this task lands in exactly the regime where the
theory makes falsifiable predictions — observable as g-purity dynamics during training,
with a floored (Matchgate) arm as control. Second, the fourier-fingerprints work
provides the FCC as a cheap, hardware-compatible descriptor of an ansatz's coefficient
correlations; here it serves as an ansatz-selection metric, and tracking the spectrum
under a trainable front end addresses that paper's open question about nonlinear
classical preprocessing.

The architecture consequence: many small shared-weight QFMs inside message passing
(permutation-equivariant by construction, tractable spectra, cheap analytic simulation)
instead of the former one-qubit-per-particle monolith; an elementwise residual MLP as
front end so cross-particle structure must come from the quantum part. This is an
inductive-bias study, not an advantage claim — the relevant spectra admit classical
surrogates, and the honest question is whether the trigonometric structure helps.

See `ROADMAP.md` for the experimental plan.