# Literature

Candidate references from the avenue review of 2026-08-31 (scaling, hard circuits,
kinematics-informed features), pending detailed review before anything cites them.
Grouped by what they bear on, one line each. The unflattening manuscript's own
bibliography (ragone, fontana, goh, cerezo_does, brod, kokcu, wiersema, mhiri, barthe,
thanasilp_subtleties, schuld, shin, peters) is not repeated — only entries new to the
project record, plus published-version updates.

## Simulability and hardness (phase 6 claim wording)

- Cerezo et al., *Does provable absence of barren plateaus imply classical
  simulability?* — now published, [Nat. Commun. 2025](https://www.nature.com/articles/s41467-025-63099-6)
  (arXiv:2312.09121). Update the cite; the "polynomial floor-free hard" corner being
  conjectured empty rests here.
- *Enabling Lie-Algebraic Classical Simulation beyond Free Fermions*,
  [arXiv:2604.16701](https://arxiv.org/abs/2604.16701) (2026). Extends g-sim to
  S_n-equivariant and U(1)-sector circuits; does **not** cover bipartite-XY or doped
  families — cite when claiming the ladder arm sits outside Lie-algebraic simulation.
- Angrisani et al., *Classically estimating observables of noiseless quantum circuits*,
  [arXiv:2409.01706](https://arxiv.org/abs/2409.01706). Average-case Pauli-propagation
  surrogates at any depth: the reason phase 6 claims "encoded-universal family", never
  "no classical surrogate".
- Anschuetz et al., *Efficient classical algorithms for simulating symmetric quantum
  systems*, arXiv:2211.16998. Permutation-symmetric circuits are simulable — bites any
  proposal to move the constellation's equivariance into the circuit.
- Liu, Arunachalam, Temme, *A rigorous and robust quantum speed-up in supervised machine
  learning*, arXiv:2010.02174. The bar for "provably hard *and* useful" feature maps
  (discrete log); calibrates what hardness can buy on a natural task.

## Equivariance and HEP baselines (phase 6 features)

- *Lorentz-Equivariant Quantum Graph Neural Network for High-Energy Physics*,
  [arXiv:2411.01641](https://arxiv.org/abs/2411.01641) (IEEE TAI 2025). Closest external
  work: few-qubit circuits as edge functions on Lorentz-invariant inputs for jet
  tagging. Cite in `RESEARCH.md` once the third-angle arm lands.
- LorentzNet, [arXiv:2201.08187](https://arxiv.org/abs/2201.08187); PELICAN,
  arXiv:2211.00454; L-GATr, [arXiv:2405.14806](https://arxiv.org/abs/2405.14806). The
  classical bar for invariant-input architectures.
- *What Do Lorentz-Equivariant Jet Taggers Learn?*,
  [arXiv:2606.21790](https://arxiv.org/abs/2606.21790) (2026). Equivariant taggers
  suppress frame-dependent features and encode invariant masses — evidence that the
  invariants are the right inputs.
- Schatzki et al., *Theoretical guarantees for permutation-equivariant quantum neural
  networks*, arXiv:2210.09974 (npj QI 2024). Trainability guarantees for S_n-equivariant
  circuits; pairs with Anschuetz above as the trainable-but-simulable caution.

## Spectra and Fourier models (phase 5 alignment program)

- *Overcoming Fourier Locking in Quantum Data Re-uploading Classifiers*,
  [arXiv:2607.11013](https://arxiv.org/abs/2607.11013) (2026). Names the locking failure
  mode; "frequency pacing" — sculpt the spectrum to where the task needs it — is the
  alignment-not-size program.
- *Quantum Spectral Model*, [arXiv:2607.22516](https://arxiv.org/abs/2607.22516) (2026).
  Input-conditioned re-uploading as spectral control.
- *Mitigating frequency learning bias via multi-stage residual learning*,
  [arXiv:2603.10083](https://arxiv.org/abs/2603.10083), and *grid-based initialization
  for frequency reachability*, [arXiv:2602.23409](https://arxiv.org/abs/2602.23409)
  (both 2026). Mitigations for the locking failure mode that keeps
  `trainable_frequencies` off (ROADMAP fixed decisions) — that reason is weaker now.
- Jaderberg et al., *Let quantum neural networks choose their own frequencies*,
  arXiv:2309.03279. The original trainable-frequency case.
- Landman et al., *Classically approximating variational QML with random Fourier
  features*, arXiv:2210.13200; Schreiber, Eisert, Meyer, *Classical surrogates of
  quantum learning models*, arXiv:2206.11740. The RFF surrogate control the phase-5
  alignment program needs anyway (README's no-advantage framing).

## Sinusoidal networks (trig-interface program, 2026-09-02)

- Zhao et al., *Quantum Implicit Neural Representations* (QIREN),
  [ICML 2024](https://proceedings.mlr.press/v235/zhao24l.html) /
  [arXiv:2406.03873](https://arxiv.org/abs/2406.03873). Data-reuploading circuits
  hybridised with classical layers as a quantum generalisation of Fourier/SIREN
  networks; claims exponentially more compact Fourier-series representation than
  classical FNNs. In substance it demonstrates QFMs on image tasks -- treat it as
  a framing reference for the hybrid, not a source to attribute learning effects
  to (user note 2026-09-02); our claim is the general one, matching the classical
  architecture to the model's trigonometric character.
- Sitzmann et al., *Implicit neural representations with periodic activation
  functions* (SIREN), arXiv:2006.09661. The omega_0 scaling and initialisation that
  keep post-sine activations well-distributed through depth — the scale discipline
  the block-2 re-encoding boundary lacks.
- Tancik et al., *Fourier features let networks learn high frequency functions*,
  arXiv:2006.10739; Rahaman et al., *On the spectral bias of neural networks*,
  arXiv:1806.08734. Why Fourier features + MLP works, and what plain MLPs miss.
- *A unified theory of sinusoidal activation families for INRs*,
  [arXiv:2502.00869](https://arxiv.org/html/2502.00869v3) (2025); *Spectral
  bottleneck in sinusoidal representation networks*,
  [arXiv:2509.09719](https://arxiv.org/pdf/2509.09719) (2025). Current theory and
  failure modes of the sine-activation family; Parascandolo (2016) is the old
  warning that naked sines train badly without the init discipline.
- *Quantum INRs for 3D scene reconstruction*,
  [arXiv:2512.12683](https://arxiv.org/pdf/2512.12683) (2025). QIREN-style models
  applied further afield; the thread is active.

## Kernels and data (pre-training gates)

- Huang et al., *Power of data in quantum machine learning*, arXiv:2011.01938. The
  geometric-difference test between quantum and classical kernels on the actual dataset:
  a cheap, principled gate before investing further in the hard arm.
- Thanasilp et al., *Exponential concentration and untrainability in quantum kernel
  methods*, arXiv:2208.11060. If a kernel reading of the edge QFM is ever wanted.
