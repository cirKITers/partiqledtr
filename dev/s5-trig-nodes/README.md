# s5 — trig interface: does classical trigonometric capacity fit the QFM arm?

The quantum arm's classical parts are purely linear, so the whole model is
trig-polynomial → linear → trig-polynomial → linear, and block 2 consumes the
node update's output directly as RY angles. This study asks whether giving the
classical side capacity *in the same trigonometric language* (SIREN; cf. QIREN, arXiv:2406.03873)
pays — and gates that question with two measurements that need no training.

**Measurement 1 — the boundary is not collapsed at init (2026-09-02).** At the
s4 configuration, block-2 encoded angles start at std 0.89 rad, sin² 0.43,
purity 0.92 μ_n — the *uniform* level, not the clustered one (the collapse
hypothesis is falsified), while the gated block-1 chart sits at 1.89 μ_n.
`node_omega` moves init purity only 0.92 → 1.01 μ_n by ω = 16. What training
does to this distribution is the open half; the block-2 instrumentation
(`block2_g_purity`, `block2_angle_stats` in `final_metrics`) now records it.

**Measurement 2 — the edge decision is invariant-shaped, not angle-trig-shaped
(`spectrum.py`, `results/spectrum.json`).** Held-out accuracy of linear probes
per sampled real edge (majority 0.381):

| probe | features | acc |
| --- | --- | --- |
| **m_ij powers 0–3** | **4** | **0.495** |
| trig on angles, k≤2, d≤2 | 85 | 0.471 |
| trig, d≤3 or k≤4 | ≤377 | 0.469–0.471 |

A cubic in the single pair invariant beats every integer-frequency trig probe on
the six encoded angles; the trig content saturates at total degree 2, and
per-feature frequencies beyond the L=2 QFM box (|ω|≤2) add nothing. So the
model's frequency box is already spectrally sufficient for the single-edge
content of these angles, and the missing edge-level information lives in m_ij —
the open question of loading m_ij in-algebra (as the angle of the cross-particle
XY bonds), in empirical form. (Bounds the edge function only; message passing
sees more.)

**The probe** (`run.py`, 4 cells × 2 seeds, XY_Cycle / boost chart / K=4 /
no preconditioner):

| cell | what it isolates |
| --- | --- |
| `linear-w1` | the s4 architecture + boundary instrumentation (control) |
| `linear-w8` | boundary scale alone (`node_omega = 8`) |
| `siren` | sine node MLP, SIREN init — the trig-interface arm |
| `elu` | same MLP, ELU — matched parameters, different activation |

Read against the s4 smoke baseline (`XY_Cycle` none: 0.555 ± 0.009). If `siren`
beats `elu`, the trigonometric coherence pays beyond capacity; if both beat
`linear` equally, it was capacity; if neither, the node update is not the
bottleneck and the m_ij result above points at the next lever.

**Result (2026-09-02):** the first branch — siren 0.587 ± .007 beats the
matched elu 0.565 ± .002 beats linear 0.555 ± .011; ω=8 alone hurts (0.530).
The sine node update on the floor-free cycle nearly matches the s4 project best
(odd-chord × mlp, 0.591 ± .012) at a 17× smaller DLA. Framing per the user's
note: the gain is the classical architecture matched to the model's
trigonometric character — not a mechanism imported from QIREN, which
demonstrates QFMs on image tasks.

**The expansion** (`run.py --expand`, 5 cells × 2 seeds): siren across the
trichotomy × {none, mlp} — ladder and odd-chord both preconditioner settings,
plus the cycle × mlp completion. Linear baselines are s4's smoke grid (3 seeds);
the capacity control lives in the probe. Reading rule, registered in advance:

- **Uniform lift** (siren − linear ≈ +0.03 on every arm) ⇒ the interface fix is
  generic architecture-matching, independent of the algebra.
- **Floor-free-specific lift** (cycle/ladder gain > odd-chord gain) ⇒ the linear
  interface was specifically handicapping floor-free arms, and s4's
  "floored arm leads" ranking is an interface artefact.
- **Preconditioner interaction**: the mlp's small positive delta (+0.005–0.019
  in s4) should persist under siren if the input-distribution mechanism and the
  node update are orthogonal; a sign flip would say the sine layer absorbs the
  preconditioner's role.
- Headline check: does odd-chord × mlp × siren pass 0.591?

**Expansion result (2026-09-03):** the floor-free-specific branch — siren lifts
cycle/ladder by +0.032/+0.044 at `none` and the floored control by only +0.006,
flipping s4's ranking (ladder 0.602 ± .001 leads the preconditioner-free field).
New project best 0.607 ± .002 (odd-chord × mlp × siren). Under siren the mlp is
neutral on floor-free arms and +0.029 on the floored one — a non-unflattening
effect by construction there.
