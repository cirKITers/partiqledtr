"""Fluksio flow declarations.

Planned flows (see ROADMAP.md):
- generate: data.generate_decays -> data.to_lcag -> data.featurize
- train: dataset -> train.fit (+ analysis.g_purity / analysis.spectrum hooks) -> metrics
- analyze: trained model -> fingerprints, FCC, ablation reports
"""
