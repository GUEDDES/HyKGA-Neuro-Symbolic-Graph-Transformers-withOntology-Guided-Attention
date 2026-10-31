# HyKGA: Neuro-Symbolic Graph Transformers with Ontology-Guided Attention for Context-Aware Social Discourse Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

Official reference implementation, W3C OWL 2 DL / SHACL ontology specification, causal anti-leakage subgraph builder, training & task-transfer pipeline, and mathematical verification suite for the paper:

> **Neuro-Symbolic Graph Transformers with Ontology-Guided Attention for Context-Aware Social Discourse Analysis**  
> *Wyssem Fathallah, Abdelweheb Gueddes, Mourad Mars, Mohammed J. AlGhamdi, Mohamed Ali Mahjoub, and Hedi Sakli*  
> MDPI *Technologies* (Manuscript ID: `technologies-4653627`)  
> **Repository:** <https://github.com/GUEDDES/HyKGA-Neuro-Symbolic-Graph-Transformers-withOntology-Guided-Attention>

---

## Repository Structure

| File / Directory | Description |
|---|---|
| `social_discourse_ontology.owl` | Formal W3C OWL 2 DL specification (RDF/XML) of the 7 pairwise-disjoint entity concepts $\mathcal{C}$ (grounded in W3C SIOC / FOAF) and 7 directed relations $\mathcal{R}$, with open-world TBox domain/range axioms ($\mathcal{A}$) and closed-world functional ($\le 1$) and irreflexive constraints ($\mathcal{S}$). |
| `compile_ontology_masks.py` | Direct XML/OWL 2 DL parser and offline/online symbolic compiler producing $\mathbf{C}_{dir} \in \{0,1\}^{7\times 7}$, the symmetric reflexive attention matrix $\mathbf{C} = \mathbf{C}_{dir} \vee \mathbf{C}_{dir}^\top \vee \mathbf{I}$ (23/49 admissible pairs; 53.1% masked), the relation mask $\mathbf{m}_{\mathcal{R}} = \mathbf{m}_{\mathcal{A}} \odot \mathbf{m}_{\mathcal{S}}$ (Eqs. 13–15), the conflict-damped training mask $\tilde{\mathbf{L}}_{\mathcal{R}}^{(train)}$ (Eq. 4), progressive axiom-degradation masks (Table 11), and **Table 3 (`tab:conflict_example`)**. |
| `causal_subgraph_builder.py` | Strictly causal, target-pair-excised graph builder implementing $E_i^{causal}$ (Eq. 18), exponential temporal decay $\alpha(\tau_i, t_e)$ (Eq. 19), target-excised 4-scale snapshot centralities $\mathbf{PE}_{graph}^{(i)}(v, t_s)$ on $G_{\le t_s}^{(-i)}$ (Eq. 20), and causal Louvain hard-negative sampling (Sec. 3.5.1). |
| `hykga_model.py` | Complete PyTorch implementation of HyKGA (Algorithms 1 & 2, Eqs. 2–33): 3-layer `CausalMultiHeadGAT`, hybrid graph-aware tokenization, 12-layer ontology-guided dual-attention Transformer, NER typing classifier $\pi_\psi$ with Shannon entropy gating ($\tau_{ent}=0.65$), Louvain-guided InfoNCE alignment, conflict-damped training loss, message-level $\le 1$ functional cardinality decoding, bilinear link predictor, and dual OVR evaluation ($\mathrm{OVR}_{proj}$ vs. $\mathrm{OVR}_{pred}$ / $\mathrm{OVR}_{pos}$). |
| `train_and_evaluate.py` | End-to-end PyTorch `Dataset` / `DataLoader` training and inference pipeline implementing the two-phase training schedule (5-epoch gold-type warmup followed by predicted-type conflict-clamped training) and 4-regime evaluation (`hard`, `soft`, `adaptive`, `posthoc`). |
| `transfer_public_benchmarks.py` | Task-transfer classification heads and W3C SIOC cross-platform mappings for **RumourEval-2019** (Twitter & Reddit 4-class stance), **PHEME** (leave-one-event-out 3-class veracity & zero-shot `PHEME-RE`), and **Twitter15** (4-class propagation tree cascades) (Section 4.5, Tables 12–13). |
| `verify_metrics_and_bounds.py` | End-to-end mathematical verifier for **Table 4** (TSAB statistics & $\le 1$ cardinality bounds), **Tables 5 & 6** ($\mathrm{OVR}_{proj}$ vs. $\mathrm{OVR}_{pred}$ and $\mathrm{OVR}_{pos}$), **Table 7** (500-bin query rank histograms over $|Q_{test}|=28{,}124$ and harmonic bounds $[\mathrm{MRR}_{\min}, \mathrm{MRR}_{\max}]$), **Tables 9 & 10** (5-seed mask isolation & anti-leakage controls), and **Table 17** (soft-masking decomposition $F_- = R^\star + E_-$ and Eq. 42 first-order optimality condition). |
| `hykga_eval.py` | Unified command-line evaluation and consistency auditing tool (`eval`, `dataset`, `audit`). |
| `records/experimental_records.json` | Complete per-seed experimental records, 500-bin link-prediction query rank distributions ($|Q_{test}| = 28{,}124$), leakage control logs, and soft-penalty sweep records. |
| `records/tsab_manifest_sample.jsonl` | Anonymized TSAB benchmark manifest sample and schema (thread IDs, message IDs, timestamps, split assignments, typed entity pairs) compliant with platform Terms of Service. |
| `records/reported_tables_audit.json` | Complete 65-row audit specification for all empirical tables in the manuscript. |
| `training_config.json` | Exact hyperparameter configuration, random seeds ($\{42, 104, 256, 512, 1024\}$), and QLoRA baseline settings. |
| `ANNOTATION_AND_SPLIT_PROTOCOL.md` | Documentation of the OWL 2 annotation guidelines, W3C SIOC mappings, stratified negative-pair construction, and chronological root-thread partitioning. |
| `run_all_verifications.py` | Single-command test runner executing all 8 verification suites. |

---

## Quick Start & One-Command Verification

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Complete Verification Suite

```bash
python run_all_verifications.py
```

Or run individual modules directly:

```bash
# 1. Parse W3C OWL 2 DL file, compile ontology masks & reproduce Table 3
python compile_ontology_masks.py

# 2. Verify causal anti-leakage across GAT neighborhoods, PE_graph snapshots (Eq. 20), and Louvain sampling
python causal_subgraph_builder.py

# 3. Run PyTorch HyKGA (3-layer GAT + 12-layer Transformer) & Proposition 1 smoke tests
python hykga_model.py

# 4. Run end-to-end training & 4-regime evaluation pipeline (hard, soft, adaptive, posthoc)
python train_and_evaluate.py

# 5. Run cross-dataset task-transfer adapters (RumourEval Twitter/Reddit, PHEME, Twitter15)
python transfer_public_benchmarks.py

# 6. Verify all empirical tables (Tables 4, 5, 6, 7, 9, 10, 17), 500-bin rank histograms, and Eq. 42 optimality
python verify_metrics_and_bounds.py

# 7. Validate TSAB manifest schema & audit all 65 reported manuscript rows
python hykga_eval.py dataset --pairs records/tsab_manifest_sample.jsonl
python hykga_eval.py audit --reported records/reported_tables_audit.json
```

---

## Citation

```bibtex
@article{fathallah2026hykga,
  author  = {Fathallah, Wyssem and Gueddes, Abdelweheb and Mars, Mourad and AlGhamdi, Mohammed J. and Mahjoub, Mohamed Ali and Sakli, Hedi},
  title   = {Neuro-Symbolic Graph Transformers with Ontology-Guided Attention for Context-Aware Social Discourse Analysis},
  journal = {Technologies},
  year    = {2026},
  publisher = {MDPI}
}
```
