#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Reproducibility & Verification Suite for HyKGA
=====================================================
Repository: https://github.com/GUEDDES/HyKGA-Neuro-Symbolic-Graph-Transformers-withOntology-Guided-Attention
Manuscript ID: technologies-4653627 (MDPI Technologies)

Executes all 8 verification suites in sequence:
  1. compile_ontology_masks.py     (Direct W3C OWL 2 DL XML parser, C_dir, C, m_A, m_S, m_R, Table 3 worked example)
  2. causal_subgraph_builder.py    (Causal subgraph, target-excised PE_graph snapshots, Louvain negative sampling)
  3. hykga_model.py                (3-layer CausalMultiHeadGAT + 12-layer Dual-Attention Transformer + <=1 cardinality decoding)
  4. train_and_evaluate.py         (End-to-end training loop with gold-type warmup -> predicted-type conflict clamp & 4-regime evaluation)
  5. transfer_public_benchmarks.py (RumourEval Twitter/Reddit SIOC transfer, PHEME veracity/RE, Twitter15 cascade heads)
  6. verify_metrics_and_bounds.py  (Tables 4, 5, 6, 7, 9, 10, 17, 500-bin query rank histograms, and Eq. 42 optimality condition)
  7. hykga_eval.py dataset         (TSAB manifest schema & functional cardinality validation)
  8. hykga_eval.py audit           (Full consistency audit of all 65 reported empirical table rows)
"""

import os
import subprocess
import sys


def run_step(title: str, cmd: list):
    print("\n" + "=" * 84)
    print(f"RUNNING: {title}")
    print("=" * 84)
    res = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if res.returncode != 0:
        sys.exit(f"[FAILED] Step '{title}' exited with code {res.returncode}")


def main():
    py = sys.executable
    run_step("1/8 W3C OWL 2 DL Parser, Symbolic Compilation & Conflict Example (Table 3)", [py, "compile_ontology_masks.py"])
    run_step("2/8 Causal Anti-Leakage Graph, Centrality Snapshots & Louvain Sampler (Eqs. 18-20)", [py, "causal_subgraph_builder.py"])
    run_step("3/8 PyTorch HyKGA Architecture (3-Layer GAT + 12-Layer Transformer) Smoke Test", [py, "hykga_model.py"])
    run_step("4/8 End-to-End Training & Multi-Regime Inference Pipeline (Algorithms 1 & 2)", [py, "train_and_evaluate.py"])
    run_step("5/8 Cross-Dataset Task-Transfer & SIOC Platform Mapping (Section 4.5, Table 12)", [py, "transfer_public_benchmarks.py"])
    run_step("6/8 Mathematical Bounds, 500-Bin Rank Records & Soft-Masking Optimality", [py, "verify_metrics_and_bounds.py"])
    run_step("7/8 TSAB Manifest Schema & Functional Cardinality Validation", [py, "hykga_eval.py", "dataset", "--pairs", "records/tsab_manifest_sample.jsonl"])
    run_step("8/8 Full Audit of All 65 Reported Empirical Table Rows", [py, "hykga_eval.py", "audit", "--reported", "records/reported_tables_audit.json"])
    print("\n" + "#" * 84)
    print("ALL 8 REPRODUCIBILITY & MATHEMATICAL VERIFICATION SUITES PASSED SUCCESSFULLY!")
    print("#" * 84)


if __name__ == "__main__":
    main()
