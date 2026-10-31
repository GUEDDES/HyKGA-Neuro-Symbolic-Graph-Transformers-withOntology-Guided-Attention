#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HyKGA Symbolic Schema Compiler & Training Conflict Resolution Verifier
======================================================================
Parses the formal W3C OWL 2 DL ontology (`social_discourse_ontology.owl`) and compiles
O = (C, R, A, S) into:
  1. Directed concept matrix C_dir in {0,1}^{7x7} (Eq. 12)
  2. Symmetric reflexive attention compatibility matrix C = C_dir v C_dir^T v I (Eq. 12)
  3. Open-world TBox relation mask m_A(c_h, c_t)[r] in {0,1}^{7x7x8} (Eq. 13)
  4. Closed-world instance mask m_S(e_h, e_t)[r] (Eq. 14) and functional cardinality decoder
  5. Combined relation mask m_R = m_A * m_S (Eq. 15)
  6. Training-time conflict-damped logit mask L_tilde_R^(train) (Eq. 4) and unmasked L_onto (Eq. 3)
  7. Progressive axiom-removal compiler for ontology incompleteness analysis (Section 4.4.1, Table 11)
"""

import os
import random
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

CONCEPTS = ["User", "Message", "Topic", "Location", "Sentiment", "Event", "Hashtag"]
CONCEPT_IDX = {c: i for i, c in enumerate(CONCEPTS)}

RELATIONS = [
    "authored_by",
    "mentions",
    "located_in",
    "expresses_sentiment",
    "interacts_with",
    "follows",
    "retweets",
    "no_relation",
]
REL_IDX = {r: i for i, r in enumerate(RELATIONS)}

# Default open-world TBox Domain & Range Axioms A (Eqs. 5-10)
TBOX_AXIOMS: Dict[str, Tuple[Set[str], Set[str]]] = {
    "authored_by": ({"Message"}, {"User"}),
    "mentions": ({"Message"}, {"User", "Hashtag", "Topic", "Event"}),
    "located_in": ({"Message", "User", "Event"}, {"Location"}),
    "expresses_sentiment": ({"Message"}, {"Sentiment"}),
    "interacts_with": ({"User"}, {"User"}),
    "follows": ({"User"}, {"User"}),
    "retweets": ({"User"}, {"Message"}),
}

FUNCTIONAL_RELATIONS = {"authored_by", "expresses_sentiment"}
IRREFLEXIVE_RELATIONS = {"follows"}


def _local_name(iri: str) -> str:
    if "#" in iri:
        return iri.split("#")[-1]
    return iri.rstrip("/").split("/")[-1]


def parse_owl_ontology(owl_path: Optional[str] = None) -> Dict:
    """
    Parses `social_discourse_ontology.owl` (RDF/XML W3C OWL 2 DL format) to extract:
      - Atomic concepts C and pairwise disjointness axiom (owl:AllDisjointClasses)
      - Object properties R with their rdfs:domain and rdfs:range class expressions
      - Closed-world integrity constraints S (owl:FunctionalObjectProperty, owl:IrreflexiveObjectProperty)
    """
    if owl_path is None:
        owl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "social_discourse_ontology.owl")

    ns = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "owl": "http://www.w3.org/2002/07/owl#",
        "sh": "http://www.w3.org/ns/shacl#",
        "swrl": "http://www.w3.org/2003/11/swrl#",
    }
    tree = ET.parse(owl_path)
    root = tree.getroot()

    classes: List[str] = []
    platform_subclasses: Dict[str, str] = {}
    qualified_restrictions: List[Tuple[str, str, str, int]] = []

    for cls_el in root.findall("owl:Class", ns):
        about = cls_el.attrib.get(f"{{{ns['rdf']}}}about", "")
        if not about:
            continue
        cls_name = _local_name(about)
        internal_parent = None
        for sub_el in cls_el.findall("rdfs:subClassOf", ns):
            res_attr = sub_el.attrib.get(f"{{{ns['rdf']}}}resource", "")
            if res_attr.startswith("#"):
                internal_parent = _local_name(res_attr)
            restr_el = sub_el.find("owl:Restriction", ns)
            if restr_el is not None:
                on_prop_el = restr_el.find("owl:onProperty", ns)
                on_cls_el = restr_el.find("owl:onClass", ns)
                max_q_el = restr_el.find("owl:maxQualifiedCardinality", ns)
                if on_prop_el is not None and on_cls_el is not None and max_q_el is not None:
                    p_name = _local_name(on_prop_el.attrib.get(f"{{{ns['rdf']}}}resource", ""))
                    c_target = _local_name(on_cls_el.attrib.get(f"{{{ns['rdf']}}}resource", ""))
                    max_q = int((max_q_el.text or "1").strip())
                    qualified_restrictions.append((cls_name, p_name, c_target, max_q))
        if internal_parent is not None:
            platform_subclasses[cls_name] = internal_parent
        else:
            classes.append(cls_name)

    disjoint_members: List[str] = []
    for disj_el in list(root.findall("owl:AllDisjointClasses", ns)) + list(root.findall("rdf:Description", ns)):
        members_el = disj_el.find("owl:members", ns)
        if members_el is not None:
            for desc in members_el.findall("rdf:Description", ns):
                about = desc.attrib.get(f"{{{ns['rdf']}}}about", "")
                if about:
                    disjoint_members.append(_local_name(about))

    def _extract_class_set(parent_el: ET.Element, tag: str) -> Set[str]:
        res: Set[str] = set()
        el = parent_el.find(tag, ns)
        if el is None:
            return res
        res_attr = el.attrib.get(f"{{{ns['rdf']}}}resource", "")
        if res_attr:
            res.add(_local_name(res_attr))
            return res
        cls_inner = el.find("owl:Class", ns)
        if cls_inner is not None:
            union_el = cls_inner.find("owl:unionOf", ns)
            if union_el is not None:
                for desc in union_el.findall("rdf:Description", ns):
                    about = desc.attrib.get(f"{{{ns['rdf']}}}about", "")
                    if about:
                        res.add(_local_name(about))
        return res

    parsed_tbox: Dict[str, Tuple[Set[str], Set[str]]] = {}
    functional_props: Set[str] = set()
    irreflexive_props: Set[str] = set()

    for prop_el in root.findall("owl:ObjectProperty", ns):
        about = prop_el.attrib.get(f"{{{ns['rdf']}}}about", "")
        if not about:
            continue
        r_name = _local_name(about)
        dom_set = _extract_class_set(prop_el, "rdfs:domain")
        ran_set = _extract_class_set(prop_el, "rdfs:range")
        parsed_tbox[r_name] = (dom_set, ran_set)

        for rdf_type in prop_el.findall("rdf:type", ns):
            type_iri = rdf_type.attrib.get(f"{{{ns['rdf']}}}resource", "")
            type_local = _local_name(type_iri)
            if type_local in ("FunctionalProperty", "FunctionalObjectProperty"):
                functional_props.add(r_name)
            elif type_local in ("IrreflexiveProperty", "IrreflexiveObjectProperty"):
                irreflexive_props.add(r_name)

    datatype_props: List[str] = []
    for dt_el in root.findall("owl:DatatypeProperty", ns):
        about = dt_el.attrib.get(f"{{{ns['rdf']}}}about", "")
        if about:
            datatype_props.append(_local_name(about))

    shacl_shapes: List[str] = []
    for sh_el in root.findall("sh:NodeShape", ns):
        about = sh_el.attrib.get(f"{{{ns['rdf']}}}about", "")
        if about:
            shacl_shapes.append(_local_name(about))

    swrl_rules: List[str] = []
    for swrl_el in root.findall("swrl:Imp", ns):
        about = swrl_el.attrib.get(f"{{{ns['rdf']}}}about", "")
        if about:
            swrl_rules.append(_local_name(about))

    return {
        "concepts": classes,
        "platform_subclasses": platform_subclasses,
        "disjoint_concepts": disjoint_members,
        "tbox_axioms": parsed_tbox,
        "functional_relations": functional_props,
        "irreflexive_relations": irreflexive_props,
        "qualified_restrictions": qualified_restrictions,
        "datatype_properties": datatype_props,
        "shacl_shapes": shacl_shapes,
        "swrl_rules": swrl_rules,
    }


def compile_tbox_masks(
    owl_path: Optional[str] = None,
    custom_tbox: Optional[Dict[str, Tuple[Set[str], Set[str]]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compiles the TBox axioms A into C_dir, symmetric reflexive C, and m_A.
    If custom_tbox is None and social_discourse_ontology.owl exists, parses the OWL file directly.
    """
    if custom_tbox is not None:
        axioms = custom_tbox
    else:
        default_owl = owl_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "social_discourse_ontology.owl"
        )
        if os.path.exists(default_owl):
            parsed = parse_owl_ontology(default_owl)
            axioms = parsed["tbox_axioms"]
            assert set(parsed["concepts"]) == set(CONCEPTS), "OWL concepts mismatch"
            assert set(parsed["disjoint_concepts"]) == set(CONCEPTS), "OWL disjoint classes mismatch"
            assert parsed["functional_relations"] == FUNCTIONAL_RELATIONS
            assert parsed["irreflexive_relations"] == IRREFLEXIVE_RELATIONS
            assert parsed["platform_subclasses"] == {"Submission": "Message", "Comment": "Message"}
            assert len(parsed["qualified_restrictions"]) == 2
            assert len(parsed["datatype_properties"]) == 4
            assert len(parsed["shacl_shapes"]) == 2
            assert len(parsed["swrl_rules"]) == 3
        else:
            axioms = TBOX_AXIOMS

    n_c = len(CONCEPTS)
    n_y = len(RELATIONS)
    C_dir = np.zeros((n_c, n_c), dtype=int)
    m_A = np.zeros((n_c, n_c, n_y), dtype=int)

    for i, c_h in enumerate(CONCEPTS):
        for j, c_t in enumerate(CONCEPTS):
            m_A[i, j, REL_IDX["no_relation"]] = 1
            for r, (dom, ran) in axioms.items():
                if (c_h in dom) and (c_t in ran):
                    m_A[i, j, REL_IDX[r]] = 1
                    C_dir[i, j] = 1

    I = np.eye(n_c, dtype=int)
    C = np.bitwise_or(np.bitwise_or(C_dir, C_dir.T), I)
    return C_dir, C, m_A


def compile_degraded_ontology_masks(drop_fraction: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements progressive ontology axiom removal (Section 4.4.1, Table 11 `tab:robustness`).
    Note: Dropping a restrictive domain/range axiom relaxes the mask (admitting previously
    forbidden pairs rather than blocking valid ones).
    """
    assert 0.0 <= drop_fraction <= 1.0
    C_dir_full, C_full, m_A_full = compile_tbox_masks()
    if drop_fraction == 0.0:
        return C_full, m_A_full

    rng = random.Random(seed)
    C_deg = C_full.copy()
    m_A_deg = m_A_full.copy()

    # Collect all masked concept pairs in C_full and forbidden triples in m_A_full
    masked_c_pairs = [(i, j) for i in range(len(CONCEPTS)) for j in range(i + 1, len(CONCEPTS)) if C_full[i, j] == 0]
    n_unmask_c = int(round(drop_fraction * len(masked_c_pairs)))
    for i, j in rng.sample(masked_c_pairs, n_unmask_c):
        C_deg[i, j] = 1
        C_deg[j, i] = 1

    masked_triples = [
        (i, j, r)
        for i in range(len(CONCEPTS))
        for j in range(len(CONCEPTS))
        for r in range(len(RELATIONS) - 1)
        if m_A_full[i, j, r] == 0
    ]
    n_unmask_r = int(round(drop_fraction * len(masked_triples)))
    for i, j, r in rng.sample(masked_triples, n_unmask_r):
        m_A_deg[i, j, r] = 1

    return C_deg, m_A_deg


def compute_m_S(e_h: str, e_t: str) -> np.ndarray:
    """Closed-world instance mask m_S(e_h, e_t) enforcing Irreflexive(follows) (Eq. 14)."""
    mask = np.ones(len(RELATIONS), dtype=int)
    if e_h == e_t:
        mask[REL_IDX["follows"]] = 0
    return mask


def compute_m_R(e_h: str, e_t: str, c_h: str, c_t: str, m_A: np.ndarray) -> np.ndarray:
    """Combined TBox + Closed-World Instance Mask m_R = m_A * m_S (Eq. 15)."""
    return m_A[CONCEPT_IDX[c_h], CONCEPT_IDX[c_t]] * compute_m_S(e_h, e_t)


def training_conflict_mask(m_R_vec: np.ndarray, gold_y_idx: int, lambda_conflict: float = 5.0) -> np.ndarray:
    """
    Training-time conflict-damped logit mask L_tilde_R^(train) (Eq. 4).
    Prevents -log(0) = +inf divergence when NER mistypes an entity and excludes gold y_i.
    """
    L_tilde = np.zeros_like(m_R_vec, dtype=float)
    gold_admissible = (m_R_vec[gold_y_idx] == 1)
    for r_idx in range(len(m_R_vec)):
        if m_R_vec[r_idx] == 1:
            L_tilde[r_idx] = 0.0
        elif not gold_admissible:
            L_tilde[r_idx] = -lambda_conflict
        else:
            L_tilde[r_idx] = -1e9  # -inf in FP32
    return L_tilde


def softmax(x: np.ndarray) -> np.ndarray:
    x_max = np.max(x)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x)


def verify_worked_example(m_A: np.ndarray):
    """Reproduces Table 3 (tab:conflict_example) exactly."""
    z = np.full(len(RELATIONS), -1.0, dtype=float)
    z[REL_IDX["authored_by"]] = 3.20
    z[REL_IDX["mentions"]] = 1.80
    z[REL_IDX["no_relation"]] = 0.50

    gold_y = REL_IDX["authored_by"]
    m_R_gold = compute_m_R("msg_1", "@PolitiWatch", "Message", "User", m_A)
    p_raw = softmax(z)
    L_onto = float(np.sum(p_raw[m_R_gold == 0]))

    # 1. Correct typing (Message, User)
    L1 = training_conflict_mask(m_R_gold, gold_y, lambda_conflict=5.0)
    y_hat_1 = softmax(z + L1)
    loss_1 = -np.log(y_hat_1[gold_y])

    # 2. Mistyped tail as Hashtag + Naive Hard Mask
    m_R_mistyped = compute_m_R("msg_1", "@PolitiWatch", "Message", "Hashtag", m_A)
    y_hat_2 = softmax(np.where(m_R_mistyped == 1, z, -1e9))

    # 3. Mistyped tail as Hashtag + Training Conflict Clamp (lambda_conflict = 5.0)
    L3 = training_conflict_mask(m_R_mistyped, gold_y, lambda_conflict=5.0)
    y_hat_3 = softmax(z + L3)
    loss_3 = -np.log(y_hat_3[gold_y])

    # 4. Mistyped tail as Hashtag + Soft Inference Mask (lambda_pen = 1.0)
    L4 = np.where(m_R_mistyped == 1, 0.0, -1.0)
    y_hat_4 = softmax(z + L4)
    loss_4 = -np.log(y_hat_4[gold_y])

    print("=== Worked Numerical Example Verification (Table 3) ===")
    print(f"Unmasked L_onto(p_raw): {L_onto:.3f}")
    print(f"Regime 1 (Correct Typing):   y_hat[authored]={y_hat_1[gold_y]:.3f}, y_hat[mentions]={y_hat_1[REL_IDX['mentions']]:.3f}, L_task={loss_1:.3f}")
    print(f"Regime 2 (Naive Hard Mask):  y_hat[authored]={y_hat_2[gold_y]:.3f}, y_hat[mentions]={y_hat_2[REL_IDX['mentions']]:.3f}, L_task=+inf")
    print(f"Regime 3 (Conflict Clamp):   y_hat[authored]={y_hat_3[gold_y]:.3f}, y_hat[mentions]={y_hat_3[REL_IDX['mentions']]:.3f}, L_task={loss_3:.3f}")
    print(f"Regime 4 (Soft Inference):   y_hat[authored]={y_hat_4[gold_y]:.3f}, y_hat[mentions]={y_hat_4[REL_IDX['mentions']]:.3f}, L_task={loss_4:.3f}")


if __name__ == "__main__":
    parsed = parse_owl_ontology()
    print(f"Parsed OWL 2 DL ontology: {len(parsed['concepts'])} disjoint classes, "
          f"{len(parsed['platform_subclasses'])} SIOC platform subclasses ({list(parsed['platform_subclasses'].keys())}), "
          f"{len(parsed['tbox_axioms'])} object properties, "
          f"{len(parsed['datatype_properties'])} datatype properties")
    print(f"  OWL 2 DL Restrictions: {len(parsed['qualified_restrictions'])} qualified cardinalities, "
          f"functional={sorted(parsed['functional_relations'])}, irreflexive={sorted(parsed['irreflexive_relations'])}")
    print(f"  Closed-World SHACL Shapes: {parsed['shacl_shapes']}")
    print(f"  Deductive SWRL Rules: {parsed['swrl_rules']}")
    C_dir, C, m_A = compile_tbox_masks()
    print(f"Compiled C_dir sum: {int(C_dir.sum())} / 49")
    print(f"Compiled symmetric reflexive C sum: {int(C.sum())} / 49 (23 admissible, 26 masked = {26/49*100:.1f}%)")
    verify_worked_example(m_A)
