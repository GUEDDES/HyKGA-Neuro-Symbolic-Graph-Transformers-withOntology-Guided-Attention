#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hykga_eval.py - Unified Metric Calculator & Mathematical Consistency Auditor for HyKGA
======================================================================================
Repository: https://github.com/GUEDDES/HyKGA-Neuro-Symbolic-Graph-Transformers-withOntology-Guided-Attention

Subcommands:
  eval     Computes P, R, Micro-F1, OVR_proj, OVR_pred, OVR_pos, OAS_A, OAS_{A u S},
           Hits@k, and MRR from per-instance prediction files (one file per model and seed),
           and outputs metrics.json + tables.tex.
  dataset  Verifies a TSAB pair manifest (split counts, TBox domain/range admissibility,
           irreflexivity of follows, and <= 1 functional cardinality per message).
  audit    Audits reported table figures (JSON) for mathematical compatibility across P, R, F1,
           OVR_proj, OVR_pred, OVR_pos, OAS_A, OAS, Hits@{1,3,10}, and MRR bounds.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict

CONCEPTS = ["User", "Message", "Topic", "Location", "Sentiment", "Event", "Hashtag"]
RELATIONS = ["authored_by", "mentions", "located_in", "expresses_sentiment",
             "interacts_with", "follows", "retweets"]
NO_REL = "no_relation"
TBOX_AXIOMS = {
    "authored_by": ({"Message"}, {"User"}),
    "mentions": ({"Message"}, {"User", "Hashtag", "Topic", "Event"}),
    "located_in": ({"Message", "User", "Event"}, {"Location"}),
    "expresses_sentiment": ({"Message"}, {"Sentiment"}),
    "interacts_with": ({"User"}, {"User"}),
    "follows": ({"User"}, {"User"}),
    "retweets": ({"User"}, {"Message"}),
}
FUNCTIONAL_RELATIONS = {"authored_by", "expresses_sentiment"}   # <= 1 per message
IRREFLEXIVE_RELATIONS = {"follows"}


def tbox_admissible(c_h, c_t, r):
    if r == NO_REL:
        return True
    dom, ran = TBOX_AXIOMS[r]
    return (c_h in dom) and (c_t in ran)


def read_rows(path):
    """Reads a CSV (, or ;) or JSONL file and returns a list of dicts."""
    with open(path, "r", encoding="utf-8-sig") as f:
        head = f.read(4096)
        f.seek(0)
        if path.lower().endswith((".jsonl", ".json")):
            return [json.loads(line) for line in f if line.strip()]
        delim = ";" if head.count(";") > head.count(",") else ","
        return list(csv.DictReader(f, delimiter=delim))


def need(row, key, path):
    if key not in row or row[key] in (None, ""):
        sys.exit(f"[ERROR] missing column '{key}' in {path}")
    return str(row[key]).strip()


def mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, s


def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > FPMIN else FPMIN)
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > FPMIN else FPMIN)
        c = 1.0 + aa / c; c = c if abs(c) > FPMIN else FPMIN
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > FPMIN else FPMIN)
        c = 1.0 + aa / c; c = c if abs(c) > FPMIN else FPMIN
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    bt = math.exp(lbt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def paired_ttest(xs, ys):
    """Two-tailed paired t-test. Returns (t, p)."""
    d = [x - y for x, y in zip(xs, ys)]
    n = len(d)
    if n < 2:
        return float("nan"), float("nan")
    m, s = mean_std(d)
    if s == 0:
        return float("inf"), 0.0
    t = m / (s / math.sqrt(n))
    df = n - 1
    p = betainc(df / 2.0, 0.5, df / (df + t * t))
    return t, p


def relation_metrics(rows, path, multi_min_ent=3, multi_max_tok=12):
    N = len(rows)
    tp = fp = fn = 0
    n_pred_pos = n_gold_pos = 0
    gold_viol = 0
    tbox_v = irr_v = func_v = 0
    any_v = 0
    viol_flags = [False] * N
    typing_err = 0
    has_pred_types = False
    by_msg_rel = defaultdict(list)

    for i, r in enumerate(rows):
        gold = need(r, "gold", path)
        pred = need(r, "pred", path)
        ch, ct = need(r, "head_type", path), need(r, "tail_type", path)
        eh, et = need(r, "head_id", path), need(r, "tail_id", path)
        for lab in (gold, pred):
            if lab != NO_REL and lab not in TBOX_AXIOMS:
                sys.exit(f"[ERROR] unknown label '{lab}' in {path} (line {i + 2})")
        if gold != NO_REL:
            n_gold_pos += 1
            if not tbox_admissible(ch, ct, gold) or (gold in IRREFLEXIVE_RELATIONS and eh == et):
                gold_viol += 1
        if pred != NO_REL:
            n_pred_pos += 1
            if pred == gold:
                tp += 1
            else:
                fp += 1
        if gold != NO_REL and pred != gold:
            fn += 1
        v = False
        if not tbox_admissible(ch, ct, pred):
            tbox_v += 1; v = True
        if pred in IRREFLEXIVE_RELATIONS and eh == et:
            irr_v += 1; v = True
        if pred in FUNCTIONAL_RELATIONS:
            by_msg_rel[(need(r, "message_id", path), pred)].append(i)
        viol_flags[i] = v
        if "head_type_pred" in r and r["head_type_pred"] not in (None, ""):
            has_pred_types = True
            if r["head_type_pred"] != ch or r.get("tail_type_pred", ct) != ct:
                typing_err += 1

    gold_func = defaultdict(int)
    for r in rows:
        if r["gold"] in FUNCTIONAL_RELATIONS:
            gold_func[(str(r["message_id"]).strip(), r["gold"])] += 1
    gold_viol += sum(c - 1 for c in gold_func.values() if c > 1)
    for (_, rel_name), idx in by_msg_rel.items():
        if len(idx) < 2:
            continue
        correct = [j for j in idx if rows[j]["gold"] == rel_name]
        keep = correct[0] if correct else idx[0]
        for j in idx:
            if j != keep:
                func_v += 1
                viol_flags[j] = True
    any_v = sum(viol_flags)

    P = tp / n_pred_pos if n_pred_pos else 0.0
    R = tp / n_gold_pos if n_gold_pos else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0

    out = dict(N=N, n_gold_pos=n_gold_pos, n_pred_pos=n_pred_pos, TP=tp, FP=fp, FN=fn,
               P=P, R=R, F1=F1,
               viol_any=any_v, viol_tbox=tbox_v, viol_irr=irr_v, viol_func=func_v,
               OVR_all=100.0 * any_v / N if N else 0.0,
               OVR_pos=100.0 * any_v / n_pred_pos if n_pred_pos else 0.0,
               share_tbox=100.0 * tbox_v / N, share_irr=100.0 * irr_v / N, share_func=100.0 * func_v / N,
               OAS_A=1.0 - tbox_v / N if N else 1.0,
               OAS=1.0 - any_v / N if N else 1.0,
               gold_violations=gold_viol)
    if has_pred_types:
        out["typing_error_rate"] = 100.0 * typing_err / N
    if rows and "n_entities" in rows[0] and "n_tokens" in rows[0]:
        sub = [k for k, r in enumerate(rows)
               if int(r["n_entities"]) >= multi_min_ent and int(r["n_tokens"]) <= multi_max_tok]
        out["n_multi"] = len(sub)
        out["OVR_multi"] = 100.0 * sum(viol_flags[k] for k in sub) / len(sub) if sub else float("nan")
    return out


def check_relation(m, label):
    err = []
    if m["gold_violations"] > 0:
        err.append(f"{label}: {m['gold_violations']} gold labels violate the ontology")
    if m["viol_any"] > m["FP"] and m["gold_violations"] == 0:
        err.append(f"{label}: violations ({m['viol_any']}) > false positives ({m['FP']})")
    if m["OVR_all"] > m["OVR_pos"] + 1e-9:
        err.append(f"{label}: OVR_pred > OVR_pos")
    if m["OAS_A"] + 1e-12 < m["OAS"]:
        err.append(f"{label}: OAS_A < OAS_(A u S)")
    return err


def mrr_bounds(h1, h3, h10, n_cand):
    lo = h1 + (h3 - h1) / 3.0 + (h10 - h3) / 10.0 + (1.0 - h10) / float(n_cand)
    hi = h1 + (h3 - h1) / 2.0 + (h10 - h3) / 4.0 + (1.0 - h10) / 11.0
    return lo, hi


def link_metrics(rows, path, n_cand_default):
    ranks = []
    n_cand = n_cand_default
    for i, r in enumerate(rows):
        rk = float(need(r, "rank", path))
        if rk < 1:
            sys.exit(f"[ERROR] rank < 1 in {path} (line {i + 2})")
        ranks.append(rk)
        if "n_candidates" in r and r["n_candidates"] not in (None, ""):
            n_cand = max(n_cand, int(r["n_candidates"]))
    Q = len(ranks)
    h = {k: sum(1 for x in ranks if x <= k) / Q for k in (1, 3, 10)}
    mrr = sum(1.0 / x for x in ranks) / Q
    lo, hi = mrr_bounds(h[1], h[3], h[10], n_cand)
    return dict(Q=Q, H1=h[1], H3=h[3], H10=h[10], MRR=mrr, MRR_min=lo, MRR_max=hi, n_cand=n_cand)


def check_link(m, label):
    err = []
    if not (m["MRR_min"] - 1e-9 <= m["MRR"] <= m["MRR_max"] + 1e-9):
        err.append(f"{label}: MRR {m['MRR']:.4f} out of bounds [{m['MRR_min']:.4f}, {m['MRR_max']:.4f}]")
    if not (m["H1"] <= m["H3"] <= m["H10"]):
        err.append(f"{label}: Hits@k non-monotonic")
    return err


FNAME_RE = re.compile(r"^(?P<model>.+?)__seed(?P<seed>\d+)\.(csv|jsonl|json)$", re.I)


def collect(dirpath):
    groups = defaultdict(dict)
    if not os.path.isdir(dirpath):
        return groups
    for fn in sorted(os.listdir(dirpath)):
        m = FNAME_RE.match(fn)
        if m:
            groups[m.group("model")][int(m.group("seed"))] = os.path.join(dirpath, fn)
        elif not fn.startswith("."):
            print(f"[WARNING] ignored file (expected MODEL__seedN.csv): {fn}")
    return groups


def fmt_pm(m, s):
    return f"{m:.2f} $\\pm$ {s:.2f}"


def cmd_eval(a):
    rel = collect(os.path.join(a.predictions, "relations"))
    lnk = collect(os.path.join(a.predictions, "link_prediction"))
    if not rel and not lnk:
        sys.exit("[ERROR] no files found in relations/ or link_prediction/")
    errors, results = [], {"relations": {}, "link_prediction": {}}
    ref_n = None
    for model, seeds in rel.items():
        per_seed = {}
        for seed, path in sorted(seeds.items()):
            m = relation_metrics(read_rows(path), path)
            if ref_n is None:
                ref_n = (m["N"], m["n_gold_pos"])
            elif (m["N"], m["n_gold_pos"]) != ref_n:
                errors.append(f"{model} seed {seed}: test population mismatch "
                              f"(N={m['N']}, positives={m['n_gold_pos']}) vs reference {ref_n}")
            errors += check_relation(m, f"{model} seed {seed}")
            per_seed[seed] = m
        agg = {}
        for k in ("P", "R", "F1", "OVR_all", "OVR_pos", "OAS_A", "OAS",
                  "share_tbox", "share_irr", "share_func", "OVR_multi", "typing_error_rate"):
            vals = [v[k] for v in per_seed.values() if k in v and not math.isnan(v[k])]
            if vals:
                agg[k] = mean_std(vals)
        results["relations"][model] = {"per_seed": per_seed, "mean_std": agg}
    for model, seeds in lnk.items():
        per_seed = {}
        for seed, path in sorted(seeds.items()):
            m = link_metrics(read_rows(path), path, a.n_candidates)
            errors += check_link(m, f"{model} seed {seed} (link)")
            per_seed[seed] = m
        agg = {k: mean_std([v[k] for v in per_seed.values()]) for k in ("H1", "H3", "H10", "MRR")}
        lo, hi = mrr_bounds(agg["H1"][0], agg["H3"][0], agg["H10"][0],
                            max(v["n_cand"] for v in per_seed.values()))
        agg["bounds"] = (lo, hi)
        results["link_prediction"][model] = {"per_seed": per_seed, "mean_std": agg}

    ref = a.reference
    if ref in results["relations"]:
        ref_seeds = results["relations"][ref]["per_seed"]
        for model, d in results["relations"].items():
            if model == ref:
                continue
            common = sorted(set(ref_seeds) & set(d["per_seed"]))
            t, p = paired_ttest([ref_seeds[s]["F1"] for s in common], [d["per_seed"][s]["F1"] for s in common])
            d["ttest_F1_vs_ref"] = {"seeds": common, "t": t, "p": p}

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    L = []
    L.append("% === Table 5: P, R, Micro-F1, OVR, OAS, MRR ===")
    for model, d in results["relations"].items():
        g = d["mean_std"]
        mrr = results["link_prediction"].get(model, {}).get("mean_std", {}).get("MRR")
        mrr_s = f"{mrr[0]:.3f}" if mrr else "---"
        pv = d.get("ttest_F1_vs_ref", {}).get("p")
        L.append(f"{model} & {fmt_pm(*g['P'])} & {fmt_pm(*g['R'])} & {fmt_pm(*g['F1'])} & "
                 f"{g['OVR_all'][0]:.1f}\\% & {g['OAS'][0]:.2f} & {mrr_s}\\\\"
                 + (f"  % p(F1 vs {ref}) = {pv:.4g}" if pv is not None else ""))
    with open(os.path.join(a.out, "tables.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    print(f"Results written to {a.out}/metrics.json and {a.out}/tables.tex")
    if errors:
        for e in errors:
            print("  - " + e)
        sys.exit(2)
    print("All consistency checks passed.")


def cmd_dataset(a):
    rows = read_rows(a.pairs)
    splits = defaultdict(lambda: defaultdict(int))
    ents = defaultdict(lambda: defaultdict(set))
    msg_func = defaultdict(int)
    errors = []
    for i, r in enumerate(rows):
        sp = need(r, "split", a.pairs)
        gold = need(r, "gold", a.pairs)
        ch, ct = need(r, "head_type", a.pairs), need(r, "tail_type", a.pairs)
        eh, et = need(r, "head_id", a.pairs), need(r, "tail_id", a.pairs)
        splits[sp][gold] += 1
        ents[sp][ch].add(eh)
        ents[sp][ct].add(et)
        if not tbox_admissible(ch, ct, gold):
            errors.append(f"line {i + 2}: gold '{gold}' violates domain/range ({ch}, {ct})")
        if gold in IRREFLEXIVE_RELATIONS and eh == et:
            errors.append(f"line {i + 2}: gold follows(u,u)")
        if gold in FUNCTIONAL_RELATIONS:
            msg_func[(need(r, "message_id", a.pairs), gold)] += 1
    for (msg, rel), c in msg_func.items():
        if c > 1:
            errors.append(f"message {msg}: {c} gold relations '{rel}' (functional cardinality <= 1 violated)")
    labels = RELATIONS + [NO_REL]
    order = [s for s in ("train", "val", "test") if s in splits] + [s for s in splits if s not in ("train", "val", "test")]
    total = {lab: sum(splits[s][lab] for s in order) for lab in labels}
    grand = sum(total.values())
    print("Relation".ljust(22) + "".join(s.rjust(10) for s in order) + "Total".rjust(10))
    for lab in labels:
        print(lab.ljust(22) + "".join(f"{splits[s][lab]:10d}" for s in order) + f"{total[lab]:10d}")
    print("TOTAL PAIRS".ljust(22) + "".join(f"{sum(splits[s].values()):10d}" for s in order) + f"{grand:10d}")
    for s in order:
        pos = sum(splits[s][l] for l in RELATIONS)
        print(f"  {s}: {pos} positive, {splits[s][NO_REL]} no_relation")
    print("\nUnique entities per concept type:")
    for c in CONCEPTS:
        print(c.ljust(22) + "".join(f"{len(ents[s][c]):10d}" for s in order))
    n_msg = len(set().union(*[ents[s]["Message"] for s in order])) if order else 0
    n_auth = sum(total[l] for l in ["authored_by"])
    if n_auth > n_msg:
        errors.append(f"{n_auth} authored_by relations for {n_msg} distinct Message entities (violates <= 1 author per message)")
    if errors:
        print(f"\n*** {len(errors)} DATASET ERROR(S) ***")
        for e in errors[:50]:
            print("  - " + e)
        sys.exit(2)
    print("\nDataset manifest is 100% consistent with all TBox and closed-world functional/irreflexive constraints -> PASS")


def cmd_audit(a):
    spec = json.load(open(a.reported, encoding="utf-8"))
    N, NPOS = spec["N_test"], spec["N_test_positive"]
    fails, warns, n_ok = [], [], 0
    for row in spec["rows"]:
        lab = f"[{row.get('table', '?')}] {row.get('model', '?')}"
        ovr = row.get("OVR_all")
        if ovr is None and row.get("OAS") is not None:
            ovr = 100.0 * (1.0 - row["OAS"])
        local = []
        if row.get("OVR_all") is not None and row.get("OAS") is not None:
            if abs(row["OAS"] - (1.0 - row["OVR_all"] / 100.0)) > 0.006:
                local.append(f"OAS={row['OAS']} != 1 - OVR_all/100 ({1.0 - row['OVR_all']/100.0:.3f})")
        if row.get("OVR_pos") is not None and row.get("P") is not None:
            mx_pos = 100.0 * (1.0 - (row["P"] - 0.005))
            if row["OVR_pos"] - 0.05 > mx_pos:
                local.append(f"OVR_pos {row['OVR_pos']}% > 1 - P = {mx_pos:.1f}%")
        if row.get("OVR_pred") is not None and row.get("OVR_pos") is not None and row.get("P") is not None and row.get("R") is not None:
            pos_share = (row["R"] * NPOS / row["P"]) / N
            exp_pred = pos_share * row["OVR_pos"]
            if abs(row["OVR_pred"] - exp_pred) > 0.2:
                local.append(f"OVR_pred {row['OVR_pred']}% != (|P_pos|/N)*OVR_pos ({exp_pred:.2f}%)")
        if row.get("OAS_A") is not None and row.get("OAS") is not None and row["OAS_A"] + 0.005 < row["OAS"] - 0.005:
            local.append("OAS_A < OAS_(A u S)")
        if all(row.get(k) is not None for k in ("H1", "H3", "H10", "MRR")):
            _, hi = mrr_bounds(min(row["H1"] + 0.005, 1), min(row["H3"] + 0.005, 1), min(row["H10"] + 0.005, 1),
                               spec.get("n_candidates", 500))
            lo2, _ = mrr_bounds(max(row["H1"] - 0.005, 0), max(row["H3"] - 0.005, 0), max(row["H10"] - 0.005, 0),
                                spec.get("n_candidates", 500))
            if not (lo2 - 0.0005 <= row["MRR"] <= hi + 0.0005):
                local.append(f"MRR {row['MRR']} out of bounds [{lo2:.3f}, {hi:.3f}]")
        if all(row.get(k) is not None for k in ("P", "R", "F1")):
            f_lo = 2 * (row["P"] - .005) * (row["R"] - .005) / (row["P"] + row["R"] - .01)
            f_hi = 2 * (row["P"] + .005) * (row["R"] + .005) / (row["P"] + row["R"] + .01)
            if not (f_lo - 0.01 <= row["F1"] <= f_hi + 0.01):
                warns.append(f"{lab}: F1={row['F1']} far from 2PR/(P+R) [{f_lo:.3f}, {f_hi:.3f}]")
        if local:
            fails += [f"{lab}: {m}" for m in local]
        else:
            n_ok += 1
    for chk in spec.get("sum_checks", []):
        s = round(sum(chk["parts"]), 6)
        if abs(s - chk["total"]) > chk.get("tol", 0):
            fails.append(f"[{chk['label']}] sum {s} != reported total {chk['total']}")
    for chk in spec.get("leq_checks", []):
        if chk["a"] > chk["b"]:
            fails.append(f"[{chk['label']}] {chk['a']} > {chk['b']}")
    print(f"Compliant rows: {n_ok} / {len(spec['rows'])}")
    if warns:
        print(f"\nWarnings ({len(warns)}):")
        for w in warns:
            print("  - " + w)
    if fails:
        print(f"\n*** {len(fails)} MATHEMATICAL INCONSISTENCY(IES) ***")
        for f_ in fails:
            print("  - " + f_)
        sys.exit(2)
    print("Zero mathematical inconsistencies detected -> PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval", help="compute metrics from prediction files")
    e.add_argument("--predictions", required=True, help="directory containing relations/ and link_prediction/")
    e.add_argument("--out", default="results", help="output directory")
    e.add_argument("--reference", default="HyKGA", help="reference model for paired t-test")
    e.add_argument("--n-candidates", type=int, default=500, help="|V_cand| for MRR_min bound")
    d = sub.add_parser("dataset", help="verify TSAB pair manifest")
    d.add_argument("--pairs", required=True)
    u = sub.add_parser("audit", help="audit reported table figures")
    u.add_argument("--reported", required=True)
    a = ap.parse_args()
    {"eval": cmd_eval, "dataset": cmd_dataset, "audit": cmd_audit}[a.cmd](a)


if __name__ == "__main__":
    main()
