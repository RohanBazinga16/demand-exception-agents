"""
Evaluation. Four layers, measured separately (retrieval quality != answer quality):

  1. Detection  — did the deterministic triage flag the IWLs with planted exceptions?
  2. Retrieval  — for a standard query per IWL, is the explaining note in the top k? (no LLM involved)
  3. Diagnosis  — did the agent (or rules) name the right cause and bucket, and is every cited note
                  (a) real, (b) actually retrieved in this run, (c) applicable to that IWL?
  4. Outcome    — on the hidden 12 weeks (W105-W116), does the human-approved reforecast beat the baseline?

ground_truth.csv and future_actuals.csv are read ONLY here.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from engine import DATA, LOC_NAMES, Engine

POSITIVE = lambda c: c != "NONE"  # noqa: E731


def load_truth():
    gt = pd.read_csv(os.path.join(DATA, "ground_truth.csv")).fillna("")
    fut = pd.read_csv(os.path.join(DATA, "future_actuals.csv"))
    return gt, fut


def _std_query(e, iwl):
    m = e.meta[iwl]
    return f"{m['sku']} {m['pack']} {LOC_NAMES[m['location']].split(' ')[0]}"


# ----------------------------------------------------------------- 1. detection
def detection_eval(e: Engine, gt: pd.DataFrame) -> dict:
    t = e.triage.merge(gt, on="iwl_id")
    pred = t.bucket != "Low-Touch"
    true = t.true_cause.apply(POSITIVE)
    tp, fp, fn = int((pred & true).sum()), int((pred & ~true).sum()), int((~pred & true).sum())
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec,
                f1=2 * prec * rec / (prec + rec) if prec + rec else 0,
                bucket_accuracy=float((t.bucket == t.expected_bucket).mean()),
                review_queue=int(pred.sum()), total=len(t),
                false_positives=t[pred & ~true].iwl_id.tolist(), missed=t[~pred & true].iwl_id.tolist(),
                confusion=pd.crosstab(t.expected_bucket, t.bucket, rownames=["expected"], colnames=["engine"]))


# ----------------------------------------------------------------- 2. retrieval
def retrieval_eval(e: Engine, gt: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    rows = []
    for r in gt[gt.planted].itertuples():
        q = _std_query(e, r.iwl_id)
        hits = e.search_notes(q, 6)
        ids = [h["note_id"] for h in hits]
        true = [x for x in r.explaining_notes.split("|") if x]
        rank = next((i + 1 for i, n in enumerate(ids) if n in true), None)
        rows.append(dict(iwl_id=r.iwl_id, query=q, explaining_notes=",".join(true) or "(none: gap)",
                         top3=",".join(ids[:3]), rank_of_correct=rank,
                         hit_at_k=(rank is not None and rank <= k) if true else None,
                         top1_misleading=bool(true) and ids[0] not in true or (not true and hits[0]["score"] >= 0.2),
                         note=r.comment))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- 3. diagnosis
def _retrieved_ids(tool_log):
    ids = set()
    for c in tool_log:
        if c["tool"] == "search_event_notes":
            ids |= set(pd.Series(c["result"]).str.findall(r"EV\d\d").iloc[0])
    return ids


def diagnosis_eval(run: dict, e: Engine, gt: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    items = (run.get("diagnosis") or {}).get("items", [])
    corpus = set(e.notes.note_id)
    retrieved = _retrieved_ids(run.get("tool_log", []))
    g = gt.set_index("iwl_id")
    flagged = set(e.triage[e.triage.bucket != "Low-Touch"].iwl_id)
    rows = []
    for it in items:
        iwl = it["iwl_id"].strip().upper()
        if iwl not in g.index:
            rows.append(dict(iwl_id=iwl, invalid_iwl=True))
            continue
        truth = g.loc[iwl]
        true_notes = {x for x in truth.explaining_notes.split("|") if x}
        cited = {c.strip().upper() for c in it.get("evidence_note_ids", [])}
        # A cancellation note counts as applicable evidence only for "no promo" conclusions
        applicable = true_notes
        rows.append(dict(
            iwl_id=iwl, true_cause=truth.true_cause, pred_cause=it["cause"],
            cause_correct=it["cause"] == truth.true_cause,
            expected_bucket=truth.expected_bucket, engine_bucket=it.get("engine_bucket"),
            final_bucket=it["final_bucket"], bucket_correct=it["final_bucket"] == truth.expected_bucket,
            cited=",".join(sorted(cited)) or "-", true_notes=",".join(sorted(true_notes)) or "(gap)",
            invented_ids=",".join(sorted(cited - corpus)) or "",
            unretrieved_citations=",".join(sorted((cited & corpus) - retrieved)) or "",
            inapplicable_citations=",".join(sorted((cited & corpus) - applicable)) or "",
            gap_handled=(not true_notes and truth.planted) and not cited,
            evidence_status=it.get("evidence_status")))
    df = pd.DataFrame(rows)
    covered = set(df.iwl_id) if len(df) else set()
    n_cit = sum(len([x for x in r.split(",") if x and x != "-"]) for r in df.get("cited", []))
    n_bad = sum(len([x for x in r.split(",") if x]) for r in df.get("inapplicable_citations", []))
    summary = dict(
        diagnosed=len(df), flagged_by_engine=len(flagged), not_covered=sorted(flagged - covered),
        cause_accuracy=float(df.cause_correct.mean()) if len(df) else 0,
        bucket_accuracy_on_flagged=float(df.bucket_correct.mean()) if len(df) else 0,
        citations=n_cit, inapplicable_citations=n_bad,
        citation_precision=(n_cit - n_bad) / n_cit if n_cit else None,
        invented_ids=int((df.invented_ids != "").sum()) if len(df) else 0,
        unretrieved_citations=int((df.unretrieved_citations != "").sum()) if len(df) else 0,
    )
    # end-to-end bucket accuracy on all 40: agent's final bucket where it diagnosed, engine bucket elsewhere
    final = e.triage.set_index("iwl_id").bucket.to_dict()
    for it in items:
        if it["iwl_id"].strip().upper() in final:
            final[it["iwl_id"].strip().upper()] = it["final_bucket"]
    summary["final_bucket_accuracy_all"] = float(np.mean([final[i] == g.loc[i].expected_bucket for i in final]))
    return df, summary


# ----------------------------------------------------------------- 4. outcome
def _wape(f, a):
    return float(np.abs(np.asarray(f) - a).sum() / max(a.sum(), 1))


def _wape4(f, a):
    """WAPE on 4-week buckets: the level at which a structurally noisy item should be judged."""
    f, a = np.asarray(f).reshape(-1, 4).sum(1), np.asarray(a).reshape(-1, 4).sum(1)
    return float(np.abs(f - a).sum() / max(a.sum(), 1))


def _verdict(new, base):
    return "better" if new < base - 1e-3 else ("worse" if new > base + 1e-3 else "same")


def outcome_eval(run: dict, gt: pd.DataFrame, fut: pd.DataFrame, approvals: dict | None = None):
    """approvals: {iwl: {"decision": "approve"|"reject", "params_json": str (optional edit)}}.
    None = counterfactual where every recommendation is applied without review."""
    e = Engine.load()
    e.run_baseline()
    recs = (run.get("recommendations") or {}).get("recommendations", [])
    A = {i: g.sort_values("week").units.to_numpy(float) for i, g in fut.groupby("iwl_id")}
    rows, final_fc = [], {i: e.baseline[i]["fcst"].copy() for i in e.Y}
    for r in recs:
        iwl = r["iwl_id"].strip().upper()
        if iwl not in e.Y:
            continue
        dec = "approve" if approvals is None else approvals.get(iwl, {}).get("decision", "pending")
        params_json = r.get("params_json", "{}")
        if approvals and approvals.get(iwl, {}).get("params_json"):
            params_json = approvals[iwl]["params_json"]
        try:
            params = json.loads(params_json or "{}")
        except Exception:  # noqa: BLE001
            params = {}
        try:
            p = e.reforecast(iwl, r["action"], params)
            new = np.array(p["reforecast"])
        except Exception:  # noqa: BLE001
            new = e.baseline[iwl]["fcst"]
        base = e.baseline[iwl]["fcst"]
        if dec == "approve":
            final_fc[iwl] = new
        rows.append(dict(iwl_id=iwl, action=r["action"], params=json.dumps(params), decision=dec,
                         true_cause=gt.set_index("iwl_id").loc[iwl].true_cause,
                         wape_baseline=_wape(base, A[iwl]), wape_reforecast=_wape(new, A[iwl]),
                         verdict=_verdict(_wape(new, A[iwl]), _wape(base, A[iwl])),
                         wape4_baseline=_wape4(base, A[iwl]), wape4_reforecast=_wape4(new, A[iwl])))
    df = pd.DataFrame(rows)
    tot_a = sum(A[i].sum() for i in e.Y)
    port_base = sum(np.abs(e.baseline[i]["fcst"] - A[i]).sum() for i in e.Y) / tot_a
    port_final = sum(np.abs(final_fc[i] - A[i]).sum() for i in e.Y) / tot_a
    return df, dict(portfolio_wape_baseline=port_base, portfolio_wape_after=port_final,
                    applied=int((df.decision == "approve").sum()) if len(df) else 0)


def full_report(run: dict, approvals: dict | None = None) -> dict:
    from engine import get_engine
    e = get_engine()
    gt, fut = load_truth()
    det = detection_eval(e, gt)
    ret = retrieval_eval(e, gt)
    diag_df, diag = diagnosis_eval(run, e, gt)
    out_auto, out_auto_s = outcome_eval(run, gt, fut, None)
    res = dict(detection=det, retrieval=ret, diagnosis_table=diag_df, diagnosis=diag,
               outcome_auto=out_auto, outcome_auto_summary=out_auto_s)
    if approvals:
        out_h, out_h_s = outcome_eval(run, gt, fut, approvals)
        res.update(outcome_human=out_h, outcome_human_summary=out_h_s)
    return res


def print_report(rep: dict):
    d = rep["detection"]
    print("\n=== 1. DETECTION (deterministic triage) ===")
    print(f"review queue {d['review_queue']}/{d['total']} IWLs | precision {d['precision']:.0%} recall {d['recall']:.0%}"
          f" | engine bucket accuracy {d['bucket_accuracy']:.0%}")
    print(f"false positives: {d['false_positives']}  missed: {d['missed']}")
    print("\n=== 2. RETRIEVAL (standard query, k=3, no LLM) ===")
    r = rep["retrieval"]
    print(r[["iwl_id", "query", "explaining_notes", "top3", "rank_of_correct", "hit_at_k", "top1_misleading"]].to_string(index=False))
    h = r.hit_at_k.dropna()
    print(f"hit@3 {h.mean():.0%} on {len(h)} IWLs with an explaining note | top-1 misleading on {int(r.top1_misleading.sum())}")
    print("\n=== 3. DIAGNOSIS ===")
    s = rep["diagnosis"]
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
    t = rep["diagnosis_table"]
    if len(t):
        print(t[["iwl_id", "true_cause", "pred_cause", "expected_bucket", "final_bucket", "cited", "true_notes",
                 "inapplicable_citations", "unretrieved_citations"]].to_string(index=False))
    print("\n=== 4. OUTCOME on hidden W105-W116 (all recommendations applied, no human filter) ===")
    o = rep["outcome_auto"]
    if len(o):
        print(o[["iwl_id", "action", "params", "true_cause", "wape_baseline", "wape_reforecast", "verdict", "wape4_baseline", "wape4_reforecast"]]
              .round(3).to_string(index=False))
    s = rep["outcome_auto_summary"]
    print(f"portfolio WAPE {s['portfolio_wape_baseline']:.1%} -> {s['portfolio_wape_after']:.1%}")
    if "outcome_human_summary" in rep:
        s = rep["outcome_human_summary"]
        print(f"with human approvals: portfolio WAPE {s['portfolio_wape_baseline']:.1%} -> {s['portfolio_wape_after']:.1%}"
              f" ({s['applied']} applied)")
