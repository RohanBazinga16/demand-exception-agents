"""
Rule-based baseline: the same pipeline with no LLM.

Uses the same tools and the same cause signatures the agents are given, but fixed
if-then rules, and it trusts the top search hit as evidence. It exists for two reasons:
  1. a comparison point — what does the LLM add over rules? (evaluation)
  2. a fallback that runs with no API key (demo safety)
"""
from __future__ import annotations

import json
import re
import time

from engine import LOC_NAMES, get_engine
import tools as T


def _query(e, iwl):
    m = e.meta[iwl]
    city = LOC_NAMES[m["location"]].split(" ")[0]
    return f"{m['sku']} {m['pack']} {city}"


def run_rules() -> dict:
    e = get_engine(reset=True)
    t0 = time.time()
    analyst = T.portfolio_profile_impl()
    fc = T.run_baseline_forecast_impl()
    T.exception_triage_impl()
    tri = e.triage.set_index("iwl_id")
    flagged = tri[tri.bucket != "Low-Touch"]
    T.search_event_notes_impl(" | ".join(_query(e, i) for i in flagged.index), k=4)

    items, reqs, recs = [], [], []
    for iwl, r in flagged.iterrows():
        m = e.meta[iwl]
        sib = [j for j, mj in e.meta.items() if j != iwl and mj["sku"] == m["sku"] and mj["location"] == m["location"]]
        sib_shift = tri.loc[sib[0], "level_shift_pct"] if sib else 0
        params, evidence = {}, []
        if r.bucket == "Structurally-Noisy":
            cause, method = "STRUCTURAL_NOISE", "aggregate"
        elif r.forward_event_notes:
            cause, method = "UPCOMING_PROMO", "promo_uplift"
            evidence = list(r.forward_event_notes)
            note = e.notes.set_index("note_id").loc[evidence[0]]
            up = re.search(r"\+(\d+)%", note.text)
            params = dict(from_week=int(note.effective_from_week), to_week=int(note.effective_to_week),
                          uplift_pct=int(up.group(1)) if up else 30)
        elif r.spike_weeks and r.exception_weeks_8w <= 3:
            cause, method = "ONE_OFF_SPIKE", "cleanse_spike"
            params = dict(spike_weeks=[w for w in r.spike_weeks if r.spike_weeks.count(w)][:3])
        elif r.pack_split_dev_pp >= 8 and abs(r.level_shift_pct) >= 10 and sib_shift * r.level_shift_pct < 0:
            cause, method, params = "PACK_MIX_SHIFT", "relevel", dict(recent_weeks=8)
        elif r.level_shift_pct >= 15:
            cause, method, params = "LEVEL_SHIFT_UP", "relevel", dict(recent_weeks=6)
        elif r.level_shift_pct <= -15:
            cause, method, params = "LEVEL_SHIFT_DOWN", "relevel", dict(recent_weeks=6)
        elif r.weekly_split_dev_pp >= 4:
            cause, method = "WEEKLY_PHASING", "rephase_weekly"
        elif r.level_shift_pct <= -12 or r.level_shift_pct >= 12:
            cause, method = "BIAS_DRIFT", "no_change"
        else:
            cause, method = "NONE", "no_change"
        bucket = r.bucket
        if bucket == "Monitor" and method not in ("rephase_weekly",):
            method, params = "no_change", {}
        if not evidence:                       # naive evidence: trust the top similarity hit
            top = e.search_notes(_query(e, iwl), 1)[0]
            if top["score"] >= 0.2:
                evidence = [top["note_id"]]
        items.append(dict(iwl_id=iwl, engine_bucket=r.bucket, final_bucket=bucket, cause=cause,
                          evidence_note_ids=evidence,
                          evidence_status="supported" if evidence else "no_evidence_found",
                          reasoning=f"Rule: {cause} from level_shift={r.level_shift_pct:+.0f}%, "
                                    f"pack_split={r.pack_split_dev_pp:.1f}, wk_split={r.weekly_split_dev_pp:.1f}, "
                                    f"outliers={r.spike_weeks}.", confidence="medium"))
        reqs.append(dict(iwl_id=iwl, method=method, params=params))
    T.reforecast_iwls_impl(json.dumps(reqs))
    for it, rq in zip(items, reqs):
        p = e.proposals[it["iwl_id"]]
        chg = (sum(p["reforecast"]) / max(sum(p["baseline"]), 1e-9) - 1) * 100
        recs.append(dict(iwl_id=it["iwl_id"], final_bucket=it["final_bucket"], cause=it["cause"],
                         action=rq["method"], params_json=json.dumps(rq["params"]),
                         evidence_note_ids=it["evidence_note_ids"], horizon_change_pct=round(chg, 1),
                         rationale=it["reasoning"], confidence="medium"))
    return dict(mode="rules", model="none (deterministic rules)", started=time.strftime("%Y-%m-%d %H:%M:%S"),
                seconds=round(time.time() - t0, 1), usage={},
                analyst_summary=analyst[:1500], forecast_summary=fc,
                diagnosis=dict(items=items, summary="Rule-based diagnosis (no LLM)."),
                recommendations=dict(recommendations=recs, planner_summary="Rule-based recommendations (no LLM)."),
                proposals=e.proposals, tool_log=e.tool_log, trace=[])
