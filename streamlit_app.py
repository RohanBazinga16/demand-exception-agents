"""
Demand Exception Agents — Streamlit app (the deployed system).

Demand Data -> Forecast -> Detect Exception -> Investigate -> Reforecast -> Recommend -> Human Approval

Run locally:   streamlit run streamlit_app.py
Deploy:        Streamlit Community Cloud, main file = streamlit_app.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import warnings

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from engine import BUCKETS, HORIZON, LOC_NAMES, N_HIST, TH, get_engine  # noqa: E402
from evaluation import full_report, load_truth  # noqa: E402

RUNS = os.path.join(ROOT, "runs")
AUDIT = os.path.join(RUNS, "audit_log.csv")
BUCKET_COLORS = {"Low-Touch": "#5B9E47", "Monitor": "#1F8FD1", "Action-Required": "#E8891A",
                 "Structurally-Noisy": "#6B7785"}

st.set_page_config(page_title="Demand Exception Agents", page_icon="📦", layout="wide")


# ------------------------------------------------------------------ helpers
@st.cache_resource
def engine():
    return get_engine()


def save_run(run):
    os.makedirs(RUNS, exist_ok=True)
    path = run.get("_path") or os.path.join(RUNS, f"{run['mode']}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    run["_path"] = path
    json.dump(run, open(path, "w"), indent=1, default=str)
    return path


def list_runs():
    return sorted(glob.glob(os.path.join(RUNS, "*.json")), reverse=True)


def recs_df(run):
    recs = (run.get("recommendations") or {}).get("recommendations", [])
    return pd.DataFrame(recs)


def diag_df(run):
    return pd.DataFrame((run.get("diagnosis") or {}).get("items", []))


def bucket_badge(b):
    return f"<span style='background:{BUCKET_COLORS.get(b, '#999')};color:white;padding:2px 8px;" \
           f"border-radius:10px;font-size:0.8em'>{b}</span>"


e = engine()
tri = e.triage
ss = st.session_state
ss.setdefault("run", None)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 📦 Demand Exception Agents")
    st.caption("Kaveri Foods Pvt Ltd — a **fictional** FMCG company. All data is synthetic.")
    st.divider()
    key_default = ""
    try:
        key_default = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:  # noqa: BLE001
        pass
    api_key = st.text_input("Gemini API key", value="", type="password",
                            help="Free key from aistudio.google.com. Never stored. "
                                 "Leave empty to use a key set by the app owner (if any).") or key_default
    model = st.selectbox("Model", ["gemini-3.8-flash", "gemini-3-flash-preview", "gemini-2.5-flash-lite", "custom"])
    if model == "custom":
        model = st.text_input("Model name", "gemini-3.8-flash")
    max_rpm = st.slider("Max requests / minute", 2, 30, 8,
                        help="Keep below your free-tier limit; the crew waits rather than failing.")
    approver = st.text_input("Approver name (for the audit log)", "Demand Planner")
    st.divider()
    runs = list_runs()
    names = ["(none)"] + [os.path.basename(p) for p in runs]
    pick = st.selectbox("Load a saved run", names)
    if pick != "(none)" and st.button("Load run"):
        ss.run = json.load(open(os.path.join(RUNS, pick)))
        ss.run["_path"] = os.path.join(RUNS, pick)
        st.success(f"Loaded {pick}")
    if ss.run:
        st.info(f"Current run: **{ss.run['mode']}** · {ss.run.get('model')} · {ss.run.get('started')}")

st.title("Agentic Demand Forecasting & Forecast Exception Management")
st.caption("Demand Data → Forecast → Detect Exception → Investigate → Reforecast → Recommend → **Human Approval**")

tabs = st.tabs(["1 · Pipeline", "2 · Demand & Forecast", "3 · Exception Triage", "4 · Run Agents",
                "5 · Human Approval", "6 · Evaluation", "7 · Audit Trail"])

# ------------------------------------------------------------------ 1. pipeline
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("IWLs (item × DC)", len(tri))
    c2.metric("History", f"{N_HIST} weeks")
    c3.metric("Event notes (RAG corpus)", len(e.notes))
    c4.metric("Review queue after triage", int((tri.bucket != "Low-Touch").sum()),
              delta=f"-{int((tri.bucket == 'Low-Touch').sum())} auto Low-Touch", delta_color="off")
    st.graphviz_chart("""
digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
  data [label="Demand history\\n40 IWLs x 104 wks", fillcolor="#EEF3F8"];
  notes [label="Event notes\\n20 docs", fillcolor="#EEF3F8"];
  a1 [label="1 Demand Analyst\\nportfolio_profile\\niwl_history", fillcolor="#D6E8F5"];
  a2 [label="2 Forecasting\\nrun_baseline_forecast\\nforecast_detail", fillcolor="#D6E8F5"];
  a3 [label="3 Forecast Diagnosis\\nexception_triage (5 KPIs, 4 lenses)\\nsearch_event_notes (mini-RAG)", fillcolor="#D6E8F5"];
  a4 [label="4 Planning Recommendation\\nreforecast_iwls", fillcolor="#D6E8F5"];
  h [label="Human approval\\napprove / edit / reject", fillcolor="#FCE6C9"];
  log [label="Audit log + evaluation", fillcolor="#E6F2E1"];
  data -> a1 -> a2 -> a3 -> a4 -> h -> log; notes -> a3; }""")
    st.markdown("""
**Design choices that matter**
- **Numbers come from tools, judgement from the model.** KPIs, lens scores, forecasts and reforecasts are deterministic
  Python. Agents choose which tool to call, read the evidence, and decide the cause and the action.
- **Sequential process, not hierarchical.** The plan is fixed in code; only the judgement inside each step is delegated.
  Demand-planning errors cost money and need an audit trail.
- **Nothing changes without a human.** Reforecasts are proposals until a planner approves them (tab 5).
- **The hidden future is never shown to any agent.** Weeks 105–116 are used only in tab 6 to score the outcome.
""")

# ------------------------------------------------------------------ 2. demand & forecast
with tabs[1]:
    c1, c2 = st.columns([1, 3])
    with c1:
        iwl = st.selectbox("IWL", e.iwls(), index=e.iwls().index("OAT1K-MUM"),
                           format_func=lambda i: f"{i} · {e.label(i)}")
        reveal = st.checkbox("Reveal hidden actuals W105–W116 (evaluation only)", value=False)
        r = tri.set_index("iwl_id").loc[iwl]
        st.markdown(bucket_badge(r.bucket), unsafe_allow_html=True)
        st.caption(r.bucket_reason)
        st.metric("8-week WAPE (lag 4)", f"{r.wape_8w:.0%}")
        st.metric("8-week bias", f"{r.bias_8w:+.0%}")
        st.metric("Level shift vs W71–96", f"{r.level_shift_pct:+.0f}%")
    with c2:
        b = e.baseline[iwl]
        weeks = list(range(N_HIST - 51, N_HIST + 1))
        rows = [dict(week=w, value=e.Y[iwl][w - 1], series="Actual") for w in weeks]
        rows += [dict(week=w, value=b["lag_fcst"][w - 1], series="Forecast (made 4 wks earlier)") for w in weeks]
        rows += [dict(week=N_HIST + 1 + i, value=v, series="Baseline forecast") for i, v in enumerate(b["fcst"])]
        run = ss.run
        if run and iwl in (run.get("proposals") or {}):
            p = run["proposals"][iwl]
            rows += [dict(week=N_HIST + 1 + i, value=v, series=f"Proposed: {p['method']}")
                     for i, v in enumerate(p["reforecast"])]
        if reveal:
            _, fut = load_truth()
            f = fut[fut.iwl_id == iwl].sort_values("week")
            rows += [dict(week=int(w), value=float(v), series="Hidden actual") for w, v in zip(f.week, f.units)]
        df = pd.DataFrame(rows)
        ch = alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=18)).encode(
            x=alt.X("week:Q", title="Week"), y=alt.Y("value:Q", title="Units"),
            color=alt.Color("series:N", legend=alt.Legend(orient="bottom")),
            strokeDash=alt.condition(alt.datum.series == "Forecast (made 4 wks earlier)",
                                     alt.value([4, 3]), alt.value([0])),
            tooltip=["series", "week", alt.Tooltip("value:Q", format=",.0f")]).properties(height=380)
        rule = alt.Chart(pd.DataFrame({"x": [N_HIST + 0.5]})).mark_rule(color="#999").encode(x="x:Q")
        st.altair_chart(ch + rule, width="stretch")
        st.caption("Grey line = as-of week W104. Everything to the right is forecast.")

# ------------------------------------------------------------------ 3. triage
with tabs[2]:
    cols = st.columns(4)
    for c, bk in zip(cols, BUCKETS):
        c.metric(bk, int((tri.bucket == bk).sum()))
    st.markdown("**Four lenses → one treatment.** Thresholds: "
                f"Structurally-Noisy if Forecastability < {TH['noisy_forecastability']}; Action-Required if an active "
                f"promo note is in the horizon, or Risk ≥ {TH['action_risk']} with Persistence ≥ "
                f"{TH['action_persistence']} or forward forecast change ≥ {int(TH['action_fwd_change'] * 100)}%; "
                f"Monitor if Risk ≥ {TH['monitor_risk']}, Persistence ≥ {TH['monitor_persistence']} or level shift "
                f"≥ {TH['monitor_level_shift']}%.")
    sc = alt.Chart(tri).mark_circle(size=140, opacity=0.85).encode(
        x=alt.X("risk:Q", title="Current Risk"), y=alt.Y("persistence:Q", title="Persistence"),
        color=alt.Color("bucket:N", scale=alt.Scale(domain=list(BUCKET_COLORS), range=list(BUCKET_COLORS.values()))),
        size=alt.Size("criticality:Q", legend=None),
        tooltip=["iwl_id", "item", "location", "bucket", alt.Tooltip("forecastability:Q", format=".0f"),
                 alt.Tooltip("criticality:Q", format=".0f"), alt.Tooltip("risk:Q", format=".0f"),
                 alt.Tooltip("persistence:Q", format=".0f")]).properties(height=330)
    st.altair_chart(sc, width="stretch")
    show = st.multiselect("Buckets", BUCKETS, default=["Monitor", "Action-Required", "Structurally-Noisy"])
    view = tri[tri.bucket.isin(show)][[
        "iwl_id", "item", "location", "bucket", "forecastability", "criticality", "risk", "persistence",
        "weekly_split_dev_pp", "sku_split_dev_pp", "pack_split_dev_pp", "fwd_fcst_change_pct", "bias_change_pp",
        "bucket_err_4w", "exception_weeks_8w", "level_shift_pct", "forward_event_notes", "bucket_reason"]]
    st.dataframe(view.round(1), width="stretch", hide_index=True)
    with st.expander("KPI definitions"):
        st.markdown("""
| KPI | Definition (window W097–W104, forecast at 4-week lag) | Feeds lens |
|---|---|---|
| Weekly split deviation (pp) | mean \\|forecast weekly share − actual weekly share\\| inside each 4-week block | Current Risk |
| SKU split deviation (pp) | \\|forecast SKU share − actual SKU share\\| within category at the DC | Current Risk (context) |
| Package-to-SKU split (pp) | \\|forecast pack share − actual pack share\\| within the SKU at the DC | Current Risk, Forecastability (split stability) |
| Forward forecast change (%) | this cycle's W105–112 forecast vs last cycle's (as of W100) | Current Risk |
| Bias change (pp) | bias of last 4 weeks − bias of W089–W096 | Persistence (diagnostic) |
""")

# ------------------------------------------------------------------ 4. run agents
with tabs[3]:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Run the 4-agent crew (Gemini)")
        st.caption("About 10–20 model calls, 2–5 minutes on the free tier.")
        if st.button("▶ Run agents", type="primary", disabled=not api_key):
            from crew import run_agents
            log_box = st.empty()
            lines = []

            def prog(msg):
                lines.append(msg.replace("\n", " ")[:160])
                try:
                    log_box.code("\n".join(lines[-12:]))
                except Exception:  # noqa: BLE001
                    pass
            with st.spinner("Agents working… (Demand Analyst → Forecasting → Diagnosis → Planning)"):
                try:
                    ss.run = run_agents(api_key=api_key, model=model, max_rpm=max_rpm, verbose=False, progress=prog)
                    path = save_run(ss.run)
                    st.success(f"Done in {ss.run['seconds']}s · saved {os.path.basename(path)}")
                except Exception as ex:  # noqa: BLE001
                    st.error(f"Run failed: {ex}")
                    st.caption("Common causes: invalid key, model name not available to your key, or free-tier "
                               "quota exhausted (pick another model in the sidebar or wait a minute).")
        if not api_key:
            st.caption("Paste a Gemini key in the sidebar to enable.")
    with c2:
        st.markdown("#### Run the rule-based baseline")
        st.caption("Same tools, fixed rules, no LLM. The comparison point for evaluation.")
        if st.button("▶ Run rules"):
            from rules import run_rules
            ss.run = run_rules()
            path = save_run(ss.run)
            st.success(f"Done · saved {os.path.basename(path)}")

    run = ss.run
    if run:
        st.divider()
        st.markdown(f"### Run output — `{run['mode']}` · {run.get('model')} · {run.get('seconds')}s")
        if run.get("usage"):
            st.caption(f"Token usage: {run['usage']}")
        with st.expander("Agent 1 · Demand Analyst", expanded=False):
            st.markdown(run.get("analyst_summary") or "-")
        with st.expander("Agent 2 · Forecasting", expanded=False):
            st.markdown(run.get("forecast_summary") or "-")
        st.markdown("#### Agent 3 · Forecast Diagnosis")
        d = diag_df(run)
        if len(d):
            st.markdown((run.get("diagnosis") or {}).get("summary", ""))
            st.dataframe(d, width="stretch", hide_index=True)
        else:
            st.warning("Diagnosis output could not be parsed into the schema.")
            st.code(run.get("diagnosis_raw", "")[:4000])
        st.markdown("#### Agent 4 · Planning Recommendation")
        rd = recs_df(run)
        if len(rd):
            st.markdown((run.get("recommendations") or {}).get("planner_summary", ""))
            st.dataframe(rd, width="stretch", hide_index=True)
        else:
            st.warning("Recommendation output could not be parsed into the schema.")
            st.code(run.get("recommendations_raw", "")[:4000])

# ------------------------------------------------------------------ 5. approval
with tabs[4]:
    run = ss.run
    if not run or not len(recs_df(run)):
        st.info("Run or load a run first (tab 4 or sidebar).")
    else:
        rd = recs_df(run).copy()
        prev = run.get("approvals") or {}
        rd.insert(0, "decision", [prev.get(i, {}).get("decision", "pending") for i in rd.iwl_id])
        rd["params_json"] = [prev.get(i, {}).get("params_json") or p for i, p in zip(rd.iwl_id, rd.params_json)]
        st.markdown("Review each proposal. Edit `params_json` to change a reforecast (e.g. a smaller promo uplift). "
                    "Only **approve** changes the forecast. Every decision is written to the audit log with your name.")
        edited = st.data_editor(
            rd[["decision", "iwl_id", "final_bucket", "cause", "action", "params_json", "horizon_change_pct",
                "evidence_note_ids", "rationale", "confidence"]],
            column_config={"decision": st.column_config.SelectboxColumn(options=["pending", "approve", "reject"]),
                           "params_json": st.column_config.TextColumn(width="medium")},
            disabled=["iwl_id", "final_bucket", "cause", "action", "horizon_change_pct", "evidence_note_ids",
                      "rationale", "confidence"],
            hide_index=True, width="stretch", key=f"editor_{run.get('_path', run['started'])}")
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("✅ Commit decisions", type="primary"):
                approvals, audit = {}, []
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                for _, r in edited.iterrows():
                    orig = rd.set_index("iwl_id").loc[r.iwl_id]
                    entry = dict(decision=r.decision, approver=approver, ts=ts)
                    if r.params_json != recs_df(run).set_index("iwl_id").loc[r.iwl_id].params_json:
                        entry["params_json"] = r.params_json
                    approvals[r.iwl_id] = entry
                    audit.append(dict(ts=ts, run=os.path.basename(run.get("_path", "unsaved")), mode=run["mode"],
                                      iwl_id=r.iwl_id, action=orig.action, params_json=r.params_json,
                                      decision=r.decision, approver=approver, evidence=orig.evidence_note_ids))
                run["approvals"] = approvals
                save_run(run)
                pd.DataFrame(audit).to_csv(AUDIT, mode="a", header=not os.path.exists(AUDIT), index=False)
                st.success(f"Committed {sum(a['decision'] == 'approve' for a in approvals.values())} approvals, "
                           f"{sum(a['decision'] == 'reject' for a in approvals.values())} rejections.")
        with c2:
            sel = st.selectbox("Preview a proposal", rd.iwl_id.tolist())
            p = (run.get("proposals") or {}).get(sel)
            if p:
                pdf = pd.DataFrame({"week": list(range(N_HIST + 1, N_HIST + HORIZON + 1)) * 2,
                                    "units": p["baseline"] + p["reforecast"],
                                    "series": ["Baseline"] * HORIZON + [f"Proposed ({p['method']})"] * HORIZON})
                st.altair_chart(alt.Chart(pdf).mark_line(point=True).encode(
                    x="week:Q", y="units:Q", color="series:N").properties(height=230), width="stretch")
                notes = rd.set_index("iwl_id").loc[sel].evidence_note_ids
                for nid in (notes if isinstance(notes, list) else []):
                    if nid in set(e.notes.note_id):
                        n = e.notes.set_index("note_id").loc[nid]
                        st.caption(f"**{nid}** ({n.location}, published W{int(n.published_week)}): {n.title} — {n.text}")
                    else:
                        st.caption(f"**{nid}**: ⚠ not found in the note corpus")

# ------------------------------------------------------------------ 6. evaluation
with tabs[5]:
    run = ss.run
    if not run:
        st.info("Run or load a run first.")
    else:
        rep = full_report(run, run.get("approvals"))
        det, diag = rep["detection"], rep["diagnosis"]
        st.markdown("#### 1 · Detection — deterministic triage vs planted exceptions")
        c = st.columns(4)
        c[0].metric("Recall", f"{det['recall']:.0%}")
        c[1].metric("Precision", f"{det['precision']:.0%}")
        c[2].metric("Engine bucket accuracy", f"{det['bucket_accuracy']:.0%}")
        c[3].metric("Review queue", f"{det['review_queue']} / {det['total']}")
        st.caption(f"False positives: {det['false_positives']} · Missed: {det['missed']}")
        st.markdown("#### 2 · Retrieval quality — standard query per IWL, k = 3 (no LLM)")
        r = rep["retrieval"]
        h = r.hit_at_k.dropna()
        c = st.columns(2)
        c[0].metric("Explaining note in top 3", f"{h.mean():.0%}", help=f"{len(h)} IWLs that have an explaining note")
        c[1].metric("Top-1 result misleading", int(r.top1_misleading.sum()),
                    help="Top hit is a note for another location, a cancelled plan, or an unrelated item")
        st.dataframe(r, width="stretch", hide_index=True)
        st.markdown(f"#### 3 · Diagnosis quality — `{run['mode']}`")
        c = st.columns(5)
        c[0].metric("Cause accuracy", f"{diag['cause_accuracy']:.0%}")
        c[1].metric("Bucket accuracy (flagged)", f"{diag['bucket_accuracy_on_flagged']:.0%}")
        cp = diag["citation_precision"]
        c[2].metric("Citation precision", "-" if cp is None else f"{cp:.0%}")
        c[3].metric("Cited but never retrieved", diag["unretrieved_citations"])
        c[4].metric("Invented note ids", diag["invented_ids"])
        if diag["not_covered"]:
            st.warning(f"Flagged IWLs the run did not diagnose: {diag['not_covered']}")
        st.dataframe(rep["diagnosis_table"], width="stretch", hide_index=True)
        st.markdown("#### 4 · Outcome on hidden W105–W116")
        a = rep["outcome_auto_summary"]
        c = st.columns(3)
        c[0].metric("Portfolio WAPE — baseline", f"{a['portfolio_wape_baseline']:.1%}")
        c[1].metric("If every recommendation were auto-applied", f"{a['portfolio_wape_after']:.1%}")
        if "outcome_human_summary" in rep:
            hs = rep["outcome_human_summary"]
            c[2].metric("With human approvals", f"{hs['portfolio_wape_after']:.1%}", help=f"{hs['applied']} applied")
        else:
            c[2].metric("With human approvals", "commit decisions in tab 5")
        st.dataframe(rep["outcome_auto"].round(3), width="stretch", hide_index=True)
        st.caption("wape4 = WAPE on 4-week buckets, the right yardstick for structurally noisy IWLs.")

        st.markdown("#### Compare saved runs (agents vs rules)")
        rows = []
        for pth in list_runs()[:12]:
            try:
                rr = json.load(open(pth))
                rp = full_report(rr, rr.get("approvals"))
                dg, oa = rp["diagnosis"], rp["outcome_auto_summary"]
                rows.append(dict(run=os.path.basename(pth), mode=rr["mode"], model=rr.get("model"),
                                 seconds=rr.get("seconds"), cause_accuracy=dg["cause_accuracy"],
                                 bucket_accuracy_all=dg["final_bucket_accuracy_all"],
                                 citation_precision=dg["citation_precision"],
                                 unretrieved_citations=dg["unretrieved_citations"],
                                 wape_auto_applied=oa["portfolio_wape_after"],
                                 wape_human=rp.get("outcome_human_summary", {}).get("portfolio_wape_after")))
            except Exception as ex:  # noqa: BLE001
                rows.append(dict(run=os.path.basename(pth), mode=f"unreadable: {ex}"))
        if rows:
            st.dataframe(pd.DataFrame(rows).round(3), width="stretch", hide_index=True)

# ------------------------------------------------------------------ 7. audit
with tabs[6]:
    st.markdown("#### Approval audit log")
    if os.path.exists(AUDIT):
        st.dataframe(pd.read_csv(AUDIT).iloc[::-1], width="stretch", hide_index=True)
    else:
        st.caption("No decisions committed yet.")
    run = ss.run
    if run:
        st.markdown(f"#### Tool calls in this run ({len(run.get('tool_log', []))})")
        tl = pd.DataFrame(run.get("tool_log", []))
        if len(tl):
            st.dataframe(tl[["ts", "tool", "args", "result_chars"]], width="stretch", hide_index=True)
            i = st.number_input("Show full result of call #", 0, len(tl) - 1, 0)
            st.code(tl.iloc[int(i)].result[:6000])
        if run.get("trace"):
            with st.expander(f"Agent step trace ({len(run['trace'])} steps)"):
                for s in run["trace"]:
                    st.text(f"[{s['ts']}] {s['step'][:1500]}")
