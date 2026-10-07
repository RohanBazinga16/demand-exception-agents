"""
The tools the agents may call. Each one is a thin wrapper over the deterministic
engine. The docstrings ARE prompts: the model decides when to call a tool from
the name, docstring and argument names alone (Session 7).

Every call is written to engine.tool_log, which is how evaluation.py checks
provenance (was a cited note ever actually retrieved?).
"""
from __future__ import annotations

import json

import numpy as np

from engine import BUCKETS, HORIZON, LOC_NAMES, METHODS, N_HIST, get_engine

try:
    from crewai.tools import tool
except Exception:  # crewai not installed: rule-based mode still works
    def tool(name):
        def deco(f):
            f.name = name
            return f
        return deco


def _r(x, n=1):
    return None if x is None else round(float(x), n)


# ---------------------------------------------------------------- implementations
def portfolio_profile_impl() -> str:
    e = get_engine()
    t = e.triage
    rows = ["iwl_id | item | loc | class | avg_units_26w | cov | zero_share | forecastability | flags"]
    for r in t.itertuples():
        avg = e.Y[r.iwl_id][-26:].mean()
        flags = []
        if r.zero_share > 0.2:
            flags.append("intermittent")
        if avg < 100:
            flags.append("low-volume")
        if r.spike_weeks:
            flags.append(f"recent outliers W{r.spike_weeks}")
        rows.append(f"{r.iwl_id} | {r.item} | {r.location} | {r.service_class} | {avg:.0f} | {r.cov:.2f} | "
                    f"{r.zero_share:.2f} | {r.forecastability:.0f} | {','.join(flags) or '-'}")
    out = (f"Portfolio: {len(t)} IWLs (10 items x 4 DCs), {N_HIST} weeks of weekly shipments, no missing weeks.\n"
           f"Forecastability 0-100 (higher = easier): median {t.forecastability.median():.0f}, "
           f"{(t.forecastability < 40).sum()} IWLs below 40.\n" + "\n".join(rows))
    e.log("portfolio_profile", {}, out)
    return out


def iwl_history_impl(iwl_id: str, weeks: int = 16) -> str:
    e = get_engine()
    iwl_id = iwl_id.strip().upper()
    if iwl_id not in e.Y:
        return f"Unknown IWL '{iwl_id}'. Valid ids look like OAT1K-MUM. Options: {', '.join(e.iwls())}"
    weeks = int(np.clip(weeks, 4, 52))
    b = e.baseline[iwl_id]
    y = e.Y[iwl_id]
    lines = [f"{iwl_id} = {e.label(iwl_id)}. Last {weeks} weeks (forecast = made 4 weeks earlier):",
             "week | week_start | actual | lag4_forecast | error_pct | deseasonalised"]
    for k in range(N_HIST - weeks, N_HIST):
        f = b["lag_fcst"][k]
        err = (f - y[k]) / max(f, 1) * 100
        lines.append(f"W{k + 1:03d} | {e.meta[iwl_id]['week_start'][k]} | {y[k]:.0f} | {f:.0f} | {err:+.0f}% | "
                     f"{b['deseas'][k]:.0f}")
    lines.append(f"Reference level (deseasonalised mean W071-W096): {b['deseas'][70:96].mean():.0f}")
    out = "\n".join(lines)
    e.log("iwl_history", {"iwl_id": iwl_id, "weeks": weeks}, out)
    return out


def run_baseline_forecast_impl() -> str:
    e = get_engine()
    e.run_baseline()
    t = e.run_triage()
    y8 = sum(e.Y[i][-8:].sum() for i in e.Y)
    f8 = sum(e.baseline[i]["lag_fcst"][-8:].sum() for i in e.Y)
    ae = sum(np.abs(e.baseline[i]["lag_fcst"][-8:] - e.Y[i][-8:]).sum() for i in e.Y)
    top = t.sort_values("bucket_err_4w", ascending=False).head(10)
    mov = t.reindex(t.fwd_fcst_change_pct.abs().sort_values(ascending=False).index).head(6)
    lines = ["Method: seasonal index (category-pooled, 52 weeks) x exponentially smoothed level (alpha 0.08). "
             f"Accuracy measured at a 4-week lag over W{N_HIST - 7:03d}-W{N_HIST:03d}.",
             f"Portfolio WAPE {ae / y8:.1%}, portfolio bias {(f8 - y8) / y8:+.1%} (positive = over-forecast).",
             "Largest 4-week bucket errors (iwl | 4w bucket error | 8w WAPE | 8w bias):"]
    lines += [f"  {r.iwl_id} | {r.bucket_err_4w:.0%} | {r.wape_8w:.0%} | {r.bias_8w:+.0%}" for r in top.itertuples()]
    lines.append(f"Largest forward-forecast changes vs last cycle (W{N_HIST + 1}-W{N_HIST + 8}):")
    lines += [f"  {r.iwl_id} | {r.fwd_fcst_change_pct:+.1f}%" for r in mov.itertuples()]
    lines.append(f"12-week forecasts for W{N_HIST + 1:03d}-W{N_HIST + HORIZON:03d} are stored for every IWL.")
    out = "\n".join(lines)
    e.log("run_baseline_forecast", {}, out)
    return out


def forecast_detail_impl(iwl_id: str) -> str:
    e = get_engine()
    iwl_id = iwl_id.strip().upper()
    if iwl_id not in e.Y:
        return f"Unknown IWL '{iwl_id}'. Options: {', '.join(e.iwls())}"
    b = e.baseline[iwl_id]
    cur = " ".join(f"W{N_HIST + 1 + i}:{v:.0f}" for i, v in enumerate(b["fcst"]))
    pri = " ".join(f"W{N_HIST + 1 + i}:{v:.0f}" for i, v in enumerate(b["prior_cycle_fcst"][:8]))
    out = f"{iwl_id} ({e.label(iwl_id)})\nCurrent cycle forecast: {cur}\nPrior cycle forecast: {pri}"
    if iwl_id in e.proposals:
        p = e.proposals[iwl_id]
        out += f"\nProposed reforecast ({p['method']} {p['params']}): " + " ".join(
            f"W{N_HIST + 1 + i}:{v:.0f}" for i, v in enumerate(p["reforecast"]))
    e.log("forecast_detail", {"iwl_id": iwl_id}, out)
    return out


def exception_triage_impl() -> str:
    e = get_engine()
    t = e.triage
    flagged = t[t.bucket != "Low-Touch"].sort_values(["bucket", "risk"], ascending=[True, False])
    lines = [
        "Deterministic triage of all 40 IWLs. KPIs over W097-W104 at 4-week lag.",
        "Columns: iwl | engine_bucket (reason) | lenses F=forecastability C=criticality R=current risk P=persistence | "
        "KPIs: wk_split(pp) sku_split(pp) pack_split(pp) fwd_fcst_chg(%) bias_chg(pp) | "
        "signals: 4w_bucket_err, exception_weeks/8, same_sign_run, level_shift(%), outlier_weeks, promo_notes_in_horizon",
    ]
    for r in flagged.itertuples():
        lines.append(
            f"{r.iwl_id} ({r.item} @ {r.location}) | {r.bucket} ({r.bucket_reason}) | "
            f"F={r.forecastability:.0f} C={r.criticality:.0f} R={r.risk:.0f} P={r.persistence:.0f} | "
            f"wk_split={r.weekly_split_dev_pp:.1f} sku_split={r.sku_split_dev_pp:.1f} pack_split={r.pack_split_dev_pp:.1f} "
            f"fwd_chg={r.fwd_fcst_change_pct:+.1f} bias_chg={r.bias_change_pp:+.1f} | "
            f"err4w={r.bucket_err_4w:.0%} exc={r.exception_weeks_8w}/8 run={r.same_sign_run} "
            f"level_shift={r.level_shift_pct:+.0f}% outliers={r.spike_weeks or '-'} "
            f"promo_notes={r.forward_event_notes or '-'}")
    low = t[t.bucket == "Low-Touch"].iwl_id.tolist()
    lines.append(f"Low-Touch ({len(low)}, no review needed): {', '.join(low)}")
    lines.append("Counts: " + ", ".join(f"{b}={int((t.bucket == b).sum())}" for b in BUCKETS))
    out = "\n".join(lines)
    e.log("exception_triage", {}, out)
    return out


def search_event_notes_impl(query: str, k: int = 3) -> str:
    e = get_engine()
    k = int(np.clip(k, 1, 6))
    blocks = []
    for q in [s.strip() for s in str(query).split("|") if s.strip()][:16]:
        hits = e.search_notes(q, k)
        lines = [f'Query "{q}" -> top {k}:']
        for h in hits:
            lines.append(f"  #{h['rank']} {h['note_id']} (score {h['score']}) published {h['published']} | "
                         f"type={h['type']} location={h['location']} scope={h['item_scope']} "
                         f"effective={h['effective']} status={h['status']}\n"
                         f"     {h['title']}: {h['text']}")
        blocks.append("\n".join(lines))
    out = "\n".join(blocks) if blocks else "Empty query."
    e.log("search_event_notes", {"query": query, "k": k}, out)
    return out


def reforecast_iwls_impl(requests_json: str) -> str:
    e = get_engine()
    try:
        reqs = json.loads(requests_json)
        if isinstance(reqs, dict):
            reqs = [reqs]
    except Exception as ex:  # noqa: BLE001
        return f"Could not parse JSON ({ex}). Pass a JSON list like " \
               '[{"iwl_id":"OAT1K-MUM","method":"relevel","params":{"recent_weeks":6}}]'
    lines = []
    for rq in reqs:
        iwl = str(rq.get("iwl_id", "")).strip().upper()
        method = rq.get("method", "")
        params = rq.get("params", {}) or {}
        if iwl not in e.Y:
            lines.append(f"{iwl}: unknown IWL")
            continue
        if method not in METHODS:
            lines.append(f"{iwl}: unknown method '{method}'. Allowed: {METHODS}")
            continue
        try:
            p = e.reforecast(iwl, method, params)
        except Exception as ex:  # noqa: BLE001
            lines.append(f"{iwl}: failed ({ex})")
            continue
        b, n = np.array(p["baseline"]), np.array(p["reforecast"])
        chg = (n.sum() / max(b.sum(), 1e-9) - 1) * 100
        lines.append(f"{iwl} | {method} {json.dumps(params)} | 12-week total {b.sum():.0f} -> {n.sum():.0f} "
                     f"({chg:+.1f}%) | first 4 weeks {b[:4].round(0).tolist()} -> {n[:4].round(0).tolist()} "
                     f"| status: PROPOSED, awaiting human approval")
    out = "\n".join(lines) or "No requests."
    e.log("reforecast_iwls", {"requests_json": requests_json}, out)
    return out


# ---------------------------------------------------------------- CrewAI tools
@tool("portfolio_profile")
def portfolio_profile() -> str:
    """Profile every IWL (Item-Week-Location) in the portfolio: average weekly volume, coefficient of
    variation, share of zero weeks, a 0-100 forecastability score and data-quality flags (intermittent,
    low-volume, recent outliers). Takes no arguments. Call this first when you need an overview of the data."""
    return portfolio_profile_impl()


@tool("iwl_history")
def iwl_history(iwl_id: str, weeks: int = 16) -> str:
    """Week-by-week actual shipments, the forecast made 4 weeks earlier, the error and the deseasonalised
    level for ONE IWL. iwl_id looks like 'OAT1K-MUM'. weeks = how many recent weeks (4-52, default 16).
    Use it to see the shape of an exception: a step change, a one-off spike, a slow drift or lumpy demand."""
    return iwl_history_impl(iwl_id, weeks)


@tool("run_baseline_forecast")
def run_baseline_forecast() -> str:
    """Run the statistical baseline forecast for all 40 IWLs and return portfolio accuracy (WAPE, bias at a
    4-week lag), the IWLs with the largest recent errors and the largest forecast changes versus the last
    planning cycle. Takes no arguments. This is the only way to produce or refresh the baseline forecast."""
    return run_baseline_forecast_impl()


@tool("forecast_detail")
def forecast_detail(iwl_id: str) -> str:
    """The 12-week forward forecast for ONE IWL (current cycle), the prior cycle's forecast for the same
    weeks, and any proposed reforecast. iwl_id looks like 'POP150-MUM'."""
    return forecast_detail_impl(iwl_id)


@tool("exception_triage")
def exception_triage() -> str:
    """Compute the five demand-planning KPIs (weekly split deviation, SKU split deviation, package-to-SKU
    split deviation, forward forecast change, bias change), score every IWL on the four lenses
    (Forecastability, Business Criticality, Current Risk, Persistence) and assign a treatment bucket:
    Low-Touch, Monitor, Action-Required or Structurally-Noisy. Returns full detail for every non-Low-Touch IWL.
    Takes no arguments. The bucket is rule-based and can be wrong: it trusts note metadata and cannot read text."""
    return exception_triage_impl()


@tool("search_event_notes")
def search_event_notes(query: str, k: int = 3) -> str:
    """Similarity search over the commercial event notes (promotions, listings, competitor moves, price
    changes, customer order patterns, supply issues). Returns the top-k notes with id, publish week, location,
    item scope, effective weeks, status and full text. Several queries can be sent at once separated by ' | ',
    e.g. 'Rolled Oats 1kg Mumbai | Fruit Muesli 400g Delhi'. WARNING: ranking is by word similarity only.
    It does not check location, dates, or whether a later note cancelled an earlier one — you must."""
    return search_event_notes_impl(query, k)


@tool("reforecast_iwls")
def reforecast_iwls(requests_json: str) -> str:
    """Propose reforecasts for one or more IWLs in a single call. requests_json is a JSON list of objects
    {"iwl_id": str, "method": str, "params": object}. Methods and params:
      relevel        {"recent_weeks": 3-26}  reset the level to the mean of the last N deseasonalised weeks
                     (use the number of weeks since the change started)
      promo_uplift   {"from_week": 105-116, "to_week": 105-116, "uplift_pct": number}  for a confirmed future promo
      cleanse_spike  {"spike_weeks": [week numbers]}  remove a past one-off spike (and its dip) from the baseline
      rephase_weekly {}  keep 4-week totals, redistribute weeks using the recent week-of-month pattern
      aggregate      {}  flat average for structurally noisy items (plan at a higher level instead)
      no_change      {}  keep the baseline
    Results are PROPOSALS only; nothing changes until a human planner approves."""
    return reforecast_iwls_impl(requests_json)


ANALYST_TOOLS = [portfolio_profile, iwl_history]
FORECAST_TOOLS = [run_baseline_forecast, forecast_detail]
DIAGNOSIS_TOOLS = [exception_triage, search_event_notes, iwl_history]
PLANNING_TOOLS = [reforecast_iwls, forecast_detail]
LOCATIONS_HELP = ", ".join(f"{k}={v}" for k, v in LOC_NAMES.items())
