"""
The 4-agent crew (CrewAI, sequential process, Gemini).

  Demand Analyst  ->  Forecasting  ->  Forecast Diagnosis  ->  Planning Recommendation  ->  [human approval]

Why sequential and not hierarchical: in demand planning a wrong step is costly and the
planner needs an audit trail, so the plan is fixed in code and only the judgement inside
each step is delegated to the model (Session 7, "use a plain workflow when...").
"""
from __future__ import annotations

import json
import os
import re
import time

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from engine import LOC_NAMES, get_engine  # noqa: E402
from schemas import DiagnosisReport, RecommendationSet  # noqa: E402
import tools as T  # noqa: E402

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
LOCS = ", ".join(f"{k} = {v}" for k, v in LOC_NAMES.items())

CAUSE_GUIDE = """Cause signatures (use the KPI values, then confirm with notes):
- LEVEL_SHIFT_UP / LEVEL_SHIFT_DOWN: level_shift beyond about +/-15%, many exception weeks, long same-sign run.
- ONE_OFF_SPIKE: outlier weeks in the window, few exception weeks, demand back to normal afterwards.
- UPCOMING_PROMO: a CONFIRMED promotion for this exact item and location inside W105-W116 that has not been cancelled.
- PACK_MIX_SHIFT: high pack_split and the sibling pack of the same item at the same DC moves the opposite way.
- WEEKLY_PHASING: high wk_split but small 4-week bucket error (the month total is right, the weeks are wrong).
- STRUCTURAL_NOISE: low forecastability, many zero weeks, lumpy orders.
- BIAS_DRIFT: steady erosion or growth without a clear step.
- NONE: nothing real is wrong (e.g. a split KPI moved only because a sibling item changed, or a promo was cancelled)."""


FALLBACK_MODELS = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3-flash-preview").split(",")
                   if m.strip()]
TRANSIENT_MARKERS = ("503", "unavailable", "high demand", "overloaded", "500 ", "internal error",
                     "deadline exceeded", "504", "429", "resource exhausted", "rate limit")
RETRY_WAITS = [5, 10, 20, 40, 60]       # seconds; ~2+ minutes of patience per call before falling back
RETRY_EVENTS: list = []                 # recorded in the run file: transient errors are part of the audit trail


def _is_transient(err: BaseException) -> bool:
    s = f"{type(err).__name__} {err}".lower()
    return any(m in s for m in TRANSIENT_MARKERS)


def make_llm(api_key: str | None = None, model: str | None = None, temperature: float = 0.1,
             fallbacks: list | None = None):
    """Gemini LLM that retries transient errors (503 overloaded, 429, 5xx) with backoff and, if the model
    stays unavailable, switches to the next fallback model. CrewAI itself only retries 429s."""
    from crewai.llms.providers.gemini.completion import GeminiCompletion

    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("No Gemini API key. Set GEMINI_API_KEY or paste it in the app sidebar.")
    os.environ["GEMINI_API_KEY"] = key
    chain = [model or DEFAULT_MODEL] + [m for m in (fallbacks if fallbacks is not None else FALLBACK_MODELS)
                                        if m != (model or DEFAULT_MODEL)]

    class RetryingGemini(GeminiCompletion):
        def call(self, *args, **kwargs):
            last = None
            while True:
                for i, wait in enumerate([0] + RETRY_WAITS):
                    if wait:
                        RETRY_EVENTS.append(dict(ts=time.strftime("%H:%M:%S"), model=self.model,
                                                 attempt=i, wait_s=wait, error=str(last)[:200]))
                        print(f"[retry] {self.model}: transient error, waiting {wait}s (attempt {i + 1})")
                        time.sleep(wait)
                    try:
                        return super().call(*args, **kwargs)
                    except Exception as ex:  # noqa: BLE001
                        if not _is_transient(ex):
                            raise
                        last = ex
                nxt = chain.index(self.model) + 1 if self.model in chain else len(chain)
                if nxt >= len(chain):
                    raise RuntimeError(f"Gemini stayed unavailable after retries on {chain}. "
                                       f"Wait a few minutes and re-run. Last error: {last}")
                print(f"[fallback] {self.model} unavailable -> switching to {chain[nxt]}")
                RETRY_EVENTS.append(dict(ts=time.strftime("%H:%M:%S"), model=self.model,
                                         switched_to=chain[nxt], error=str(last)[:200]))
                self.model = chain[nxt]

    return RetryingGemini(model=chain[0], api_key=key, temperature=temperature)


def build_crew(llm, max_rpm: int = 8, verbose: bool = True, step_callback=None):
    from crewai import Agent, Crew, Process, Task

    analyst = Agent(
        role="Demand Analyst",
        goal="Describe the demand data honestly: volume, volatility, intermittency and data-quality issues, "
             "so that nobody judges an IWL by a standard it cannot meet.",
        backstory="You are a demand analyst at Kaveri Foods, an Indian FMCG company with four distribution "
                  f"centres ({LOCS}). You never state a number you did not get from a tool.",
        tools=T.ANALYST_TOOLS, llm=llm, max_iter=4, allow_delegation=False, verbose=verbose,
        step_callback=step_callback)

    forecaster = Agent(
        role="Forecasting Agent",
        goal="Produce the statistical baseline forecast and report where it is least accurate and where it "
             "moved most since the last planning cycle.",
        backstory="You run the weekly statistical forecast. You report accuracy at a 4-week lag because that "
                  "is the frozen planning horizon. You never adjust numbers yourself; only tools produce forecasts.",
        tools=T.FORECAST_TOOLS, llm=llm, max_iter=4, allow_delegation=False, verbose=verbose,
        step_callback=step_callback)

    diagnostician = Agent(
        role="Forecast Diagnosis Agent",
        goal="For every IWL that the triage did not mark Low-Touch, find the most likely cause of the "
             "exception and the evidence for it, and correct the triage bucket where the evidence says so.",
        backstory="You are a sceptical demand-planning investigator. Similarity search returns notes that are "
                  "NEARBY, not necessarily CORRECT: you always check that a note is for the same item, the same "
                  "location and the right weeks, and whether a later note cancelled it. If no applicable note "
                  "exists you say 'no_evidence_found' rather than borrowing a note that does not apply. "
                  "A confident wrong cause is worse than an honest 'unexplained'.",
        tools=T.DIAGNOSIS_TOOLS, llm=llm, max_iter=12, allow_delegation=False, verbose=verbose,
        step_callback=step_callback)

    planner = Agent(
        role="Planning Recommendation Agent",
        goal="Turn each diagnosis into one concrete, reversible reforecast proposal that a human planner "
             "can verify and approve in under a minute.",
        backstory="You are a senior demand planner. You prefer the smallest change that fixes the cause. You "
                  "never chase a one-off spike, never add a promotion that is not confirmed, and for "
                  "structurally noisy items you recommend planning at an aggregate level instead of weekly fixes.",
        tools=T.PLANNING_TOOLS, llm=llm, max_iter=6, allow_delegation=False, verbose=verbose,
        step_callback=step_callback)

    t1 = Task(
        description="Call portfolio_profile once. Summarise: (1) data coverage, (2) which IWLs are "
                    "intermittent or low-volume and should not be judged on weekly accuracy, (3) any data-quality "
                    "flags such as recent outlier weeks. Under 180 words. Use only numbers from the tool.",
        expected_output="A short data profile with a list of structurally hard-to-forecast IWLs.",
        agent=analyst)

    t2 = Task(
        description="Call run_baseline_forecast once. Report portfolio WAPE and bias, the IWLs with the largest "
                    "4-week bucket errors, and the largest forecast changes vs the last cycle. Note one "
                    "limitation of the baseline method. Under 180 words.",
        expected_output="Accuracy summary, top error IWLs, top forecast movers, one limitation.",
        agent=forecaster, context=[t1])

    t3 = Task(
        description=f"""Step 1: call exception_triage once.
Step 2: for EVERY IWL whose engine bucket is not Low-Touch, call search_event_notes. Batch them: put several
queries in one call separated by ' | ' (use item name, pack and city, e.g. 'Rolled Oats 1kg Mumbai'). Use k=4.
Use iwl_history only if the KPI signals are ambiguous. Keep total tool calls under 8.
Step 3: decide for each flagged IWL: cause, final_bucket, evidence.
{CAUSE_GUIDE}
Evidence rules: cite a note only if it was returned by your searches AND its location matches the IWL's DC
(or is 'ALL'), its scope matches the item, and its dates fit. If a later note cancels or supersedes it, the
earlier note is not evidence; use evidence_status 'conflicting' or downgrade the bucket. If nothing applies,
evidence_status = 'no_evidence_found' and evidence_note_ids = [].
final_bucket may differ from engine_bucket; explain why in reasoning. Include every non-Low-Touch IWL.""",
        expected_output="A DiagnosisReport covering every non-Low-Touch IWL.",
        agent=diagnostician, context=[t1, t2], output_pydantic=DiagnosisReport)

    t4 = Task(
        description="""For every IWL in the DiagnosisReport choose an action:
- final_bucket Action-Required: relevel (recent_weeks = weeks since the shift started, 4-10) for level shifts,
  pack-mix shifts and drifts; cleanse_spike (spike_weeks from the outlier weeks) for one-off spikes;
  promo_uplift (weeks and uplift % taken from the confirmed note) for upcoming promotions.
- final_bucket Monitor: no_change, or rephase_weekly if the cause is WEEKLY_PHASING.
- final_bucket Structurally-Noisy: aggregate.
- final_bucket Low-Touch: no_change.
Call reforecast_iwls ONCE with a JSON list containing every IWL (one call, not one per IWL).
Then return one Recommendation per IWL. Copy horizon_change_pct from the tool output. params_json must be
the exact params you sent. Keep each rationale short and verifiable (KPI value + note id).""",
        expected_output="A RecommendationSet with one recommendation per diagnosed IWL.",
        agent=planner, context=[t3], output_pydantic=RecommendationSet)

    return Crew(agents=[analyst, forecaster, diagnostician, planner], tasks=[t1, t2, t3, t4],
                process=Process.sequential, verbose=verbose, max_rpm=max_rpm)


# ------------------------------------------------------------------ output parsing
def _coerce(model_cls, task_output):
    """Structure is enforceable, but providers sometimes return text: recover the JSON if needed."""
    if task_output is None:
        return None
    p = getattr(task_output, "pydantic", None)
    if p is not None:
        return p.model_dump()
    jd = getattr(task_output, "json_dict", None)
    if jd:
        return model_cls.model_validate(jd).model_dump()
    raw = getattr(task_output, "raw", "") or ""
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            return model_cls.model_validate(json.loads(m.group(0))).model_dump()
        except Exception:  # noqa: BLE001
            pass
    return None


def run_agents(api_key: str | None = None, model: str | None = None, max_rpm: int = 8,
               verbose: bool = True, llm=None, progress=None, fallbacks: list | None = None) -> dict:
    """Run the full crew and return a run record (also used by the Streamlit app)."""
    e = get_engine(reset=True)            # fresh state: empty tool log, no proposals
    trace = []
    RETRY_EVENTS.clear()

    def step_cb(step):
        txt = str(step)
        trace.append(dict(ts=time.strftime("%H:%M:%S"), step=txt[:3000]))
        if progress:
            progress(txt[:300])

    try:  # never block on CrewAI's interactive "view your traces?" prompt (Colab / terminal approvals)
        from crewai.events.listeners.tracing.utils import set_suppress_tracing_messages
        set_suppress_tracing_messages(True)
    except Exception:  # noqa: BLE001
        pass
    import warnings
    warnings.filterwarnings("ignore", message=".*callbacks cannot be serialized.*")
    llm = llm or make_llm(api_key, model, fallbacks=fallbacks)
    crew = build_crew(llm, max_rpm=max_rpm, verbose=verbose, step_callback=step_cb)
    t0 = time.time()
    crew.kickoff()
    secs = time.time() - t0
    outs = [t.output for t in crew.tasks]
    usage = {}
    try:
        um = crew.usage_metrics
        usage = um.model_dump() if hasattr(um, "model_dump") else dict(um)
    except Exception:  # noqa: BLE001
        pass
    return dict(
        mode="agents", model=getattr(llm, "model", str(model)), started=time.strftime("%Y-%m-%d %H:%M:%S"),
        seconds=round(secs, 1), usage=usage,
        analyst_summary=getattr(outs[0], "raw", ""), forecast_summary=getattr(outs[1], "raw", ""),
        diagnosis=_coerce(DiagnosisReport, outs[2]), diagnosis_raw=getattr(outs[2], "raw", ""),
        recommendations=_coerce(RecommendationSet, outs[3]), recommendations_raw=getattr(outs[3], "raw", ""),
        proposals=e.proposals, tool_log=e.tool_log, trace=trace, retry_events=list(RETRY_EVENTS))
