# Agentic Demand Forecasting & Forecast Exception Management

**AI for Managers (MBAEx, IIM Calcutta) · Group project · Agentic workflow track (Session 7)**

```
Demand Data → Forecast → Detect Exception → Investigate → Reforecast → Recommend → Human Approval
   Agent 1      Agent 2   ─────────── Agent 3 ───────────   ─── Agent 4 ───      planner (UI / CLI)
```

Four CrewAI agents on Gemini, seven tools, one human approval step. A demand planner's weekly
exception review: 40 Item-Week-Locations (IWLs) are triaged through four lenses (Forecastability,
Business Criticality, Current Risk, Persistence) into four treatment buckets. Only the IWLs that need a
human are investigated, explained with evidence from commercial event notes, and given a reforecast
proposal that a planner approves, edits or rejects.

All data is **synthetic**. *Kaveri Foods Pvt Ltd* is fictional; no real company, retailer or person is
described and no client data was used.

---

## 1. Run it

| Option | Who it is for | Steps |
|---|---|---|
| **Hosted app** | Anyone with the link | Open the Streamlit link → paste a Gemini key in the sidebar (or use the owner's key if configured) → tab 4 *Run agents* → tab 5 approve → tab 6 evaluation |
| **Colab** | Professor / other participants | Open `notebooks/run_in_colab.ipynb` in Colab → run all cells (uploads the zip or clones the repo) |
| **Local** | Builders | `pip install -r requirements.txt` then the commands below |

```bash
python data/generate_data.py               # (optional) regenerate the synthetic data, seed 42
python run.py --mode rules                 # rule-based baseline, no API key, < 1 second
export GEMINI_API_KEY=...                  # free key: https://aistudio.google.com
python run.py --mode agents --approve ask  # 4-agent crew + approve/reject/edit in the terminal
python run.py --report runs/<file>.json    # re-score a saved run
streamlit run streamlit_app.py             # the app
python tests/test_crew_wiring.py           # offline wiring test (no key, scripted stand-in LLM)
```

Free-tier notes: a run makes roughly 10–20 model calls and takes 2–5 minutes. `--max-rpm` (CLI) or
the sidebar slider keeps requests under the free-tier limit. If you get a 404 for the model name or
quota errors, pass `--model <name>` with any model your key lists in AI Studio (default `gemini-3.8-flash`; Google retired `gemini-2.5-flash` for new keys).

### Deploy (what "deployed" means for this brief)
1. Push this folder to a public GitHub repo.
2. share.streamlit.io → *New app* → pick the repo → main file `streamlit_app.py` → *Advanced settings*: Python 3.12.
3. Optional: *Secrets* → `GEMINI_API_KEY = "..."` so visitors can run without their own key (uses your quota).
4. Commit at least one real agent run JSON into `runs/` so the app shows agent output even when quota is exhausted.

---

## 2. What is in the repo

```
data/generate_data.py      synthetic data generator (seed 42) + the planted exceptions
data/*.csv                 demand_history (40 IWLs x 104 wks), item_master, event_notes (20),
                           future_actuals (W105-116, EVALUATION ONLY), ground_truth (EVALUATION ONLY)
src/engine.py              deterministic core: baseline forecast, 5 KPIs, 4 lenses, buckets,
                           TF-IDF note retrieval, reforecast methods
src/tools.py               the 7 CrewAI tools (docstrings are prompts) + tool-call log
src/schemas.py             pydantic output schemas for agents 3 and 4
src/crew.py                the 4 agents, 4 tasks, sequential crew, Gemini LLM
src/rules.py               rule-based baseline (same tools, no LLM) - comparison + no-key fallback
src/evaluation.py          4-layer evaluation (detection, retrieval, diagnosis, outcome)
run.py                     CLI runner; saves every run to runs/*.json (audit trail)
streamlit_app.py           the deployed app (7 tabs incl. human approval and evaluation)
notebooks/run_in_colab.ipynb
tests/test_crew_wiring.py  proves agents->tools->schemas wiring without an API key
docs/Design_Note_Demand_Exception_Agents.docx   design note draft (yellow [FILL] = your run's numbers)
docs/architecture.png / .svg / .dot             the single architecture diagram
```

### Agents and tools

| Agent | Tools | Output |
|---|---|---|
| 1 Demand Analyst | `portfolio_profile`, `iwl_history` | data profile, hard-to-forecast IWLs |
| 2 Forecasting | `run_baseline_forecast`, `forecast_detail` | accuracy at 4-week lag, top errors and movers |
| 3 Forecast Diagnosis | `exception_triage`, `search_event_notes`, `iwl_history` | `DiagnosisReport`: cause, final bucket, cited notes, evidence status |
| 4 Planning Recommendation | `reforecast_iwls`, `forecast_detail` | `RecommendationSet`: action, params, horizon change, rationale |
| Human planner | app tab 5 / `--approve ask` | approve / edit / reject → `runs/*.json` + `runs/audit_log.csv` |

Design rule: **numbers come from tools, judgement from the model.** The LLM never computes a KPI or a
forecast. Process is **sequential** (workflow with agentic steps), not hierarchical: errors are costly and
need an audit trail.

### The five KPIs (window W097–W104, forecast at a 4-week lag)

| KPI | Definition |
|---|---|
| Weekly split deviation (pp) | mean \|forecast weekly share − actual weekly share\| inside each 4-week block |
| SKU split deviation (pp) | \|forecast SKU share − actual SKU share\| within category at the DC |
| Package-to-SKU split (pp) | \|forecast pack share − actual pack share\| within the SKU at the DC |
| Forward forecast change (%) | this cycle's W105–112 forecast vs last cycle's (as of W100) |
| Bias change (pp) | bias of the last 4 weeks − bias of W089–W096 |

### Four lenses → treatment bucket
* **Forecastability** (volatility, zero-week share, history depth, pack-split stability)
* **Business criticality** (revenue rank, service class A/B/C, shelf life / waste sensitivity)
* **Current risk** (4-week bucket error, recency of exception weeks, forward-looking KPIs, active promo notes in horizon)
* **Persistence** (exception weeks in last 8, consecutive same-sign errors)

Structurally-Noisy if Forecastability < 40 → Action-Required if an active promo note is in the horizon, or
Risk ≥ 50 with Persistence ≥ 45 or forward forecast change ≥ 10% (downgraded to Monitor if criticality < 30)
→ Monitor if Risk ≥ 30, Persistence ≥ 45 or level shift ≥ 12% → otherwise Low-Touch.

---

## 3. Data and planted exceptions

40 IWLs = 10 items (5 SKUs × 2 packs) × 4 DCs (Mumbai, Delhi NCR, Bengaluru, Kolkata), weekly shipments
W001–W104 (w/c 7 Oct 2024 – 28 Sep 2026), festive seasonality. 14 IWLs carry a planted condition:

| IWL | Planted condition | Correct outcome | Why it is hard |
|---|---|---|---|
| OAT1K-MUM | listing gain, +40% from W97 | Action, relevel | — |
| MUS400-DEL | competitor launch, −30 to −38% | Action, relevel | sibling pack's split KPI moves too |
| GRA12-MUM | −30% step, **no note explains it** | Action, relevel, *no evidence* | top search hit is an old Delhi stock-out note |
| GRA6-BLR | BOGO spike W101–102 | Action, **cleanse** spike | spike has leaked into the baseline |
| POP150-MUM | confirmed promo W106–108 | Action, promo uplift | KPIs cannot see the future |
| POP150-DEL | promo planned (EV05) then **cancelled** (EV06) | Low-Touch, no change | EV05 still says "active"; it outranks EV06 |
| OAT500-KOL / OAT1K-KOL | price rise moves mix 500g → 1kg | Action, relevel both | top search hit for Kolkata 1kg is the *Mumbai* listing |
| MUS750-MUM | monthly order cycle (weekly phasing) | Monitor | month total is right, weeks are wrong |
| CAK250-KOL, CAK500-KOL, CAK500-BLR | intermittent institutional demand | Structurally-Noisy, aggregate | weekly fixes cannot help |
| GRA12-DEL | slow erosion, no event | Monitor | — |
| OAT500-BLR | nothing (distractor notes exist) | Low-Touch | Mumbai-only offer and a national article rank top |

---

## 4. Evaluation (retrieval quality and answer quality are measured separately)

1. **Detection** — triage vs planted exceptions: precision / recall / bucket accuracy. No LLM.
2. **Retrieval** — one standard query per IWL: is the explaining note in the top 3? Is the top hit misleading? No LLM.
3. **Diagnosis** — cause accuracy, bucket accuracy, and **citation checks**: is each cited note real, was it
   actually returned by a search in this run (tool log), and does it apply to that item/location/date?
4. **Outcome** — on the hidden weeks W105–W116: WAPE of baseline vs reforecast, (a) if every
   recommendation were auto-applied, (b) with the human's approvals.

Reference numbers from the deterministic parts (reproducible, seed 42):

| Measure | Value |
|---|---|
| Triage recall / precision | 100% / 86% (2 false positives: MUS750-DEL sibling effect, POP150-DEL cancelled promo) |
| Review queue | 14 of 40 IWLs (26 auto Low-Touch) |
| Retrieval hit@3 | 100% (10 IWLs with an explaining note) — but **top-1 is misleading for 7 of 14** |
| Rules baseline: cause accuracy / citation precision | 71% / 50% |
| Rules baseline: portfolio WAPE W105–116 | 10.1% → 8.8% (the cancelled promo is wrongly uplifted: 5% → 20% WAPE on POP150-DEL) |

Agent numbers come from your own live run — see `runs/` and app tab 6. Do not quote the wiring test.

---

## 5. Known limitations (state these precisely)
* Synthetic data; thresholds were calibrated on the same data they are evaluated on. Out-of-sample behaviour is unknown.
* The baseline is a simple seasonal-index × smoothed-level model, chosen for transparency, not accuracy.
* Similarity retrieval (TF-IDF) does not understand location, dates or cancellation; the Diagnosis agent is told to check, and the evaluation measures whether it did.
* LLM output is non-deterministic: two runs can differ. Run at least twice and report both.
* Structurally noisy IWLs stay at ~80–90% weekly WAPE whatever the method; the value is removing them from weekly review, not fixing them.
* Free-tier Gemini limits the number of runs per day.

---

## 6. Team checklist for submission day
1. Get a Gemini key (aistudio.google.com) and run `python run.py --mode agents --approve ask` **twice**. Both runs save to `runs/`.
2. Open the app (`streamlit run streamlit_app.py`), load each run, approve/reject in tab 5, read tab 6. Copy the numbers into the
   yellow `[FILL]` cells of the design note. Pick the surprising result your runs show most clearly.
3. Push the repo (including your two agent runs in `runs/`) to GitHub and deploy on Streamlit Community Cloud. Put both links in the note.
4. Record a backup screen capture of one full live run in case the free-tier quota is exhausted during the demo.
5. Submit: repo link + app link + design note (export the .docx to PDF, max 6 pages).

## 7. Fifteen-minute demo plan
| Min | What | Where |
|---|---|---|
| 0–2 | The planner's problem: noisy exception queue; whose problem; cost of a wrong call | slide / tab 1 |
| 2–4 | Data and the four lenses; 40 IWLs → 14 in the queue | tabs 2–3 |
| 4–8 | Live run of the crew; narrate the tool calls agent by agent | tab 4 (or a saved run if quota is low) |
| 8–10 | Human approval: approve a level shift, edit a promo uplift, reject one; show the audit log | tabs 5, 7 |
| 10–13 | Evaluation: retrieval vs answer quality; agents vs rules; **the surprising result** | tab 6 |
| 13–15 | Governance and limitations; what we would need to deploy it on real data | design note |
