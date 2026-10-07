"""
Deterministic engine: everything numeric lives here, never in the LLM.

  * baseline statistical forecast (seasonal index x exponentially smoothed level)
  * the five demand-planning KPIs
  * the four IWL lenses (Forecastability, Business Criticality, Current Risk,
    Persistence) and the treatment-bucket logic
  * event-note retrieval (TF-IDF mini-RAG over event_notes.csv)
  * reforecast methods

The engine reads demand_history.csv, item_master.csv and event_notes.csv only.
It never reads future_actuals.csv or ground_truth.csv (evaluation.py does).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

N_HIST = 104                 # last history week (the "as-of" week for this cycle)
HORIZON = 12                 # forecast weeks 105..116
PRIOR_CYCLE_ORIGIN = 100     # last planning cycle, for forward-forecast-change
LAG = 4                      # accuracy measured at a 4-week lag (frozen horizon)
ALPHA = 0.08                 # smoothing of the statistical baseline
WIN = 8                      # exception review window = last 8 weeks
EXC_THRESHOLD = 0.25         # a week is an "exception week" if |error| > 25%

LOC_NAMES = {"MUM": "Mumbai", "DEL": "Delhi NCR", "BLR": "Bengaluru", "KOL": "Kolkata (East)"}
BUCKETS = ["Low-Touch", "Monitor", "Action-Required", "Structurally-Noisy"]
CAUSES = ["LEVEL_SHIFT_UP", "LEVEL_SHIFT_DOWN", "ONE_OFF_SPIKE", "UPCOMING_PROMO",
          "PACK_MIX_SHIFT", "WEEKLY_PHASING", "STRUCTURAL_NOISE", "BIAS_DRIFT", "NONE"]
METHODS = ["relevel", "promo_uplift", "cleanse_spike", "rephase_weekly", "aggregate", "no_change"]

# Treatment thresholds (documented in the design note; calibrated on this synthetic set)
TH = dict(noisy_forecastability=40, action_risk=50, action_persistence=45,
          action_fwd_change=0.10, monitor_risk=30, monitor_persistence=45, monitor_level_shift=12,
          low_criticality=30)


def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


@dataclass
class Engine:
    hist: pd.DataFrame = None
    items: pd.DataFrame = None
    notes: pd.DataFrame = None
    Y: dict = field(default_factory=dict)          # iwl -> np.array of 104 actuals
    SI: dict = field(default_factory=dict)         # category -> 52 seasonal indices
    meta: dict = field(default_factory=dict)       # iwl -> dict(item, sku, pack, cat, loc, ...)
    baseline: dict = field(default_factory=dict)   # iwl -> dict(level path, lag forecasts, fcst)
    triage: pd.DataFrame = None
    proposals: dict = field(default_factory=dict)  # iwl -> reforecast proposal
    tool_log: list = field(default_factory=list)
    _tfidf: object = None
    _note_mat: object = None

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, data_dir: str = DATA) -> "Engine":
        e = cls()
        e.hist = pd.read_csv(os.path.join(data_dir, "demand_history.csv"))
        e.items = pd.read_csv(os.path.join(data_dir, "item_master.csv"))
        e.notes = pd.read_csv(os.path.join(data_dir, "event_notes.csv"))
        e.hist = e.hist[e.hist.week <= N_HIST]  # safety: never look past the as-of week
        im = e.items.set_index("item_id")
        for iwl, g in e.hist.sort_values("week").groupby("iwl_id"):
            e.Y[iwl] = g.units.to_numpy(dtype=float)
            it = im.loc[g.item_id.iloc[0]]
            e.meta[iwl] = dict(item_id=g.item_id.iloc[0], sku=it.sku, pack=it.pack,
                               category=it.category, location=g.location.iloc[0],
                               service_class=it.service_class,
                               shelf_life_days=int(it.shelf_life_days),
                               unit_price_inr=float(it.unit_price_inr),
                               week_start=g.week_start.tolist())
        e._build_seasonality()
        e._build_retriever()
        return e

    def log(self, tool: str, args: dict, result: str):
        self.tool_log.append(dict(ts=time.strftime("%H:%M:%S"), tool=tool,
                                  args=json.dumps(args, default=str)[:300],
                                  result_chars=len(result), result=result))

    def iwls(self):
        return sorted(self.Y)

    def label(self, iwl):
        m = self.meta[iwl]
        return f"{m['sku']} {m['pack']} @ {LOC_NAMES[m['location']]}"

    # ----------------------------------------------------------- seasonality
    def _build_seasonality(self):
        """Category-pooled weekly seasonal index (52 positions), median of yearly-normalised ratios."""
        for cat in self.items.category.unique():
            ratios = []
            for iwl, y in self.Y.items():
                if self.meta[iwl]["category"] != cat or (y == 0).mean() > 0.2:
                    continue
                for yr in range(2):
                    blk = y[yr * 52:(yr + 1) * 52]
                    ratios.append(blk / blk.mean())
            si = np.median(np.vstack(ratios), axis=0)
            si = (np.roll(si, 1) + si + np.roll(si, -1)) / 3.0   # circular 3-week smoothing
            self.SI[cat] = si / si.mean()

    def si(self, iwl, t):
        """Seasonal index for week t (1-based, may exceed 104)."""
        return self.SI[self.meta[iwl]["category"]][(t - 1) % 52]

    # -------------------------------------------------------------- baseline
    def _level_path(self, y_deseas, alpha=ALPHA, skip=()):
        L = np.zeros(len(y_deseas))
        L[0] = y_deseas[:8].mean()
        for i in range(1, len(y_deseas)):
            if (i + 1) in skip:                       # week numbers to ignore (cleansing)
                L[i] = L[i - 1]
            else:
                L[i] = alpha * y_deseas[i] + (1 - alpha) * L[i - 1]
        return L

    def run_baseline(self):
        for iwl, y in self.Y.items():
            t = np.arange(1, N_HIST + 1)
            si = np.array([self.si(iwl, k) for k in t])
            d = y / si
            L = self._level_path(d)
            lagf = np.full(N_HIST, np.nan)            # forecast for week k made at week k-LAG
            for k in range(LAG + 1, N_HIST + 1):
                lagf[k - 1] = L[k - 1 - LAG] * si[k - 1]
            fut_t = np.arange(N_HIST + 1, N_HIST + HORIZON + 1)
            fcst = L[N_HIST - 1] * np.array([self.si(iwl, k) for k in fut_t])
            prior = L[PRIOR_CYCLE_ORIGIN - 1] * np.array([self.si(iwl, k) for k in fut_t])
            self.baseline[iwl] = dict(deseas=d, level=L, lag_fcst=lagf, fcst=fcst,
                                      prior_cycle_fcst=prior)
        return self.baseline

    # ------------------------------------------------------------------ KPIs
    def _kpis(self, iwl):
        y, b, m = self.Y[iwl], self.baseline[iwl], self.meta[iwl]
        f = b["lag_fcst"]
        w = slice(N_HIST - WIN, N_HIST)                  # weeks 97..104
        w4 = slice(N_HIST - 4, N_HIST)
        fw, yw = f[w], y[w]
        wape8 = np.abs(fw - yw).sum() / max(yw.sum(), 1)
        bias8 = (fw - yw).sum() / max(yw.sum(), 1)
        bucket_err4 = abs(f[w4].sum() - y[w4].sum()) / max(y[w4].sum(), 1)
        bias4 = (f[w4] - y[w4]).sum() / max(y[w4].sum(), 1)
        prior_w = slice(N_HIST - 16, N_HIST - 8)         # weeks 89..96
        bias_prior = (f[prior_w] - y[prior_w]).sum() / max(y[prior_w].sum(), 1)

        # KPI 1: weekly split deviation (pp) — share of each week inside 4-week blocks
        devs = []
        for s in (N_HIST - 8, N_HIST - 4):
            fb, yb = f[s:s + 4], y[s:s + 4]
            if yb.sum() > 0 and fb.sum() > 0:
                devs.append(np.abs(fb / fb.sum() - yb / yb.sum()).mean())
        weekly_split = 100 * float(np.mean(devs)) if devs else 0.0

        # KPI 2: SKU split deviation (pp) — SKU share within category at this location
        # KPI 3: package-to-SKU split deviation (pp) — pack share within SKU at this location
        def share(level_key, level_val, within_key, within_val, arr_name):
            num = den = 0.0
            for j, mj in self.meta.items():
                if mj["location"] != m["location"] or mj[within_key] != within_val:
                    continue
                arr = self.Y[j][w] if arr_name == "a" else self.baseline[j]["lag_fcst"][w]
                den += arr.sum()
                if mj[level_key] == level_val:
                    num += arr.sum()
            return num / den if den else 0.0
        sku_split = 100 * abs(share("sku", m["sku"], "category", m["category"], "f")
                              - share("sku", m["sku"], "category", m["category"], "a"))
        pack_split = 100 * abs(share("item_id", m["item_id"], "sku", m["sku"], "f")
                               - share("item_id", m["item_id"], "sku", m["sku"], "a"))

        # KPI 4: forward forecast change — this cycle vs last cycle, overlapping weeks 105..112
        fwd_change = b["fcst"][:8].sum() / max(b["prior_cycle_fcst"][:8].sum(), 1e-9) - 1
        # KPI 5: bias change (pp) — last 4 weeks vs the 8 weeks before the review window
        bias_change = 100 * (bias4 - bias_prior)

        # supporting signals (measurements, not diagnoses)
        err = (fw - yw) / np.maximum(fw, 1)
        exc = np.abs(err) > EXC_THRESHOLD
        run, sign = 0, np.sign(err[-1])
        for e_ in err[::-1]:
            if np.sign(e_) == sign and abs(e_) > 0.05:
                run += 1
            else:
                break
        d = b["deseas"]
        ref = d[N_HIST - 34:N_HIST - 8]                  # weeks 71..96
        level_shift = d[N_HIST - 6:].mean() / max(ref.mean(), 1e-9) - 1
        recent_z = (d[N_HIST - 8:] - ref.mean()) / max(ref.std(), 1e-9)
        spike_weeks = [int(N_HIST - 8 + 1 + i) for i, z in enumerate(recent_z) if z > 3]

        return dict(wape_8w=wape8, bias_8w=bias8, bucket_err_4w=bucket_err4,
                    weekly_split_dev_pp=weekly_split, sku_split_dev_pp=sku_split,
                    pack_split_dev_pp=pack_split, fwd_fcst_change_pct=100 * fwd_change,
                    bias_change_pp=bias_change, exception_weeks_8w=int(exc.sum()),
                    exceptions_last4=int(exc[-4:].sum()), same_sign_run=int(run),
                    level_shift_pct=100 * level_shift, spike_weeks=spike_weeks)

    # ---------------------------------------------------------- four lenses
    def forward_events(self, iwl):
        """Structured metadata check: active promo for this item+location inside the horizon."""
        m = self.meta[iwl]
        n = self.notes
        hit = n[(n.type == "promo") & (n.location == m["location"]) & (n.item_scope == m["item_id"])
                & (n.status == "active") & (n.effective_from_week.between(N_HIST + 1, N_HIST + HORIZON))]
        return hit.note_id.tolist()

    def _lenses(self, iwl, k):
        y, m = self.Y[iwl], self.meta[iwl]
        d = self.baseline[iwl]["deseas"]
        stable = d[52:N_HIST - 8]                        # weeks 53..96: before the review window
        cov = stable.std() / max(stable.mean(), 1e-9)
        zero_share = float((y[52:] == 0).mean())
        # split stability: std of pack share within SKU, 4-week blocks, weeks 53..96
        sib = [j for j, mj in self.meta.items() if mj["location"] == m["location"] and mj["sku"] == m["sku"]]
        tot = sum(self.Y[j] for j in sib)
        shares = [y[s:s + 4].sum() / max(tot[s:s + 4].sum(), 1) for s in range(52, N_HIST - 8, 4)]
        split_std = float(np.std(shares))
        forecastability = 100 * (0.45 * _clip01(1 - cov / 0.6) + 0.30 * (1 - _clip01(zero_share / 0.4))
                                 + 0.10 * min(1, len(y) / 52) + 0.15 * _clip01(1 - split_std / 0.10))

        rev = {j: self.Y[j][-26:].sum() * self.meta[j]["unit_price_inr"] for j in self.Y}
        vol_pct = sum(v <= rev[iwl] for v in rev.values()) / len(rev)
        svc = {"A": 1.0, "B": 0.6, "C": 0.2}[m["service_class"]]
        perish = _clip01((365 - m["shelf_life_days"]) / 275)
        criticality = 100 * (0.5 * vol_pct + 0.3 * svc + 0.2 * perish)

        fwd_ev = self.forward_events(iwl)
        severity = _clip01(k["bucket_err_4w"] / 0.25)
        recency = k["exceptions_last4"] / 4
        forward = max(_clip01(abs(k["fwd_fcst_change_pct"]) / 25), _clip01(k["weekly_split_dev_pp"] / 12),
                      _clip01(k["pack_split_dev_pp"] / 12), 1.0 if fwd_ev else 0.0)
        risk = 100 * (0.5 * severity + 0.2 * recency + 0.3 * forward)

        persistence = 100 * (0.5 * k["exception_weeks_8w"] / WIN + 0.5 * min(k["same_sign_run"], WIN) / WIN)
        return dict(forecastability=forecastability, criticality=criticality, risk=risk,
                    persistence=persistence, cov=cov, zero_share=zero_share,
                    forward_event_notes=fwd_ev)

    def _bucket(self, L, k):
        if L["forecastability"] < TH["noisy_forecastability"]:
            return "Structurally-Noisy", "forecastability below threshold"
        if L["forward_event_notes"]:
            return "Action-Required", f"active promo note in horizon: {','.join(L['forward_event_notes'])}"
        act = L["risk"] >= TH["action_risk"] and (
            L["persistence"] >= TH["action_persistence"] or abs(k["fwd_fcst_change_pct"]) >= 100 * TH["action_fwd_change"])
        if act and L["criticality"] < TH["low_criticality"]:
            return "Monitor", "action-level signal but low business criticality"
        if act:
            return "Action-Required", "high current risk with persistence or forward forecast change"
        if L["risk"] >= TH["monitor_risk"] or L["persistence"] >= TH["monitor_persistence"]:
            return "Monitor", "moderate risk or repeating errors"
        if abs(k["level_shift_pct"]) >= TH["monitor_level_shift"]:
            return "Monitor", "deseasonalised level has moved; errors not yet persistent"
        return "Low-Touch", "stable and low risk"

    def run_triage(self):
        if not self.baseline:
            self.run_baseline()
        rows = []
        for iwl in self.iwls():
            k = self._kpis(iwl)
            L = self._lenses(iwl, k)
            bucket, why = self._bucket(L, k)
            m = self.meta[iwl]
            rows.append(dict(iwl_id=iwl, item=f"{m['sku']} {m['pack']}", location=m["location"],
                             service_class=m["service_class"], **k, **L, bucket=bucket, bucket_reason=why))
        self.triage = pd.DataFrame(rows)
        return self.triage

    # ------------------------------------------------------------- retrieval
    def _note_doc(self, r):
        loc = LOC_NAMES.get(r.location, "all regions")
        return f"{r.title}. {r.text} Location: {loc}. Scope: {r.item_scope}. Type: {r.type}."

    def _build_retriever(self):
        docs = [self._note_doc(r) for r in self.notes.itertuples()]
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self._note_mat = self._tfidf.fit_transform(docs)

    def search_notes(self, query: str, k: int = 3):
        """Similarity search only. It does not know about cancellations or locations — by design."""
        q = self._tfidf.transform([query])
        s = cosine_similarity(q, self._note_mat).ravel()
        order = np.argsort(-s)[:k]
        out = []
        for rank, i in enumerate(order, 1):
            r = self.notes.iloc[i]
            out.append(dict(rank=rank, score=round(float(s[i]), 3), note_id=r.note_id,
                            published=f"W{int(r.published_week):03d}", type=r.type, location=r.location,
                            item_scope=r.item_scope,
                            effective=f"W{int(r.effective_from_week):03d}-" + (
                                f"W{int(r.effective_to_week):03d}" if pd.notna(r.effective_to_week) else "open")
                            if pd.notna(r.effective_from_week) else "n/a",
                            status=r.status, title=r.title, text=r.text))
        return out

    # ------------------------------------------------------------ reforecast
    def reforecast(self, iwl, method, params=None):
        params = params or {}
        b, y = self.baseline[iwl], self.Y[iwl]
        fut_t = np.arange(N_HIST + 1, N_HIST + HORIZON + 1)
        si_f = np.array([self.si(iwl, k) for k in fut_t])
        base = b["fcst"].copy()
        d = b["deseas"]
        if method == "relevel":
            n = int(params.get("recent_weeks", 6))
            n = int(np.clip(n, 3, 26))
            new = d[-n:].mean() * si_f
        elif method == "promo_uplift":
            w0, w1 = int(params.get("from_week", 0)), int(params.get("to_week", 0))
            up = float(params.get("uplift_pct", 0)) / 100
            new = base.copy()
            for i, t in enumerate(fut_t):
                if w0 <= t <= w1:
                    new[i] *= (1 + up)
        elif method == "cleanse_spike":
            weeks = [int(x) for x in params.get("spike_weeks", [])]
            skip = set(weeks) | {w + 1 for w in weeks if w + 1 <= N_HIST and params.get("include_dip", True)}
            L = self._level_path(d, skip=skip)
            new = L[-1] * si_f
        elif method == "rephase_weekly":
            # learn week-of-month profile from the last 12 weeks, keep 4-week totals unchanged
            ws = self.meta[iwl]["week_start"]
            wom = lambda s: 1 if int(s[-2:]) <= 7 else 2  # noqa: E731  first week of month vs rest
            recent = [(wom(ws[i]), d[i]) for i in range(N_HIST - 12, N_HIST)]
            m1 = np.mean([v for g, v in recent if g == 1]) if any(g == 1 for g, _ in recent) else 1
            m2 = np.mean([v for g, v in recent if g == 2]) if any(g == 2 for g, _ in recent) else 1
            fdates = pd.date_range(pd.Timestamp(ws[-1]) + pd.Timedelta(weeks=1), periods=HORIZON, freq="7D")
            prof = np.array([m1 if dt.day <= 7 else m2 for dt in fdates])
            prof = prof / prof.mean()
            new = base * prof
        elif method == "aggregate":
            new = d[-52:].mean() * si_f        # long-window level x seasonality: stop chasing weekly noise
        elif method == "no_change":
            new = base.copy()
        else:
            raise ValueError(f"unknown method {method}; allowed {METHODS}")
        new = np.maximum(new, 0)
        self.proposals[iwl] = dict(method=method, params=params, baseline=base.round(1).tolist(),
                                   reforecast=new.round(1).tolist())
        return self.proposals[iwl]


_ENGINE = None


def get_engine(reset: bool = False) -> Engine:
    global _ENGINE
    if _ENGINE is None or reset:
        _ENGINE = Engine.load()
        _ENGINE.run_baseline()
        _ENGINE.run_triage()
    return _ENGINE
