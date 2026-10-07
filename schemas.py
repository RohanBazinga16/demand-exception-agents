"""Output schemas the agents must fill. Structure is enforceable; truth is not —
evaluation.py checks the content (e.g. that every cited note was actually retrieved)."""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

Bucket = Literal["Low-Touch", "Monitor", "Action-Required", "Structurally-Noisy"]
Cause = Literal["LEVEL_SHIFT_UP", "LEVEL_SHIFT_DOWN", "ONE_OFF_SPIKE", "UPCOMING_PROMO",
                "PACK_MIX_SHIFT", "WEEKLY_PHASING", "STRUCTURAL_NOISE", "BIAS_DRIFT", "NONE"]
Method = Literal["relevel", "promo_uplift", "cleanse_spike", "rephase_weekly", "aggregate", "no_change"]
Evidence = Literal["supported", "no_evidence_found", "conflicting"]
Confidence = Literal["low", "medium", "high"]


class DiagnosisItem(BaseModel):
    iwl_id: str = Field(description="IWL id exactly as in the triage table, e.g. OAT1K-MUM")
    engine_bucket: Bucket = Field(description="Bucket assigned by the deterministic triage tool")
    final_bucket: Bucket = Field(description="Your bucket after investigation (may differ from engine_bucket)")
    cause: Cause = Field(description="Most likely cause of the exception, from the allowed list")
    evidence_note_ids: List[str] = Field(default_factory=list,
                                         description="Event note ids that support the cause. Only ids returned by "
                                                     "search_event_notes in this run AND applicable to this item, "
                                                     "location and dates. Empty if none.")
    evidence_status: Evidence = Field(description="supported / no_evidence_found / conflicting")
    reasoning: str = Field(description="Two or three sentences citing the KPI values and notes used")
    confidence: Confidence


class DiagnosisReport(BaseModel):
    items: List[DiagnosisItem]
    summary: str = Field(description="Three to five sentences for the planning lead")


class Recommendation(BaseModel):
    iwl_id: str
    final_bucket: Bucket
    cause: Cause
    action: Method = Field(description="Reforecast method that was run with reforecast_iwls")
    params_json: str = Field(description='Parameters passed to the reforecast, as JSON, e.g. {"recent_weeks": 6}')
    evidence_note_ids: List[str] = Field(default_factory=list)
    horizon_change_pct: float = Field(description="Change in 12-week forecast total vs baseline, in percent, "
                                                  "as reported by the reforecast tool")
    rationale: str = Field(description="One to three sentences a planner can verify")
    confidence: Confidence


class RecommendationSet(BaseModel):
    recommendations: List[Recommendation]
    planner_summary: str = Field(description="What the planner should approve first, and why, in under 120 words")
