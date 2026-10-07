"""
Offline wiring test for the CrewAI crew — NO API key, NO real model.

A scripted stand-in LLM issues native tool calls the way Gemini does, so this proves that
agents, tools, task context, the tool log and the pydantic outputs are wired correctly.
It says nothing about answer quality: its "answers" are copied from the rule-based baseline.
Never report numbers from this test as agent results.

  python tests/test_crew_wiring.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from crewai.llms.base_llm import BaseLLM  # noqa: E402

from crew import run_agents  # noqa: E402
from rules import run_rules  # noqa: E402

RULES = run_rules()

SCRIPT = {
    "Demand Analyst": [("portfolio_profile", {})],
    "Forecasting Agent": [("run_baseline_forecast", {})],
    "Forecast Diagnosis Agent": [("exception_triage", {}),
                                 ("search_event_notes", {"query": "Rolled Oats 1kg Mumbai | Masala Popcorn 150g Delhi",
                                                         "k": 4})],
    "Planning Recommendation Agent": [("reforecast_iwls", {"requests_json": json.dumps(
        [dict(iwl_id=r["iwl_id"], method=r["action"], params=json.loads(r["params_json"]))
         for r in RULES["recommendations"]["recommendations"]])})],
}
FINAL = {
    "Demand Analyst": "Stub profile.",
    "Forecasting Agent": "Stub forecast summary.",
    "Forecast Diagnosis Agent": json.dumps(RULES["diagnosis"]),
    "Planning Recommendation Agent": json.dumps(RULES["recommendations"]),
}


class ScriptedLLM(BaseLLM):
    calls: int = 0

    def supports_function_calling(self) -> bool:
        return True

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None):
        self.calls += 1
        role = getattr(from_agent, "role", None)
        msgs = messages if isinstance(messages, list) else [{"role": "user", "content": messages}]
        n_tool_results = sum(1 for m in msgs if m.get("role") == "tool")
        if response_model is not None:            # converter / structured-output call
            text = FINAL.get(role) or next((v for v in FINAL.values() if v.startswith("{") and
                                            set(response_model.model_fields) <= set(json.loads(v))), "{}")
            return response_model.model_validate_json(text)
        plan = SCRIPT.get(role, [])
        if tools and n_tool_results < len(plan):
            name, args = plan[n_tool_results]
            return [{"id": f"call_{self.calls}", "function": {"name": name, "arguments": json.dumps(args)}}]
        return FINAL.get(role, "done")


if __name__ == "__main__":
    llm = ScriptedLLM(model="scripted-test")
    run = run_agents(llm=llm, verbose=False, max_rpm=1000)
    tools_called = [c["tool"] for c in run["tool_log"]]
    print("LLM calls:", llm.calls)
    print("tools called:", tools_called)
    assert {"portfolio_profile", "run_baseline_forecast", "exception_triage", "search_event_notes",
            "reforecast_iwls"} <= set(tools_called), "a tool was never reached"
    assert run["diagnosis"] and len(run["diagnosis"]["items"]) == 14, "diagnosis schema not parsed"
    assert run["recommendations"] and len(run["recommendations"]["recommendations"]) == 14, "recs not parsed"
    assert len(run["proposals"]) == 14, "reforecast proposals not stored"
    print("WIRING TEST PASSED")
